"""Compile the CUDA device code to PTX without a GPU.

``clang++`` in CUDA mode can lower ``__global__`` code to PTX for a named
architecture given only CUDA's headers and libdevice -- no driver, no device,
and no ``nvcc``. The headers and libdevice are available from PyPI
(``nvidia-cuda-runtime-cu12``, ``nvidia-cuda-nvcc-cu12``,
``nvidia-cuda-cccl-cu12``, ``nvidia-curand-cu12``), so the handwritten kernel
can be checked in ordinary CPU-only CI rather than sitting unverified until
someone with a GPU builds it.

Only ``rmsnorm_kernel.cuh`` is compiled this way. The ATen launch site in
``rmsnorm.cu`` cannot be: ATen requires C++20, and CUDA's ``crt/host_defines.h``
collides with libstdc++'s use of ``__attribute__((__noinline__))`` when the
CUDA headers are processed first, which is the collision that header's own
comments describe. nvcc avoids it; clang does not. That side is boilerplate
and is compiled by ``torch.utils.cpp_extension.load`` on first use.
"""

from __future__ import annotations

import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path

CSRC = Path(__file__).resolve().parent.parent / "kernelforge" / "csrc"

#: Subdirectories of the installed ``nvidia`` namespace package that carry the
#: headers clang's CUDA wrapper expects to find in a toolkit.
_WHEEL_INCLUDE_DIRS = (
    "cuda_runtime/include",
    "cuda_nvcc/include",
    "cuda_cccl/include",
    "curand/include",
)


@dataclass(frozen=True)
class CudaToolchain:
    compiler: Path
    cuda_path: Path
    stdlib_includes: tuple[Path, ...]
    #: Which libstdc++ was selected, for the test report. The choice is
    #: load-bearing (see `_libstdcxx_candidates`) so it must not be invisible.
    stdlib_label: str = "clang default"


def _nvidia_root() -> Path | None:
    site = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if site.is_dir():
        return site
    # pip --user installs land outside purelib.
    for candidate in Path.home().glob(".local/lib/python3.*/site-packages/nvidia"):
        if candidate.is_dir():
            return candidate
    return None


def _libstdcxx_candidates() -> list[tuple[str, tuple[Path, ...]]]:
    """Candidate libstdc++ header sets to try, newest version first.

    Clang has to be told where libstdc++ lives, and *which* version it picks
    matters more than it looks. Clang's CUDA wrapper header includes ``<cmath>``
    for every CUDA translation unit, so the standard library is pulled into the
    NVPTX device compilation whether the kernel uses it or not. libstdc++ 14's
    ``<limits>`` then declares ``numeric_limits<__float128>`` unconditionally --
    the macro that guards it is baked into the installed headers when GCC is
    built for x86 -- and ``__float128`` does not exist on NVPTX, so the device
    pass fails with 18 errors before reaching any of our code.

    Taking the newest version therefore cannot be right: it works on a runner
    with GCC 13 and fails on one with GCC 14. No compiler flag avoids it
    (``-std=`` variants, ``-nostdinc++``, ``--gcc-toolchain`` and letting clang
    choose were all tried and all fail), so the only option is to pick a
    version that works -- which means trying them, not assuming.

    The empty set is included last: on a machine where clang's default happens
    to be compatible, nothing needs to be passed at all.
    """
    base = Path("/usr/include/c++")
    candidates: list[tuple[str, tuple[Path, ...]]] = []
    if base.is_dir():
        for version in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
            paths = (
                version,
                Path("/usr/include/x86_64-linux-gnu/c++") / version.name,
            )
            candidates.append((f"libstdc++-{version.name}", tuple(p for p in paths if p.is_dir())))
    candidates.append(("clang default", ()))
    return candidates


#: The smallest CUDA translation unit there is. Enough to decide a toolchain:
#: clang's CUDA wrapper drags in the standard library regardless of content, so
#: an empty kernel fails on an incompatible libstdc++ exactly as the real one
#: does, in a fraction of the time.
_PROBE_SOURCE = "__global__ void kernelforge_toolchain_probe() {}\n"


def _assemble_cuda_path(destination: Path) -> Path | None:
    """Build the directory layout clang expects from a real CUDA install.

    A system toolkit is used as-is when present; otherwise the pieces are
    gathered from the PyPI wheels into ``destination`` by symlink.
    """
    system = Path("/usr/local/cuda")
    if (system / "nvvm/libdevice").is_dir() and (system / "include/cuda_runtime.h").exists():
        return system

    nvidia = _nvidia_root()
    if nvidia is None:
        return None
    libdevice = nvidia / "cuda_nvcc/nvvm/libdevice"
    if not libdevice.is_dir():
        return None

    include = destination / "include"
    include.mkdir(parents=True, exist_ok=True)
    found_runtime = False
    for relative in _WHEEL_INCLUDE_DIRS:
        source = nvidia / relative
        if not source.is_dir():
            continue
        for entry in source.iterdir():
            link = include / entry.name
            if not link.exists():
                link.symlink_to(entry)
            found_runtime = found_runtime or entry.name == "cuda_runtime.h"
    if not found_runtime:
        return None

    nvvm = destination / "nvvm"
    nvvm.mkdir(parents=True, exist_ok=True)
    if not (nvvm / "libdevice").exists():
        (nvvm / "libdevice").symlink_to(libdevice)

    ptxas = nvidia / "cuda_nvcc/bin/ptxas"
    if ptxas.exists():
        bindir = destination / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        if not (bindir / "ptxas").exists():
            (bindir / "ptxas").symlink_to(ptxas)

    (destination / "version.txt").write_text("CUDA Version 12.9.0\n")
    return destination


def find_toolchain(workdir: Path) -> CudaToolchain | None:
    """Locate a *working* CUDA device compiler, or ``None`` to skip.

    Each candidate standard library is verified by compiling an empty kernel
    rather than assumed to work. Returning a toolchain that cannot compile is
    worse than returning nothing: the failure would surface as ten confusing
    errors inside the kernel tests instead of one clear skip.
    """
    compiler = shutil.which("clang++")
    if compiler is None:
        return None
    cuda_path = _assemble_cuda_path(workdir / "cuda-root")
    if cuda_path is None:
        return None

    probe = workdir / "toolchain_probe.cu"
    probe.write_text(_PROBE_SOURCE)
    for index, (label, includes) in enumerate(_libstdcxx_candidates()):
        toolchain = CudaToolchain(
            compiler=Path(compiler),
            cuda_path=cuda_path,
            stdlib_includes=includes,
            stdlib_label=label,
        )
        try:
            compile_device_code(
                toolchain, probe, workdir / f"toolchain_probe_{index}.ptx", arch="sm_80"
            )
        except AssertionError:
            continue
        return toolchain
    return None


def compile_device_code(toolchain: CudaToolchain, source: Path, output: Path, *, arch: str) -> str:
    """Compile ``source`` to PTX for ``arch`` and return the PTX text."""
    command = [
        str(toolchain.compiler),
        "-x",
        "cuda",
        "--cuda-device-only",
        f"--cuda-path={toolchain.cuda_path}",
        f"--cuda-gpu-arch={arch}",
        "--no-cuda-version-check",
        "-std=c++17",
        # Optimised, as a real build is: at -O0 clang leaves __shfl_down_sync
        # out of line and the reduction unrolling cannot be inspected.
        "-O3",
        f"-I{CSRC}",
        "-S",
        "-o",
        str(output),
        str(source),
    ]
    for include in toolchain.stdlib_includes:
        command[-3:-3] = ["-isystem", str(include)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise AssertionError(f"clang failed to compile {source.name} for {arch}:\n{result.stderr}")
    return output.read_text()


#: A translation unit that instantiates the kernel for every dtype the
#: pybind entry point accepts (see the TORCH_CHECK in rmsnorm.cpp).
INSTANTIATION_SOURCE = """
#include "rmsnorm_kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template __global__ void kernelforge::rmsnorm_forward_kernel<float>(
    const float*, const float*, float*, int, long long, float);
template __global__ void kernelforge::rmsnorm_forward_kernel<__half>(
    const __half*, const __half*, __half*, int, long long, float);
template __global__ void kernelforge::rmsnorm_forward_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, long long, float);
"""
