"""Tests for the handwritten CUDA RMSNorm.

Split by what each needs:

* the compile tests lower the device code to PTX with clang and inspect the
  generated instructions -- no GPU, and no ``nvcc``;
* the ``gpu``-marked test builds the real extension and checks it against the
  same reference the Triton kernel is held to.
"""

from __future__ import annotations

import pytest
import torch
from cuda_compile_harness import (
    INSTANTIATION_SOURCE,
    compile_device_code,
    find_toolchain,
)

from kernelforge.kernels import cuda_rmsnorm as extension
from kernelforge.testing import assert_verified

ARCHITECTURES = ("sm_80", "sm_90")


@pytest.fixture(scope="module")
def ptx_by_arch(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("cuda-compile")
    toolchain = find_toolchain(workdir)
    if toolchain is None:
        pytest.skip(
            "no CUDA device compiler available; install clang and "
            "nvidia-cuda-runtime-cu12 + nvidia-cuda-nvcc-cu12 to enable"
        )
    source = workdir / "instantiate.cu"
    source.write_text(INSTANTIATION_SOURCE)
    return {
        arch: compile_device_code(toolchain, source, workdir / f"{arch}.ptx", arch=arch)
        for arch in ARCHITECTURES
    }


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_kernel_compiles_for_every_dtype(ptx_by_arch, arch):
    """One kernel entry per dtype the pybind entry point accepts.

    AT_DISPATCH_FLOATING_TYPES_AND2 would also instantiate double, which this
    kernel cannot serve honestly -- it accumulates in float and rescales with
    rsqrtf. The binding refuses double rather than downgrading it silently, so
    three instantiations is the complete set.
    """
    ptx = ptx_by_arch[arch]
    assert ptx.count(".visible .entry") == 3
    for mangled in ("IfE", "I6__halfE", "I13__nv_bfloat16E"):
        assert mangled in ptx, f"no instantiation for {mangled} in {arch} PTX"


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_warp_reduction_is_fully_unrolled(ptx_by_arch, arch):
    """Five shuffle steps per kernel, which is log2(32).

    The reduction is written as a loop with ``#pragma unroll``. If the pragma
    stopped applying, this would drop to one shuffle inside a branch -- still
    correct, but a branch per step in the hottest part of a memory-bound
    kernel.
    """
    ptx = ptx_by_arch[arch]
    assert ptx.count("shfl.sync.down") == 5 * 3


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_block_reduction_uses_exactly_two_barriers(ptx_by_arch, arch):
    """Two ``__syncthreads()`` per kernel and not one more.

    One after the per-warp partials are published, one after the reciprocal
    RMS is broadcast. A third would mean an accidental barrier inside a loop;
    fewer would mean a race on the shared-memory slot.
    """
    assert ptx_by_arch[arch].count("bar.sync") == 2 * 3


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_reciprocal_square_root_is_computed_once_per_row(ptx_by_arch, arch):
    assert ptx_by_arch[arch].count("rsqrt") == 3


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_device_code_is_fully_inlined(ptx_by_arch, arch):
    """No out-of-line device functions in an optimised build.

    A ``.func`` in the PTX would mean a real call in a kernel whose cost is
    entirely memory traffic and a handful of instructions.
    """
    ptx = ptx_by_arch[arch]
    assert ".func" not in ptx


def test_availability_check_does_not_attempt_a_build():
    """``available()`` is called whenever baselines are listed.

    It has to be cheap and side-effect free, so on a machine with no CUDA it
    must answer False rather than trying to compile anything.
    """
    assert extension.available() is False or torch.cuda.is_available()


def test_sources_are_packaged():
    for path in extension.SOURCES:
        assert path.exists(), f"missing CUDA source {path}"


@pytest.mark.gpu
def test_extension_matches_the_reference(device):
    """Build the extension for real and check it against the same reference."""
    if not extension.available():
        pytest.skip(f"no CUDA toolkit found (CUDA_HOME={extension.cuda_home()})")
    from kernelforge.kernels.rmsnorm import rmsnorm_reference

    torch.manual_seed(0)
    for rows, cols in [(1, 1), (7, 127), (128, 768), (1024, 4096), (33, 8192)]:
        for dtype in (torch.float16, torch.float32):
            x = torch.randn(rows, cols, device=device, dtype=dtype)
            gamma = torch.randn(cols, device=device, dtype=dtype) * 0.1 + 1.0
            assert_verified(
                rmsnorm_reference(x, gamma.to(dtype)),
                extension.cuda_rmsnorm(x, gamma.to(dtype)),
                dtype=dtype,
                context=f"cuda rmsnorm {rows}x{cols} {dtype}",
            )
