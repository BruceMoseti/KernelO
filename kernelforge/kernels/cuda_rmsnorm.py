"""Loader for the handwritten CUDA RMSNorm.

Built on demand with ``torch.utils.cpp_extension.load`` rather than at install
time, so that ``pip install kernelforge`` does not require a CUDA toolchain.
The first call compiles and caches; later calls in the same process reuse the
loaded module.

The CUDA implementation is a *baseline*, not a tunable operator: its block size
follows from the row width, so there is nothing to search over. It appears
alongside PyTorch eager, ``torch.compile`` and Triton in
``kernelforge compare rmsnorm``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

CSRC = Path(__file__).resolve().parent.parent / "csrc"
SOURCES = (CSRC / "rmsnorm.cu", CSRC / "rmsnorm.cpp")
#: Not passed to the compiler, but included by rmsnorm.cu, so a wheel that
#: omitted it would fail at build time rather than at import.
HEADERS = (CSRC / "rmsnorm_kernel.cuh",)
EXTENSION_NAME = "kernelforge_rmsnorm_cuda"

_module: Any = None


def cuda_home() -> str | None:
    """Where the CUDA toolchain is, if PyTorch can find one."""
    from torch.utils.cpp_extension import CUDA_HOME

    return os.environ.get("CUDA_HOME") or CUDA_HOME


def available() -> bool:
    """Whether building the extension is possible here.

    Cheap: checks for a device and a toolchain, and never attempts a build.
    Callers that list baselines need to ask this on every run.
    """
    if not torch.cuda.is_available():
        return False
    if not all(path.exists() for path in (*SOURCES, *HEADERS)):
        return False
    home = cuda_home()
    return bool(home) and Path(home, "bin", "nvcc").exists()


def load(*, verbose: bool = False) -> Any:
    """Compile and load the extension, caching the result."""
    global _module
    if _module is not None:
        return _module
    if not torch.cuda.is_available():
        raise RuntimeError("the CUDA RMSNorm extension requires a CUDA device")
    if not cuda_home():
        raise RuntimeError("no CUDA toolkit found; set CUDA_HOME to a toolkit containing bin/nvcc")
    from torch.utils.cpp_extension import load as load_extension

    _module = load_extension(
        name=EXTENSION_NAME,
        sources=[str(path) for path in SOURCES],
        extra_cflags=["-O3"],
        # Deliberately no --use_fast_math: it would change the numerics of the
        # reciprocal square root, and the whole point is to compare this
        # against the Triton kernel at matching precision.
        extra_cuda_cflags=["-O3", "--extended-lambda"],
        verbose=verbose,
    )
    return _module


def cuda_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, *, eps: float = 1e-5) -> torch.Tensor:
    return load().rmsnorm_forward(x, gamma, eps)
