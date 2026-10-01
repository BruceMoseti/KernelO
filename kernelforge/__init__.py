"""KernelForge: hardware-aware GPU kernel autotuning for transformer inference.

Nothing exported here imports Triton or touches CUDA, so the package is safe to
import on a CPU-only machine; the kernels themselves are reached through
:func:`kernelforge.kernels.get_operator`.
"""

from __future__ import annotations

from kernelforge.benchmark.runner import TimingResult, benchmark
from kernelforge.kernels import get_operator, operator_names
from kernelforge.runtime.env import DeviceCaps, Environment, capture_environment, device_caps
from kernelforge.testing import VerificationResult, verify
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.tuner import Tuner, TuningResult

__version__ = "0.1.0"

__all__ = [
    "DeviceCaps",
    "Environment",
    "KernelConfig",
    "Problem",
    "TimingResult",
    "Tuner",
    "TuningResult",
    "VerificationResult",
    "__version__",
    "benchmark",
    "capture_environment",
    "device_caps",
    "get_operator",
    "operator_names",
    "verify",
]
