"""Hardware and software context recorded with every measurement.

A GPU timing is not interpretable without the GPU model, compute capability, driver,
CUDA, PyTorch and Triton versions, so every benchmark result and tuning run carries
an `Environment`.
"""

from __future__ import annotations

import functools
import platform
import subprocess
from dataclasses import dataclass

import torch
import triton


@dataclass(frozen=True)
class Environment:
    """Hardware and software context. GPU fields are None when no CUDA device is present."""

    gpu: str | None
    compute_capability: str | None
    sm_count: int | None
    gpu_memory_bytes: int | None
    l2_cache_bytes: int | None
    driver: str | None
    cuda: str | None  # CUDA version PyTorch was built with; None for CPU-only builds.
    torch: str
    triton: str
    python: str
    platform: str
    cpu: str
    # PyTorch settings that change what cuBLAS computes for the PyTorch baselines.
    torch_matmul_allow_tf32: bool
    torch_matmul_fp16_reduced_precision: bool
    torch_matmul_bf16_reduced_precision: bool


def collect_environment() -> Environment:
    """Describe the current CUDA device (if any) and the software stack."""
    gpu = compute_capability = driver = None
    sm_count = gpu_memory_bytes = l2_cache_bytes = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = props.name
        compute_capability = f"{props.major}.{props.minor}"
        sm_count = props.multi_processor_count
        gpu_memory_bytes = props.total_memory
        l2_cache_bytes = props.L2_cache_size
        driver = _nvidia_driver_version()
    return Environment(
        gpu=gpu,
        compute_capability=compute_capability,
        sm_count=sm_count,
        gpu_memory_bytes=gpu_memory_bytes,
        l2_cache_bytes=l2_cache_bytes,
        driver=driver,
        cuda=torch.version.cuda,
        torch=torch.__version__,
        triton=triton.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        cpu=_cpu_model(),
        torch_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        torch_matmul_fp16_reduced_precision=(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        torch_matmul_bf16_reduced_precision=(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    )


@functools.cache
def _nvidia_driver_version() -> str | None:
    # PyTorch exposes only the CUDA driver API version, not the driver release (e.g. 550.54.15).
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else None


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()
