"""Hardware-aware cache keys for tuned configurations.

A tuned config is reused only when the GPU model, compute capability, Triton version, operation,
dtype and exact shape all match. The spec's key is GPU architecture + operation + dtype +
dimensions. The model name is used rather than the compute capability alone because GPUs that
share an architecture can differ in SM count and bandwidth: an RTX 4090 and an L4 are both
sm_89. The Triton version is included because the compiler decides the generated code, and so
the ranking.

Kernel source is not part of the key; after editing a kernel, run `kernelforge cache clear`.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelforge.runtime.environment import Environment
from kernelforge.tuning.config import Problem


@dataclass(frozen=True)
class CacheKey:
    gpu: str
    compute_capability: str
    triton: str
    operation: str
    dtype: str
    shape: str  # canonical JSON, e.g. {"K": 4096, "M": 2048, "N": 4096}


def cache_key(problem: Problem, environment: Environment) -> CacheKey:
    if environment.gpu is None or environment.compute_capability is None:
        raise ValueError("tuned configs are cached per GPU, and this environment has no GPU")
    return CacheKey(
        gpu=environment.gpu,
        compute_capability=environment.compute_capability,
        triton=environment.triton,
        operation=problem.operation,
        dtype=problem.dtype_name,
        shape=problem.shape_json(),
    )
