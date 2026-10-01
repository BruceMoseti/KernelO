"""Kernel configurations and tuning problems.

Nothing here is specific to one kernel. A MatMul config carries BLOCK_M, BLOCK_N and BLOCK_K;
an RMSNorm config would carry its own parameters (for example BLOCK_SIZE and ROWS_PER_PROGRAM)
in the same `params` mapping.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


@dataclass(frozen=True)
class KernelConfig:
    """One point in a kernel's search space: compile-time parameters plus launch settings."""

    kernel: str
    params: Mapping[str, int]
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def params_json(self) -> str:
        """Canonical JSON of `params`, used as the identity of a config in the database."""
        return json.dumps(dict(self.params), sort_keys=True)

    def __str__(self) -> str:
        params = " ".join(f"{name}={value}" for name, value in self.params.items())
        return f"{params} num_warps={self.num_warps} num_stages={self.num_stages}"


@dataclass(frozen=True)
class Problem:
    """A workload to tune: one operation at one shape and dtype."""

    operation: str
    shape: Mapping[str, int]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", MappingProxyType(dict(self.shape)))

    @property
    def dtype_name(self) -> str:
        return DTYPE_NAMES[self.dtype]

    def shape_json(self) -> str:
        """Canonical JSON of `shape`, used as the identity of a problem in the database."""
        return json.dumps(dict(self.shape), sort_keys=True)

    def __str__(self) -> str:
        dims = " x ".join(str(size) for size in self.shape.values())
        return f"{self.operation} {dims} {self.dtype_name}"
