"""Problem and configuration descriptors.

Two deliberate decisions here.

**Configurations are open key/value sets, not fixed fields.** A GEMM is tuned
over ``BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M``; an RMSNorm is tuned over
``BLOCK_SIZE/ROWS_PER_PROGRAM``. Giving ``KernelConfig`` named GEMM fields
would force every other operator to carry meaningless ones, so the parameter
set is a mapping and each operator's search space owns its schema.

**Problems carry named dimensions.** ``M/N/K`` for a GEMM and ``rows/cols``
for a normalisation are not interchangeable, and the names end up in cache
keys and database rows, so they travel with the value.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from kernelforge.dtypes import dtype_name, itemsize, parse_dtype

# Keys that Triton consumes as launch parameters rather than as `tl.constexpr`
# kernel arguments. Everything else in a config is passed through as meta.
LAUNCH_KEYS = frozenset({"num_warps", "num_stages"})


class KernelConfig(Mapping[str, int]):
    """An immutable, hashable set of tuning parameters for one operation."""

    __slots__ = ("operation", "_params")

    def __init__(self, operation: str, **params: int) -> None:
        for key, value in params.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"config parameter {key!r} must be an int, got {value!r}")
        self.operation = operation
        self._params: dict[str, int] = dict(sorted(params.items()))

    def __getitem__(self, key: str) -> int:
        return self._params[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __hash__(self) -> int:
        return hash((self.operation, tuple(self._params.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KernelConfig):
            return NotImplemented
        return self.operation == other.operation and self._params == other._params

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"KernelConfig({self.operation!r}, {body})"

    @property
    def meta(self) -> dict[str, int]:
        """Parameters passed to the kernel as ``tl.constexpr`` arguments."""
        return {k: v for k, v in self._params.items() if k not in LAUNCH_KEYS}

    @property
    def launch(self) -> dict[str, int]:
        """Parameters passed to the Triton launcher (``num_warps``/``num_stages``)."""
        return {k: v for k, v in self._params.items() if k in LAUNCH_KEYS}

    def as_dict(self) -> dict[str, int]:
        return dict(self._params)

    def to_json(self) -> str:
        return json.dumps(self._params, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """Short content hash, used as the database identity of a config."""
        payload = f"{self.operation}|{self.to_json()}"
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_json(cls, operation: str, payload: str) -> KernelConfig:
        return cls(operation, **json.loads(payload))

    def render(self, order: tuple[str, ...] = ()) -> str:
        """Aligned multi-line rendering for CLI output."""
        keys = [k for k in order if k in self._params]
        keys += [k for k in self._params if k not in keys]
        width = max((len(k) for k in keys), default=0)
        return "\n".join(f"{k + ':':<{width + 1}} {self._params[k]:>4}" for k in keys)


@dataclass(frozen=True)
class Problem:
    """One concrete workload: an operation at a set of dimensions and a dtype."""

    operation: str
    dims: tuple[tuple[str, int], ...]
    dtype: torch.dtype

    @classmethod
    def create(cls, operation: str, dtype: str | torch.dtype, **dims: int) -> Problem:
        for key, value in dims.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"dimension {key!r} must be a positive int, got {value!r}")
        return cls(operation=operation, dims=tuple(dims.items()), dtype=parse_dtype(dtype))

    def __getitem__(self, key: str) -> int:
        return self.dims_dict[key]

    @property
    def dims_dict(self) -> dict[str, int]:
        return dict(self.dims)

    @property
    def shape_key(self) -> str:
        """Dimensions in declaration order, e.g. ``2048x4096x4096``."""
        return "x".join(str(v) for _, v in self.dims)

    @property
    def dtype_name(self) -> str:
        return dtype_name(self.dtype)

    @property
    def itemsize(self) -> int:
        return itemsize(self.dtype)

    @property
    def cache_key(self) -> str:
        return f"{self.operation}/{self.dtype_name}/{self.shape_key}"

    def describe(self) -> str:
        shape = " x ".join(f"{v}" for _, v in self.dims)
        names = "/".join(k for k, _ in self.dims)
        return f"{self.operation} {names}={shape} {self.dtype_name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "dims": self.dims_dict,
            "dtype": self.dtype_name,
            "shape_key": self.shape_key,
        }
