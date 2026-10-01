"""Canonical dtype names.

Shape/dtype pairs end up in cache keys, SQLite rows and CLI flags, so the
textual name has to round-trip exactly. ``str(torch.float16)`` is
``"torch.float16"``, which is awkward in a filename and in a ``--dtype`` flag,
hence the short names used throughout KernelForge.
"""

from __future__ import annotations

import torch

_NAME_TO_DTYPE: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_DTYPE_TO_NAME: dict[torch.dtype, str] = {v: k for k, v in _NAME_TO_DTYPE.items()}

SUPPORTED_DTYPES: tuple[str, ...] = tuple(_NAME_TO_DTYPE)


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        if name not in _DTYPE_TO_NAME:
            raise ValueError(f"unsupported dtype {name!r}; expected one of {SUPPORTED_DTYPES}")
        return name
    try:
        return _NAME_TO_DTYPE[name.lower()]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {name!r}; expected one of {SUPPORTED_DTYPES}"
        ) from None


def dtype_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_TO_NAME[dtype]
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype!r}") from None


def itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()
