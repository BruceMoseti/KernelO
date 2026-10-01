"""Kernel registry.

Operators are resolved lazily by name. The ``@triton.jit`` decorator runs at
module import, so importing a kernel module requires Triton to be installed;
keeping the registry a table of module paths means ``import kernelforge`` and
every CPU-side test still work on a machine without it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernelforge.kernels.base import Operator

_OPERATORS: dict[str, tuple[str, str]] = {
    "vector_add": ("kernelforge.kernels.vector_add", "VectorAddOperator"),
    "softmax": ("kernelforge.kernels.softmax", "SoftmaxOperator"),
    "matmul": ("kernelforge.kernels.matmul", "MatmulOperator"),
    "rmsnorm": ("kernelforge.kernels.rmsnorm", "RMSNormOperator"),
    "fused_linear": ("kernelforge.kernels.fused_linear", "FusedLinearOperator"),
}


def operator_names() -> tuple[str, ...]:
    return tuple(_OPERATORS)


def get_operator(name: str) -> Operator:
    try:
        module_path, class_name = _OPERATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown operator {name!r}; available: {', '.join(operator_names())}"
        ) from None
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()
