"""Ahead-of-time compilation of Triton kernels, for machines without a GPU.

``triton.compile`` lowers a kernel to PTX and cubin for a named target without
ever touching a device, which makes a real static check of the kernels
possible in ordinary CI: a typo, a bad ``tl.*`` call, an illegal tile shape or
a configuration that overruns shared memory all fail here. The compiler also
reports the shared memory it allocated, which is how the search space's
shared-memory model is validated against the thing it is modelling.

A kernel has to be compiled the way the JIT compiles it for that to hold. The
JIT specializes every launch on its argument values -- an integer equal to 1
becomes a constant, and integers divisible by 16 and 16-byte-aligned pointers
carry a divisibility hint -- and the software pipeliner relies on those facts
to issue asynchronous copies. Without them it keeps one buffer per operand
whatever ``num_stages`` says, which is not the kernel a GPU runs.

This is a test helper rather than part of the package: it exists to check the
kernels, not to run them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
from triton._C.libtriton import native_specialize_impl
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.compiler import ASTSource

#: (name, compute capability). sm80 is Ampere (A100), sm90 is Hopper (H100).
TARGETS: tuple[tuple[str, int], ...] = (("sm80", 80), ("sm90", 90))

PTR_TYPES = {"fp16": "*fp16", "bf16": "*bf16", "fp32": "*fp32"}


@dataclass(frozen=True)
class CompiledKernel:
    name: str
    target: str
    shared_bytes: int
    num_warps: int
    ptx: str

    def uses_tensor_cores(self) -> bool:
        return "mma.sync" in self.ptx


def compile_for_target(
    kernel,
    signature: dict[str, str],
    constexprs: dict[str, object],
    *,
    capability: int = 80,
    num_warps: int = 4,
    num_stages: int = 3,
    attrs: dict | None = None,
) -> CompiledKernel:
    """Compile ``kernel`` for a CUDA target and return what the compiler reported."""
    source = ASTSource(fn=kernel, signature=signature, constexprs=constexprs, attrs=attrs)
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", capability, 32),
        options={"num_warps": num_warps, "num_stages": num_stages},
    )
    return CompiledKernel(
        name=kernel.__name__,
        target=f"sm{capability}",
        shared_bytes=int(compiled.metadata.shared),
        num_warps=int(compiled.metadata.num_warps),
        ptx=compiled.asm["ptx"],
    )


def specialize(kernel, args: list) -> tuple[dict[str, str], dict[str, object], dict]:
    """Signature, constexprs and divisibility hints the JIT derives from ``args``."""
    signature: dict[str, str] = {}
    constexprs: dict[str, object] = {}
    attrs: dict = {}
    for index, (name, value) in enumerate(zip(kernel.arg_names, args, strict=False)):
        kind, hint = native_specialize_impl(BaseBackend, value, False, True, True)
        if kind == "constexpr":
            constexprs[name] = hint
        elif hint == "D":
            attrs[(index,)] = [["tt.divisibility", 16]]
        signature[name] = kind
    return signature, constexprs, attrs


def gemm_args(dtype: torch.dtype, m: int, n: int, k: int, *, bias: bool = False) -> list:
    """Runtime arguments of a GEMM launch on contiguous row-major operands."""
    operand = torch.empty(0, dtype=dtype)
    pointers = [operand] * (4 if bias else 3)
    return [*pointers, m, n, k, k, 1, n, 1, n, 1]


def gemm_constexprs(config, dtype: str) -> dict[str, object]:
    return {
        "BLOCK_M": config["BLOCK_M"],
        "BLOCK_N": config["BLOCK_N"],
        "BLOCK_K": config["BLOCK_K"],
        "GROUP_M": config["GROUP_M"],
    }


def row_signature(dtype: str, *, gamma: bool, eps: bool) -> dict[str, str]:
    ptr = PTR_TYPES[dtype]
    sig: dict[str, str] = {"x_ptr": ptr}
    if gamma:
        sig["gamma_ptr"] = ptr
    sig["y_ptr"] = ptr
    sig["x_row_stride"] = "i32"
    sig["y_row_stride"] = "i32"
    sig["n_rows"] = "i32"
    sig["n_cols"] = "i32"
    if eps:
        sig["eps"] = "fp32"
    sig["BLOCK_SIZE"] = "constexpr"
    sig["ROWS_PER_PROGRAM"] = "constexpr"
    return sig
