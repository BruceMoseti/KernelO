"""Ahead-of-time compilation of Triton kernels, for machines without a GPU.

``triton.compile`` lowers a kernel to PTX and cubin for a named target without
ever touching a device, which makes a real static check of the kernels
possible in ordinary CI: a typo, a bad ``tl.*`` call, an illegal tile shape or
a configuration that overruns shared memory all fail here. The compiler also
reports the shared memory it allocated, which is how the search space's
shared-memory model is validated against the thing it is modelling.

This is a test helper rather than part of the package: it exists to check the
kernels, not to run them.
"""

from __future__ import annotations

from dataclasses import dataclass

import triton
from triton.backends.compiler import GPUTarget
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
) -> CompiledKernel:
    """Compile ``kernel`` for a CUDA target and return what the compiler reported."""
    source = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
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


def gemm_signature(dtype: str, *, bias: bool = False) -> dict[str, str]:
    ptr = PTR_TYPES[dtype]
    sig = {
        "a_ptr": ptr,
        "b_ptr": ptr,
        "c_ptr": ptr,
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
        "GROUP_M": "constexpr",
        "INPUT_PRECISION": "constexpr",
    }
    if bias:
        keys = list(sig)
        renamed = {
            "a_ptr": "x_ptr",
            "b_ptr": "w_ptr",
            "c_ptr": "y_ptr",
            "stride_am": "stride_xm",
            "stride_ak": "stride_xk",
            "stride_bk": "stride_wk",
            "stride_bn": "stride_wn",
            "stride_cm": "stride_ym",
            "stride_cn": "stride_yn",
        }
        sig = {renamed.get(k, k): sig[k] for k in keys}
        # bias_ptr sits between w_ptr and y_ptr in the kernel signature.
        ordered = {}
        for key, value in sig.items():
            if key == "y_ptr":
                ordered["bias_ptr"] = ptr
            ordered[key] = value
        sig = ordered
    return sig


def gemm_constexprs(config, dtype: str) -> dict[str, object]:
    return {
        "BLOCK_M": config["BLOCK_M"],
        "BLOCK_N": config["BLOCK_N"],
        "BLOCK_K": config["BLOCK_K"],
        "GROUP_M": config["GROUP_M"],
        "INPUT_PRECISION": "ieee",
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
