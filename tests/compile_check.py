"""Compile kernels for NVIDIA GPU targets without a GPU; a helper for tests/test_compile.py.

It must run in its own process with TRITON_INTERPRET unset, because Triton decides at import
time whether @triton.jit functions are interpreted. It reads a JSON list of requests from stdin
and writes one JSON result per request to stdout. Each launch is specialized the way Triton's JIT
specializes a real one: integers equal to 1 become constants, integers divisible by 16 and
aligned pointers get divisibility hints.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import torch
import triton
from triton._C.libtriton import native_specialize_impl
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.compiler import ASTSource

from kernelforge.kernels.matmul import FP32_INPUT_PRECISION, GROUP_M, _matmul_kernel
from kernelforge.kernels.softmax import _softmax_kernel
from kernelforge.kernels.vector_add import BLOCK_SIZE, _vector_add_kernel

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _launch(request: dict[str, Any]) -> tuple[Any, list[Any], dict[str, Any]]:
    """Kernel, runtime arguments and constexprs of a launch on contiguous operands."""
    dtype, shape, params = DTYPES[request["dtype"]], request["shape"], request.get("params", {})
    tensor = torch.empty(0, dtype=dtype)
    if request["kernel"] == "matmul":
        m, n, k = shape["M"], shape["N"], shape["K"]
        precision = FP32_INPUT_PRECISION if dtype == torch.float32 else "tf32"
        constexprs = {**params, "GROUP_M": GROUP_M, "INPUT_PRECISION": precision}
        return _matmul_kernel, [tensor, tensor, tensor, m, n, k, k, 1, n, 1, n, 1], constexprs
    if request["kernel"] == "softmax":
        cols = shape["cols"]
        constexprs = {"BLOCK_SIZE": triton.next_power_of_2(cols)}
        return _softmax_kernel, [tensor, tensor, cols, cols, cols], constexprs
    return _vector_add_kernel, [tensor, tensor, tensor, shape["n"]], {"BLOCK_SIZE": BLOCK_SIZE}


def compile_request(request: dict[str, Any]) -> dict[str, Any]:
    kernel, args, constexprs = _launch(request)
    signature: dict[str, str] = {}
    attrs: dict[tuple[int, ...], list[list[Any]]] = {}
    for index, (name, value) in enumerate(zip(kernel.arg_names, args, strict=False)):
        kind, hint = native_specialize_impl(BaseBackend, value, False, True, True)
        if kind == "constexpr":
            constexprs[name] = hint
        elif hint == "D":
            attrs[(index,)] = [["tt.divisibility", 16]]
        signature[name] = kind
    signature.update({name: "constexpr" for name in constexprs})
    compiled = triton.compile(
        ASTSource(fn=kernel, signature=signature, constexprs=constexprs, attrs=attrs),
        target=GPUTarget("cuda", request["arch"], 32),
        options={"num_warps": request["num_warps"], "num_stages": request["num_stages"]},
    )
    ptx, ttgir = compiled.asm["ptx"], compiled.asm["ttgir"]
    return {
        "shared": compiled.metadata.shared,
        "mma_sync": "mma.sync" in ptx,
        "wgmma": "wgmma" in ptx,
        "async_copy": "async_copy" in ttgir,
    }


def main() -> None:
    results = []
    for request in json.load(sys.stdin):
        try:
            results.append(compile_request(request))
        except Exception as error:
            results.append({"error": f"{type(error).__name__}: {error}"})
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
