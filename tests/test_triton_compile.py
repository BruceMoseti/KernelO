"""Static checks on the Triton kernels, no GPU required.

``triton.compile`` lowers a kernel to PTX and cubin for a named target without
a device, which turns a class of kernel bugs into ordinary CI failures:
undefined names, illegal tile shapes, bad ``tl.*`` calls, and configurations
that overrun shared memory. It also lets the fp16 GEMM's use of tensor cores
be asserted from the generated PTX.

What these tests cannot see: anything that depends on running. The standalone
compile entry point does not run the software pipeliner, so the shared-memory
figures here are single-buffer and the ``num_stages`` factor in the search
space's estimate is checked on a GPU instead -- see
``tests/test_kernels_gpu.py::test_shared_memory_model``.
"""

from __future__ import annotations

import pytest

from kernelforge.dtypes import itemsize, parse_dtype
from kernelforge.tuning.config import KernelConfig, Problem

triton = pytest.importorskip("triton", reason="Triton is required for the compile checks")

from compile_harness import (  # noqa: E402  (after importorskip by design)
    TARGETS,
    compile_for_target,
    gemm_constexprs,
    gemm_signature,
    row_signature,
)

MATMUL_PROBLEM = Problem.create("matmul", "fp16", M=2048, N=4096, K=4096)


def _compile_gemm(kernel, config: KernelConfig, dtype: str, capability: int, *, bias: bool):
    return compile_for_target(
        kernel,
        gemm_signature(dtype, bias=bias),
        gemm_constexprs(config, dtype),
        capability=capability,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


@pytest.mark.parametrize("target_name,capability", TARGETS)
def test_default_configs_compile(target_name, capability):
    """Every kernel's shipped default configuration compiles on Ampere and Hopper."""
    from kernelforge.kernels import fused_linear, matmul, rmsnorm, softmax, vector_add

    _compile_gemm(matmul.matmul_kernel, matmul.DEFAULT_CONFIG, "fp16", capability, bias=False)
    _compile_gemm(
        fused_linear.fused_linear_gelu_kernel,
        fused_linear.DEFAULT_CONFIG,
        "fp16",
        capability,
        bias=True,
    )

    rms_config = rmsnorm.default_config(4096)
    compile_for_target(
        rmsnorm.rmsnorm_kernel,
        row_signature("fp16", gamma=True, eps=True),
        rms_config.meta,
        capability=capability,
        num_warps=rms_config["num_warps"],
    )

    sm_config = softmax.default_config(2048)
    compile_for_target(
        softmax.softmax_kernel,
        row_signature("fp16", gamma=False, eps=False),
        sm_config.meta,
        capability=capability,
        num_warps=sm_config["num_warps"],
    )

    compile_for_target(
        vector_add.vector_add_kernel,
        {
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp16",
            "n_elements": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        vector_add.DEFAULT_CONFIG.meta,
        capability=capability,
        num_warps=vector_add.DEFAULT_CONFIG["num_warps"],
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_gemm_compiles_for_every_dtype(dtype):
    from kernelforge.kernels.matmul import matmul_kernel

    config = KernelConfig(
        "matmul", BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3
    )
    result = _compile_gemm(matmul_kernel, config, dtype, 80, bias=False)
    assert result.ptx


def test_fp16_gemm_uses_tensor_cores():
    """The fp16 GEMM must reach the tensor cores.

    ``input_precision="ieee"`` is passed unconditionally so that fp32 does not
    silently fall back to TF32 (see the module docstring in
    ``kernelforge/kernels/matmul.py``). For fp16 operands the flag is
    irrelevant to the hardware path, and this asserts that: the Ampere
    ``m16n8k16`` MMA instruction is still what gets emitted.
    """
    from kernelforge.kernels.matmul import matmul_kernel

    config = KernelConfig(
        "matmul", BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=3
    )
    result = _compile_gemm(matmul_kernel, config, "fp16", 80, bias=False)
    assert result.uses_tensor_cores(), "fp16 GEMM did not emit an mma.sync instruction"
    assert "mma.sync.aligned.m16n8k16" in result.ptx


def test_fused_epilogue_costs_one_exponential_per_element():
    """The GELU rewrite should cost exactly one transcendental per element.

    ``0.5*x*(1+tanh(z))`` was rewritten as ``x/(1+exp(-2z))``. The epilogue is
    fully unrolled over the accumulator values a thread holds, so a correct
    rewrite emits one hardware ``ex2`` and one divide per element. A ``tanh``
    expansion, or a rewrite that lost the algebraic cancellation, would emit
    more.
    """
    from kernelforge.kernels.fused_linear import DEFAULT_CONFIG, fused_linear_gelu_kernel

    result = _compile_gemm(fused_linear_gelu_kernel, DEFAULT_CONFIG, "fp16", 80, bias=True)
    per_thread = (DEFAULT_CONFIG["BLOCK_M"] * DEFAULT_CONFIG["BLOCK_N"]) // (
        DEFAULT_CONFIG["num_warps"] * 32
    )
    assert result.ptx.count("ex2.approx.f32") == per_thread
    assert result.ptx.count("div.full.f32") == per_thread


def test_shared_memory_estimate_bounds_the_tile():
    """The search space's tile footprint matches what the compiler allocates.

    Compared against the single-buffer figure, which is what the standalone
    compiler emits. The allocation is sometimes padded for the swizzled
    operand layout, so the assertion is a factor-of-two band rather than
    equality.
    """
    from kernelforge.kernels.matmul import matmul_kernel
    from kernelforge.tuning.search import MatmulSearchSpace

    space = MatmulSearchSpace()
    for dtype in ("fp16", "fp32"):
        width = itemsize(parse_dtype(dtype))
        problem = Problem.create("matmul", dtype, M=2048, N=4096, K=4096)
        for bm, bn, bk in [(32, 32, 32), (64, 64, 32), (128, 128, 64), (32, 128, 32)]:
            config = KernelConfig(
                "matmul",
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_M=8,
                num_warps=4,
                num_stages=1,
            )
            result = _compile_gemm(matmul_kernel, config, dtype, 80, bias=False)
            tile_bytes = bk * (bm + bn) * width
            assert tile_bytes <= result.shared_bytes <= 2 * tile_bytes, (
                f"{dtype} {bm}x{bn}x{bk}: compiler allocated {result.shared_bytes} B "
                f"for a {tile_bytes} B tile"
            )
            # At num_stages=1 the filter's estimate is exactly the tile size;
            # the multi-stage factor is checked on a GPU.
            assert space.shared_memory_bytes(config, problem) == tile_bytes


def test_every_candidate_in_the_budget_compiles(a100_caps):
    """The configurations the tuner would actually try all compile.

    This is the check that makes the search-space filters trustworthy: if a
    filter let through a tile shape Triton rejects, the tuner would spend its
    budget on configurations that can never run.
    """
    from kernelforge.kernels.matmul import matmul_kernel
    from kernelforge.tuning.search import MatmulSearchSpace

    candidates = MatmulSearchSpace().candidates(MATMUL_PROBLEM, a100_caps)
    assert candidates, "search space produced no candidates"
    sampled = candidates[::6]
    assert len(sampled) >= 6
    for config in sampled:
        result = _compile_gemm(matmul_kernel, config, "fp16", 80, bias=False)
        assert result.uses_tensor_cores(), f"{config!r} did not reach the tensor cores"


@pytest.mark.slow
def test_full_candidate_budget_compiles(a100_caps):
    """The exhaustive version of the above; slow, so it is opt-in."""
    from kernelforge.kernels.matmul import matmul_kernel
    from kernelforge.tuning.search import MatmulSearchSpace

    for config in MatmulSearchSpace().candidates(MATMUL_PROBLEM, a100_caps):
        _compile_gemm(matmul_kernel, config, "fp16", 80, bias=False)


@pytest.mark.parametrize("cols", [128, 1024, 4096, 8192])
def test_row_kernels_compile_across_widths(cols):
    from kernelforge.kernels import rmsnorm, softmax

    rms_config = rmsnorm.default_config(cols)
    compile_for_target(
        rmsnorm.rmsnorm_kernel,
        row_signature("fp16", gamma=True, eps=True),
        rms_config.meta,
        num_warps=rms_config["num_warps"],
    )
    sm_config = softmax.default_config(cols)
    compile_for_target(
        softmax.softmax_kernel,
        row_signature("fp16", gamma=False, eps=False),
        sm_config.meta,
        num_warps=sm_config["num_warps"],
    )
