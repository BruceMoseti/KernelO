"""Static checks on the Triton kernels, no GPU required.

``triton.compile`` lowers a kernel to PTX and cubin for a named target without
a device, which turns a class of kernel bugs into ordinary CI failures:
undefined names, illegal tile shapes, bad ``tl.*`` calls, and configurations
that overrun shared memory. It also lets the fp16 GEMM's use of tensor cores
be asserted from the generated PTX.

What these tests cannot see: anything that depends on running. Each GEMM is
compiled specialized the way the JIT specializes a launch on contiguous
operands, which is what lets the software pipeliner run, so the shared memory
reported here is the multi-stage allocation a GPU would see.
"""

from __future__ import annotations

import dataclasses

import pytest

from kernelforge.dtypes import itemsize, parse_dtype
from kernelforge.tuning.config import KernelConfig, Problem

triton = pytest.importorskip("triton", reason="Triton is required for the compile checks")

from compile_harness import (  # noqa: E402  (after importorskip by design)
    TARGETS,
    compile_for_target,
    gemm_args,
    gemm_constexprs,
    row_signature,
    specialize,
)

MATMUL_PROBLEM = Problem.create("matmul", "fp16", M=2048, N=4096, K=4096)


def _compile_gemm(
    kernel, config: KernelConfig, dtype: str, capability: int, *, bias: bool, dims=None
):
    dims = dims or MATMUL_PROBLEM.dims_dict
    args = gemm_args(parse_dtype(dtype), dims["M"], dims["N"], dims["K"], bias=bias)
    signature, constexprs, attrs = specialize(kernel, args)
    constexprs.update(gemm_constexprs(config, dtype))
    signature.update(dict.fromkeys(constexprs, "constexpr"))
    return compile_for_target(
        kernel,
        signature,
        constexprs,
        capability=capability,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        attrs=attrs,
    )


def _caps_for(caps, capability: int):
    """``caps`` as if the device had compute capability ``capability``."""
    return dataclasses.replace(caps, compute_capability=f"{capability // 10}.{capability % 10}")


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


@pytest.mark.parametrize("capability", [80, 89, 90, 120])
def test_gemm_k_loop_is_software_pipelined(capability):
    """Each pipeline stage adds one operand tile to the compiler's allocation.

    This is the ``num_stages`` factor of the search space's shared-memory
    estimate, observed directly: the K loop is pipelined with asynchronous
    copies, and the allocation grows by ``BLOCK_K * (BLOCK_M + BLOCK_N) *
    itemsize`` per stage.
    """
    from kernelforge.kernels.matmul import matmul_kernel

    allocated = []
    for stages in (2, 3, 4):
        config = KernelConfig(
            "matmul",
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=64,
            GROUP_M=8,
            num_warps=4,
            num_stages=stages,
        )
        result = _compile_gemm(matmul_kernel, config, "fp16", capability, bias=False)
        assert "cp.async" in result.ptx, f"num_stages={stages}: the K loop was not pipelined"
        allocated.append(result.shared_bytes)
    tile_bytes = 64 * (128 + 128) * itemsize(parse_dtype("fp16"))
    assert [b - a for a, b in zip(allocated, allocated[1:], strict=False)] == [tile_bytes] * 2


@pytest.mark.parametrize("capability", [80, 89, 90, 120])
def test_shared_memory_estimate_bounds_the_compiler(capability, a100_caps):
    """The search space's estimate is never below what the compiler allocates.

    Checked at every pipeline depth and in both regimes of the estimate: tiles
    whose pipeline buffers dominate, and a configuration whose epilogue stages
    a larger output tile through shared memory than the pipeline holds.
    """
    from kernelforge.kernels.matmul import matmul_kernel
    from kernelforge.tuning.search import MatmulSearchSpace

    space = MatmulSearchSpace()
    caps = _caps_for(a100_caps, capability)
    for dtype, (bm, bn, bk, warps) in [
        ("fp16", (128, 128, 64, 4)),
        ("fp32", (64, 64, 32, 4)),
        ("fp16", (64, 128, 32, 8)),
    ]:
        problem = Problem.create("matmul", dtype, **MATMUL_PROBLEM.dims_dict)
        for stages in (2, 3, 4):
            config = KernelConfig(
                "matmul",
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_M=8,
                num_warps=warps,
                num_stages=stages,
            )
            result = _compile_gemm(matmul_kernel, config, dtype, capability, bias=False)
            estimate = space.shared_memory_bytes(config, problem, caps)
            assert result.shared_bytes <= estimate, (
                f"sm{capability} {dtype} {config!r}: compiler allocated "
                f"{result.shared_bytes} B, estimate {estimate} B"
            )


def test_shared_memory_filter_admits_what_fits_an_rtx_4090(rtx4090_caps):
    """A tile the compiler fits on an RTX 4090 must not be rejected for it.

    128x128x64 over four stages keeps three operand tiles in flight on sm_89:
    96 KiB against the card's 99 KiB opt-in limit. Counting one tile per
    stage, as on Hopper, would reject it.
    """
    from kernelforge.kernels.matmul import matmul_kernel
    from kernelforge.tuning.search import MatmulSearchSpace

    config = KernelConfig(
        "matmul", BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=4
    )
    result = _compile_gemm(matmul_kernel, config, "fp16", 89, bias=False)
    assert result.shared_bytes <= rtx4090_caps.max_shared_memory_per_block
    assert MatmulSearchSpace().reject_reason(config, MATMUL_PROBLEM, rtx4090_caps) is None


def test_every_candidate_in_the_budget_compiles(a100_caps):
    """The configurations the tuner would actually try all compile.

    This is the check that makes the search-space filters trustworthy: if a
    filter let through a tile shape Triton rejects, the tuner would spend its
    budget on configurations that can never run.
    """
    from kernelforge.kernels.matmul import matmul_kernel
    from kernelforge.tuning.search import MatmulSearchSpace

    space = MatmulSearchSpace()
    candidates = space.candidates(MATMUL_PROBLEM, a100_caps)
    assert candidates, "search space produced no candidates"
    sampled = candidates[::6]
    assert len(sampled) >= 6
    for config in sampled:
        result = _compile_gemm(matmul_kernel, config, "fp16", 80, bias=False)
        assert result.uses_tensor_cores(), f"{config!r} did not reach the tensor cores"
        assert result.shared_bytes <= space.shared_memory_bytes(config, MATMUL_PROBLEM, a100_caps)


@pytest.mark.slow
@pytest.mark.parametrize(
    "dims",
    [
        {"M": 2048, "N": 4096, "K": 4096},
        # A decode-shaped GEMM selects an entirely different part of the grid:
        # only BLOCK_M=16 survives the overshoot rule, and the budget never
        # reaches those tiles for a large shape because they have the least
        # reuse. Without this case they would go uncompiled.
        {"M": 1, "N": 11008, "K": 4096},
    ],
    ids=["square", "decode"],
)
def test_full_candidate_budget_compiles(dims, a100_caps):
    """Every candidate the tuner would try, exhaustively. Slow, so opt-in."""
    from kernelforge.kernels.matmul import matmul_kernel
    from kernelforge.tuning.search import MatmulSearchSpace

    problem = Problem.create("matmul", "fp16", **dims)
    space = MatmulSearchSpace()
    candidates = space.candidates(problem, a100_caps)
    assert candidates, f"no candidates for {dims}"
    for config in candidates:
        result = _compile_gemm(matmul_kernel, config, "fp16", 80, bias=False, dims=dims)
        assert result.uses_tensor_cores(), f"{config!r} did not reach the tensor cores"
        assert result.shared_bytes <= space.shared_memory_bytes(config, problem, a100_caps)


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
