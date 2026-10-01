"""Fused linear + bias + GELU correctness.

Tests marked ``gpu`` need a CUDA device. The rest also run on CPU through
Triton's interpreter (``TRITON_INTERPRET=1``), which is how CI executes them.

Two things specific to this kernel:

* The GELU was rewritten algebraically, so it is checked against
  ``F.gelu(approximate="tanh")`` across the full input range including the
  saturating tails, not just at random normal inputs.
* Fusion must not change the answer. The test that matters is the one
  comparing against the unfused three-kernel sequence a model would otherwise
  run.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernelforge.benchmark import workloads
from kernelforge.testing import assert_verified, exact_fp32_matmul
from kernelforge.tuning.config import KernelConfig, Problem

pytest.importorskip("triton")

DTYPES = (torch.float16, torch.bfloat16, torch.float32)

#: Triton's interpreter does bf16 arithmetic on the raw storage bits, so the
#: bf16 kernel can only be checked on a GPU.
DTYPE_PARAMS = [
    pytest.param(dtype, marks=pytest.mark.gpu) if dtype == torch.bfloat16 else dtype
    for dtype in DTYPES
]

#: Shapes with more multiply-adds than this are too slow for the interpreter.
INTERPRETER_MAX_MACS = 1 << 29


def _gpu_only_if_large(shape: tuple[int, int, int]):
    m, n, k = shape
    return pytest.param(shape, marks=pytest.mark.gpu) if m * n * k > INTERPRETER_MAX_MACS else shape


CORRECTNESS_SHAPES = [
    _gpu_only_if_large(tuple(p.dims_dict.values()))
    for p in workloads.problems("fused_linear", "correctness", "fp16")
]


def inputs(m, k, n, dtype, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, k, device=device, dtype=dtype, generator=gen)
    w = torch.randn(k, n, device=device, dtype=torch.float32, generator=gen)
    w = (w * (k**-0.5)).to(dtype)
    bias = torch.randn(n, device=device, dtype=dtype, generator=gen)
    return x, w, bias


def config(block_m, block_n, block_k, warps, stages) -> KernelConfig:
    return KernelConfig(
        "fused_linear",
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=warps,
        num_stages=stages,
    )


@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES)
@pytest.mark.parametrize("dtype", DTYPE_PARAMS)
def test_fusion_matches_the_unfused_sequence(shape, dtype, device):
    """The comparison that matters: same answer as matmul + bias + gelu."""
    from kernelforge.kernels.fused_linear import (
        DEFAULT_CONFIG,
        fused_linear_gelu,
        linear_gelu_reference,
    )

    m, n, k = shape
    x, w, bias = inputs(m, k, n, dtype, device)
    with exact_fp32_matmul():
        expected = linear_gelu_reference(x, w, bias)
    assert_verified(
        expected,
        fused_linear_gelu(x, w, bias, config=DEFAULT_CONFIG),
        dtype=dtype,
        context=f"fused_linear {m}x{n}x{k} {dtype}",
    )


@pytest.mark.parametrize(
    "tile", [(32, 32, 32, 4, 3), (64, 128, 32, 8, 3), (128, 128, 64, 8, 3), (32, 128, 32, 4, 2)]
)
@pytest.mark.parametrize(
    "shape",
    [
        (127, 129, 65),
        (257, 255, 511),
        # A full 4096x4096 weight per tiling is too slow for the interpreter.
        pytest.param((1, 4096, 4096), marks=pytest.mark.gpu),
    ],
)
def test_boundary_masking_across_tile_shapes(tile, shape, device):
    from kernelforge.kernels.fused_linear import fused_linear_gelu, linear_gelu_reference

    m, n, k = shape
    x, w, bias = inputs(m, k, n, torch.float16, device)
    with exact_fp32_matmul():
        expected = linear_gelu_reference(x, w, bias)
    assert_verified(
        expected,
        fused_linear_gelu(x, w, bias, config=config(*tile)),
        dtype=torch.float16,
        context=f"fused_linear {m}x{n}x{k} tile={tile}",
    )


def test_gelu_rewrite_matches_pytorch_across_the_whole_range(device):
    """``x/(1+exp(-2z))`` against ``0.5*x*(1+tanh(z))``, including the tails.

    The rewrite relies on ``tanh(z) = 2*sigmoid(2z)-1``. Away from zero the two
    forms are easy to get right; the tails are where an algebra slip shows up,
    and where an overflowing exponential would produce NaN rather than the
    correct limit of 0.
    """
    from kernelforge.kernels.fused_linear import fused_linear_gelu

    # An identity weight and zero bias make the kernel's epilogue the only
    # thing under test: the pre-activation is exactly the input value.
    values = torch.cat(
        [
            torch.linspace(-60.0, 60.0, 512, device=device),
            torch.tensor([-1e4, -100.0, -1e-7, 0.0, 1e-7, 100.0, 1e4], device=device),
        ]
    ).to(torch.float32)
    n = values.numel()
    x = values.reshape(1, n).contiguous()
    w = torch.eye(n, device=device, dtype=torch.float32)
    bias = torch.zeros(n, device=device, dtype=torch.float32)

    out = fused_linear_gelu(x, w, bias, config=config(32, 32, 32, 4, 2))
    expected = F.gelu(x, approximate="tanh")
    assert torch.isfinite(out).all(), "GELU produced non-finite values"
    assert_verified(expected, out, dtype=torch.float32, threshold=1e-4)


def test_bias_is_added_before_the_activation(device):
    """Order matters: gelu(xw + b) is not gelu(xw) + b.

    With a zero weight the output must be gelu(bias) exactly, which pins the
    ordering.
    """
    from kernelforge.kernels.fused_linear import fused_linear_gelu

    n = 64
    x = torch.randn(8, 32, device=device, dtype=torch.float32)
    w = torch.zeros(32, n, device=device, dtype=torch.float32)
    bias = torch.linspace(-4.0, 4.0, n, device=device, dtype=torch.float32)
    out = fused_linear_gelu(x, w, bias, config=config(32, 32, 32, 4, 2))
    expected = F.gelu(bias, approximate="tanh").expand(8, n)
    assert_verified(expected, out, dtype=torch.float32, threshold=1e-5)


def test_transposed_weight_needs_no_copy(device):
    """``nn.Linear`` holds (out, in); the kernel reads strides."""
    from kernelforge.kernels.fused_linear import fused_linear_gelu, linear_gelu_reference

    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(64, 128, device=device, dtype=torch.float16, generator=gen)
    weight = torch.randn(256, 128, device=device, dtype=torch.float16, generator=gen) * 0.05
    bias = torch.randn(256, device=device, dtype=torch.float16, generator=gen)
    view = weight.t()
    assert not view.is_contiguous()
    with exact_fp32_matmul():
        expected = linear_gelu_reference(x, view, bias)
    assert_verified(expected, fused_linear_gelu(x, view, bias), dtype=torch.float16)


#: Cells of padding on every side of a guarded tensor: at least the largest
#: block size, so a whole-tile overrun lands in the padding rather than past it.
GUARD = 128
#: Exactly representable in every dtype, and not a value a stray store could
#: write: anything computed from a NaN-padded operand is NaN.
OUT_SENTINEL = -1024.0


def guarded(rows, cols, dtype, device, fill):
    """A ``rows x cols`` view into a buffer padded with ``GUARD`` cells of ``fill``."""
    buffer = torch.full((rows + 2 * GUARD, cols + 2 * GUARD), fill, device=device, dtype=dtype)
    return buffer, buffer[GUARD : GUARD + rows, GUARD : GUARD + cols]


@pytest.mark.parametrize("tile", [(32, 32, 32, 4, 3), (128, 128, 64, 8, 3)])
@pytest.mark.parametrize("w_layout", ["row-major", "column-major"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (33, 47, 61), (70, 17, 129)])
def test_no_access_outside_the_operands(shape, w_layout, tile, device):
    """Operands sit inside NaN and the output inside a sentinel value.

    A load past an operand's edge pulls NaN into the result, and a store past
    the output's edge overwrites the sentinel. Without the padding, a stray
    store lands outside the allocation, where no comparison can see it.
    """
    from kernelforge.kernels.fused_linear import fused_linear_gelu, linear_gelu_reference

    m, n, k = shape
    x_values, w_values, bias_values = inputs(m, k, n, torch.float16, device)
    _, x = guarded(m, k, torch.float16, device, float("nan"))
    if w_layout == "row-major":
        _, w = guarded(k, n, torch.float16, device, float("nan"))
    else:
        w = guarded(n, k, torch.float16, device, float("nan"))[1].t()
    bias = guarded(1, n, torch.float16, device, float("nan"))[1][0]
    x.copy_(x_values)
    w.copy_(w_values)
    bias.copy_(bias_values)
    out_buffer, out = guarded(m, n, torch.float16, device, OUT_SENTINEL)

    fused_linear_gelu(x, w, bias, config=config(*tile), out=out)
    assert_verified(
        linear_gelu_reference(x_values.double(), w_values.double(), bias_values.double()),
        out,
        dtype=torch.float16,
        context=f"fused_linear {m}x{n}x{k} tile={tile} {w_layout} w",
    )
    out.fill_(OUT_SENTINEL)
    assert (out_buffer == OUT_SENTINEL).all(), "fused_linear_gelu wrote outside its output"


def test_every_candidate_agrees_with_the_reference(device):
    from kernelforge.kernels.fused_linear import fused_linear_gelu, linear_gelu_reference
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import FusedLinearSearchSpace

    m, n, k = 257, 513, 129
    problem = Problem.create("fused_linear", "fp16", M=m, N=n, K=k)
    x, w, bias = inputs(m, k, n, torch.float16, device)
    with exact_fp32_matmul():
        expected = linear_gelu_reference(x, w, bias)
    for candidate in FusedLinearSearchSpace().candidates(problem, device_caps(device)):
        assert_verified(
            expected,
            fused_linear_gelu(x, w, bias, config=candidate),
            dtype=torch.float16,
            context=f"candidate {candidate!r}",
        )


@pytest.mark.gpu
def test_fusion_reduces_the_kernel_launch_count(device):
    """The structural claim behind the fused kernel, measured.

    The unfused sequence launches a GEMM, an add and an activation; the fused
    kernel launches one. This asserts the count rather than a latency, because
    the count is what fusion changes by construction.
    """
    from kernelforge.kernels.fused_linear import (
        DEFAULT_CONFIG,
        fused_linear_gelu,
        linear_gelu_reference,
    )
    from kernelforge.profiling import profile_kernels

    x, w, bias = inputs(1024, 1024, 1024, torch.float16, device)
    unfused = profile_kernels(
        lambda: linear_gelu_reference(x, w, bias), label="unfused", warmup=5, iterations=10
    )
    fused = profile_kernels(
        lambda: fused_linear_gelu(x, w, bias, config=DEFAULT_CONFIG),
        label="fused",
        warmup=5,
        iterations=10,
    )
    assert fused.launches_per_iteration == pytest.approx(1.0, abs=0.01)
    # At least three: a GEMM, a bias add and an activation. Not pinned to
    # exactly three because cuBLAS may split a GEMM across kernels, which is
    # its choice and not something this assertion should depend on.
    assert unfused.launches_per_iteration >= 3.0


def test_inputs_are_validated(device):
    from kernelforge.kernels.fused_linear import fused_linear_gelu

    x = torch.zeros(4, 8, device=device, dtype=torch.float16)
    w = torch.zeros(8, 16, device=device, dtype=torch.float16)
    bias = torch.zeros(16, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="bias must have shape"):
        fused_linear_gelu(x, w, torch.zeros(8, device=device, dtype=torch.float16))
    with pytest.raises(ValueError, match="shape mismatch"):
        fused_linear_gelu(x, torch.zeros(9, 16, device=device, dtype=torch.float16), bias)
    with pytest.raises(ValueError, match="dtype mismatch"):
        fused_linear_gelu(x, w.float(), bias)
    with pytest.raises(ValueError, match="out must be"):
        fused_linear_gelu(x, w, bias, out=torch.empty(4, 15, device=device, dtype=torch.float16))
    with pytest.raises(ValueError, match="out must be"):
        fused_linear_gelu(x, w, bias, out=torch.empty(4, 16, device=device, dtype=torch.float32))
