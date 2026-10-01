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
from kernelforge.testing import assert_verified
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
    expected = linear_gelu_reference(x.double(), w.double(), bias.double())
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
    expected = linear_gelu_reference(x.double(), w.double(), bias.double())
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
    expected = F.gelu(x.double(), approximate="tanh")
    assert torch.isfinite(out).all(), "GELU produced non-finite values"
    # Beyond |x| = 8, GELU is 0 or x to within 1e-20. The curve is checked on
    # its own because the bound's absolute term scales with rms(ref): next to
    # tails of 1e4 it would allow an error of 0.1 on a negative lobe whose
    # magnitude never exceeds 0.17.
    curve = x.abs() <= 8.0
    assert_verified(expected[curve], out[curve], dtype=torch.float32, context="GELU curve")
    assert_verified(expected[~curve], out[~curve], dtype=torch.float32, context="GELU tails")


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
    expected = linear_gelu_reference(x.double(), view.double(), bias.double())
    assert_verified(expected, fused_linear_gelu(x, view, bias), dtype=torch.float16)


def test_every_candidate_agrees_with_the_reference(device):
    from kernelforge.kernels.fused_linear import fused_linear_gelu, linear_gelu_reference
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import FusedLinearSearchSpace

    m, n, k = 257, 513, 129
    problem = Problem.create("fused_linear", "fp16", M=m, N=n, K=k)
    x, w, bias = inputs(m, k, n, torch.float16, device)
    expected = linear_gelu_reference(x.double(), w.double(), bias.double())
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
