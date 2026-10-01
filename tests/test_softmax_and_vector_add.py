"""Softmax and vector-add correctness. Marked ``gpu`` throughout.

Softmax gets a specific test for the numerical stability shift: without
subtracting the row maximum, fp16 ``exp`` overflows above x ~ 11, which is an
ordinary magnitude for an attention logit. The test feeds exactly that.
"""

from __future__ import annotations

import pytest
import torch

from kernelforge.benchmark import workloads
from kernelforge.testing import assert_verified
from kernelforge.tuning.config import KernelConfig, Problem

pytest.importorskip("triton")

pytestmark = pytest.mark.gpu

DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@pytest.mark.parametrize("shape", workloads._ROW_SHAPES["correctness"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_across_awkward_shapes(shape, dtype, device):
    from kernelforge.kernels.softmax import default_config, softmax

    rows, cols = shape
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(rows, cols, device=device, dtype=dtype, generator=gen)
    assert_verified(
        torch.softmax(x, dim=-1),
        softmax(x, config=default_config(cols)),
        dtype=dtype,
        context=f"softmax {rows}x{cols} {dtype}",
    )


def test_softmax_survives_logits_that_would_overflow_fp16(device):
    """The reason for subtracting the row maximum.

    fp16 tops out at 65504, so exp(x) overflows for x above about 11. These
    logits reach 60, which the naive formulation turns into inf/inf = NaN.
    """
    from kernelforge.kernels.softmax import default_config, softmax

    x = torch.full((4, 512), 60.0, device=device, dtype=torch.float16)
    x[:, 0] = 80.0
    out = softmax(x, config=default_config(512))
    assert torch.isfinite(out).all(), "softmax produced non-finite values"
    assert_verified(torch.softmax(x, dim=-1), out, dtype=torch.float16)
    assert torch.allclose(out.float().sum(dim=-1), torch.ones(4, device=device), atol=1e-2)


def test_softmax_handles_a_constant_row(device):
    """Every output should be 1/n, and the max subtraction makes it exactly so."""
    from kernelforge.kernels.softmax import default_config, softmax

    x = torch.full((2, 128), -3.5, device=device, dtype=torch.float32)
    out = softmax(x, config=default_config(128))
    assert torch.allclose(out, torch.full_like(out, 1.0 / 128), atol=1e-6)


@pytest.mark.parametrize("rows_per_program", [1, 2, 4, 8])
def test_softmax_rows_per_program(rows_per_program, device):
    from kernelforge.kernels.softmax import softmax
    from kernelforge.tuning.search import next_power_of_2

    rows, cols = 37, 257
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(rows, cols, device=device, dtype=torch.float16, generator=gen)
    config = KernelConfig(
        "softmax",
        BLOCK_SIZE=next_power_of_2(cols),
        ROWS_PER_PROGRAM=rows_per_program,
        num_warps=4,
    )
    assert_verified(torch.softmax(x, dim=-1), softmax(x, config=config), dtype=torch.float16)


def test_softmax_candidates_all_agree(device):
    from kernelforge.kernels.softmax import softmax
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import SoftmaxSearchSpace

    rows, cols = 129, 1024
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(rows, cols, device=device, dtype=torch.float16, generator=gen)
    expected = torch.softmax(x, dim=-1)
    problem = Problem.create("softmax", "fp16", rows=rows, cols=cols)
    for candidate in SoftmaxSearchSpace().candidates(problem, device_caps(device)):
        assert_verified(
            expected, softmax(x, config=candidate), dtype=torch.float16, context=f"{candidate!r}"
        )


def test_softmax_rejects_non_2d_input(device):
    from kernelforge.kernels.softmax import softmax

    with pytest.raises(ValueError, match="2D"):
        softmax(torch.zeros(2, 3, 4, device=device, dtype=torch.float16))


@pytest.mark.parametrize("n", workloads._VECTOR_SHAPES["correctness"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_add_across_sizes(n, dtype, device):
    from kernelforge.kernels.vector_add import DEFAULT_CONFIG, vector_add

    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(n, device=device, dtype=dtype, generator=gen)
    b = torch.randn(n, device=device, dtype=dtype, generator=gen)
    assert_verified(a + b, vector_add(a, b, config=DEFAULT_CONFIG), dtype=dtype, context=f"n={n}")


def test_vector_add_candidates_all_agree(device):
    from kernelforge.kernels.vector_add import vector_add
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import VectorAddSearchSpace

    n = 1_000_003
    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(n, device=device, dtype=torch.float16, generator=gen)
    b = torch.randn(n, device=device, dtype=torch.float16, generator=gen)
    expected = a + b
    problem = Problem.create("vector_add", "fp16", n=n)
    for candidate in VectorAddSearchSpace().candidates(problem, device_caps(device)):
        assert_verified(expected, vector_add(a, b, config=candidate), dtype=torch.float16)


def test_vector_add_validates_inputs(device):
    from kernelforge.kernels.vector_add import vector_add

    a = torch.zeros(8, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="shape mismatch"):
        vector_add(a, torch.zeros(9, device=device, dtype=torch.float16))
    with pytest.raises(ValueError, match="dtype mismatch"):
        vector_add(a, torch.zeros(8, device=device, dtype=torch.float32))
