"""Softmax and vector-add correctness.

Tests marked ``gpu`` need a CUDA device. The rest also run on CPU through
Triton's interpreter (``TRITON_INTERPRET=1``), which is how CI executes them.

Softmax gets a specific test for the numerical stability shift: the kernel
exponentiates in fp32, so without subtracting the row maximum ``exp`` overflows
above x ~ 88.7. The test feeds logits above that.
"""

from __future__ import annotations

import pytest
import torch

from kernelforge.benchmark import workloads
from kernelforge.testing import assert_verified
from kernelforge.tuning.config import KernelConfig, Problem

pytest.importorskip("triton")

DTYPES = (torch.float16, torch.bfloat16, torch.float32)

#: Vector add does its arithmetic in the input dtype, and Triton's interpreter
#: does bf16 arithmetic on the raw storage bits, so bf16 needs a GPU. Softmax
#: converts to fp32 first and runs in every dtype on CPU.
VECTOR_ADD_DTYPES = [
    pytest.param(dtype, marks=pytest.mark.gpu) if dtype == torch.bfloat16 else dtype
    for dtype in DTYPES
]

CORRECTNESS_ROW_SHAPES = [
    tuple(p.dims_dict.values()) for p in workloads.problems("softmax", "correctness", "fp16")
]
CORRECTNESS_ELEMENT_COUNTS = [
    p["n"] for p in workloads.problems("vector_add", "correctness", "fp16")
]


@pytest.mark.parametrize("shape", CORRECTNESS_ROW_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_across_awkward_shapes(shape, dtype, device):
    from kernelforge.kernels.softmax import default_config, softmax

    rows, cols = shape
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(rows, cols, device=device, dtype=dtype, generator=gen)
    assert_verified(
        torch.softmax(x.double(), dim=-1),
        softmax(x, config=default_config(cols)),
        dtype=dtype,
        context=f"softmax {rows}x{cols} {dtype}",
    )


def test_softmax_survives_logits_that_would_overflow_fp32(device):
    """The reason for subtracting the row maximum.

    The kernel exponentiates in fp32, where exp(x) overflows for x above about
    88.7. These logits reach 100, which the naive formulation turns into
    inf/inf = NaN.
    """
    from kernelforge.kernels.softmax import default_config, softmax

    x = torch.full((4, 512), 90.0, device=device, dtype=torch.float16)
    x[:, 0] = 100.0
    out = softmax(x, config=default_config(512))
    assert torch.isfinite(out).all(), "softmax produced non-finite values"
    assert_verified(torch.softmax(x.double(), dim=-1), out, dtype=torch.float16)
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
    assert_verified(
        torch.softmax(x.double(), dim=-1), softmax(x, config=config), dtype=torch.float16
    )


def test_softmax_candidates_all_agree(device):
    from kernelforge.kernels.softmax import softmax
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import SoftmaxSearchSpace

    rows, cols = 129, 1024
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(rows, cols, device=device, dtype=torch.float16, generator=gen)
    expected = torch.softmax(x.double(), dim=-1)
    problem = Problem.create("softmax", "fp16", rows=rows, cols=cols)
    for candidate in SoftmaxSearchSpace().candidates(problem, device_caps(device)):
        assert_verified(
            expected, softmax(x, config=candidate), dtype=torch.float16, context=f"{candidate!r}"
        )


def test_softmax_rejects_non_2d_input(device):
    from kernelforge.kernels.softmax import softmax

    with pytest.raises(ValueError, match="2D"):
        softmax(torch.zeros(2, 3, 4, device=device, dtype=torch.float16))


@pytest.mark.parametrize("n", CORRECTNESS_ELEMENT_COUNTS)
@pytest.mark.parametrize("dtype", VECTOR_ADD_DTYPES)
def test_vector_add_across_sizes(n, dtype, device):
    from kernelforge.kernels.vector_add import DEFAULT_CONFIG, vector_add

    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(n, device=device, dtype=dtype, generator=gen)
    b = torch.randn(n, device=device, dtype=dtype, generator=gen)
    assert_verified(
        a.double() + b.double(),
        vector_add(a, b, config=DEFAULT_CONFIG),
        dtype=dtype,
        context=f"n={n}",
    )


def test_vector_add_candidates_all_agree(device):
    from kernelforge.kernels.vector_add import vector_add
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import VectorAddSearchSpace

    n = 1_000_003
    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(n, device=device, dtype=torch.float16, generator=gen)
    b = torch.randn(n, device=device, dtype=torch.float16, generator=gen)
    expected = a.double() + b.double()
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
