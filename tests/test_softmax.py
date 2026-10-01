"""Softmax correctness.

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

#: Softmax converts to fp32 first and runs in every dtype on CPU.
DTYPES = (torch.float16, torch.bfloat16, torch.float32)

CORRECTNESS_ROW_SHAPES = [
    tuple(p.dims_dict.values()) for p in workloads.problems("softmax", "correctness", "fp16")
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


@pytest.mark.parametrize("seed", range(24))
def test_softmax_randomised_shapes(seed, device):
    """Random shapes, dtypes and configurations. Catches what a fixed list does not.

    Widths are log-uniform up to the single-pass limit of 16384. A row of 16
    columns or fewer has no candidates, since every configuration would idle
    threads, so it runs the default configuration, as dispatch would.
    """
    import random

    from kernelforge.kernels.softmax import default_config, softmax
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import SoftmaxSearchSpace

    # Seeded by name, so that each kernel's suite draws its own shapes.
    rng = random.Random(f"softmax-{seed}")
    rows, cols = rng.randint(1, 600), round(2 ** rng.uniform(0, 14))
    dtype = rng.choice(DTYPES)
    problem = Problem.create("softmax", dtype, rows=rows, cols=cols)
    candidates = SoftmaxSearchSpace().candidates(problem, device_caps(device))
    chosen = rng.choice(candidates) if candidates else default_config(cols)

    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, cols, device=device, dtype=dtype, generator=gen)
    assert_verified(
        torch.softmax(x.double(), dim=-1),
        softmax(x, config=chosen),
        dtype=dtype,
        context=f"softmax {rows}x{cols} {dtype} {chosen!r}",
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
