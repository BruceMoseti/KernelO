"""RMSNorm correctness.

Marked ``gpu`` throughout. Beyond the usual awkward row widths, two things get
specific attention:

* **The fp32 reduction.** A kernel that accumulates the sum of squares in the
  input dtype gives visibly wrong answers for wide rows in fp16, so there is a
  test that would fail if the upcast were removed.
* **ROWS_PER_PROGRAM.** Processing several rows per program is where a
  row-indexing bug hides, and it only shows up when the row count is not a
  multiple of the stride.
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

CORRECTNESS_SHAPES = [
    tuple(p.dims_dict.values()) for p in workloads.problems("rmsnorm", "correctness", "fp16")
]


def inputs(rows, cols, dtype, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, cols, device=device, dtype=dtype, generator=gen)
    gamma = torch.randn(cols, device=device, dtype=torch.float32, generator=gen)
    return x, (gamma * 0.1 + 1.0).to(dtype)


@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_default_config_across_awkward_shapes(shape, dtype, device):
    from kernelforge.kernels.rmsnorm import default_config, rmsnorm, rmsnorm_reference

    rows, cols = shape
    x, gamma = inputs(rows, cols, dtype, device)
    assert_verified(
        rmsnorm_reference(x, gamma),
        rmsnorm(x, gamma, config=default_config(cols)),
        dtype=dtype,
        context=f"rmsnorm {rows}x{cols} {dtype}",
    )


@pytest.mark.parametrize("rows_per_program", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [1, 3, 7, 17, 64, 129])
def test_rows_per_program_with_an_indivisible_row_count(rows_per_program, rows, device):
    """The tail program handles fewer rows than its stride."""
    from kernelforge.kernels.rmsnorm import rmsnorm, rmsnorm_reference
    from kernelforge.tuning.search import next_power_of_2

    cols = 513
    x, gamma = inputs(rows, cols, torch.float16, device)
    config = KernelConfig(
        "rmsnorm",
        BLOCK_SIZE=next_power_of_2(cols),
        ROWS_PER_PROGRAM=rows_per_program,
        num_warps=4,
    )
    assert_verified(
        rmsnorm_reference(x, gamma),
        rmsnorm(x, gamma, config=config),
        dtype=torch.float16,
        context=f"rmsnorm rows={rows} rpp={rows_per_program}",
    )


def test_wide_fp16_rows_need_the_fp32_reduction(device):
    """The test that fails if the upcast is removed.

    A sum of squares over 8192 unit-variance fp16 values reaches ~8192, where
    11 bits of mantissa cannot resolve the individual terms, and the resulting
    RMS is visibly wrong. The reference reduces in fp32, so a kernel that did
    not would miss the tolerance.
    """
    from kernelforge.kernels.rmsnorm import default_config, rmsnorm, rmsnorm_reference

    x, gamma = inputs(64, 8192, torch.float16, device)
    result = assert_verified(
        rmsnorm_reference(x, gamma),
        rmsnorm(x, gamma, config=default_config(8192)),
        dtype=torch.float16,
    )
    assert result.error < 5e-3

    # And a genuinely fp16 reduction really is worse, so the test has teeth.
    #
    # `dtype=torch.float16` on the sum is load-bearing: PyTorch's CUDA
    # reduction for half promotes to fp32 internally, so `x.pow(2).mean(-1)`
    # would *also* accumulate in fp32 and the comparison would isolate nothing.
    from kernelforge.testing import verify

    naive_sum = x.pow(2).sum(dim=-1, keepdim=True, dtype=torch.float16)
    naive_rms = torch.rsqrt(naive_sum.to(torch.float32) / x.shape[-1] + 1e-5)
    naive = (x * naive_rms * gamma).to(torch.float16)
    assert verify(rmsnorm_reference(x, gamma), naive, dtype=torch.float16).error > result.error


def test_all_zero_row_is_finite(device):
    """eps inside the square root is what keeps this defined."""
    from kernelforge.kernels.rmsnorm import default_config, rmsnorm, rmsnorm_reference

    x = torch.zeros(4, 256, device=device, dtype=torch.float16)
    gamma = torch.ones(256, device=device, dtype=torch.float16)
    out = rmsnorm(x, gamma, config=default_config(256))
    assert torch.isfinite(out).all()
    assert_verified(rmsnorm_reference(x, gamma), out, dtype=torch.float16)


def test_every_candidate_agrees_with_the_reference(device):
    from kernelforge.kernels.rmsnorm import rmsnorm, rmsnorm_reference
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import RMSNormSearchSpace

    rows, cols = 333, 1024
    problem = Problem.create("rmsnorm", "fp16", rows=rows, cols=cols)
    x, gamma = inputs(rows, cols, torch.float16, device)
    expected = rmsnorm_reference(x, gamma)
    for candidate in RMSNormSearchSpace().candidates(problem, device_caps(device)):
        assert_verified(
            expected,
            rmsnorm(x, gamma, config=candidate),
            dtype=torch.float16,
            context=f"candidate {candidate!r}",
        )


def test_single_pass_limit_is_an_explicit_error(device):
    """A row too wide to hold must fail loudly, not silently truncate."""
    from kernelforge.kernels.rmsnorm import rmsnorm
    from kernelforge.tuning.search import RMSNormSearchSpace

    cols = RMSNormSearchSpace.MAX_BLOCK_SIZE * 2
    x = torch.zeros(1, cols, device=device, dtype=torch.float16)
    gamma = torch.ones(cols, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="single-pass limit"):
        rmsnorm(x, gamma)


def test_undersized_block_is_rejected(device):
    from kernelforge.kernels.rmsnorm import rmsnorm

    x, gamma = inputs(8, 1024, torch.float16, device)
    config = KernelConfig("rmsnorm", BLOCK_SIZE=256, ROWS_PER_PROGRAM=1, num_warps=4)
    with pytest.raises(ValueError, match="cannot hold a row"):
        rmsnorm(x, gamma, config=config)


def test_gamma_shape_is_validated(device):
    from kernelforge.kernels.rmsnorm import rmsnorm

    x = torch.zeros(4, 128, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="gamma must have shape"):
        rmsnorm(x, torch.ones(64, device=device, dtype=torch.float16))
