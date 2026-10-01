"""Correctness-gate tests.

The gate decides which configurations the tuner is allowed to rank, so it has
to fail on the failure modes that actually occur in GPU kernels: a masking bug
that leaves part of the output untouched, a NaN from an unstable reduction, and
a wrong shape.
"""

from __future__ import annotations

import pytest
import torch

from kernelforge.testing import (
    ELEMENTWISE_TOLERANCES,
    ERROR_THRESHOLDS,
    assert_verified,
    exact_fp32_matmul,
    threshold_for,
    verify,
)


def test_identical_tensors_pass_with_zero_error():
    x = torch.randn(64, 64)
    result = verify(x, x.clone())
    assert result.passed
    assert result.error == 0.0
    assert result.mismatched == 0
    assert result.total == 64 * 64


def test_rounding_to_fp16_passes():
    """A correct fp16 kernel differs from an fp32 reference by output rounding."""
    reference = torch.randn(256, 256)
    result = verify(reference, reference.to(torch.float16), dtype=torch.float16)
    assert result.passed


def test_rounding_to_fp16_fails_the_fp32_threshold():
    """The threshold is selected by dtype, and it matters."""
    reference = torch.randn(256, 256)
    result = verify(reference, reference.to(torch.float16).to(torch.float32))
    assert not result.passed
    assert "normalised error" in result.reason


def test_unwritten_output_region_is_caught():
    """The failure mode of a GEMM with a broken boundary mask.

    A kernel that writes only whole tiles leaves the ragged edge at whatever
    the output buffer held. Zeroing a single column of a 128-column output is
    a 0.8% error by element count, and the gate has to catch it.
    """
    reference = torch.randn(128, 128)
    broken = reference.clone()
    broken[:, -1] = 0.0
    result = verify(reference, broken)
    assert not result.passed
    assert result.mismatched > 0


def test_non_finite_output_is_caught_before_any_arithmetic():
    reference = torch.randn(32, 32)
    broken = reference.clone()
    broken[0, 0] = float("nan")
    result = verify(reference, broken)
    assert not result.passed
    assert "non-finite" in result.reason
    assert result.error == float("inf")

    broken[0, 0] = float("inf")
    assert not verify(reference, broken).passed


def test_shape_mismatch_is_reported_as_such():
    result = verify(torch.randn(8, 8), torch.randn(8, 9))
    assert not result.passed
    assert "shape mismatch" in result.reason


def test_zero_reference_falls_back_to_absolute_error():
    zeros = torch.zeros(16, 16)
    assert verify(zeros, zeros.clone()).passed
    assert not verify(zeros, torch.full((16, 16), 0.5)).passed


def test_empty_tensors_are_trivially_verified():
    assert verify(torch.empty(0), torch.empty(0)).passed


def test_error_is_scale_invariant():
    """One threshold has to hold across magnitudes.

    A GEMM's output grows like sqrt(K), so an absolute tolerance tuned at one
    K is wrong at another. Scaling both tensors must not change the verdict.
    """
    reference = torch.randn(128, 128)
    output = reference + 1e-4 * torch.randn(128, 128)
    small = verify(reference, output)
    large = verify(reference * 1000, output * 1000)
    # Equal to the precision of the fp32 arithmetic doing the scaling.
    assert small.error == pytest.approx(large.error, rel=1e-3)


@pytest.mark.parametrize("dtype", list(ERROR_THRESHOLDS))
def test_every_supported_dtype_has_a_threshold(dtype):
    assert threshold_for(dtype) > 0


def test_missing_threshold_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="no correctness threshold"):
        threshold_for(torch.int32)


def test_explicit_threshold_overrides_the_table():
    reference = torch.randn(32, 32)
    output = reference + 0.1
    assert not verify(reference, output).passed
    assert verify(reference, output, threshold=10.0).passed


def test_assert_verified_raises_with_context():
    with pytest.raises(AssertionError, match="BLOCK_M=64"):
        assert_verified(torch.zeros(4), torch.ones(4), context="BLOCK_M=64")


def test_assert_verified_returns_the_result_on_success():
    x = torch.randn(4)
    assert assert_verified(x, x.clone()).passed


def test_exact_fp32_matmul_restores_the_previous_setting():
    """TF32 is a global switch; the context manager must not leak."""
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with exact_fp32_matmul():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before


def test_exact_fp32_matmul_restores_on_exception():
    torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(RuntimeError), exact_fp32_matmul():
        raise RuntimeError("boom")
    assert torch.backends.cuda.matmul.allow_tf32 is True


# --- Against an exact (float64) reference ------------------------------------
#
# Each bug class below is simulated in PyTorch at a realistic size. The
# normalised maximum passes every one of them; the elementwise bound must not.


def _gemm_operands(m, n, k, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(m, k, generator=gen).to(dtype), torch.randn(k, n, generator=gen).to(dtype)


def _fp32_accumulated(a, b):
    """What a correct kernel produces: fp32 accumulation, rounded once to the output format."""
    return (a.float() @ b.float()).to(a.dtype)


@pytest.mark.parametrize("k", [16, 127, 1025, 4096])
@pytest.mark.parametrize("dtype", list(ELEMENTWISE_TOLERANCES))
def test_correct_gemm_passes_the_elementwise_bound_with_headroom(dtype, k):
    a, b = _gemm_operands(128, 96, k, dtype)
    result = verify(a.double() @ b.double(), _fp32_accumulated(a, b), dtype=dtype)
    assert result.passed, result.reason
    # Rounding to fp16 or bf16 uses at most half of the one-ulp budget.
    assert result.max_error_ratio < 0.6


@pytest.mark.parametrize("k", [256, 1024, 2048])
def test_fp16_accumulation_is_rejected(k):
    """fp16-accumulate MMA runs at twice the rate on GeForce tensor cores, so a
    kernel that took it would also win the ranking if the gate let it through."""
    a, b = _gemm_operands(1024, 1024, k, torch.float16)
    acc = torch.zeros(1024, 1024, dtype=torch.float16)
    for k0 in range(0, k, 16):
        acc = (acc.float() + a[:, k0 : k0 + 16].float() @ b[k0 : k0 + 16].float()).half()
    result = verify(a.double() @ b.double(), acc, dtype=torch.float16)
    assert not result.passed
    assert "elements outside" in result.reason


@pytest.mark.parametrize("dtype", list(ELEMENTWISE_TOLERANCES))
def test_dropped_k_term_is_rejected(dtype):
    a, b = _gemm_operands(128, 96, 1025, dtype)
    dropped = (a[:, :-1].float() @ b[:-1].float()).to(dtype)
    assert not verify(a.double() @ b.double(), dropped, dtype=dtype).passed


def test_tf32_inputs_fail_the_fp32_bound():
    a, b = _gemm_operands(128, 96, 1025, torch.float32)

    def tf32(t):
        return ((t.view(torch.int32) + 0x1000) & ~0x1FFF).view(torch.float32)

    assert not verify(a.double() @ b.double(), tf32(a) @ tf32(b), dtype=torch.float32).passed


def test_rmsnorm_averaging_over_the_padded_block_is_rejected():
    """Dividing the sum of squares by BLOCK_SIZE=1024 instead of 1020 columns."""
    x = torch.randn(2048, 1020, generator=torch.Generator().manual_seed(0)).half().double()

    def rmsnorm(width):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) / width + 1e-5)

    assert verify(rmsnorm(1020), rmsnorm(1020).half(), dtype=torch.float16).passed
    assert not verify(rmsnorm(1020), rmsnorm(1024).half(), dtype=torch.float16).passed


def test_softmax_padded_with_zero_instead_of_minus_infinity_is_rejected():
    """Padding lanes loaded as 0 add exp(0 - max) each to the denominator."""
    x = torch.randn(2048, 1020, generator=torch.Generator().manual_seed(0)).half().double()
    padded = torch.cat([x, torch.zeros(2048, 4, dtype=torch.float64)], dim=-1)
    buggy = torch.softmax(padded, dim=-1)[:, :1020]
    reference = torch.softmax(x, dim=-1)
    assert verify(reference, reference.half(), dtype=torch.float16).passed
    assert not verify(reference, buggy.half(), dtype=torch.float16).passed


def test_erf_gelu_is_rejected_where_tanh_gelu_is_specified():
    x = torch.randn(512, 1024, generator=torch.Generator().manual_seed(0)).half().double()
    reference = torch.nn.functional.gelu(x, approximate="tanh")
    erf = torch.nn.functional.gelu(x)
    assert verify(reference, reference.half(), dtype=torch.float16).passed
    assert not verify(reference, erf.half(), dtype=torch.float16).passed


def test_subnormal_outputs_are_held_only_to_the_format_spacing():
    """fp16 softmax over a long row has outputs below fp16's smallest normal,
    which fp16 represents only to its subnormal spacing; without that floor
    this correctly rounded result would be rejected."""
    x = torch.randn(8, 32768, generator=torch.Generator().manual_seed(0)).half()
    reference = torch.softmax(x.double(), dim=-1)
    result = verify(reference, torch.softmax(x.float(), dim=-1).half(), dtype=torch.float16)
    assert result.passed, result.reason
    assert result.max_error_ratio < 0.6


def test_elementwise_bound_is_scale_invariant():
    a, b = _gemm_operands(64, 64, 512, torch.float16)
    base = verify(a.double() @ b.double(), _fp32_accumulated(a, b), dtype=torch.float16)
    a = a * 64
    scaled = verify(a.double() @ b.double(), _fp32_accumulated(a, b), dtype=torch.float16)
    assert scaled.max_error_ratio == pytest.approx(base.max_error_ratio, rel=1e-6)


def test_non_finite_exact_reference_fails_rather_than_passing_everything():
    reference = torch.zeros(4, dtype=torch.float64)
    reference[0] = float("inf")
    result = verify(reference, torch.zeros(4, dtype=torch.float32))
    assert not result.passed
    assert "non-finite" in result.reason


def test_explicit_threshold_keeps_the_normalised_comparison():
    reference = torch.randn(32, 32, dtype=torch.float64)
    output = (reference + 0.1).float()
    assert not verify(reference, output).passed
    result = verify(reference, output, threshold=10.0)
    assert result.passed
    assert result.max_error_ratio is None
