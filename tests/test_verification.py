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
