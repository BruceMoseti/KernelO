"""Correctness gate.

The autotuner must never rank an incorrect configuration, so every candidate
passes through :func:`verify` before it is allowed near the benchmark loop.

**Against an exact reference: elementwise.** A reference computed in float64
from the same low-precision inputs the kernel received is exact for this
purpose: upcasting is lossless and float64's own rounding is negligible, so
the whole difference belongs to the kernel. Every element must then satisfy

    |out - ref| <= rtol * |ref| + atol * rms(ref) + spacing

``rtol`` covers rounding to the output format plus fp32 arithmetic inside
the kernel; ``atol * rms(ref)`` covers outputs near zero from cancellation in
a reduction, whose error is set by the fp32 accumulator rather than by the
element's own size, and keeps the bound scale invariant; ``spacing`` is the
format's subnormal spacing, below which no output can be more accurate. A
normalised maximum lets an error hide wherever the reference is small, which
is how a GEMM accumulating in fp16 passed it; an elementwise bound does not.

**Between two implementations: the normalised infinity norm.** Any other
reference is a second implementation with rounding of its own, compared as
follows.

**Why the normalised infinity norm rather than ``torch.allclose``.**
``allclose`` needs an absolute tolerance that depends on the magnitude of the
data, which for a GEMM grows like ``sqrt(K)``: a tolerance tuned for
``K=512`` either rejects correct kernels at ``K=8192`` or waves through broken
ones at ``K=128``. The gate used here is scale invariant,

    err = max|out - ref| / max|ref|

so one threshold per dtype holds across every shape. Elementwise mismatch
counts are still computed and reported, because "0.4% of elements differ" and
"every element differs slightly" are very different bugs.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

# Thresholds on the normalised infinity-norm error. Derived from the rounding
# behaviour of each format, with headroom for a differing summation order:
# a correct kernel differs from the reference only by output rounding
# (fp16: 2**-11 ~ 4.9e-4, bf16: 2**-8 ~ 3.9e-3) plus accumulation order.
# fp32 assumes the reference ran with TF32 disabled -- see `exact_fp32_matmul`.
ERROR_THRESHOLDS: dict[torch.dtype, float] = {
    torch.float32: 1e-5,
    torch.float16: 5e-3,
    torch.bfloat16: 2e-2,
}

# (rtol, atol) of the elementwise bound against a float64 reference. rtol is
# one ulp of the output format for fp16 and bf16: twice the half-ulp of a
# correctly rounded result, with the rest left for fp32 arithmetic before the
# rounding. fp32 output has no final rounding to absorb that arithmetic, so its
# rtol is 2**-16 (128 ulps), enough for approximate exp and division and for a
# different summation order. atol is relative to rms(ref): fp32 accumulation
# error grows like sqrt(K) * 2**-24 * rms(ref), and 2**-12 leaves headroom for
# several thousand terms while staying far below a dropped term (~rms/sqrt(K)).
# All kernels here accumulate in fp32; the bound assumes it.
ELEMENTWISE_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (2**-16, 2**-12),
    torch.float16: (2**-10, 2**-12),
    torch.bfloat16: (2**-7, 2**-12),
}


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    error: float
    max_abs_err: float
    mismatched: int
    total: int
    threshold: float
    reason: str = ""
    #: Against a float64 reference only: the worst element's error divided by
    #: its allowance. The comparison passes at 1.0 or below, and ``threshold``
    #: is then 1.0.
    max_error_ratio: float | None = None

    def __str__(self) -> str:
        if self.passed and self.max_error_ratio is not None:
            return f"ok (worst element at {self.max_error_ratio:.2f}x its allowance)"
        if self.passed:
            return f"ok (err={self.error:.3e} <= {self.threshold:.1e})"
        return f"FAIL ({self.reason})"


def threshold_for(dtype: torch.dtype) -> float:
    try:
        return ERROR_THRESHOLDS[dtype]
    except KeyError:
        raise ValueError(
            f"no correctness threshold defined for {dtype}; "
            f"known dtypes: {sorted(str(d) for d in ERROR_THRESHOLDS)}"
        ) from None


def verify(
    reference: torch.Tensor,
    output: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    threshold: float | None = None,
) -> VerificationResult:
    """Compare a kernel output against a reference.

    A float64 reference is taken as exact and checked elementwise; any other
    reference is compared by normalised error (see the module docstring). An
    explicit ``threshold`` always selects the normalised comparison.

    ``dtype`` selects the tolerance and defaults to the output dtype; pass it
    explicitly when the reference was computed in higher precision than the
    kernel it is checking.
    """
    gate_dtype = dtype if dtype is not None else output.dtype
    limit = threshold if threshold is not None else threshold_for(gate_dtype)
    exact = reference.dtype == torch.float64 and threshold is None

    if reference.shape != output.shape:
        return VerificationResult(
            passed=False,
            error=float("inf"),
            max_abs_err=float("inf"),
            mismatched=output.numel(),
            total=output.numel(),
            threshold=limit,
            reason=f"shape mismatch: reference {tuple(reference.shape)} vs output {tuple(output.shape)}",
        )

    # Compare in fp32, or in float64 against an exact reference: the error
    # itself must not be subject to the rounding of the format under test.
    work_dtype = torch.float64 if exact else torch.float32
    ref = reference.detach().to(work_dtype)
    out = output.detach().to(work_dtype)
    total = out.numel()

    if total == 0:
        return VerificationResult(True, 0.0, 0.0, 0, 0, limit)

    if not torch.isfinite(out).all():
        n_bad = int((~torch.isfinite(out)).sum())
        return VerificationResult(
            passed=False,
            error=float("inf"),
            max_abs_err=float("inf"),
            mismatched=n_bad,
            total=total,
            threshold=limit,
            reason=f"output contains {n_bad}/{total} non-finite values",
        )

    diff = (out - ref).abs()
    max_abs_err = float(diff.max())
    scale = float(ref.abs().max())
    # An all-zero reference is a degenerate but legitimate case (e.g. a zero
    # bias); fall back to the absolute error so the division stays meaningful.
    error = max_abs_err / scale if scale > 0 else max_abs_err
    if exact:
        return _verify_elementwise(ref, diff, gate_dtype, error, max_abs_err)
    passed = error <= limit
    # The element count is measured against the same bound as the verdict, so
    # a failing comparison can never report that nothing differs.
    budget = limit * scale if scale > 0 else limit
    mismatched = int((diff > budget).sum())

    return VerificationResult(
        passed=passed,
        error=error,
        max_abs_err=max_abs_err,
        mismatched=mismatched,
        total=total,
        threshold=limit,
        reason=(
            ""
            if passed
            else (
                f"normalised error {error:.3e} exceeds {limit:.1e} "
                f"(max|diff|={max_abs_err:.3e}, {mismatched}/{total} elements off)"
            )
        ),
    )


def _verify_elementwise(
    ref: torch.Tensor, diff: torch.Tensor, dtype: torch.dtype, error: float, max_abs_err: float
) -> VerificationResult:
    total = ref.numel()
    # An infinite reference would make the rms term, and so every allowance,
    # infinite: nothing could fail.
    if not torch.isfinite(ref).all():
        return VerificationResult(
            False, error, max_abs_err, total, total, 1.0, "reference contains non-finite values"
        )
    rtol, atol = ELEMENTWISE_TOLERANCES[dtype]
    finfo = torch.finfo(dtype)
    rms = ref.square().mean().sqrt()
    allowed = rtol * ref.abs() + (atol * rms + finfo.smallest_normal * finfo.eps)
    ratio = float((diff / allowed).max())
    mismatched = int((diff > allowed).sum())
    passed = mismatched == 0
    return VerificationResult(
        passed=passed,
        error=error,
        max_abs_err=max_abs_err,
        mismatched=mismatched,
        total=total,
        threshold=1.0,
        reason=(
            ""
            if passed
            else (
                f"{mismatched}/{total} elements outside the {dtype} bound "
                f"(worst at {ratio:.2f}x its allowance, max|diff|={max_abs_err:.3e})"
            )
        ),
        max_error_ratio=ratio,
    )


def assert_verified(
    reference: torch.Tensor,
    output: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    threshold: float | None = None,
    context: str = "",
) -> VerificationResult:
    result = verify(reference, output, dtype=dtype, threshold=threshold)
    if not result.passed:
        prefix = f"{context}: " if context else ""
        raise AssertionError(f"{prefix}{result.reason}")
    return result


@contextlib.contextmanager
def exact_fp32_matmul():
    """Disable TF32 for the duration of the block.

    On Ampere and later, PyTorch may run an fp32 matmul through TF32 tensor
    cores, which carries ~1e-3 relative error -- a hundred times the fp32
    threshold above. Any fp32 reference must therefore be computed with TF32
    off, or the "reference" is itself the least accurate number in the
    comparison.
    """
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
