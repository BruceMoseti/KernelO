"""Correctness gate.

The autotuner must never rank an incorrect configuration, so every candidate
passes through :func:`verify` before it is allowed near the benchmark loop.

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


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    error: float
    max_abs_err: float
    mismatched: int
    total: int
    threshold: float
    reason: str = ""

    def __str__(self) -> str:
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

    ``dtype`` selects the threshold and defaults to the output dtype; pass it
    explicitly when the reference was computed in higher precision than the
    kernel it is checking.
    """
    gate_dtype = dtype if dtype is not None else output.dtype
    limit = threshold if threshold is not None else threshold_for(gate_dtype)

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

    # Compare in fp32: the error itself must not be subject to the rounding of
    # the format under test.
    ref = reference.detach().to(torch.float32)
    out = output.detach().to(torch.float32)
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
