"""Correctness checks for kernel outputs.

Every tuning candidate and every kernel test goes through `verify`: an output that fails is
discarded and never benchmarked or ranked.

The reference must be computed in float64 from the same (low-precision) inputs the kernel saw.
Upcasting to float64 is exact and the float64 computation's own error is negligible, so the
measured error belongs to the kernel alone. An element passes when

    |actual - reference| <= rtol * |reference| + atol * rms(reference) + spacing

- `rtol` covers rounding the result to the output dtype (at most half an ulp) plus error from
  fp32 arithmetic inside the kernel. For fp16 and bf16 it is one ulp of the output dtype, twice
  the rounding bound. fp32 has no lower-precision rounding step, so its rtol is 2^-16 (128 fp32
  ulps), which covers approximate exp and division (a few ulps each), softmax's argument
  reduction (about |x| ulps), and tree reductions (about log2(n) ulps).
- `atol * rms(reference)` covers outputs near zero produced by cancellation in reductions
  (MatMul), whose absolute error is set by the fp32 accumulator, not by the output's own size.
  fp32 accumulation error grows roughly like sqrt(K) * 2^-24 * rms(reference) with
  round-to-nearest; tensor cores truncate inside each MMA, which grows faster. 2^-12 leaves
  headroom for K up to several thousand under pessimistic truncation estimates. It is still far
  below the error of real bugs: a dropped K term (about rms / sqrt(K)), fp16 accumulation, or
  TF32 inputs when IEEE fp32 was intended. Scaling by rms(reference) makes the check
  scale-invariant.
- `spacing` is the subnormal spacing of the output dtype (smallest normal times eps). No output
  can be more accurate than this; it matters for fp16, for example in softmax over long rows.

All kernels in this package accumulate in fp32. The tolerances assume that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Tolerance:
    rtol: float  # relative to |reference|, elementwise
    atol: float  # relative to rms(reference)


TOLERANCES: dict[torch.dtype, Tolerance] = {
    torch.float32: Tolerance(rtol=2**-16, atol=2**-12),
    torch.float16: Tolerance(rtol=2**-10, atol=2**-12),
    torch.bfloat16: Tolerance(rtol=2**-7, atol=2**-12),
}


@dataclass(frozen=True)
class Verification:
    passed: bool
    mismatches: int  # elements outside tolerance
    max_abs_error: float
    max_error_ratio: float  # max over elements of |error| / allowed error; passed iff <= 1

    def describe(self) -> str:
        return (
            f"{self.mismatches} elements outside tolerance "
            f"(max error {self.max_abs_error:.3g}, {self.max_error_ratio:.3g}x the allowed error)"
        )


def verify(reference: torch.Tensor, actual: torch.Tensor, dtype: torch.dtype) -> Verification:
    """Compare a kernel output of `dtype` against a float64 reference."""
    if reference.dtype != torch.float64:
        raise ValueError(f"reference must be float64, got {reference.dtype}")
    if actual.dtype != dtype:
        raise ValueError(f"output dtype {actual.dtype} does not match {dtype}")
    if actual.shape != reference.shape:
        raise ValueError(f"output shape {tuple(actual.shape)} != {tuple(reference.shape)}")
    if not torch.isfinite(reference).all():
        raise ValueError("reference contains non-finite values")

    tolerance = TOLERANCES[dtype]
    finfo = torch.finfo(dtype)
    rms = reference.square().mean().sqrt()
    floor = tolerance.atol * rms + finfo.smallest_normal * finfo.eps
    allowed = tolerance.rtol * reference.abs() + floor
    error = (actual.double() - reference).abs().nan_to_num(nan=float("inf"))
    within = error <= allowed
    return Verification(
        passed=bool(within.all()),
        mismatches=int((~within).sum()),
        max_abs_error=float(error.max()),
        max_error_ratio=float((error / allowed).max()),
    )
