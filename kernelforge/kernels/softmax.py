"""Fused row-wise softmax: y = exp(x - max(x)) / sum(exp(x - max(x))) over the last dimension.

"Fused" means each row is read from global memory once and written once: the max, the
exponentials, the sum and the division all happen in registers, so the kernel moves the
minimum 2 * rows * cols elements.

One program owns one row and holds it entirely in registers as a BLOCK_SIZE vector,
BLOCK_SIZE = next_power_of_2(cols). Lanes past the end of the row load -inf, so they contribute
exp(-inf) = 0 to the sum and never win the max. The row maximum is subtracted before
exponentiating: exp(x) overflows fp32 for x > 88, while exp(x - max) <= 1. Arithmetic is fp32
for every input dtype, and the result is rounded to the input dtype on store.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    shifted = x - tl.max(x, axis=0)
    numerator = tl.exp(shifted)
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def num_warps_for(block_size: int) -> int:
    """About 8 elements per thread (256 per warp), clamped to 1..16 warps."""
    return min(max(block_size // 256, 1), 16)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension of a 2-D tensor whose rows are contiguous."""
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D tensor, got {x.ndim}-D")
    if x.stride(1) != 1:
        raise ValueError("rows must be contiguous (stride 1 along the last dimension)")
    n_rows, n_cols = x.shape
    out = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
    block_size = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](
        x,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps_for(block_size),
    )
    return out
