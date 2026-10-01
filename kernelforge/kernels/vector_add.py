"""Elementwise vector addition, out[i] = x[i] + y[i].

One program handles BLOCK_SIZE consecutive elements. The final program is partial whenever the
length is not a multiple of BLOCK_SIZE, so loads and stores are masked to `offsets < n`.
Offsets are 32-bit, so tensors must have fewer than 2**31 elements.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024
NUM_WARPS = 4


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add two contiguous tensors of the same shape, dtype and device."""
    if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
        raise ValueError("x and y must have the same shape, dtype and device")
    if not (x.is_contiguous() and y.is_contiguous()):
        raise ValueError("x and y must be contiguous")
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return out
