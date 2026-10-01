"""Tiled matrix multiplication, C = A @ B, with A: M x K, B: K x N, C: M x N.

Tiling. Each program computes one BLOCK_M x BLOCK_N tile of C. It walks K in BLOCK_K steps,
loading a BLOCK_M x BLOCK_K tile of A and a BLOCK_K x BLOCK_N tile of B and accumulating their
product into an fp32 accumulator held in registers. Every element loaded from A is reused for
BLOCK_N outputs and every element of B for BLOCK_M outputs, instead of being fetched once per
output. Larger tiles raise that reuse but need more registers and shared memory, and they give
fewer programs to spread across the SMs. The tuner explores that trade-off.

Masking. Tiles on the right and bottom edges, and the last K step, extend past the matrix
whenever a dimension is not a multiple of the block size. Loads are masked with `other=0`, so
out-of-range products contribute nothing, and stores are masked to the matrix bounds. The
Triton tutorial instead wraps row and column indices with `% M` and `% N`, which keeps the loads
unmasked but computes duplicate rows that are then discarded. Here every load is masked, so no
program ever reads outside the matrix, at the cost of a little predicate arithmetic.

Pipelining. With num_stages > 1, Triton prefetches upcoming K tiles with asynchronous copies
(cp.async), which move 16-byte-aligned vectors. For 16-bit dtypes it uses them for an operand
only when it can prove that the operand's contiguous dimension and leading stride are multiples
of 16 elements: K for A, N for B (M does not matter). Other shapes stay correct, but that
operand is loaded synchronously every iteration. fp32's 4-byte copies do not need this
alignment. tests/test_compile.py checks this by compiling for sm_80 and sm_90.

Program order (L2 reuse). The grid is 1-D. With row-by-row numbering, the W programs resident
at once would cover W tiles of one tile row: one row panel of A, but W column panels of B.
Instead, consecutive ids walk down a band of GROUP_M tile rows before moving to the next tile
column. The resident programs then cover a GROUP_M x (W / GROUP_M) block of tiles, which needs
GROUP_M panels of A and W / GROUP_M panels of B. For W = 128 that is 8 + 16 panels instead of
1 + 128, so more loads hit in L2. GROUP_M is fixed at 8, as in the Triton tutorial, and is not
tuned.

Accumulation is fp32 for every input dtype. fp32 inputs use input_precision="ieee" (true fp32
FMA, matching PyTorch's default of TF32 off) rather than Triton's TF32 default. Offsets are
32-bit, so each operand must span fewer than 2**31 elements.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP_M = 8
FP32_INPUT_PRECISION = "ieee"
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tile_rows = tl.cdiv(M, BLOCK_M)
    tile_cols = tl.cdiv(N, BLOCK_N)
    programs_per_band = GROUP_M * tile_cols
    first_row = (pid // programs_per_band) * GROUP_M
    rows_in_band = tl.minimum(tile_rows - first_row, GROUP_M)
    pid_in_band = pid % programs_per_band
    tile_m = first_row + pid_in_band % rows_in_band
    tile_n = pid_in_band // rows_in_band

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    row_in_bounds = rows[:, None] < M
    col_in_bounds = cols[None, :] < N
    a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
    b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_in_bounds = (k0 + ks) < K
        a = tl.load(a_ptrs, mask=row_in_bounds & k_in_bounds[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_in_bounds[:, None] & col_in_bounds, other=0.0)
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=row_in_bounds & col_in_bounds)


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a @ b for 2-D tensors of any strides. Defaults are the fixed baseline config.

    `out`, if given, receives the result and may have any strides.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes for matmul: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("a and b must have the same dtype and device")
    if a.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {a.dtype}; expected one of {SUPPORTED_DTYPES}")
    (m, k), n = a.shape, b.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    elif out.shape != (m, n) or out.dtype != a.dtype or out.device != a.device:
        raise ValueError(f"out must be a {m}x{n} {a.dtype} tensor on {a.device}")
    # input_precision only affects fp32 x fp32 dots; "tf32" is Triton's default and is a no-op
    # for 16-bit operands.
    input_precision = FP32_INPUT_PRECISION if a.dtype == torch.float32 else "tf32"
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _matmul_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=GROUP_M,
        INPUT_PRECISION=input_precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
