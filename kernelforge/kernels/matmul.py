"""Blocked GEMM: ``C = A @ B``.

Written from the blocked-GEMM model rather than transcribed, so the reasoning
behind each piece is recorded here.

**Why tile at all.** Computing one output element reads a row of A and a
column of B: 2K elements for K multiply-adds, an arithmetic intensity of 1
FLOP per element loaded. No GPU can feed that. A ``BLOCK_M x BLOCK_N`` tile
instead loads ``BLOCK_K * (BLOCK_M + BLOCK_N)`` elements to produce
``BLOCK_M * BLOCK_N`` results, so intensity rises to
``BLOCK_M*BLOCK_N / (BLOCK_M + BLOCK_N)`` per K-step -- 64 for a 128x128 tile
against 1 for the naive version. That ratio is exactly the ``reuse`` term in
the search-space priority function.

**Why group the program ordering.** With the obvious row-major program order,
the programs resident on the GPU at any moment span one strip of C, which
touches ``BLOCK_M`` rows of A and *all* of B. Launching in ``GROUP_M``-row
groups instead makes the concurrent working set a square-ish block of C, so
the A and B strips it reads are both small enough to stay in L2 and get reused
by neighbouring programs. Same arithmetic, same number of global loads issued,
far more of them served by cache.

**Why fp32 accumulate.** ``tl.dot`` accumulates in fp32 even for fp16 inputs,
which is what makes a K=4096 reduction usable in half precision. The cast back
to the output dtype happens once, in the epilogue.

**Why ``input_precision="ieee"``.** For fp16 and bf16 operands this is a
no-op, confirmed by the ``mma.sync.aligned.m16n8k16`` instruction still
appearing in the generated PTX (asserted in ``tests/test_triton_compile.py``).
For fp32 operands it stops Triton from quietly computing in TF32, whose 10-bit
mantissa carries ~1e-3 relative error. Without it, "fp32" would mean TF32 here
and fp32 in the PyTorch reference, and the correctness comparison would be
measuring the precision gap rather than the kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernelforge.benchmark import metrics
from kernelforge.kernels.base import Operator, compiled_baseline
from kernelforge.testing import exact_fp32_matmul
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import MatmulSearchSpace


@triton.jit
def matmul_kernel(
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
):
    # --- L2-aware program ordering -------------------------------------
    # Map the 1D program id onto a tile of C in GROUP_M-row groups. The last
    # group is short whenever num_pid_m is not a multiple of GROUP_M, which
    # `group_size_m` accounts for.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- operand pointers ----------------------------------------------
    # The `% M` / `% N` wrap keeps every load in bounds without masking the M
    # and N axes in the inner loop: out-of-range rows read some other valid
    # row, contribute to accumulator lanes that the epilogue's store mask
    # discards, and cost nothing. Only the K axis needs a real mask, because a
    # short final K step must contribute zero rather than wrapped data.
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- epilogue ------------------------------------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


#: Phase 6 starting point: one fixed configuration, correct before fast. Also
#: serves as the untuned "Triton baseline" that tuning is reported against.
DEFAULT_CONFIG = KernelConfig(
    "matmul", BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3
)


def _check_operands(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2D operands, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shape mismatch for matmul: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} vs {b.dtype}")


def matmul(a: torch.Tensor, b: torch.Tensor, *, config: KernelConfig | None = None) -> torch.Tensor:
    _check_operands(a, b)
    cfg = config or DEFAULT_CONFIG
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        **cfg.meta,
        **cfg.launch,
    )
    return c


# --- Triton's own autotuner, as a comparison baseline ---------------------
# The point of KernelForge is not to re-skin `@triton.autotune`, so the two are
# measured against each other. This is the tutorial-style approach: a short
# hand-written config list, no feasibility filtering, no correctness gate, and
# a cache keyed only on M/N/K.
_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
        num_warps=warps,
        num_stages=stages,
    )
    for bm, bn, bk, warps, stages in [
        (128, 256, 64, 8, 3),
        (64, 256, 32, 4, 4),
        (128, 128, 32, 4, 4),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
        (128, 32, 32, 4, 4),
        (64, 32, 32, 2, 5),
        (32, 64, 32, 2, 5),
    ]
]

matmul_kernel_autotuned = triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])(
    matmul_kernel
)


def matmul_triton_autotune(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _check_operands(a, b)
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    matmul_kernel_autotuned[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


class MatmulOperator(Operator):
    name = "matmul"
    config_order = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "num_warps", "num_stages")

    def search_space(self) -> MatmulSearchSpace:
        return MatmulSearchSpace()

    def make_inputs(
        self, problem: Problem, device: torch.device, *, seed: int = 0
    ) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device=device).manual_seed(seed)
        dims = problem.dims_dict
        a = torch.randn(dims["M"], dims["K"], device=device, dtype=problem.dtype, generator=gen)
        b = torch.randn(dims["K"], dims["N"], device=device, dtype=problem.dtype, generator=gen)
        return a, b

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b = inputs
        # TF32 off: see the module docstring. For fp16/bf16 this changes
        # nothing, so it is applied unconditionally for one reference path.
        with exact_fp32_matmul():
            return torch.matmul(a, b)

    def run(self, config: KernelConfig, *inputs: torch.Tensor) -> torch.Tensor:
        a, b = inputs
        return matmul(a, b, config=config)

    def default_config(self, problem: Problem) -> KernelConfig:
        return DEFAULT_CONFIG

    def flops(self, problem: Problem) -> int:
        dims = problem.dims_dict
        return metrics.matmul_flops(dims["M"], dims["N"], dims["K"])

    def bytes_moved(self, problem: Problem) -> int:
        dims = problem.dims_dict
        return metrics.matmul_bytes(dims["M"], dims["N"], dims["K"], problem.itemsize)

    def baselines(self, problem: Problem, inputs):
        a, b = inputs
        return {
            "torch_eager": lambda: torch.matmul(a, b),
            "torch_compile": compiled_baseline(torch.matmul, inputs),
            "triton_baseline": lambda: matmul(a, b, config=DEFAULT_CONFIG),
            "triton_autotune": lambda: matmul_triton_autotune(a, b),
        }
