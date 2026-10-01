"""Fused row-wise softmax.

    y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

Subtracting the row maximum is not an optimisation, it is a requirement: the
largest fp16 value is 65504, so ``exp`` overflows for x > 11 and the naive
``exp(x) / sum(exp(x))`` returns NaN on inputs that occur routinely in
attention logits. Shifting by the maximum leaves the result mathematically
unchanged while bounding every exponent at zero.

The kernel is single-pass: one program loads an entire row into registers,
reduces it twice (max, then sum) without going back to global memory, and
writes it out. PyTorch's eager softmax reads and writes the row several times,
which is the whole source of the speedup on a memory-bound operation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernelforge.benchmark import metrics
from kernelforge.kernels.base import Operator, compiled_baseline
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import SoftmaxSearchSpace, next_power_of_2


@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    x_row_stride,
    y_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    for i in tl.static_range(ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + i
        if row < n_rows:
            # Masked lanes load -inf so that they lose the max reduction and
            # contribute exp(-inf) = 0 to the sum, which keeps both reductions
            # correct without a separate code path for the tail.
            x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            x = x - tl.max(x, axis=0)
            numerator = tl.exp(x)
            denominator = tl.sum(numerator, axis=0)
            tl.store(y_ptr + row * y_row_stride + cols, numerator / denominator, mask=mask)


def default_config(n_cols: int) -> KernelConfig:
    block = next_power_of_2(n_cols)
    # One warp per 256 columns up to 8 warps: enough threads to keep loads in
    # flight without giving each thread so few elements that the reduction
    # tree dominates.
    warps = max(1, min(8, block // 256))
    return KernelConfig("softmax", BLOCK_SIZE=block, ROWS_PER_PROGRAM=1, num_warps=warps)


def softmax(x: torch.Tensor, *, config: KernelConfig | None = None) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    if x.stride(1) != 1:
        raise ValueError(
            "x must have a contiguous last dimension; this kernel indexes columns "
            "directly off the row pointer. Pass x.contiguous() -- explicitly, so the "
            "copy is not hidden inside a measured region."
        )
    n_rows, n_cols = x.shape
    cfg = config or default_config(n_cols)
    block = cfg["BLOCK_SIZE"]
    if block < n_cols:
        raise ValueError(
            f"BLOCK_SIZE={block} cannot hold a row of {n_cols}; "
            "this kernel is single-pass by design"
        )
    if block > SoftmaxSearchSpace.MAX_BLOCK_SIZE:
        raise ValueError(
            f"row width {n_cols} exceeds the single-pass limit of "
            f"{SoftmaxSearchSpace.MAX_BLOCK_SIZE}"
        )
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_rows, cfg["ROWS_PER_PROGRAM"]),)
    softmax_kernel[grid](
        x,
        out,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        **cfg.meta,
        **cfg.launch,
    )
    return out


class SoftmaxOperator(Operator):
    name = "softmax"
    config_order = ("BLOCK_SIZE", "ROWS_PER_PROGRAM", "num_warps")

    def search_space(self) -> SoftmaxSearchSpace:
        return SoftmaxSearchSpace()

    def make_inputs(
        self, problem: Problem, device: torch.device, *, seed: int = 0
    ) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device=device).manual_seed(seed)
        dims = problem.dims_dict
        x = torch.randn(
            dims["rows"], dims["cols"], device=device, dtype=problem.dtype, generator=gen
        )
        return (x,)

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        return torch.softmax(x, dim=-1)

    def run(self, config: KernelConfig, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        return softmax(x, config=config)

    def default_config(self, problem: Problem) -> KernelConfig:
        return default_config(problem.dims_dict["cols"])

    def flops(self, problem: Problem) -> int:
        dims = problem.dims_dict
        # max, subtract, exp, sum, divide: five passes over each element. An
        # approximation -- softmax is bandwidth bound, so GB/s is the metric
        # that matters and this only exists for completeness.
        return 5 * dims["rows"] * dims["cols"]

    def bytes_moved(self, problem: Problem) -> int:
        dims = problem.dims_dict
        return metrics.elementwise_bytes(dims["rows"] * dims["cols"], problem.itemsize, tensors=2)

    def is_memory_bound(self) -> bool:
        return True

    def baselines(self, problem: Problem, inputs):
        (x,) = inputs
        return {
            "torch_eager": lambda: torch.softmax(x, dim=-1),
            "torch_compile": compiled_baseline(lambda t: torch.softmax(t, dim=-1), inputs),
            "triton_baseline": lambda: softmax(x, config=default_config(problem.dims_dict["cols"])),
        }
