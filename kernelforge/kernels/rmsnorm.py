"""RMS normalisation, the memory-bound counterpart to the GEMM.

    rms(x) = sqrt(mean(x_i^2) + eps)
    y_i    = x_i / rms(x) * gamma_i

Why this operator earns its place next to a GEMM: a GEMM is a compute
throughput problem, where the work is to keep the tensor cores fed. RMSNorm
does O(1) arithmetic per element, so its ceiling is the memory system and the
only question that matters is whether it reads x once. The optimisation
targets are therefore completely different, and a tuner that only knows how to
tune tiling would have nothing to say here -- which is why the search space is
owned by the operator and not by the tuner.

Two numerical decisions:

* The sum of squares accumulates in fp32 regardless of the input dtype. In
  fp16, ``sum(x^2)`` over 4096 elements of unit variance reaches ~4096, and
  fp16 has 11 bits of mantissa, so the small terms stop contributing long
  before the reduction finishes. This is the same upcast every production
  implementation performs.
* ``eps`` is added inside the square root, which is what makes the gradient and
  the value finite for an all-zero row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernelforge.benchmark import metrics
from kernelforge.kernels.base import Operator, compiled_baseline
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import RMSNormSearchSpace, next_power_of_2

DEFAULT_EPS = 1e-5


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    gamma_ptr,
    y_ptr,
    x_row_stride,
    y_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # gamma is shared by every row, so it is loaded once per program rather
    # than once per row. With ROWS_PER_PROGRAM=8 that removes seven redundant
    # loads of the full vector from the inner loop.
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for i in tl.static_range(ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + i
        if row < n_rows:
            x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
            # Masked lanes read 0.0, which contributes nothing to a sum of
            # squares, so the tail needs no special case. Dividing by n_cols
            # rather than BLOCK_SIZE keeps the mean correct for partial rows.
            mean_square = tl.sum(x * x, axis=0) / n_cols
            inv_rms = 1.0 / tl.sqrt(mean_square + eps)
            tl.store(y_ptr + row * y_row_stride + cols, x * inv_rms * gamma, mask=mask)


def default_config(n_cols: int) -> KernelConfig:
    block = next_power_of_2(n_cols)
    warps = max(1, min(8, block // 256))
    return KernelConfig("rmsnorm", BLOCK_SIZE=block, ROWS_PER_PROGRAM=1, num_warps=warps)


def rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    config: KernelConfig | None = None,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    n_rows, n_cols = x.shape
    if gamma.shape != (n_cols,):
        raise ValueError(f"gamma must have shape ({n_cols},), got {tuple(gamma.shape)}")
    cfg = config or default_config(n_cols)
    block = cfg["BLOCK_SIZE"]
    if block < n_cols:
        raise ValueError(
            f"BLOCK_SIZE={block} cannot hold a row of {n_cols}; "
            "this kernel is single-pass by design"
        )
    if block > RMSNormSearchSpace.MAX_BLOCK_SIZE:
        raise ValueError(
            f"hidden size {n_cols} exceeds the single-pass limit of "
            f"{RMSNormSearchSpace.MAX_BLOCK_SIZE}"
        )
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_rows, cfg["ROWS_PER_PROGRAM"]),)
    rmsnorm_kernel[grid](
        x,
        gamma,
        out,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        eps,
        **cfg.meta,
        **cfg.launch,
    )
    return out


def rmsnorm_reference(
    x: torch.Tensor, gamma: torch.Tensor, *, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """PyTorch RMSNorm with an fp32 reduction.

    The reduction, the rescale and the gamma multiply all happen in fp32 and
    the result is rounded once, which is the same sequence the Triton kernel
    performs. Some implementations (Llama's, for one) round after the rescale
    and then multiply by gamma in the low precision; that choice differs by a
    single rounding step and is well inside the fp16 tolerance either way.
    """
    xf = x.to(torch.float32)
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv_rms * gamma.to(torch.float32)).to(x.dtype)


class RMSNormOperator(Operator):
    name = "rmsnorm"
    config_order = ("BLOCK_SIZE", "ROWS_PER_PROGRAM", "num_warps")

    def search_space(self) -> RMSNormSearchSpace:
        return RMSNormSearchSpace()

    def make_inputs(
        self, problem: Problem, device: torch.device, *, seed: int = 0
    ) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device=device).manual_seed(seed)
        dims = problem.dims_dict
        x = torch.randn(
            dims["rows"], dims["cols"], device=device, dtype=problem.dtype, generator=gen
        )
        # gamma starts near 1, as it does after initialisation in a real model.
        gamma = torch.randn(dims["cols"], device=device, dtype=problem.dtype, generator=gen)
        gamma = (gamma * 0.1 + 1.0).to(problem.dtype)
        return x, gamma

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        x, gamma = inputs
        return rmsnorm_reference(x, gamma)

    def run(self, config: KernelConfig, *inputs: torch.Tensor) -> torch.Tensor:
        x, gamma = inputs
        return rmsnorm(x, gamma, config=config)

    def default_config(self, problem: Problem) -> KernelConfig:
        return default_config(problem.dims_dict["cols"])

    def flops(self, problem: Problem) -> int:
        dims = problem.dims_dict
        # square, accumulate, rescale, scale by gamma.
        return 4 * dims["rows"] * dims["cols"]

    def bytes_moved(self, problem: Problem) -> int:
        dims = problem.dims_dict
        return metrics.rmsnorm_bytes(dims["rows"], dims["cols"], problem.itemsize)

    def is_memory_bound(self) -> bool:
        return True

    def baselines(self, problem: Problem, inputs):
        x, gamma = inputs
        out = {
            "torch_eager": lambda: rmsnorm_reference(x, gamma),
            "torch_compile": compiled_baseline(rmsnorm_reference, inputs),
            "triton_baseline": lambda: rmsnorm(
                x, gamma, config=default_config(problem.dims_dict["cols"])
            ),
        }
        # The handwritten CUDA implementation joins the comparison only when a
        # toolchain is present; everything else still runs without one.
        from kernelforge.kernels import cuda_rmsnorm as cuda_ext

        if cuda_ext.available():
            out["cuda"] = lambda: cuda_ext.cuda_rmsnorm(x, gamma, eps=DEFAULT_EPS)
        return out
