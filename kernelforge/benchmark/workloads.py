"""Workloads: how to build inputs, compute the float64 reference, count work, and run a config."""

from __future__ import annotations

import torch

from kernelforge.benchmark.metrics import matmul_flops
from kernelforge.kernels.matmul import matmul
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import MATMUL_SEARCH_SPACE, SearchSpace

# The fixed configuration from Phase 6, used as the "Triton baseline".
MATMUL_BASELINE = KernelConfig(
    "matmul", {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3
)


def matmul_problem(m: int, n: int, k: int, dtype: torch.dtype) -> Problem:
    return Problem("matmul", {"M": m, "N": n, "K": k}, dtype)


class MatmulWorkload:
    """C = A @ B with contiguous row-major A (M x K) and B (K x N)."""

    @property
    def search_space(self) -> SearchSpace:
        return MATMUL_SEARCH_SPACE

    def make_inputs(self, problem: Problem, device: torch.device) -> tuple[torch.Tensor, ...]:
        m, n, k = (problem.shape[dim] for dim in "MNK")
        generator = torch.Generator(device=device).manual_seed(0)
        a = torch.randn(m, k, generator=generator, device=device).to(problem.dtype)
        b = torch.randn(k, n, generator=generator, device=device).to(problem.dtype)
        return a, b

    def reference(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        a, b = inputs
        return a.double() @ b.double()

    def run(self, config: KernelConfig, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        a, b = inputs
        return matmul(
            a,
            b,
            block_m=config.params["BLOCK_M"],
            block_n=config.params["BLOCK_N"],
            block_k=config.params["BLOCK_K"],
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    def flops(self, problem: Problem) -> int:
        return matmul_flops(problem.shape["M"], problem.shape["N"], problem.shape["K"])

    def bytes_moved(self, problem: Problem) -> int:
        m, n, k = (problem.shape[dim] for dim in "MNK")
        return (m * k + k * n + m * n) * (torch.finfo(problem.dtype).bits // 8)


MATMUL = MatmulWorkload()
