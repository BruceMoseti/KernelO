"""Elementwise add: ``c[i] = a[i] + b[i]``.

The simplest kernel in the repository, and it is here for two reasons: it is
the smallest thing that exercises the full pipeline (search space ->
compile -> verify -> benchmark -> database), and it is a pure bandwidth test,
so the measured GB/s is a useful sanity check against the card's spec sheet.
There is nothing to optimise beyond picking a block size, so the search space
is correspondingly tiny.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from kernelforge.benchmark import metrics
from kernelforge.kernels.base import Operator, compiled_baseline
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import VectorAddSearchSpace


@triton.jit
def vector_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # The last program is partial whenever n_elements is not a multiple of
    # BLOCK_SIZE, which is the common case; the mask suppresses both the loads
    # and the store for those lanes.
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


DEFAULT_CONFIG = KernelConfig("vector_add", BLOCK_SIZE=1024, num_warps=4)


def vector_add(
    a: torch.Tensor, b: torch.Tensor, *, config: KernelConfig | None = None
) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} vs {b.dtype}")
    cfg = config or DEFAULT_CONFIG
    a, b = a.contiguous(), b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, cfg["BLOCK_SIZE"]),)
    vector_add_kernel[grid](a, b, out, n, **cfg.meta, **cfg.launch)
    return out


class VectorAddOperator(Operator):
    name = "vector_add"
    config_order = ("BLOCK_SIZE", "num_warps")

    def search_space(self) -> VectorAddSearchSpace:
        return VectorAddSearchSpace()

    def make_inputs(
        self, problem: Problem, device: torch.device, *, seed: int = 0
    ) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device=device).manual_seed(seed)
        n = problem.dims_dict["n"]
        a = torch.randn(n, device=device, dtype=problem.dtype, generator=gen)
        b = torch.randn(n, device=device, dtype=problem.dtype, generator=gen)
        return a, b

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b = inputs
        return a + b

    def run(self, config: KernelConfig, *inputs: torch.Tensor) -> torch.Tensor:
        a, b = inputs
        return vector_add(a, b, config=config)

    def default_config(self, problem: Problem) -> KernelConfig:
        return DEFAULT_CONFIG

    def flops(self, problem: Problem) -> int:
        return problem.dims_dict["n"]

    def bytes_moved(self, problem: Problem) -> int:
        return metrics.vector_add_bytes(problem.dims_dict["n"], problem.itemsize)

    def is_memory_bound(self) -> bool:
        return True

    def baselines(
        self, problem: Problem, inputs: tuple[torch.Tensor, ...]
    ) -> dict[str, Callable[[], torch.Tensor]]:
        a, b = inputs
        return {
            "torch_eager": lambda: a + b,
            "torch_compile": compiled_baseline(torch.add, inputs),
            "triton_baseline": lambda: vector_add(a, b, config=DEFAULT_CONFIG),
        }
