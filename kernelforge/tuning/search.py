"""Search spaces: a kernel's candidate configurations and the rules that prune them.

A `SearchSpace` is a grid of parameter values plus named rules. The tuner only calls
`candidates()`, so a new kernel defines its own grid and rules without any change to the tuner.
For example, RMSNorm would use BLOCK_SIZE, ROWS_PER_PROGRAM and num_warps.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import triton

from kernelforge.tuning.config import KernelConfig, Problem


@dataclass(frozen=True)
class DeviceLimits:
    """Hardware limits that decide which configurations are feasible or sensible."""

    sm_count: int
    max_shared_memory_bytes: int  # per thread block (the opt-in maximum)


def device_limits() -> DeviceLimits:
    """Limits of the current CUDA device, as reported by Triton's driver."""
    props = triton.runtime.driver.active.utils.get_device_properties(torch.cuda.current_device())
    return DeviceLimits(
        sm_count=props["multiprocessor_count"], max_shared_memory_bytes=props["max_shared_mem"]
    )


@dataclass(frozen=True)
class Rule:
    """A named reason to discard a candidate; `rejects` returns True for configs to drop."""

    name: str
    reason: str
    rejects: Callable[[KernelConfig, Problem, DeviceLimits], bool]


@dataclass(frozen=True)
class Candidates:
    configs: list[KernelConfig]
    pruned: dict[str, int]  # rule name -> number of configs it removed

    @property
    def grid_size(self) -> int:
        return len(self.configs) + sum(self.pruned.values())


@dataclass(frozen=True)
class SearchSpace:
    kernel: str
    params: Mapping[str, Sequence[int]]
    num_warps: Sequence[int]
    num_stages: Sequence[int]
    dtypes: Sequence[torch.dtype]
    rules: Sequence[Rule]

    def grid(self) -> list[KernelConfig]:
        """Every combination of parameter values, before pruning."""
        names = list(self.params)
        combinations = itertools.product(*self.params.values(), self.num_warps, self.num_stages)
        return [
            KernelConfig(self.kernel, dict(zip(names, values[:-2], strict=True)), *values[-2:])
            for values in combinations
        ]

    def candidates(self, problem: Problem, device: DeviceLimits) -> Candidates:
        """Configs that no rule rejects. Each pruned config counts against the first rule."""
        if problem.dtype not in self.dtypes:
            raise ValueError(f"{self.kernel} does not support {problem.dtype}")
        kept: list[KernelConfig] = []
        pruned: Counter[str] = Counter()
        for config in self.grid():
            rule = next((r for r in self.rules if r.rejects(config, problem, device)), None)
            if rule is None:
                kept.append(config)
            else:
                pruned[rule.name] += 1
        return Candidates(kept, dict(pruned))


# MatMul. The grid is the spec's: 4 x 4 x 3 tile shapes x 3 warp counts x 3 stage counts = 432.
MATMUL_BLOCKS_MN = (16, 32, 64, 128)
MATMUL_BLOCKS_K = (16, 32, 64)
MAX_ACCUMULATORS_PER_THREAD = 128  # of the 255 registers a thread can address
MMA_FRAGMENT_PER_THREAD = 4  # one 16x8 tensor-core output fragment spread over a 32-thread warp


def _tiles(problem: Problem, block_m: int, block_n: int) -> int:
    return triton.cdiv(problem.shape["M"], block_m) * triton.cdiv(problem.shape["N"], block_n)


def _block_limit(size: int) -> int:
    """Largest useful block for a dimension: its size rounded up to a power of two, at least 16."""
    return max(16, triton.next_power_of_2(size))


def _admissible_tiles(problem: Problem) -> list[tuple[int, int]]:
    limit_m, limit_n = _block_limit(problem.shape["M"]), _block_limit(problem.shape["N"])
    return [
        (block_m, block_n)
        for block_m in MATMUL_BLOCKS_MN
        for block_n in MATMUL_BLOCKS_MN
        if block_m <= limit_m and block_n <= limit_n
    ]


def _accumulators_per_thread(config: KernelConfig) -> float:
    return config.params["BLOCK_M"] * config.params["BLOCK_N"] / (32 * config.num_warps)


def _exceeds_shared_memory(config: KernelConfig, problem: Problem, device: DeviceLimits) -> bool:
    # Triton's software pipeline keeps num_stages copies of an A tile and a B tile in shared memory.
    p = config.params
    itemsize = torch.finfo(problem.dtype).bits // 8
    needed = (p["BLOCK_M"] + p["BLOCK_N"]) * p["BLOCK_K"] * itemsize * config.num_stages
    return needed > device.max_shared_memory_bytes


def _exceeds_register_budget(config: KernelConfig, problem: Problem, device: DeviceLimits) -> bool:
    return _accumulators_per_thread(config) > MAX_ACCUMULATORS_PER_THREAD


def _leaves_warps_idle(config: KernelConfig, problem: Problem, device: DeviceLimits) -> bool:
    return _accumulators_per_thread(config) < MMA_FRAGMENT_PER_THREAD


def _tile_exceeds_problem(config: KernelConfig, problem: Problem, device: DeviceLimits) -> bool:
    p, shape = config.params, problem.shape
    return (
        p["BLOCK_M"] > _block_limit(shape["M"])
        or p["BLOCK_N"] > _block_limit(shape["N"])
        or p["BLOCK_K"] > _block_limit(shape["K"])
    )


def _underfills_gpu(config: KernelConfig, problem: Problem, device: DeviceLimits) -> bool:
    smallest = min(MATMUL_BLOCKS_MN)
    target = min(device.sm_count, _tiles(problem, smallest, smallest))
    return _tiles(problem, config.params["BLOCK_M"], config.params["BLOCK_N"]) < target


def _low_reuse(config: KernelConfig, problem: Problem, device: DeviceLimits) -> bool:
    block_m, block_n = config.params["BLOCK_M"], config.params["BLOCK_N"]
    return any(
        big_m >= block_m
        and big_n >= block_n
        and big_m * big_n >= 4 * block_m * block_n
        and _tiles(problem, big_m, big_n) >= device.sm_count
        for big_m, big_n in _admissible_tiles(problem)
    )


MATMUL_RULES = (
    Rule(
        "shared_memory",
        "num_stages x (A tile + B tile) exceeds the per-block shared memory limit",
        _exceeds_shared_memory,
    ),
    Rule(
        "register_budget",
        f"more than {MAX_ACCUMULATORS_PER_THREAD} fp32 accumulators per thread would spill",
        _exceeds_register_budget,
    ),
    Rule(
        "idle_warps",
        "too few outputs for every warp to own one 16x8 tensor-core fragment",
        _leaves_warps_idle,
    ),
    Rule(
        "tile_exceeds_problem",
        "a block exceeds its dimension rounded up to a power of two",
        _tile_exceeds_problem,
    ),
    Rule(
        "underfills_gpu",
        "fewer output tiles than SMs, when smaller tiles could occupy every SM",
        _underfills_gpu,
    ),
    Rule(
        "low_reuse",
        "a tile with 4x the area still occupies every SM, with half the memory traffic per FLOP",
        _low_reuse,
    ),
)

MATMUL_SEARCH_SPACE = SearchSpace(
    kernel="matmul",
    params={"BLOCK_M": MATMUL_BLOCKS_MN, "BLOCK_N": MATMUL_BLOCKS_MN, "BLOCK_K": MATMUL_BLOCKS_K},
    num_warps=(2, 4, 8),
    num_stages=(2, 3, 4),
    dtypes=(torch.float32, torch.float16, torch.bfloat16),
    rules=MATMUL_RULES,
)
