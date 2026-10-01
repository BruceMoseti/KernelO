import pytest
import torch
import triton

from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import (
    MATMUL_SEARCH_SPACE,
    MAX_ACCUMULATORS_PER_THREAD,
    MMA_FRAGMENT_PER_THREAD,
    DeviceLimits,
    Rule,
    SearchSpace,
)

# SM counts and opt-in shared memory per block from NVIDIA's published specifications.
A100 = DeviceLimits(sm_count=108, max_shared_memory_bytes=166_912)
H100_SXM = DeviceLimits(sm_count=132, max_shared_memory_bytes=232_448)
RTX_4090 = DeviceLimits(sm_count=128, max_shared_memory_bytes=101_376)
T4 = DeviceLimits(sm_count=40, max_shared_memory_bytes=65_536)
DEVICES = [A100, H100_SXM, RTX_4090, T4]
SHAPES = [
    (2048, 4096, 4096),
    (4096, 4096, 4096),
    (2048, 11008, 4096),
    (1024, 1024, 1024),
    (511, 769, 1025),
    (512, 512, 512),
    (16, 4096, 4096),
    (128, 128, 128),
    (1, 1, 1),
]


def _problem(m: int, n: int, k: int, dtype: torch.dtype = torch.float16) -> Problem:
    return Problem("matmul", {"M": m, "N": n, "K": k}, dtype)


def _tile(config: KernelConfig) -> tuple[int, int]:
    return config.params["BLOCK_M"], config.params["BLOCK_N"]


def test_grid_is_the_spec_space() -> None:
    grid = MATMUL_SEARCH_SPACE.grid()
    assert len(grid) == 4 * 4 * 3 * 3 * 3
    assert len({(c.params_json(), c.num_warps, c.num_stages) for c in grid}) == len(grid)


@pytest.mark.parametrize(
    ("device", "expected"), [(A100, 72), (H100_SXM, 72), (RTX_4090, 70), (T4, 56)]
)
def test_candidate_count_for_the_spec_example_workload(device: DeviceLimits, expected: int) -> None:
    candidates = MATMUL_SEARCH_SPACE.candidates(_problem(2048, 4096, 4096), device)
    assert len(candidates.configs) == expected
    assert candidates.grid_size == 432
    assert {_tile(c) for c in candidates.configs} == {(64, 128), (128, 64), (128, 128)}


@pytest.mark.parametrize("device", DEVICES)
def test_known_good_configs_survive(device: DeviceLimits) -> None:
    survivors = MATMUL_SEARCH_SPACE.candidates(_problem(2048, 4096, 4096), device).configs
    # The spec's illustrative best config and the in-space configs of Triton's matmul tutorial.
    for block_m, block_n, block_k, num_warps, num_stages in [
        (64, 128, 32, 8, 4),
        (128, 128, 32, 4, 4),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
    ]:
        params = {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}
        assert KernelConfig("matmul", params, num_warps, num_stages) in survivors


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("device", DEVICES)
def test_survivors_respect_hardware_and_problem_limits(
    device: DeviceLimits, shape: tuple[int, int, int], dtype: torch.dtype
) -> None:
    problem = _problem(*shape, dtype)
    candidates = MATMUL_SEARCH_SPACE.candidates(problem, device)
    assert candidates.configs, "pruning must never remove every candidate"
    itemsize = torch.finfo(dtype).bits // 8
    for config in candidates.configs:
        block_m, block_n, block_k = (config.params[p] for p in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        shared = (block_m + block_n) * block_k * itemsize * config.num_stages
        assert shared <= device.max_shared_memory_bytes
        accumulators = block_m * block_n / (32 * config.num_warps)
        assert MMA_FRAGMENT_PER_THREAD <= accumulators <= MAX_ACCUMULATORS_PER_THREAD
        for block, size in zip((block_m, block_n, block_k), shape, strict=True):
            assert block <= max(16, triton.next_power_of_2(size))


def test_mid_size_problem_keeps_only_tiles_that_occupy_every_sm() -> None:
    survivors = MATMUL_SEARCH_SPACE.candidates(_problem(512, 512, 512), A100).configs
    assert all(
        triton.cdiv(512, bm) * triton.cdiv(512, bn) >= 108 for bm, bn in map(_tile, survivors)
    )


def test_tiny_problem_maximizes_parallelism() -> None:
    # Even 16x16 tiles give only 64 tiles of 128x128, fewer than the A100's 108 SMs.
    survivors = MATMUL_SEARCH_SPACE.candidates(_problem(128, 128, 128), A100).configs
    assert {_tile(c) for c in survivors} == {(16, 16)}


def test_skinny_problem_never_pads_a_dimension_beyond_a_power_of_two() -> None:
    survivors = MATMUL_SEARCH_SPACE.candidates(_problem(16, 4096, 4096), A100).configs
    assert {c.params["BLOCK_M"] for c in survivors} == {16}


def test_fp32_prunes_more_for_shared_memory_than_fp16() -> None:
    problem_fp16 = _problem(2048, 4096, 4096, torch.float16)
    problem_fp32 = _problem(2048, 4096, 4096, torch.float32)
    fp16 = MATMUL_SEARCH_SPACE.candidates(problem_fp16, RTX_4090)
    fp32 = MATMUL_SEARCH_SPACE.candidates(problem_fp32, RTX_4090)
    assert fp32.pruned["shared_memory"] > fp16.pruned["shared_memory"]
    assert len(fp32.configs) < len(fp16.configs)


def test_unsupported_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not support"):
        MATMUL_SEARCH_SPACE.candidates(_problem(64, 64, 64, torch.float64), A100)


def test_search_space_is_not_matmul_specific() -> None:
    # An RMSNorm-shaped space, with no BLOCK_M/N/K, uses the same machinery.
    space = SearchSpace(
        kernel="rmsnorm",
        params={"BLOCK_SIZE": (256, 512, 1024), "ROWS_PER_PROGRAM": (1, 2)},
        num_warps=(4, 8),
        num_stages=(1,),
        dtypes=(torch.float16,),
        rules=(
            Rule(
                "block_wider_than_row",
                "a block wider than the hidden size only adds masked lanes",
                lambda config, problem, device: (
                    config.params["BLOCK_SIZE"] > triton.next_power_of_2(problem.shape["hidden"])
                ),
            ),
        ),
    )
    problem = Problem("rmsnorm", {"rows": 64, "hidden": 512}, torch.float16)
    candidates = space.candidates(problem, A100)
    assert len(candidates.configs) == 8
    assert candidates.pruned == {"block_wider_than_row": 4}
