"""Search-space tests.

The filters are the reason KernelForge's tuning cost is bounded, so they are
tested directly: each rule is checked against a configuration it is supposed
to reject, and the end-to-end candidate count is checked against the budget.
Because the filters read a ``DeviceCaps`` value object rather than
``torch.cuda``, all of this runs without a GPU -- including the test that a
configuration feasible on an A100 is correctly rejected on an RTX 4090.
"""

from __future__ import annotations

import pytest

from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import (
    DEFAULT_MAX_CANDIDATES,
    MatmulSearchSpace,
    RMSNormSearchSpace,
    VectorAddSearchSpace,
    next_power_of_2,
    search_space,
)

GEMM = Problem.create("matmul", "fp16", M=2048, N=4096, K=4096)


def gemm_config(**overrides) -> KernelConfig:
    params = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    params.update(overrides)
    return KernelConfig("matmul", **params)


@pytest.mark.parametrize(
    "n,expected", [(1, 1), (2, 2), (3, 4), (127, 128), (128, 128), (4097, 8192)]
)
def test_next_power_of_2(n, expected):
    assert next_power_of_2(n) == expected


def test_next_power_of_2_rejects_non_positive():
    with pytest.raises(ValueError):
        next_power_of_2(0)


def test_baseline_config_is_accepted(a100_caps):
    assert MatmulSearchSpace().reject_reason(gemm_config(), GEMM, a100_caps) is None


def test_candidate_count_respects_the_budget(a100_caps):
    result = MatmulSearchSpace().generate(GEMM, a100_caps)
    assert result.generated == 432
    assert result.feasible < result.generated, "filters rejected nothing"
    assert len(result.candidates) == DEFAULT_MAX_CANDIDATES
    assert len(set(result.candidates)) == len(result.candidates), "duplicate candidates"


def test_candidate_selection_is_deterministic(a100_caps):
    space = MatmulSearchSpace()
    assert space.candidates(GEMM, a100_caps) == space.candidates(GEMM, a100_caps)


def test_budget_is_honoured_when_tightened(a100_caps):
    assert len(MatmulSearchSpace().candidates(GEMM, a100_caps, max_candidates=5)) == 5


def test_shared_memory_filter_is_device_specific(a100_caps, rtx4090_caps):
    """The same tile is feasible on an A100 and not on an RTX 4090.

    An fp32 128x128 tile with BLOCK_K=64 over three pipeline stages keeps two
    operand tiles in flight on both cards: 128 KiB of shared memory, which fits
    in the A100's 163 KiB opt-in budget but not the 4090's 99 KiB. This is the
    concrete reason the cache key carries the board name.
    """
    fp32_problem = Problem.create("matmul", "fp32", M=2048, N=4096, K=4096)
    config = gemm_config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3)
    space = MatmulSearchSpace()
    assert space.shared_memory_bytes(config, fp32_problem, a100_caps) == 2 * 64 * 256 * 4
    assert space.reject_reason(config, fp32_problem, a100_caps) is None
    reason = space.reject_reason(config, fp32_problem, rtx4090_caps)
    assert reason is not None and "shared memory" in reason


def test_register_pressure_filter_rejects_oversized_accumulators(a100_caps):
    """A 128x128 tile on 2 warps is 256 fp32 accumulators per thread.

    The rule caps the accumulator at 128, deliberately well under the 255
    registers a thread can address, because addressing, the software pipeline
    and the epilogue all need their share. 256 would exceed even the
    architectural limit, so the accumulator alone would spill to local memory.
    """
    config = gemm_config(BLOCK_M=128, BLOCK_N=128, num_warps=2)
    reason = MatmulSearchSpace().reject_reason(config, GEMM, a100_caps)
    assert reason is not None and "register pressure" in reason


def test_tiny_tiles_are_rejected(a100_caps):
    reason = MatmulSearchSpace().reject_reason(gemm_config(BLOCK_M=16, BLOCK_N=16), GEMM, a100_caps)
    assert reason is not None and "tile too small" in reason


def test_coalescing_filter_depends_on_dtype(a100_caps):
    """BLOCK_K=16 is 32 contiguous bytes in fp16 and 64 in fp32.

    A tile row shorter than half a 128-byte transaction wastes most of each
    cache line it touches, so the rule rejects the fp16 case and accepts the
    fp32 one.
    """
    space = MatmulSearchSpace()
    fp16_reason = space.reject_reason(gemm_config(BLOCK_K=16), GEMM, a100_caps)
    assert fp16_reason is not None and "uncoalesced A tile" in fp16_reason

    fp32_problem = Problem.create("matmul", "fp32", M=2048, N=4096, K=4096)
    assert space.reject_reason(gemm_config(BLOCK_K=16), fp32_problem, a100_caps) is None


def test_tile_overshoot_filter_rejects_tiles_larger_than_the_problem(a100_caps):
    thin = Problem.create("matmul", "fp16", M=16, N=4096, K=4096)
    reason = MatmulSearchSpace().reject_reason(gemm_config(BLOCK_M=128), thin, a100_caps)
    assert reason is not None and "tile overshoot" in reason


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_decode_shaped_gemms_still_produce_candidates(m, a100_caps):
    """The single-token GEMM must be tunable.

    The smallest BLOCK_M in the grid is 16, so an unconditional overshoot rule
    rejects *every* candidate for M < 8 and tuning returns nothing. That is the
    shape single-stream decoding issues most, and the shape the workload suite,
    the experiment script and the warp-count case study all depend on.
    """
    space = MatmulSearchSpace()
    problem = Problem.create("matmul", "fp16", M=m, N=11008, K=4096)
    candidates = space.candidates(problem, a100_caps)
    assert candidates, f"no candidates for M={m}"
    # The smallest tile has to survive, since nothing smaller exists.
    assert min(c["BLOCK_M"] for c in candidates) == min(space.BLOCK_M)


def test_overshoot_rule_still_bites_when_a_smaller_tile_exists(a100_caps):
    """Relaxing the rule at the floor must not disable it above the floor."""
    space = MatmulSearchSpace()
    problem = Problem.create("matmul", "fp16", M=1, N=11008, K=4096)
    for config in space.candidates(problem, a100_caps):
        assert config["BLOCK_M"] == min(space.BLOCK_M)


@pytest.mark.parametrize("operation", ["matmul", "fused_linear"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (128, 1, 128), (4096, 8, 4096), (8, 24, 64)])
def test_narrow_gemms_still_produce_candidates(operation, dtype, m, n, k, a100_caps):
    """A narrow N must not empty the grid the way a small M once did.

    The size heuristics all prefer a wider tile or fewer warps, and the
    overshoot rule caps how wide a tile may get. For N=1 the two meet: every
    tile wide enough to fill a 128-byte transaction overshoots, and every tile
    that does not overshoot is rejected as too small. Every one of these
    shapes reported ``feasible: 0`` and made ``kernelforge tune`` exit 1,
    including ``1x1x1`` and ``128x1x128`` from the ``correctness`` suite.
    """
    space = search_space(operation)
    problem = Problem.create(operation, dtype, M=m, N=n, K=k)
    assert space.candidates(problem, a100_caps), f"no candidates for {m}x{n}x{k} {dtype}"


def test_relaxing_the_size_heuristics_does_not_widen_a_tunable_shape(a100_caps):
    """The carve-out must only apply where the grid has nothing else to offer.

    The flagship shape admits bigger tiles and more warps, so every size
    heuristic still bites and the counts stay where the README quotes them.
    """
    result = MatmulSearchSpace().generate(GEMM, a100_caps)
    assert (result.feasible, result.after_prune) == (180, 180)


@pytest.mark.parametrize("operation", ["rmsnorm", "softmax"])
@pytest.mark.parametrize("cols", [1, 3, 16, 31])
def test_rows_narrower_than_a_warp_still_produce_candidates(operation, cols, a100_caps):
    """A row that cannot fill one warp is still a row worth measuring.

    ``BLOCK_SIZE`` is the row width rounded up, so below 32 columns every warp
    count in the grid leaves lanes idle -- including the smallest. Rejecting
    all of them left ``rmsnorm rows/cols=1 x 1`` from the ``correctness`` suite
    with nothing to tune.
    """
    space = search_space(operation)
    problem = Problem.create(operation, "fp16", rows=64, cols=cols)
    candidates = space.candidates(problem, a100_caps)
    assert candidates, f"no candidates for a {cols}-column row"
    assert {c["num_warps"] for c in candidates} == {min(space.NUM_WARPS)}


def test_idle_thread_rule_still_bites_above_the_fewest_warps(a100_caps):
    """Relaxing the rule at the floor must not disable it above the floor."""
    space = RMSNormSearchSpace()
    problem = Problem.create("rmsnorm", "fp16", rows=64, cols=8)
    reason = space.reject_reason(
        KernelConfig("rmsnorm", BLOCK_SIZE=8, ROWS_PER_PROGRAM=1, num_warps=4),
        problem,
        a100_caps,
    )
    assert reason is not None and "idle threads" in reason


@pytest.mark.parametrize(
    "operation", ["matmul", "fused_linear", "rmsnorm", "softmax", "vector_add"]
)
@pytest.mark.parametrize("suite", ["smoke", "sweep", "transformer", "correctness"])
def test_every_shipped_workload_shape_is_tunable(operation, suite, a100_caps):
    """No shape the repository ships may leave the tuner with nothing to run.

    The suites are the shapes the benchmark, the experiment script and the
    kernel tests all draw from, so a shape that generates no candidates is a
    `kernelforge tune` that exits 1 on a workload this project claims to cover.
    """
    from kernelforge.benchmark import workloads

    space = search_space(operation)
    for problem in workloads.problems(operation, suite, "fp16"):
        assert space.candidates(problem, a100_caps), f"no candidates for {problem.describe()}"


def test_pipeline_filter_rejects_stages_the_k_loop_cannot_fill(a100_caps):
    shallow = Problem.create("matmul", "fp16", M=2048, N=2048, K=32)
    reason = MatmulSearchSpace().reject_reason(
        gemm_config(BLOCK_K=32, num_stages=4), shallow, a100_caps
    )
    assert reason is not None and "pipeline cannot fill" in reason


def test_utilisation_prune_drops_tilings_that_leave_sms_idle(a100_caps):
    """A 512x512 GEMM admits both 256-program and 16-program tilings.

    The coarse ones occupy a tenth of an A100 and are dropped.
    """
    space = MatmulSearchSpace()
    problem = Problem.create("matmul", "fp16", M=512, N=512, K=2048)
    result = space.generate(problem, a100_caps)
    assert result.after_prune < result.feasible
    for config in result.candidates:
        assert space.num_programs(config, problem) >= a100_caps.sm_count // 2


def test_utilisation_prune_is_skipped_when_nothing_can_fill_the_gpu(a100_caps):
    """For a small enough problem the least-bad option still needs measuring."""
    space = MatmulSearchSpace()
    tiny = Problem.create("matmul", "fp16", M=64, N=64, K=256)
    result = space.generate(tiny, a100_caps)
    assert result.after_prune == result.feasible
    assert result.candidates, "no candidates left for a small problem"


def test_large_problems_are_ranked_by_reuse(a100_caps):
    """When every tiling fills the GPU, arithmetic intensity decides."""
    space = MatmulSearchSpace()
    best = space.candidates(GEMM, a100_caps)[0]
    assert space.num_programs(best, GEMM) >= a100_caps.sm_count
    assert best["BLOCK_M"] * best["BLOCK_N"] >= 64 * 128


def test_small_problems_are_ranked_by_parallelism(a100_caps):
    """When no tiling fills the GPU, occupying more SMs beats reuse.

    A 256x256 output is 4 programs at 128x128 and 64 at 32x32. Ranking by
    reuse alone would spend the whole budget on tilings that use four of the
    A100's 108 multiprocessors.
    """
    space = MatmulSearchSpace()
    problem = Problem.create("matmul", "fp16", M=256, N=256, K=1024)
    candidates = space.candidates(problem, a100_caps)
    top = candidates[0]
    assert space.num_programs(top, problem) == max(
        space.num_programs(c, problem) for c in candidates
    )
    assert space.num_programs(top, problem) > 4


def test_rmsnorm_space_is_shaped_differently_from_the_gemm_space(a100_caps):
    """The tuner must not assume every kernel has BLOCK_M/BLOCK_N/BLOCK_K.

    A single-pass row kernel has to hold the whole row, so BLOCK_SIZE is
    determined by the problem and only the thread count and rows-per-program
    are free. The resulting space is an order of magnitude smaller.
    """
    problem = Problem.create("rmsnorm", "fp16", rows=4096, cols=4096)
    candidates = RMSNormSearchSpace().candidates(problem, a100_caps)
    assert candidates
    assert {k for c in candidates for k in c} == {
        "BLOCK_SIZE",
        "ROWS_PER_PROGRAM",
        "num_warps",
    }
    assert {c["BLOCK_SIZE"] for c in candidates} == {4096}
    assert len(candidates) < DEFAULT_MAX_CANDIDATES


def test_rmsnorm_space_scales_block_size_to_the_row(a100_caps):
    for cols, expected in [(512, 512), (768, 1024), (4096, 4096)]:
        problem = Problem.create("rmsnorm", "fp16", rows=1024, cols=cols)
        candidates = RMSNormSearchSpace().candidates(problem, a100_caps)
        assert {c["BLOCK_SIZE"] for c in candidates} == {expected}


@pytest.mark.parametrize("operation", ["rmsnorm", "softmax"])
def test_default_config_is_feasible_at_the_single_pass_limit(operation, a100_caps):
    """The widest accepted row must not hand the default config a bad setting.

    The default allocates one warp per 256 columns up to 8 warps, so at the
    limit a thread holds exactly ``MAX_ELEMS_PER_THREAD`` elements. Raising
    the width limit without revisiting the heuristic would produce a default
    configuration the search space itself rejects.
    """
    # The default heuristic lives with the kernel, so this one test needs
    # Triton importable; the rest of this module does not.
    pytest.importorskip("triton")
    from kernelforge.kernels import get_operator

    space = search_space(operation)
    cols = space.MAX_BLOCK_SIZE
    problem = Problem.create(operation, "fp16", rows=1024, cols=cols)
    config = get_operator(operation).default_config(problem)
    assert config["BLOCK_SIZE"] == cols
    per_thread = cols / (config["num_warps"] * a100_caps.warp_size)
    assert per_thread <= space.MAX_ELEMS_PER_THREAD
    assert space.reject_reason(config, problem, a100_caps) is None


def test_rmsnorm_rejects_rows_per_program_larger_than_the_batch(a100_caps):
    problem = Problem.create("rmsnorm", "fp16", rows=2, cols=512)
    for config in RMSNormSearchSpace().candidates(problem, a100_caps):
        assert config["ROWS_PER_PROGRAM"] <= 2


def test_vector_add_space_rejects_idle_threads(a100_caps):
    problem = Problem.create("vector_add", "fp16", n=1_000_003)
    for config in VectorAddSearchSpace().candidates(problem, a100_caps):
        assert config["BLOCK_SIZE"] >= config["num_warps"] * 32


def test_search_space_rejects_a_mismatched_problem(a100_caps):
    with pytest.raises(ValueError, match="handles 'matmul'"):
        MatmulSearchSpace().generate(Problem.create("rmsnorm", "fp16", rows=1, cols=1), a100_caps)


def test_search_space_lookup_by_name():
    assert isinstance(search_space("matmul"), MatmulSearchSpace)
    with pytest.raises(ValueError, match="no search space"):
        search_space("flash_attention")


def test_explain_reports_every_stage(a100_caps):
    text = MatmulSearchSpace().explain(GEMM, a100_caps)
    for expected in ("grid points", "feasible", "after prune", "selected (budget)"):
        assert expected in text
