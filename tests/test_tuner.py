"""Tuner tests.

The tuner is generic over :class:`Operator`, so it can be driven end to end on
a CPU with a purpose-built operator whose configurations fail in each of the
ways a real kernel configuration fails: one raises at launch, one returns the
wrong answer, one is correct but slow, one is correct and fast. That makes the
property that matters testable without a GPU -- an incorrect configuration is
never ranked, however fast it is.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from kernelforge.db import ResultsDB
from kernelforge.kernels.base import Operator
from kernelforge.runtime.env import DeviceCaps
from kernelforge.tuning.cache import ConfigCache
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.search import SearchSpace
from kernelforge.tuning.tuner import (
    STATUS_INCORRECT,
    STATUS_OK,
    STATUS_RUNTIME_ERROR,
    Tuner,
)

# Behaviours keyed by the MODE parameter of the fake operator.
MODE_CORRECT_FAST = 0
MODE_RAISES = 1
MODE_WRONG_ANSWER = 2
MODE_CORRECT_SLOW = 3
MODE_INFEASIBLE = 4


class _FakeSpace(SearchSpace):
    operation = "cpu_double"

    def grid(self, problem: Problem) -> Iterator[KernelConfig]:
        for mode in range(5):
            yield KernelConfig(self.operation, MODE=mode)

    def reject_reason(self, config, problem, caps) -> str | None:
        if config["MODE"] == MODE_INFEASIBLE:
            return "infeasible by construction"
        return None


class _FakeOperator(Operator):
    """Doubles a vector. Each MODE fails in a different, deliberate way."""

    name = "cpu_double"
    config_order = ("MODE",)

    def search_space(self) -> _FakeSpace:
        return _FakeSpace()

    def make_inputs(self, problem, device, *, seed=0):
        gen = torch.Generator(device=device).manual_seed(seed)
        return (
            torch.randn(problem.dims_dict["n"], device=device, dtype=problem.dtype, generator=gen),
        )

    def reference(self, *inputs):
        (x,) = inputs
        return x * 2

    def run(self, config, *inputs):
        (x,) = inputs
        mode = config["MODE"]
        if mode == MODE_RAISES:
            raise RuntimeError("simulated launch failure")
        if mode == MODE_WRONG_ANSWER:
            return x * 3
        if mode == MODE_CORRECT_SLOW:
            out = x.clone()
            for _ in range(40):
                out = out * 1.0
            return out * 2
        return x * 2

    def default_config(self, problem):
        return KernelConfig("cpu_double", MODE=MODE_CORRECT_FAST)

    def flops(self, problem):
        return problem.dims_dict["n"]

    def bytes_moved(self, problem):
        return 2 * problem.dims_dict["n"] * problem.itemsize

    def baselines(self, problem, inputs):
        (x,) = inputs
        return {"torch_eager": lambda: x * 2}


PROBLEM = Problem.create("cpu_double", "fp32", n=200_000)


@pytest.fixture
def tuned():
    tuner = Tuner(warmup=2, iterations=5, flush_l2=False, measure_baselines=True)
    return tuner.tune(_FakeOperator(), PROBLEM, device="cpu")


def test_counts_cover_every_stage(tuned):
    assert tuned.generated == 5
    assert tuned.feasible == 4
    assert tuned.tested == 4
    assert tuned.correct == 2


def test_each_failure_mode_is_classified(tuned):
    by_mode = {o.config["MODE"]: o for o in tuned.outcomes}
    assert by_mode[MODE_CORRECT_FAST].status == STATUS_OK
    assert by_mode[MODE_RAISES].status == STATUS_RUNTIME_ERROR
    assert "simulated launch failure" in by_mode[MODE_RAISES].error
    assert by_mode[MODE_WRONG_ANSWER].status == STATUS_INCORRECT
    assert by_mode[MODE_CORRECT_SLOW].status == STATUS_OK
    assert MODE_INFEASIBLE not in by_mode


def test_incorrect_configurations_are_never_ranked(tuned):
    """The central guarantee: ranking only ever sees verified candidates."""
    ranked_modes = [o.config["MODE"] for o in tuned.ranked()]
    assert MODE_WRONG_ANSWER not in ranked_modes
    assert MODE_RAISES not in ranked_modes
    assert set(ranked_modes) == {MODE_CORRECT_FAST, MODE_CORRECT_SLOW}


def test_failed_candidates_carry_no_timing(tuned):
    for outcome in tuned.outcomes:
        if not outcome.ok:
            assert outcome.timing is None
            with pytest.raises(ValueError, match="no timing"):
                _ = outcome.median_ms


def test_best_is_the_fastest_correct_candidate(tuned):
    assert tuned.best is not None
    assert tuned.best.config["MODE"] == MODE_CORRECT_FAST
    assert tuned.best_config == KernelConfig("cpu_double", MODE=MODE_CORRECT_FAST)
    assert tuned.ranked()[0].median_ms <= tuned.ranked()[-1].median_ms


def test_rejection_reasons_are_retained(tuned):
    assert len(tuned.rejections) == 1
    config, reason = tuned.rejections[0]
    assert config["MODE"] == MODE_INFEASIBLE
    assert reason == "infeasible by construction"


def test_status_counts_sum_to_the_tested_count(tuned):
    assert sum(tuned.status_counts().values()) == tuned.tested


def test_baselines_are_measured_with_identical_settings(tuned):
    assert "torch_eager" in tuned.baselines
    baseline = tuned.baselines["torch_eager"]
    assert baseline.warmup == 2
    assert baseline.iterations == 5
    assert tuned.speedup_over("torch_eager") > 0
    assert tuned.speedup_over("does_not_exist") is None


def test_derived_metrics_use_the_operator_models(tuned):
    ms = tuned.best.median_ms
    assert tuned.tflops(ms) == pytest.approx(PROBLEM.dims_dict["n"] / (ms * 1e-3) / 1e12)
    assert tuned.gbps(ms) > 0


def test_environment_is_captured_with_the_result(tuned):
    assert tuned.environment.torch_version == torch.__version__
    assert tuned.elapsed_s > 0


def test_a_space_with_no_feasible_candidates_yields_no_best():
    class _Empty(_FakeSpace):
        def reject_reason(self, config, problem, caps):
            return "nothing is feasible"

    class _EmptyOperator(_FakeOperator):
        def search_space(self):
            return _Empty()

    result = Tuner(warmup=1, iterations=2, measure_baselines=False).tune(
        _EmptyOperator(), PROBLEM, device="cpu"
    )
    assert result.tested == 0
    assert result.best is None
    assert result.ranked() == []


def test_results_are_persisted_to_the_database(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    tuner = Tuner(warmup=1, iterations=2, flush_l2=False, db=db)
    result = tuner.tune(_FakeOperator(), PROBLEM, device="cpu")

    counts = db.counts()
    assert counts["runs"] == 1
    assert counts["problems"] == 1
    # One row per tested candidate plus one per baseline.
    assert counts["results"] == result.tested + len(result.baselines)

    rows = db.rows(operation="cpu_double")
    statuses = {row["status"] for row in rows}
    assert {STATUS_OK, STATUS_INCORRECT, STATUS_RUNTIME_ERROR} <= statuses
    # Failed candidates are recorded, with their error, and without a latency.
    failed = [r for r in rows if r["status"] == STATUS_RUNTIME_ERROR]
    assert failed and failed[0]["median_us"] is None and failed[0]["error"]
    db.close()


def test_best_config_is_written_to_the_cache(tmp_path):
    cache = ConfigCache(tmp_path / "configs.json")
    tuner = Tuner(warmup=1, iterations=2, flush_l2=False, cache=cache)
    result = tuner.tune(_FakeOperator(), PROBLEM, device="cpu")

    stored = cache.get(PROBLEM, result.environment.device_key)
    assert stored == result.best_config
    entry = ConfigCache(tmp_path / "configs.json").entries()[0]
    assert entry.candidates_tested == result.tested
    assert entry.median_us > 0


def test_log_callback_reports_the_phases():
    lines = []
    Tuner(warmup=1, iterations=2, measure_baselines=False, log=lines.append).tune(
        _FakeOperator(), PROBLEM, device="cpu"
    )
    joined = "\n".join(lines)
    assert "Verifying candidates" in joined
    assert "2 / 4 configurations passed correctness" in joined
    assert "Benchmarking" in joined


def test_tuner_honours_the_candidate_budget():
    result = Tuner(warmup=1, iterations=2, max_candidates=2, measure_baselines=False).tune(
        _FakeOperator(), PROBLEM, device="cpu"
    )
    assert result.tested == 2


def test_device_caps_fall_back_without_cuda():
    """The filters need capabilities even when no device is attached."""
    caps = DeviceCaps(
        name="NVIDIA Test",
        compute_capability="9.0",
        total_memory_bytes=1,
        sm_count=1,
        max_shared_memory_per_block=1,
        shared_memory_per_sm=1,
        registers_per_sm=1,
        max_threads_per_sm=1,
        warp_size=32,
        l2_cache_bytes=1,
    )
    assert caps.arch == "sm90"
    assert caps.key == "NVIDIA_Test_sm90"
