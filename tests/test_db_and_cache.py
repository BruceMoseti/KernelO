"""Results database and config cache tests."""

from __future__ import annotations

import dataclasses
import json

import pytest

from kernelforge.benchmark.runner import summarize
from kernelforge.db import SCHEMA_VERSION, Measurement, ResultsDB
from kernelforge.runtime.env import capture_environment
from kernelforge.testing import VerificationResult
from kernelforge.tuning.cache import CACHE_VERSION, ConfigCache
from kernelforge.tuning.config import KernelConfig, Problem

PROBLEM = Problem.create("matmul", "fp16", M=2048, N=4096, K=4096)
CONFIG = KernelConfig(
    "matmul", BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=4
)


def timing(median_ms: float):
    return summarize([median_ms], warmup=25, iterations=1, timer="cuda_event", flushed_l2=True)


def passing_verification():
    return VerificationResult(
        passed=True, error=1e-4, max_abs_err=1e-3, mismatched=0, total=100, threshold=5e-3
    )


@pytest.fixture
def db(tmp_path):
    with ResultsDB(tmp_path / "results.db") as handle:
        yield handle


def test_schema_version_is_recorded(tmp_path):
    path = tmp_path / "results.db"
    with ResultsDB(path):
        pass
    import sqlite3

    connection = sqlite3.connect(path)
    assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    connection.close()


def test_mismatched_schema_version_is_refused(tmp_path):
    import sqlite3

    path = tmp_path / "results.db"
    with ResultsDB(path):
        pass
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="schema version"):
        ResultsDB(path)


def test_problems_and_configs_are_deduplicated(db):
    assert db.problem_id(PROBLEM) == db.problem_id(PROBLEM)
    assert db.config_id(CONFIG) == db.config_id(CONFIG)
    other = Problem.create("matmul", "fp32", M=2048, N=4096, K=4096)
    assert db.problem_id(other) != db.problem_id(PROBLEM)
    assert db.counts()["problems"] == 2


def test_identical_parameters_for_different_operators_do_not_collide(db):
    gemm = KernelConfig("matmul", BLOCK_M=64)
    fused = KernelConfig("fused_linear", BLOCK_M=64)
    assert db.config_id(gemm) != db.config_id(fused)


def test_measurement_round_trips(db):
    env = capture_environment()
    run_id = db.start_run(env, notes="unit test", version="0.1.0")
    db.record(
        run_id,
        PROBLEM,
        Measurement(
            label="kernelforge",
            status="ok",
            config=CONFIG,
            verification=passing_verification(),
            timing=timing(0.5),
            tflops=137.4,
        ),
    )
    (row,) = db.rows()
    assert row["label"] == "kernelforge"
    assert row["median_us"] == pytest.approx(500.0)
    assert row["correct"] == 1
    assert row["tflops"] == pytest.approx(137.4)
    assert row["operation"] == "matmul"
    assert row["shape_key"] == "2048x4096x4096"
    assert json.loads(row["params_json"])["BLOCK_M"] == 64
    # Provenance travels with the row, not in a separate note.
    assert row["torch_version"] == env.torch_version


def test_failed_candidates_are_recorded_without_timings(db):
    run_id = db.start_run(capture_environment())
    db.record(
        run_id,
        PROBLEM,
        Measurement(
            label="kernelforge",
            status="compile_error",
            config=CONFIG,
            error="OutOfResources: shared memory",
        ),
    )
    (row,) = db.rows()
    assert row["status"] == "compile_error"
    assert row["median_us"] is None
    assert "OutOfResources" in row["error"]


def test_baselines_are_stored_without_a_config(db):
    run_id = db.start_run(capture_environment())
    db.record(run_id, PROBLEM, Measurement(label="torch_eager", status="ok", timing=timing(1.0)))
    (row,) = db.rows()
    assert row["params_json"] is None
    assert row["label"] == "torch_eager"


def test_best_per_label_takes_the_fastest_correct_row(db):
    run_id = db.start_run(capture_environment())
    for median, label, correct in [
        (0.9, "kernelforge", True),
        (0.4, "kernelforge", False),
        (0.7, "kernelforge", True),
        (1.5, "torch_eager", True),
    ]:
        verification = (
            passing_verification()
            if correct
            else VerificationResult(False, 1.0, 1.0, 10, 10, 5e-3, "wrong")
        )
        db.record(
            run_id,
            PROBLEM,
            Measurement(
                label=label,
                status="ok" if correct else "incorrect",
                config=CONFIG if label == "kernelforge" else None,
                verification=verification,
                timing=timing(median),
            ),
        )
    best = {row["label"]: row["median_us"] for row in db.best_per_label()}
    # 0.4 ms was faster but incorrect, so it must not win.
    assert best["kernelforge"] == pytest.approx(700.0)
    assert best["torch_eager"] == pytest.approx(1500.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_name", "NVIDIA H100 80GB HBM3"),
        ("torch_version", "2.99.0"),
        ("triton_version", "9.9.0"),
    ],
)
def test_best_per_label_never_ranks_across_environments(db, field, value):
    environment = capture_environment()
    for env, median in (
        (environment, 1.0),
        (dataclasses.replace(environment, **{field: value}), 0.5),
    ):
        db.record(
            db.start_run(env),
            PROBLEM,
            Measurement(
                label="kernelforge",
                status="ok",
                verification=passing_verification(),
                timing=timing(median),
            ),
        )
    medians = sorted(row["median_us"] for row in db.best_per_label())
    assert medians == [pytest.approx(500.0), pytest.approx(1000.0)]


def test_rows_can_be_filtered(db):
    run_id = db.start_run(capture_environment())
    other = Problem.create("rmsnorm", "fp16", rows=4096, cols=4096)
    db.record(run_id, PROBLEM, Measurement(label="kernelforge", status="ok", timing=timing(1.0)))
    db.record(run_id, other, Measurement(label="kernelforge", status="ok", timing=timing(2.0)))
    assert len(db.rows(operation="matmul")) == 1
    assert len(db.rows(dtype="fp16")) == 2
    assert db.operations() == ["matmul", "rmsnorm"]


# --- cache ---------------------------------------------------------------
def test_cache_round_trips(tmp_path):
    path = tmp_path / "configs.json"
    cache = ConfigCache(path)
    assert cache.get(PROBLEM, "A100_sm80") is None

    cache.put(PROBLEM, "A100_sm80", CONFIG, median_us=421.0, timestamp="2026-01-01T00:00:00")
    assert cache.get(PROBLEM, "A100_sm80") == CONFIG
    # Reloading from disk must give the same answer.
    assert ConfigCache(path).get(PROBLEM, "A100_sm80") == CONFIG


def test_cache_is_keyed_by_device(tmp_path):
    """A configuration tuned on one board must not be served for another.

    Same architecture, different resources: an entry stored for a 4090 must
    miss for a 4080.
    """
    cache = ConfigCache(tmp_path / "configs.json")
    cache.put(PROBLEM, "NVIDIA_GeForce_RTX_4090_sm89", CONFIG)
    assert cache.get(PROBLEM, "NVIDIA_GeForce_RTX_4090_sm89") == CONFIG
    assert cache.get(PROBLEM, "NVIDIA_GeForce_RTX_4080_sm89") is None


def test_cache_is_keyed_by_dtype_and_shape(tmp_path):
    cache = ConfigCache(tmp_path / "configs.json")
    cache.put(PROBLEM, "A100_sm80", CONFIG)
    assert cache.get(Problem.create("matmul", "fp32", M=2048, N=4096, K=4096), "A100_sm80") is None
    assert cache.get(Problem.create("matmul", "fp16", M=2048, N=4096, K=2048), "A100_sm80") is None
    assert cache.get(Problem.create("rmsnorm", "fp16", rows=2048, cols=4096), "A100_sm80") is None


def test_cache_overwrites_a_retuned_entry(tmp_path):
    cache = ConfigCache(tmp_path / "configs.json")
    cache.put(PROBLEM, "A100_sm80", CONFIG, median_us=500.0)
    better = KernelConfig(
        "matmul", BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=3
    )
    cache.put(PROBLEM, "A100_sm80", better, median_us=400.0)
    assert cache.get(PROBLEM, "A100_sm80") == better
    assert len(cache) == 1


def test_cache_entries_are_listable_and_clearable(tmp_path):
    path = tmp_path / "configs.json"
    cache = ConfigCache(path)
    cache.put(PROBLEM, "A100_sm80", CONFIG, median_us=421.0)
    cache.put(
        Problem.create("rmsnorm", "fp16", rows=4096, cols=4096),
        "A100_sm80",
        KernelConfig("rmsnorm", BLOCK_SIZE=4096, ROWS_PER_PROGRAM=2, num_warps=8),
    )
    entries = cache.entries()
    assert [e.operation for e in entries] == ["matmul", "rmsnorm"]
    assert "421.0 us" in entries[0].describe()

    assert cache.clear() == 2
    assert not path.exists()
    assert ConfigCache(path).entries() == []


def test_cache_version_mismatch_is_refused(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text(json.dumps({"version": CACHE_VERSION + 1, "entries": {}}))
    with pytest.raises(RuntimeError, match="cache clear"):
        ConfigCache(path)


def test_corrupt_cache_is_reported_not_silently_discarded(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text("{not json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        ConfigCache(path)


def test_cache_writes_are_atomic(tmp_path):
    """A write must not leave a stray temporary file behind."""
    path = tmp_path / "configs.json"
    cache = ConfigCache(path)
    cache.put(PROBLEM, "A100_sm80", CONFIG)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["configs.json"]
