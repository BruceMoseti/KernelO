"""Report-generation tests.

Reports are rendered from a seeded database rather than from a GPU run, so
the rendering is covered on any machine. The numbers here are fixtures, chosen
to exercise the code paths; nothing in this file is a measurement.
"""

from __future__ import annotations

import pytest

from kernelforge.benchmark import report
from kernelforge.benchmark.runner import summarize
from kernelforge.db import Measurement, ResultsDB
from kernelforge.runtime.env import capture_environment
from kernelforge.testing import VerificationResult
from kernelforge.tuning.config import KernelConfig, Problem

pytest.importorskip("pandas", reason="report generation needs the 'report' extra")
pytest.importorskip("matplotlib", reason="report generation needs the 'report' extra")


def timing(median_ms: float):
    return summarize(
        [median_ms * 0.98, median_ms, median_ms * 1.05],
        warmup=25,
        iterations=3,
        timer="cuda_event",
        flushed_l2=True,
    )


def ok():
    return VerificationResult(True, 1e-4, 1e-3, 0, 100, 5e-3)


@pytest.fixture
def seeded_db(tmp_path):
    """A database shaped like a real tuning run, with invented latencies."""
    db = ResultsDB(tmp_path / "results.db")
    run_id = db.start_run(capture_environment(), notes="fixture", version="0.1.0")

    for shape, eager_ms, mine_ms in [
        ((1024, 1024, 1024), 0.30, 0.26),
        ((2048, 4096, 4096), 2.40, 2.10),
    ]:
        problem = Problem.create("matmul", "fp16", M=shape[0], N=shape[1], K=shape[2])
        flops = 2 * shape[0] * shape[1] * shape[2]
        for label, ms in (
            ("torch_eager", eager_ms),
            ("torch_compile", eager_ms * 0.97),
            ("triton_baseline", eager_ms * 1.4),
            ("kernelforge", mine_ms),
        ):
            db.record(
                run_id,
                problem,
                Measurement(
                    label=label,
                    status="ok",
                    config=(
                        KernelConfig(
                            "matmul",
                            BLOCK_M=64,
                            BLOCK_N=128,
                            BLOCK_K=32,
                            GROUP_M=8,
                            num_warps=8,
                            num_stages=4,
                        )
                        if label == "kernelforge"
                        else None
                    ),
                    verification=ok(),
                    timing=timing(ms),
                    tflops=flops / (ms * 1e-3) / 1e12,
                ),
            )
        # A spread of tile shapes, so the heatmap has a plane to draw.
        for block_m, block_n, ms in [
            (32, 32, mine_ms * 1.6),
            (64, 64, mine_ms * 1.1),
            (64, 128, mine_ms),
            (128, 128, mine_ms * 1.05),
        ]:
            db.record(
                run_id,
                problem,
                Measurement(
                    label="kernelforge",
                    status="ok",
                    config=KernelConfig(
                        "matmul",
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=32,
                        GROUP_M=8,
                        num_warps=4,
                        num_stages=3,
                    ),
                    verification=ok(),
                    timing=timing(ms),
                    tflops=flops / (ms * 1e-3) / 1e12,
                ),
            )

    rms = Problem.create("rmsnorm", "fp16", rows=4096, cols=4096)
    rms_bytes = (2 * 4096 * 4096 + 4096) * 2
    for label, ms in (("torch_eager", 0.21), ("kernelforge", 0.09)):
        db.record(
            run_id,
            rms,
            Measurement(
                label=label,
                status="ok",
                verification=ok(),
                timing=timing(ms),
                gbps=rms_bytes / (ms * 1e-3) / 1e9,
            ),
        )
    yield db
    db.close()


def test_report_writes_summary_tables_and_figures(seeded_db, tmp_path):
    artifacts = report.generate(seeded_db, tmp_path / "reports")
    assert artifacts.summary.exists()
    names = {p.name for p in artifacts.figures}
    assert "matmul_latency.png" in names
    assert "matmul_tflops.png" in names
    assert "rmsnorm_bandwidth.png" in names
    assert "tuning_heatmap.png" in names
    assert {p.name for p in artifacts.tables} == {"matmul.csv", "rmsnorm.csv"}
    for path in artifacts.figures:
        assert path.stat().st_size > 1000, f"{path.name} looks empty"


def test_summary_contains_provenance_and_speedups(seeded_db, tmp_path):
    report.generate(seeded_db, tmp_path / "reports")
    text = (tmp_path / "reports" / "summary.md").read_text()
    assert "## Environment" in text
    assert "PyTorch" in text
    assert "## matmul" in text
    assert "speedup vs eager" in text
    # 0.30 / 0.26 = 1.15x
    assert "1.15x" in text
    assert "TFLOP/s" in text
    # Memory-bound operators are reported in GB/s instead.
    assert "GB/s" in text


def test_memory_bound_operators_get_a_bandwidth_figure(seeded_db, tmp_path):
    artifacts = report.generate(seeded_db, tmp_path / "reports", operations=["rmsnorm"])
    names = {p.name for p in artifacts.figures}
    assert "rmsnorm_bandwidth.png" in names
    assert "rmsnorm_tflops.png" not in names


def test_empty_database_produces_a_report_that_says_so(tmp_path):
    with ResultsDB(tmp_path / "empty.db") as db:
        artifacts = report.generate(db, tmp_path / "reports")
    assert "No correct, timed measurements" in artifacts.summary.read_text()
    assert artifacts.figures == []
    assert artifacts.notes


def test_incorrect_results_are_excluded_from_reports(tmp_path):
    """A fast-but-wrong row must never reach a chart."""
    with ResultsDB(tmp_path / "results.db") as db:
        run_id = db.start_run(capture_environment())
        problem = Problem.create("matmul", "fp16", M=256, N=256, K=256)
        db.record(
            run_id,
            problem,
            Measurement(
                label="kernelforge",
                status="incorrect",
                verification=VerificationResult(False, 1.0, 1.0, 50, 100, 5e-3, "wrong"),
                timing=timing(0.01),
            ),
        )
        db.record(
            run_id,
            problem,
            Measurement(label="torch_eager", status="ok", verification=ok(), timing=timing(0.10)),
        )
        artifacts = report.generate(db, tmp_path / "reports")
    text = artifacts.summary.read_text()
    assert "torch_eager" in text or "PyTorch eager" in text
    assert "0.01" not in text


def test_heatmap_is_skipped_when_there_are_no_tuned_candidates(tmp_path):
    with ResultsDB(tmp_path / "results.db") as db:
        run_id = db.start_run(capture_environment())
        problem = Problem.create("rmsnorm", "fp16", rows=512, cols=512)
        db.record(
            run_id,
            problem,
            Measurement(label="torch_eager", status="ok", verification=ok(), timing=timing(0.1)),
        )
        artifacts = report.generate(db, tmp_path / "reports")
    assert any("tuning_heatmap" in note for note in artifacts.notes)
    assert "tuning_heatmap.png" not in {p.name for p in artifacts.figures}
