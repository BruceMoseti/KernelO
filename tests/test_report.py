"""Report-generation tests.

Reports are rendered from a seeded database rather than from a GPU run, so
the rendering is covered on any machine. The numbers here are fixtures, chosen
to exercise the code paths; nothing in this file is a measurement.
"""

from __future__ import annotations

import dataclasses

import pytest

from kernelforge.benchmark import metrics, report
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


def test_summary_rows_are_ordered_by_problem_size(seeded_db, tmp_path):
    """Not by the shape string, which would sort 512x512x512 after 2048x4096x4096."""
    report.generate(seeded_db, tmp_path / "reports")
    text = (tmp_path / "reports" / "summary.md").read_text()
    assert text.index("| 1024x1024x1024 ") < text.index("| 2048x4096x4096 ")


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


def test_two_dtypes_at_one_shape_render_as_separate_series(tmp_path):
    """A shape alone is not a unique series.

    The same shape measured in fp16 and bf16 used to collapse onto one x-axis
    label, handing matplotlib a two-element Series where it expects a scalar.
    Both the CI workflow and the experiment script take a dtype parameter and
    write the same database, so this is an ordinary path.
    """
    with ResultsDB(tmp_path / "results.db") as db:
        run_id = db.start_run(capture_environment())
        for dtype, ms in (("fp16", 1.0), ("bf16", 1.2)):
            problem = Problem.create("matmul", dtype, M=1024, N=1024, K=1024)
            for label, value in (("torch_eager", ms), ("kernelforge", ms * 0.9)):
                db.record(
                    run_id,
                    problem,
                    Measurement(
                        label=label,
                        status="ok",
                        verification=ok(),
                        timing=timing(value),
                        tflops=1.0,
                    ),
                )
        artifacts = report.generate(db, tmp_path / "reports")

    text = artifacts.summary.read_text()
    assert "fp16" in text and "bf16" in text
    figure = next(p for p in artifacts.figures if p.name == "matmul_latency.png")
    assert figure.stat().st_size > 1000


def test_mixing_flushed_and_unflushed_timings_is_reported(tmp_path):
    """The two are not comparable, so the report must not stay quiet about it."""
    with ResultsDB(tmp_path / "results.db") as db:
        run_id = db.start_run(capture_environment())
        problem = Problem.create("matmul", "fp16", M=512, N=512, K=512)
        for label, flushed in (("torch_eager", True), ("kernelforge", False)):
            db.record(
                run_id,
                problem,
                Measurement(
                    label=label,
                    status="ok",
                    verification=ok(),
                    timing=summarize(
                        [1.0],
                        warmup=25,
                        iterations=1,
                        timer="cuda_event",
                        flushed_l2=flushed,
                    ),
                ),
            )
        artifacts = report.generate(db, tmp_path / "reports")
    assert any("unflushed" in note for note in artifacts.notes)
    assert "Warning" in artifacts.summary.read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_name", "NVIDIA H100 80GB HBM3"),
        ("torch_version", "2.99.0"),
        ("triton_version", "9.9.0"),
    ],
)
def test_rows_from_different_environments_are_never_compared(field, value, tmp_path):
    """A speedup divides two rows only if one GPU and one software stack produced both.

    An H100 KernelForge row over an A100 eager row used to print "2.00x" under
    an environment table that named only the A100.
    """
    a100 = dataclasses.replace(
        capture_environment(),
        gpu_name="NVIDIA A100-SXM4-80GB",
        gpu_arch="8.0",
        device_key="NVIDIA_A100_SXM4_80GB_sm80",
    )
    problem = Problem.create("matmul", "fp16", M=2048, N=4096, K=4096)
    with ResultsDB(tmp_path / "results.db") as db:
        run_id = db.start_run(dataclasses.replace(a100, **{field: value}))
        db.record(
            run_id,
            problem,
            Measurement(label="kernelforge", status="ok", verification=ok(), timing=timing(1.0)),
        )
        run_id = db.start_run(a100)
        for label, ms in (("torch_eager", 2.0), ("kernelforge", 1.6)):
            db.record(
                run_id,
                problem,
                Measurement(label=label, status="ok", verification=ok(), timing=timing(ms)),
            )
        artifacts = report.generate(db, tmp_path / "reports")

    text = artifacts.summary.read_text()
    assert "2.00x" not in text
    assert "1.25x" in text
    assert "NVIDIA A100-SXM4-80GB" in text and value in text


def test_runs_in_one_environment_are_compared(tmp_path):
    """``tune`` and ``benchmark`` record separate runs of one environment."""
    environment = capture_environment()
    problem = Problem.create("matmul", "fp16", M=512, N=512, K=512)
    with ResultsDB(tmp_path / "results.db") as db:
        for label, ms in (("kernelforge", 1.0), ("torch_eager", 2.0)):
            db.record(
                db.start_run(environment),
                problem,
                Measurement(label=label, status="ok", verification=ok(), timing=timing(ms)),
            )
        artifacts = report.generate(db, tmp_path / "reports")
    assert "2.00x" in artifacts.summary.read_text()


@pytest.mark.parametrize(
    "operation,dtype,dims,throughput",
    [
        ("matmul", "fp16", {"M": 4096, "N": 4096, "K": 4096}, {"tflops": 156.0}),
        ("matmul", "fp32", {"M": 4096, "N": 4096, "K": 4096}, {"tflops": 9.75}),
        ("rmsnorm", "fp16", {"rows": 4096, "cols": 4096}, {"gbps": 1019.5}),
    ],
)
def test_throughput_is_a_fraction_of_the_published_peak(
    operation, dtype, dims, throughput, tmp_path
):
    """Half of each roof on the A100-SXM4-80GB datasheet: 312 and 19.5 TFLOP/s, 2039 GB/s.

    fp32 is held against the FP32 roof rather than TF32's, because the kernels
    pin fp32 ``tl.dot`` to IEEE.
    """
    a100 = dataclasses.replace(
        capture_environment(), gpu_name="NVIDIA A100-SXM4-80GB", gpu_arch="8.0"
    )
    with ResultsDB(tmp_path / "results.db") as db:
        db.record(
            db.start_run(a100),
            Problem.create(operation, dtype, **dims),
            Measurement(
                label="kernelforge",
                status="ok",
                verification=ok(),
                timing=timing(1.0),
                **throughput,
            ),
        )
        artifacts = report.generate(db, tmp_path / "reports")
    text = artifacts.summary.read_text()
    assert "| 50.0% |" in text
    assert metrics.PUBLISHED_PEAKS["NVIDIA A100-SXM4-80GB"].source in text


def test_a_gpu_without_a_published_peak_gets_no_utilisation_figure(tmp_path):
    """An unknown roof is left blank, never estimated."""
    unknown = dataclasses.replace(capture_environment(), gpu_name="NVIDIA Unlisted GPU")
    with ResultsDB(tmp_path / "results.db") as db:
        db.record(
            db.start_run(unknown),
            Problem.create("matmul", "fp16", M=512, N=512, K=512),
            Measurement(
                label="kernelforge",
                status="ok",
                verification=ok(),
                timing=timing(1.0),
                tflops=100.0,
            ),
        )
        artifacts = report.generate(db, tmp_path / "reports")
    text = artifacts.summary.read_text()
    assert "| 100.0 | - | - |" in text
    assert "No published peak on file for NVIDIA Unlisted GPU" in text


def test_every_published_peak_cites_an_nvidia_document():
    for gpu, peak in metrics.PUBLISHED_PEAKS.items():
        assert peak.source.startswith("https://") and "nvidia.com/" in peak.source, gpu


def test_provenance_comes_from_the_latest_run_not_the_latest_result(tmp_path):
    """A run that recorded nothing still happened."""
    with ResultsDB(tmp_path / "results.db") as db:
        db.start_run(capture_environment(), notes="empty run")
        assert db.runs()
        artifacts = report.generate(db, tmp_path / "reports")
        assert "No runs recorded." not in artifacts.summary.read_text()


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
