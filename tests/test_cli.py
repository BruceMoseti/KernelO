import contextlib
import dataclasses
import io
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import triton

from kernelforge.benchmark.metrics import summarize
from kernelforge.benchmark.runner import BenchmarkResult
from kernelforge.benchmark.workloads import MATMUL_BASELINE, matmul_problem
from kernelforge.cli.main import main
from kernelforge.runtime.environment import collect_environment
from kernelforge.tuning.cache import CacheKey
from kernelforge.tuning.database import TuningDatabase
from kernelforge.tuning.search import DeviceLimits
from kernelforge.tuning.tuner import Measurement

SYNTHETIC_GPU = dataclasses.replace(
    collect_environment(),
    gpu="Synthetic test GPU",
    compute_capability="0.0",
    driver=None,
    cuda=None,
    triton=triton.__version__,
)

PROBLEM = matmul_problem(2048, 4096, 4096, torch.float16)
KEY = CacheKey("Synthetic test GPU", "0.0", "3.8.0", "matmul", "fp16", PROBLEM.shape_json())


def _database_with_one_cached_config(path: Path) -> None:
    with TuningDatabase(path) as database:
        run_id = database.start_run(collect_environment())
        database.record(
            run_id, PROBLEM, Measurement("kernelforge", MATMUL_BASELINE, False, error="x")
        )
        database.store_cache(KEY, MATMUL_BASELINE, 123.0, run_id)


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "tune" in capsys.readouterr().out


def test_rejects_unknown_dtype() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["tune", "matmul", "--m", "8", "--n", "8", "--k", "8", "--dtype", "int8"])
    assert exit_info.value.code == 2


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU-only path")
def test_tune_refuses_to_run_without_a_gpu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["tune", "matmul", "--m", "8", "--n", "8", "--k", "8", "--db", str(tmp_path / "t.db")]
    )
    assert code == 1
    assert "needs a CUDA GPU" in capsys.readouterr().err
    assert not (tmp_path / "t.db").exists()


def test_cache_commands_on_a_missing_database_create_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "missing.db"
    assert main(["cache", "list", "--db", str(path)]) == 0
    assert main(["cache", "clear", "--db", str(path)]) == 0
    assert "No tuning database" in capsys.readouterr().out
    assert not path.exists()


def test_cache_list_shows_entries(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "tuning.db"
    _database_with_one_cached_config(path)
    assert main(["cache", "list", "--db", str(path)]) == 0
    out = capsys.readouterr().out
    assert "1 cached configuration(s)" in out
    assert "Synthetic test GPU (sm 0.0, Triton 3.8.0) | matmul fp16 K=4096 M=2048 N=4096" in out
    assert str(MATMUL_BASELINE) in out


def test_cache_clear_removes_entries_and_keeps_measurements(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tuning.db"
    _database_with_one_cached_config(path)
    assert main(["cache", "clear", "--db", str(path)]) == 0
    assert "Removed 1 cached configuration(s)" in capsys.readouterr().out
    with TuningDatabase(path) as database:
        assert database.cache_entries() == []
        assert database.execute("SELECT count(*) FROM results") == [(1,)]
    main(["cache", "list", "--db", str(path)])
    assert "0 cached configuration(s)" in capsys.readouterr().out


def test_tune_matmul_flow_and_report_format(
    tmp_path: Path, device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The full command on CPU: kernels run in the interpreter; the GPU-only pieces (CUDA timer,
    # Triton's device query, the environment) are stand-ins, and torch.compile is bypassed. Only
    # the flow and the report structure are checked; the stand-in latencies mean nothing.
    from kernelforge.cli import main as cli

    def stand_in_benchmark(
        fn: Callable[[], object], *, warmup: int, iterations: int
    ) -> BenchmarkResult:
        fn()
        return BenchmarkResult(
            summarize([50.0, 50.0]), (50.0, 50.0), warmup, iterations, 0, SYNTHETIC_GPU
        )

    monkeypatch.setattr(cli, "benchmark", stand_in_benchmark)
    monkeypatch.setattr(cli, "device_limits", lambda: DeviceLimits(108, 166_912))
    monkeypatch.setattr(cli, "collect_environment", lambda: SYNTHETIC_GPU)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    db = tmp_path / "tuning.db"
    args = cli._parser().parse_args(
        [
            "tune",
            "matmul",
            "--m",
            "48",
            "--n",
            "40",
            "--k",
            "80",
            "--dtype",
            "fp32",
            "--db",
            str(db),
        ]
    )

    lines = _captured_lines(lambda: cli._run_tune_matmul(args, device))
    assert lines[:4] == [
        "GPU: Synthetic test GPU (compute capability 0.0, driver None, CUDA None)",
        f"Software: PyTorch {torch.__version__}, Triton {SYNTHETIC_GPU.triton}",
        "Shape: 48 x 40 x 80",
        "dtype: FP32",
    ]
    best = lines.index("Best configuration")
    assert [line.split(":")[0] for line in lines[best + 2 : best + 7]] == [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "num_warps",
        "num_stages",
    ]
    labels = ("PyTorch eager:", "torch.compile:", "Triton baseline:", "KernelForge:")
    assert all(any(line.startswith(label) for line in lines) for label in labels)
    assert any(line.startswith("Speedup vs Triton baseline:") for line in lines)
    assert any(line.startswith("Throughput:") for line in lines)
    with TuningDatabase(db) as database:
        assert len(database.cache_entries()) == 1
        implementations = database.execute("SELECT DISTINCT implementation FROM results ORDER BY 1")
        assert [name for (name,) in implementations] == [
            "kernelforge",
            "torch",
            "torch.compile",
            "triton-baseline",
        ]

    assert "Cache hit" in "\n".join(_captured_lines(lambda: cli._run_tune_matmul(args, device)))


def _captured_lines(run: Callable[[], int]) -> list[str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert run() == 0
    return buffer.getvalue().splitlines()


@pytest.mark.gpu
def test_tune_matmul_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["tune", "matmul", "--m", "512", "--n", "512", "--k", "512", "--dtype", "fp16"]
    args += ["--warmup", "2", "--iterations", "10", "--db", str(tmp_path / "tuning.db")]
    assert main(args) == 0
    first = capsys.readouterr().out
    for expected in (
        "configurations passed correctness",
        "Best configuration",
        "PyTorch eager:",
        "torch.compile:",
        "Triton baseline:",
        "KernelForge:",
        "TFLOP/s",
    ):
        assert expected in first
    assert "rejected:" not in first
    assert main(args) == 0
    assert "Cache hit" in capsys.readouterr().out
    assert main(["cache", "list", "--db", str(tmp_path / "tuning.db")]) == 0
    assert "1 cached configuration(s)" in capsys.readouterr().out
