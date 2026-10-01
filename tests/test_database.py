import json
from pathlib import Path

import pytest
import torch

from kernelforge.benchmark.metrics import summarize
from kernelforge.benchmark.runner import BenchmarkResult
from kernelforge.benchmark.workloads import matmul_problem
from kernelforge.runtime.environment import collect_environment
from kernelforge.tuning.cache import CacheKey, cache_key
from kernelforge.tuning.config import KernelConfig
from kernelforge.tuning.database import TuningDatabase, default_path
from kernelforge.tuning.tuner import Measurement

CONFIG = KernelConfig("matmul", {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, 8, 4)
PROBLEM = matmul_problem(2048, 4096, 4096, torch.float16)
KEY = CacheKey("Synthetic test GPU", "0.0", "3.8.0", "matmul", "fp16", PROBLEM.shape_json())


def _timed(config: KernelConfig, latency_us: float) -> Measurement:
    environment = collect_environment()
    result = BenchmarkResult(
        summarize([latency_us, latency_us]), (latency_us,) * 2, 1, 2, 0, environment
    )
    return Measurement("kernelforge", config, True, max_error_ratio=0.1, benchmark=result)


def test_records_runs_problems_configs_and_results(tmp_path: Path) -> None:
    with TuningDatabase(tmp_path / "tuning.db") as database:
        run_id = database.start_run(collect_environment())
        database.record(run_id, PROBLEM, _timed(CONFIG, 42.0))
        database.record(run_id, PROBLEM, Measurement("kernelforge", CONFIG, False, error="wrong"))
        database.record(run_id, PROBLEM, Measurement("torch", None, False, error="wrong"))
        assert database.execute("SELECT count(*) FROM problems") == [(1,)]
        assert database.execute("SELECT count(*) FROM configs") == [(1,)]
        rows = database.execute(
            "SELECT implementation, correct, error, median_us, config_id IS NULL FROM results"
            " ORDER BY result_id"
        )
        assert rows == [
            ("kernelforge", 1, None, 42.0, 0),
            ("kernelforge", 0, "wrong", None, 0),
            ("torch", 0, "wrong", None, 1),
        ]
        (environment_json,) = database.execute("SELECT environment_json FROM runs")[0]
        assert json.loads(str(environment_json))["torch"] == torch.__version__


def test_config_parameters_are_queryable(tmp_path: Path) -> None:
    with TuningDatabase(tmp_path / "tuning.db") as database:
        run_id = database.start_run(collect_environment())
        database.record(run_id, PROBLEM, _timed(CONFIG, 42.0))
        assert database.execute("SELECT json_extract(params_json, '$.BLOCK_N') FROM configs") == [
            (128,)
        ]


def test_cache_store_lookup_replace_list_and_clear(tmp_path: Path) -> None:
    other = KernelConfig("matmul", {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, 4, 4)
    with TuningDatabase(tmp_path / "tuning.db") as database:
        run_id = database.start_run(collect_environment())
        database.record(run_id, PROBLEM, _timed(CONFIG, 42.0))
        assert database.cached(KEY) is None
        database.store_cache(KEY, CONFIG, 42.0, run_id)
        entry = database.cached(KEY)
        assert entry is not None and entry.config == CONFIG and entry.median_us == 42.0
        database.store_cache(KEY, other, 40.0, run_id)
        assert [e.config for e in database.cache_entries()] == [other]
        assert database.clear_cache() == 1
        assert database.cache_entries() == []
        assert database.execute("SELECT count(*) FROM results") == [(1,)]


def test_database_persists_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "tuning.db"
    with TuningDatabase(path) as database:
        run_id = database.start_run(collect_environment())
        database.record(run_id, PROBLEM, _timed(CONFIG, 42.0))
        database.store_cache(KEY, CONFIG, 42.0, run_id)
    with TuningDatabase(path) as database:
        assert database.cached(KEY) is not None


def test_cache_key_needs_a_gpu() -> None:
    environment = collect_environment()
    if environment.gpu is None:
        with pytest.raises(ValueError, match="no GPU"):
            cache_key(PROBLEM, environment)
    else:
        key = cache_key(PROBLEM, environment)
        assert (key.gpu, key.triton, key.dtype) == (environment.gpu, environment.triton, "fp16")


def test_default_path_follows_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_path() == tmp_path / "kernelforge" / "tuning.db"
