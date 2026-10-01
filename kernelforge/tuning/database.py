"""SQLite store for tuning runs, every measurement, and the tuned-config cache.

Every candidate the tuner tries is recorded, including the ones rejected for compile errors or
wrong results; their timing columns stay NULL because they were never benchmarked. Config
parameters are stored as JSON so that kernels with different parameters (MatMul's BLOCK_M/N/K,
RMSNorm's BLOCK_SIZE/ROWS_PER_PROGRAM) share one schema; `json_extract(params_json, '$.BLOCK_M')`
queries them. `cache clear` deletes only the cache table, so the measurement history survives.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from kernelforge.runtime.environment import Environment
from kernelforge.tuning.cache import CacheKey
from kernelforge.tuning.config import KernelConfig, Problem

if TYPE_CHECKING:
    from kernelforge.tuning.tuner import Measurement

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    gpu TEXT,
    compute_capability TEXT,
    driver TEXT,
    cuda TEXT,
    torch TEXT NOT NULL,
    triton TEXT NOT NULL,
    environment_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS problems (
    problem_id INTEGER PRIMARY KEY,
    operation TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    dtype TEXT NOT NULL,
    UNIQUE (operation, shape_json, dtype)
);
CREATE TABLE IF NOT EXISTS configs (
    config_id INTEGER PRIMARY KEY,
    kernel TEXT NOT NULL,
    params_json TEXT NOT NULL,
    num_warps INTEGER NOT NULL,
    num_stages INTEGER NOT NULL,
    UNIQUE (kernel, params_json, num_warps, num_stages)
);
CREATE TABLE IF NOT EXISTS results (
    result_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs (run_id),
    problem_id INTEGER NOT NULL REFERENCES problems (problem_id),
    config_id INTEGER REFERENCES configs (config_id),
    implementation TEXT NOT NULL,
    correct INTEGER NOT NULL,
    error TEXT,
    max_error_ratio REAL,
    median_us REAL,
    mean_us REAL,
    p95_us REAL,
    p99_us REAL,
    min_us REAL,
    max_us REAL,
    std_us REAL,
    warmup INTEGER,
    iterations INTEGER,
    l2_flush_bytes INTEGER,
    tflops REAL,
    gbps REAL
);
CREATE TABLE IF NOT EXISTS cache (
    gpu TEXT NOT NULL,
    compute_capability TEXT NOT NULL,
    triton TEXT NOT NULL,
    operation TEXT NOT NULL,
    dtype TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    config_id INTEGER NOT NULL REFERENCES configs (config_id),
    run_id INTEGER NOT NULL REFERENCES runs (run_id),
    median_us REAL NOT NULL,
    PRIMARY KEY (gpu, compute_capability, triton, operation, dtype, shape_json)
);
"""


def default_path() -> Path:
    """$XDG_CACHE_HOME/kernelforge/tuning.db, defaulting to ~/.cache/kernelforge/tuning.db."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "kernelforge" / "tuning.db"


@dataclass(frozen=True)
class CacheEntry:
    key: CacheKey
    config: KernelConfig
    median_us: float
    run_id: int


class TuningDatabase:
    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(SCHEMA)

    def __enter__(self) -> TuningDatabase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
        """Run a read query; for reports and tests."""
        return self._connection.execute(sql, parameters).fetchall()

    def start_run(self, environment: Environment) -> int:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with self._connection:
            cursor = self._connection.execute(
                "INSERT INTO runs (timestamp, gpu, compute_capability, driver, cuda, torch, triton,"
                " environment_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp,
                    environment.gpu,
                    environment.compute_capability,
                    environment.driver,
                    environment.cuda,
                    environment.torch,
                    environment.triton,
                    json.dumps(dataclasses.asdict(environment)),
                ),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def record(self, run_id: int, problem: Problem, measurement: Measurement) -> None:
        benchmark = measurement.benchmark
        stats = benchmark.stats if benchmark is not None else None
        with self._connection:
            problem_id = self._problem_id(problem)
            config = measurement.config
            config_id = None if config is None else self._config_id(config)
            self._connection.execute(
                "INSERT INTO results (run_id, problem_id, config_id, implementation, correct,"
                " error, max_error_ratio, median_us, mean_us, p95_us, p99_us, min_us, max_us,"
                " std_us, warmup, iterations, l2_flush_bytes, tflops, gbps)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    problem_id,
                    config_id,
                    measurement.implementation,
                    measurement.correct,
                    measurement.error,
                    measurement.max_error_ratio,
                    *(dataclasses.astuple(stats) if stats is not None else (None,) * 7),
                    benchmark.warmup if benchmark is not None else None,
                    benchmark.iterations if benchmark is not None else None,
                    benchmark.l2_flush_bytes if benchmark is not None else None,
                    measurement.tflops,
                    measurement.gbps,
                ),
            )

    def cached(self, key: CacheKey) -> CacheEntry | None:
        row = self._connection.execute(
            "SELECT kernel, params_json, num_warps, num_stages, median_us, run_id"
            " FROM cache JOIN configs USING (config_id) WHERE gpu = ? AND compute_capability = ?"
            " AND triton = ? AND operation = ? AND dtype = ? AND shape_json = ?",
            dataclasses.astuple(key),
        ).fetchone()
        if row is None:
            return None
        kernel, params_json, num_warps, num_stages, median_us, run_id = row
        config = KernelConfig(kernel, json.loads(params_json), num_warps, num_stages)
        return CacheEntry(key, config, median_us, run_id)

    def store_cache(
        self, key: CacheKey, config: KernelConfig, median_us: float, run_id: int
    ) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO cache (gpu, compute_capability, triton, operation, dtype,"
                " shape_json, config_id, run_id, median_us) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*dataclasses.astuple(key), self._config_id(config), run_id, median_us),
            )

    def cache_entries(self) -> list[CacheEntry]:
        rows = self._connection.execute(
            "SELECT gpu, compute_capability, triton, operation, dtype, shape_json, kernel,"
            " params_json, num_warps, num_stages, median_us, run_id FROM cache"
            " JOIN configs USING (config_id) ORDER BY gpu, operation, dtype, shape_json"
        ).fetchall()
        return [
            CacheEntry(
                CacheKey(*row[:6]),
                KernelConfig(row[6], json.loads(row[7]), row[8], row[9]),
                median_us=row[10],
                run_id=row[11],
            )
            for row in rows
        ]

    def clear_cache(self) -> int:
        """Delete every cached config; measurements are kept. Returns the number removed."""
        with self._connection:
            return self._connection.execute("DELETE FROM cache").rowcount

    def _problem_id(self, problem: Problem) -> int:
        identity = (problem.operation, problem.shape_json(), problem.dtype_name)
        self._connection.execute(
            "INSERT OR IGNORE INTO problems (operation, shape_json, dtype) VALUES (?, ?, ?)",
            identity,
        )
        row = self._connection.execute(
            "SELECT problem_id FROM problems WHERE operation = ? AND shape_json = ? AND dtype = ?",
            identity,
        ).fetchone()
        return int(row[0])

    def _config_id(self, config: KernelConfig) -> int:
        identity = (config.kernel, config.params_json(), config.num_warps, config.num_stages)
        self._connection.execute(
            "INSERT OR IGNORE INTO configs (kernel, params_json, num_warps, num_stages)"
            " VALUES (?, ?, ?, ?)",
            identity,
        )
        row = self._connection.execute(
            "SELECT config_id FROM configs WHERE kernel = ? AND params_json = ?"
            " AND num_warps = ? AND num_stages = ?",
            identity,
        ).fetchone()
        return int(row[0])
