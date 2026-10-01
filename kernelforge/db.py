"""SQLite store for tuning and benchmark results.

Every row is attributable: a result points at the problem, the configuration
and the *run*, and a run carries the GPU, driver, CUDA, PyTorch and Triton
versions it was produced on. That is what makes a number from three weeks ago
comparable, or knowably incomparable, to one from today.

One deliberate departure from a fixed-column schema: configurations are stored
as a JSON parameter set plus a content digest rather than as
``block_m/block_n/block_k/warps/stages`` columns. Those columns cannot
represent an RMSNorm config (``BLOCK_SIZE``, ``ROWS_PER_PROGRAM``), and adding
a nullable column per operator parameter turns the table into a sparse matrix
that every query has to special-case. The digest gives configs a stable
identity for joins, and SQLite's ``json_extract`` remains available should a
query ever need a predicate on a single parameter.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.runtime.env import Environment
from kernelforge.testing import VerificationResult
from kernelforge.tuning.config import KernelConfig, Problem

SCHEMA_VERSION = 2
DEFAULT_DB_PATH = Path("results/kernelforge.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    hostname            TEXT,
    platform            TEXT,
    python_version      TEXT,
    torch_version       TEXT,
    triton_version      TEXT,
    cuda_version        TEXT,
    driver_version      TEXT,
    gpu_name            TEXT,
    gpu_arch            TEXT,
    gpu_memory_bytes    INTEGER,
    sm_count            INTEGER,
    device_key          TEXT    NOT NULL,
    torch_matmul_fp16_reduced_precision INTEGER,
    torch_matmul_bf16_reduced_precision INTEGER,
    kernelforge_version TEXT,
    notes               TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS problems (
    problem_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation  TEXT NOT NULL,
    shape_key  TEXT NOT NULL,
    dtype      TEXT NOT NULL,
    dims_json  TEXT NOT NULL,
    UNIQUE (operation, shape_key, dtype)
);

CREATE TABLE IF NOT EXISTS configs (
    config_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    operation   TEXT NOT NULL,
    digest      TEXT NOT NULL,
    params_json TEXT NOT NULL,
    UNIQUE (operation, digest)
);

CREATE TABLE IF NOT EXISTS results (
    result_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES runs (run_id),
    problem_id INTEGER NOT NULL REFERENCES problems (problem_id),
    config_id  INTEGER          REFERENCES configs (config_id),
    label      TEXT    NOT NULL,
    status     TEXT    NOT NULL,
    correct    INTEGER,
    rel_error  REAL,
    median_us  REAL,
    mean_us    REAL,
    std_us     REAL,
    min_us     REAL,
    max_us     REAL,
    p95_us     REAL,
    p99_us     REAL,
    warmup     INTEGER,
    iterations INTEGER,
    timer      TEXT,
    flushed_l2 INTEGER,
    tflops     REAL,
    gbps       REAL,
    error      TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_results_problem ON results (problem_id, label);
CREATE INDEX IF NOT EXISTS idx_results_run     ON results (run_id);
"""

_ROWS_QUERY = """
SELECT
    r.result_id, r.label, r.status, r.correct, r.rel_error,
    r.median_us, r.mean_us, r.std_us, r.min_us, r.max_us, r.p95_us, r.p99_us,
    r.warmup, r.iterations, r.timer, r.flushed_l2, r.tflops, r.gbps, r.error,
    p.operation, p.shape_key, p.dtype, p.dims_json,
    c.params_json, c.digest,
    run.run_id, run.timestamp, run.gpu_name, run.gpu_arch, run.device_key,
    run.torch_version, run.triton_version, run.cuda_version, run.driver_version
FROM results AS r
JOIN problems AS p   ON p.problem_id = r.problem_id
JOIN runs     AS run ON run.run_id   = r.run_id
LEFT JOIN configs AS c ON c.config_id = r.config_id
"""


@dataclass(frozen=True)
class Measurement:
    """One row to record: what ran, whether it was right, and how fast."""

    label: str
    status: str
    config: KernelConfig | None = None
    verification: VerificationResult | None = None
    timing: Any = None  # TimingResult; typed loosely to keep this module import-light
    tflops: float | None = None
    gbps: float | None = None
    error: str = ""


def default_db_path() -> Path:
    return Path(os.environ.get("KERNELFORGE_DB", DEFAULT_DB_PATH))


class ResultsDB:
    """Thin, explicit wrapper over the results database."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._check_schema_version()
        self._conn.commit()

    def _check_schema_version(self) -> None:
        current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if current == 0:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        elif current != SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} was written by schema version {current}, "
                f"this build expects {SCHEMA_VERSION}"
            )

    def __enter__(self) -> ResultsDB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # --- writes ---------------------------------------------------------
    def start_run(self, env: Environment, *, notes: str = "", version: str = "") -> int:
        cur = self._conn.execute(
            """
            INSERT INTO runs (
                timestamp, hostname, platform, python_version, torch_version,
                triton_version, cuda_version, driver_version, gpu_name, gpu_arch,
                gpu_memory_bytes, sm_count, device_key,
                torch_matmul_fp16_reduced_precision, torch_matmul_bf16_reduced_precision,
                kernelforge_version, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                env.timestamp,
                env.hostname,
                env.platform,
                env.python_version,
                env.torch_version,
                env.triton_version,
                env.cuda_version,
                env.driver_version,
                env.gpu_name,
                env.gpu_arch,
                env.gpu_memory_bytes,
                env.sm_count,
                env.device_key,
                env.torch_matmul_fp16_reduced_precision,
                env.torch_matmul_bf16_reduced_precision,
                version,
                notes,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def problem_id(self, problem: Problem) -> int:
        self._conn.execute(
            "INSERT OR IGNORE INTO problems (operation, shape_key, dtype, dims_json) "
            "VALUES (?,?,?,?)",
            (
                problem.operation,
                problem.shape_key,
                problem.dtype_name,
                json.dumps(problem.dims_dict, sort_keys=True),
            ),
        )
        row = self._conn.execute(
            "SELECT problem_id FROM problems WHERE operation=? AND shape_key=? AND dtype=?",
            (problem.operation, problem.shape_key, problem.dtype_name),
        ).fetchone()
        return int(row["problem_id"])

    def config_id(self, config: KernelConfig) -> int:
        self._conn.execute(
            "INSERT OR IGNORE INTO configs (operation, digest, params_json) VALUES (?,?,?)",
            (config.operation, config.digest, config.to_json()),
        )
        row = self._conn.execute(
            "SELECT config_id FROM configs WHERE operation=? AND digest=?",
            (config.operation, config.digest),
        ).fetchone()
        return int(row["config_id"])

    def record(self, run_id: int, problem: Problem, measurement: Measurement) -> int:
        timing = measurement.timing
        verification = measurement.verification
        cur = self._conn.execute(
            """
            INSERT INTO results (
                run_id, problem_id, config_id, label, status, correct, rel_error,
                median_us, mean_us, std_us, min_us, max_us, p95_us, p99_us,
                warmup, iterations, timer, flushed_l2, tflops, gbps, error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                self.problem_id(problem),
                self.config_id(measurement.config) if measurement.config is not None else None,
                measurement.label,
                measurement.status,
                None if verification is None else int(verification.passed),
                None if verification is None else verification.error,
                None if timing is None else timing.median_ms * 1e3,
                None if timing is None else timing.mean_ms * 1e3,
                None if timing is None else timing.std_ms * 1e3,
                None if timing is None else timing.min_ms * 1e3,
                None if timing is None else timing.max_ms * 1e3,
                None if timing is None else timing.p95_ms * 1e3,
                None if timing is None else timing.p99_ms * 1e3,
                None if timing is None else timing.warmup,
                None if timing is None else timing.iterations,
                None if timing is None else timing.timer,
                None if timing is None else int(timing.flushed_l2),
                measurement.tflops,
                measurement.gbps,
                measurement.error,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_many(
        self, run_id: int, problem: Problem, measurements: Iterable[Measurement]
    ) -> None:
        for measurement in measurements:
            self.record(run_id, problem, measurement)

    # --- reads ----------------------------------------------------------
    def rows(self, *, operation: str | None = None, dtype: str | None = None) -> list[dict]:
        clauses, params = [], []
        if operation:
            clauses.append("p.operation = ?")
            params.append(operation)
        if dtype:
            clauses.append("p.dtype = ?")
            params.append(dtype)
        query = _ROWS_QUERY
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY r.result_id"
        return [dict(row) for row in self._conn.execute(query, params)]

    def best_per_label(self, *, operation: str | None = None) -> list[dict]:
        """Fastest correct measurement for each (problem, label) pair."""
        rows = self.rows(operation=operation)
        best: dict[tuple[str, str, str, str], dict] = {}
        for row in rows:
            if row["median_us"] is None or row["correct"] == 0:
                continue
            key = (row["operation"], row["shape_key"], row["dtype"], row["label"])
            if key not in best or row["median_us"] < best[key]["median_us"]:
                best[key] = row
        return sorted(
            best.values(), key=lambda r: (r["operation"], r["dtype"], r["shape_key"], r["label"])
        )

    def runs(self) -> list[dict]:
        """Every recorded run, oldest first.

        Distinct from :meth:`rows`, which returns *results* joined to their
        run: a run that recorded nothing still happened, and the provenance
        table in a report should reflect the latest run rather than the latest
        run that happened to produce a measurement.
        """
        return [dict(row) for row in self._conn.execute("SELECT * FROM runs ORDER BY run_id")]

    def counts(self) -> dict[str, int]:
        tables = ("runs", "problems", "configs", "results")
        return {
            t: int(self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tables
        }

    def operations(self) -> list[str]:
        return [
            row[0]
            for row in self._conn.execute("SELECT DISTINCT operation FROM problems ORDER BY 1")
        ]
