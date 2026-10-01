"""Helpers shared by the GPU benchmark scripts in this directory."""

from __future__ import annotations

import csv
import dataclasses
import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from kernelforge.benchmark.metrics import gbps
from kernelforge.benchmark.runner import benchmark
from kernelforge.runtime.environment import collect_environment
from kernelforge.testing import verify

Row = dict[str, Any]


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU; benchmarks produce no numbers without one")


def context_columns() -> Row:
    """Columns identical for every row of one run: timestamp and hardware/software context."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return {"timestamp": timestamp, **dataclasses.asdict(collect_environment())}


def measure(
    fn: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    dtype: torch.dtype,
    *,
    num_bytes: int,
    warmup: int,
    iterations: int,
) -> Row:
    """Verify fn's output against the reference, then time it. Incorrect outputs are not timed."""
    verification = verify(reference, fn(), dtype)
    row: Row = {"correct": verification.passed, "max_error_ratio": verification.max_error_ratio}
    if not verification.passed:
        return row
    result = benchmark(fn, warmup=warmup, iterations=iterations)
    row.update(dataclasses.asdict(result.stats))
    row["gbps"] = gbps(num_bytes, result.stats.median_us)
    row.update(
        warmup=result.warmup, iterations=result.iterations, l2_flush_bytes=result.l2_flush_bytes
    )
    return row


def write_csv(rows: list[Row], path: Path) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
