"""Latency statistics and throughput metrics."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyStats:
    """Distribution of per-call latencies, in microseconds."""

    median_us: float
    mean_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    std_us: float


def summarize(samples_us: Sequence[float]) -> LatencyStats:
    """Summarize latency samples.

    Percentiles interpolate linearly between order statistics (NumPy's default method);
    the standard deviation is the sample standard deviation. No samples are discarded.
    """
    if len(samples_us) < 2:
        raise ValueError(f"need at least 2 samples, got {len(samples_us)}")
    cuts = statistics.quantiles(samples_us, n=100, method="inclusive")
    return LatencyStats(
        median_us=statistics.median(samples_us),
        mean_us=statistics.fmean(samples_us),
        p95_us=cuts[94],
        p99_us=cuts[98],
        min_us=min(samples_us),
        max_us=max(samples_us),
        std_us=statistics.stdev(samples_us),
    )


def tflops(flops: int, time_us: float) -> float:
    """Throughput in TFLOP/s (10^12 FLOP/s)."""
    return flops / (time_us * 1e-6) / 1e12


def gbps(num_bytes: int, time_us: float) -> float:
    """Bandwidth in GB/s (10^9 bytes/s, the unit GPU memory bandwidth is quoted in)."""
    return num_bytes / (time_us * 1e-6) / 1e9


def matmul_flops(m: int, n: int, k: int) -> int:
    """FLOPs of C = A @ B with A: m x k and B: k x n, counting a multiply-add as 2."""
    return 2 * m * n * k
