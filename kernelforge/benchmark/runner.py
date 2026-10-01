"""GPU kernel microbenchmarking with CUDA events.

Methodology (details and rationale in docs/BENCHMARKING.md):

- Each timed call is bracketed by two CUDA events recorded on the current stream. Elapsed times
  are read after one synchronize at the end, so neither host timers nor per-call synchronization
  enter the measurement.
- Before every call, warmup and timed alike, the L2 cache is flushed by zeroing a buffer of
  max(256 MiB, 2 x L2) outside the timed region, so every call starts with a cold L2.
- Warmup calls run the same loop body as timed calls; the first one also JIT-compiles.
- The flushes keep the GPU behind the host, so each start event is followed by its kernel rather
  than by host launch latency. This is checked on every run: if the last start event has already
  executed when the host finishes enqueuing the last call, the GPU may have waited on the host
  inside a timed region, and a warning is emitted.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import torch

from kernelforge.benchmark.metrics import LatencyStats, summarize
from kernelforge.runtime.environment import Environment, collect_environment

_MIN_FLUSH_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class BenchmarkResult:
    """Latency distribution of one callable plus the context needed to interpret it."""

    stats: LatencyStats
    samples_us: tuple[float, ...]
    warmup: int
    iterations: int
    l2_flush_bytes: int
    environment: Environment


def benchmark(
    fn: Callable[[], object], *, warmup: int = 25, iterations: int = 200
) -> BenchmarkResult:
    """Time `fn` on the current CUDA device. `fn` must launch its work on the current stream."""
    if warmup < 1:
        raise ValueError("warmup must be >= 1 so that JIT compilation is never timed")
    if iterations < 2:
        raise ValueError("iterations must be >= 2 to form a latency distribution")
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark() needs a CUDA GPU; GPU kernel time can't be measured on CPU")

    flush_bytes = l2_flush_bytes()
    flush = torch.empty(flush_bytes // 4, dtype=torch.int32, device="cuda")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    torch.cuda.synchronize()
    for _ in range(warmup):
        flush.zero_()
        fn()
    for start, end in zip(starts, ends, strict=True):
        flush.zero_()
        start.record()
        fn()
        end.record()
    gpu_caught_up = starts[-1].query()
    torch.cuda.synchronize()
    if gpu_caught_up:
        warnings.warn(
            "the GPU drained its queue during timing, so some samples may include host launch "
            "latency rather than only GPU execution time",
            RuntimeWarning,
            stacklevel=2,
        )

    samples_us = tuple(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends, strict=True))
    return BenchmarkResult(
        stats=summarize(samples_us),
        samples_us=samples_us,
        warmup=warmup,
        iterations=iterations,
        l2_flush_bytes=flush_bytes,
        environment=collect_environment(),
    )


def l2_flush_bytes() -> int:
    """Size of the buffer zeroed before each call: twice the L2 size, and at least 256 MiB."""
    l2_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    return max(_MIN_FLUSH_BYTES, 2 * l2_bytes)
