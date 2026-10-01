"""GPU microbenchmark timing.

CUDA launches are asynchronous, so the naive

    t0 = time.perf_counter(); kernel(); t1 = time.perf_counter()

measures the time to *enqueue* work, not to execute it. This module times with
CUDA events recorded on the active stream and synchronises once, after the
whole measurement loop, so that the per-iteration cost of synchronising is not
folded into the samples.

Two methodological details that materially move GEMM numbers:

* **L2 flush.** Re-running the same kernel on the same tensors leaves the
  inputs resident in L2, which inflates throughput for problems whose working
  set fits. A buffer of max(256 MiB, 2 x L2) is zeroed before each timed
  iteration: L2 replacement is not strict LRU, so one L2's worth of writes
  does not reliably evict the inputs, and 256 MiB is what Triton's
  ``do_bench`` uses. The zeroing is enqueued *before* ``start_event``, and the
  stream is in-order, so it is not part of the measured interval.
* **Distribution, not mean.** A single mean hides bimodal behaviour from clock
  throttling and from the occasional preempted launch, so the full set of
  samples is summarised as median/p95/p99.

The flushes also keep the GPU behind the host, so each start event is followed
by its kernel rather than by the host's launch overhead. If the GPU has already
reached the last start event when the host finishes enqueuing, it waited on the
host at some point, and a ``RuntimeWarning`` says so.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from kernelforge.runtime.env import device_caps

DEFAULT_WARMUP = 25
DEFAULT_ITERATIONS = 200

# torch.cuda.Event.elapsed_time() resolves to roughly half a microsecond.
# Medians at or below this are reported but should not be compared between
# implementations; the fix is a larger problem, not more iterations.
TIMER_RESOLUTION_MS = 0.002

#: Smallest L2 flush buffer, matching ``triton.testing.do_bench``.
MIN_L2_FLUSH_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class TimingResult:
    """Summary of one timing run. All times are milliseconds."""

    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    warmup: int
    iterations: int
    timer: str
    flushed_l2: bool
    samples_ms: tuple[float, ...] = field(default=(), repr=False)

    @property
    def at_timer_resolution(self) -> bool:
        return self.timer == "cuda_event" and self.median_ms <= TIMER_RESOLUTION_MS

    @property
    def median_us(self) -> float:
        return self.median_ms * 1e3

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "timer": self.timer,
            "flushed_l2": self.flushed_l2,
        }


def summarize(
    samples_ms: list[float] | np.ndarray,
    *,
    warmup: int,
    iterations: int,
    timer: str,
    flushed_l2: bool,
    keep_samples: bool = False,
) -> TimingResult:
    """Summarise raw per-iteration timings.

    Percentiles use NumPy's default linear interpolation between order
    statistics. Separated from the measurement loop so the statistics can be
    tested against known inputs without a GPU.
    """
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty set of timing samples")
    p95, p99 = (float(x) for x in np.percentile(arr, [95, 99]))
    return TimingResult(
        median_ms=float(np.median(arr)),
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        p95_ms=p95,
        p99_ms=p99,
        warmup=warmup,
        iterations=iterations,
        timer=timer,
        flushed_l2=flushed_l2,
        samples_ms=tuple(float(x) for x in arr) if keep_samples else (),
    )


def _l2_flush_buffer(device: torch.device) -> torch.Tensor:
    size = max(MIN_L2_FLUSH_BYTES, 2 * device_caps(device).l2_cache_bytes)
    return torch.empty(size, dtype=torch.int8, device=device)


def benchmark(
    fn,
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    device: torch.device | str | None = None,
    flush_l2: bool = True,
    keep_samples: bool = False,
) -> TimingResult:
    """Time ``fn`` and return the latency distribution.

    ``fn`` takes no arguments; bind inputs with a lambda or ``functools.partial``
    so that argument marshalling is outside the measured region.

    Falls back to ``time.perf_counter`` when no CUDA device is in play. That
    path exists to keep the harness testable and to time CPU reference
    implementations, and it is labelled as such in ``TimingResult.timer`` so a
    CPU timing can never be mistaken for a GPU one.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")

    resolved = torch.device(device) if device is not None else None
    use_cuda = torch.cuda.is_available() and (resolved is None or resolved.type == "cuda")

    for _ in range(warmup):
        fn()

    if not use_cuda:
        samples = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
        return summarize(
            samples,
            warmup=warmup,
            iterations=iterations,
            timer="perf_counter",
            flushed_l2=False,
            keep_samples=keep_samples,
        )

    cuda_device = resolved if resolved is not None else torch.device("cuda")
    torch.cuda.synchronize(cuda_device)

    cache = _l2_flush_buffer(cuda_device) if flush_l2 else None
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        if cache is not None:
            cache.zero_()
        start[i].record()
        fn()
        end[i].record()

    gpu_caught_up = start[-1].query()
    torch.cuda.synchronize(cuda_device)
    if gpu_caught_up:
        warnings.warn(
            "the GPU drained its queue during timing, so some samples may include host "
            "launch latency rather than only GPU execution time",
            RuntimeWarning,
            stacklevel=2,
        )
    samples = [start[i].elapsed_time(end[i]) for i in range(iterations)]
    return summarize(
        samples,
        warmup=warmup,
        iterations=iterations,
        timer="cuda_event",
        flushed_l2=flush_l2,
        keep_samples=keep_samples,
    )


def cross_check_median_ms(fn, *, warmup: int = DEFAULT_WARMUP, min_run_time: float = 1.0) -> float:
    """Independent median latency from ``torch.utils.benchmark``.

    Used to validate this module's timing loop rather than to produce reported
    numbers: ``tests/test_benchmark_runner.py`` asserts the two agree on a
    GPU. PyTorch's timer handles accelerator synchronisation and adaptive
    replication itself, which makes it a good second opinion.
    """
    import torch.utils.benchmark as torch_benchmark

    for _ in range(warmup):
        fn()
    timer = torch_benchmark.Timer(stmt="fn()", globals={"fn": fn})
    return float(timer.blocked_autorange(min_run_time=min_run_time).median) * 1e3
