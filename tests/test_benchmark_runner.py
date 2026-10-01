"""Timing-harness tests.

The statistics are separated from the measurement loop so they can be checked
against known inputs. The GPU-marked tests cover the parts that need a device:
that CUDA events are actually used, and that the harness agrees with
PyTorch's own benchmark utility -- which is the real check that the timing
loop is correct rather than merely self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kernelforge.benchmark.runner import (
    TIMER_RESOLUTION_MS,
    benchmark,
    cross_check_median_ms,
    summarize,
)


def test_summarize_computes_the_distribution():
    samples = [1.0, 2.0, 3.0, 4.0, 100.0]
    result = summarize(samples, warmup=5, iterations=5, timer="cuda_event", flushed_l2=True)
    assert result.median_ms == 3.0
    assert result.min_ms == 1.0
    assert result.max_ms == 100.0
    assert result.mean_ms == pytest.approx(22.0)
    assert result.p95_ms == pytest.approx(float(np.percentile(samples, 95)))
    assert result.p99_ms == pytest.approx(float(np.percentile(samples, 99)))
    assert result.std_ms == pytest.approx(float(np.std(samples, ddof=1)))


def test_median_is_robust_to_an_outlier_the_mean_is_not():
    """Why the tuner ranks on the median.

    One preempted launch among two hundred moves the mean by an order of
    magnitude and leaves the median untouched.
    """
    clean = [2.0] * 199 + [2.0]
    spiked = [2.0] * 199 + [400.0]
    a = summarize(clean, warmup=0, iterations=200, timer="cuda_event", flushed_l2=True)
    b = summarize(spiked, warmup=0, iterations=200, timer="cuda_event", flushed_l2=True)
    assert a.median_ms == b.median_ms == 2.0
    assert b.mean_ms > a.mean_ms * 1.9


def test_summarize_rejects_no_samples():
    with pytest.raises(ValueError, match="empty"):
        summarize([], warmup=0, iterations=0, timer="cuda_event", flushed_l2=False)


def test_single_sample_has_zero_standard_deviation():
    result = summarize([7.0], warmup=0, iterations=1, timer="perf_counter", flushed_l2=False)
    assert result.std_ms == 0.0
    assert result.median_ms == result.mean_ms == 7.0


def test_samples_are_dropped_unless_requested():
    samples = [1.0, 2.0, 3.0]
    kwargs = {"warmup": 0, "iterations": 3, "timer": "perf_counter", "flushed_l2": False}
    assert summarize(samples, **kwargs).samples_ms == ()
    assert summarize(samples, keep_samples=True, **kwargs).samples_ms == (1.0, 2.0, 3.0)


def test_timer_resolution_flag_only_applies_to_cuda_events():
    fast = summarize(
        [TIMER_RESOLUTION_MS / 2], warmup=0, iterations=1, timer="cuda_event", flushed_l2=True
    )
    assert fast.at_timer_resolution
    slow = summarize([1.0], warmup=0, iterations=1, timer="cuda_event", flushed_l2=True)
    assert not slow.at_timer_resolution
    cpu = summarize(
        [TIMER_RESOLUTION_MS / 2], warmup=0, iterations=1, timer="perf_counter", flushed_l2=False
    )
    assert not cpu.at_timer_resolution


def test_benchmark_runs_the_requested_number_of_iterations():
    calls = []
    result = benchmark(lambda: calls.append(1), warmup=3, iterations=7, device="cpu")
    assert len(calls) == 10
    assert result.iterations == 7
    assert result.warmup == 3


def test_cpu_path_is_labelled_so_it_cannot_be_mistaken_for_a_gpu_timing():
    result = benchmark(lambda: None, warmup=1, iterations=3, device="cpu")
    assert result.timer == "perf_counter"
    assert result.flushed_l2 is False


@pytest.mark.parametrize("warmup,iterations", [(-1, 10), (0, 0), (0, -5)])
def test_benchmark_validates_its_arguments(warmup, iterations):
    with pytest.raises(ValueError):
        benchmark(lambda: None, warmup=warmup, iterations=iterations, device="cpu")


def test_timing_result_serialises_without_samples():
    result = benchmark(lambda: None, warmup=0, iterations=3, device="cpu")
    payload = result.as_dict()
    assert payload["timer"] == "perf_counter"
    assert "samples_ms" not in payload
    assert payload["median_ms"] == result.median_ms


@pytest.mark.gpu
def test_gpu_path_uses_cuda_events(device):
    x = torch.randn(4096, 4096, device=device)
    result = benchmark(lambda: x * 2, warmup=5, iterations=20, device=device)
    assert result.timer == "cuda_event"
    assert result.flushed_l2 is True
    assert result.median_ms > 0


@pytest.mark.gpu
def test_l2_flush_changes_the_measurement_of_a_cache_resident_problem(device):
    """A working set that fits in L2 measures faster when the cache is warm.

    This is the methodological point the flush exists for: without it, a
    benchmark of a small GEMM reports a bandwidth the kernel would never see
    in a real model, where the inputs are not already resident.
    """
    a = torch.randn(512, 512, device=device, dtype=torch.float16)
    b = torch.randn(512, 512, device=device, dtype=torch.float16)
    warm = benchmark(lambda: a @ b, warmup=25, iterations=100, device=device, flush_l2=False)
    cold = benchmark(lambda: a @ b, warmup=25, iterations=100, device=device, flush_l2=True)
    assert cold.median_ms >= warm.median_ms


@pytest.mark.gpu
def test_harness_agrees_with_pytorch_benchmark(device):
    """Cross-check the timing loop against ``torch.utils.benchmark``.

    Two independent implementations of GPU timing should land within 15% of
    each other on a kernel long enough to be well above event resolution. If
    they do not, the harness is measuring something other than the kernel.
    """
    a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.float16)

    def run():
        return a @ b

    ours = benchmark(run, warmup=25, iterations=200, device=device, flush_l2=False)
    theirs = cross_check_median_ms(run, min_run_time=0.5)
    assert ours.median_ms == pytest.approx(theirs, rel=0.15)
