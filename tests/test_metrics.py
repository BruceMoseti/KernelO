import numpy as np
import pytest

from kernelforge.benchmark.metrics import gbps, matmul_flops, summarize, tflops


def test_summary_matches_numpy() -> None:
    samples = np.random.default_rng(0).lognormal(mean=3.0, sigma=0.3, size=200).tolist()
    stats = summarize(samples)
    assert stats.median_us == pytest.approx(np.median(samples))
    assert stats.mean_us == pytest.approx(np.mean(samples))
    assert stats.p95_us == pytest.approx(np.percentile(samples, 95))
    assert stats.p99_us == pytest.approx(np.percentile(samples, 99))
    assert stats.min_us == min(samples)
    assert stats.max_us == max(samples)
    assert stats.std_us == pytest.approx(np.std(samples, ddof=1))


def test_summary_keeps_outliers() -> None:
    stats = summarize([1.0] * 99 + [1000.0])
    assert stats.median_us == 1.0
    assert stats.max_us == 1000.0


def test_summary_needs_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        summarize([1.0])


def test_throughput_units() -> None:
    assert tflops(10**12, time_us=1e6) == pytest.approx(1.0)
    assert gbps(10**9, time_us=1e6) == pytest.approx(1.0)
    assert matmul_flops(3, 5, 7) == 2 * 3 * 5 * 7
