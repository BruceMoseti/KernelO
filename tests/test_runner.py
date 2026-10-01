import warnings
from collections.abc import Callable

import pytest
import torch

from kernelforge.benchmark.runner import benchmark


def test_rejects_settings_that_would_time_compilation_or_lack_a_distribution() -> None:
    with pytest.raises(ValueError, match="warmup"):
        benchmark(lambda: None, warmup=0)
    with pytest.raises(ValueError, match="iterations"):
        benchmark(lambda: None, iterations=1)


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU-only path")
def test_refuses_to_time_without_cuda() -> None:
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        benchmark(lambda: None)


def _large_fp16_matmul() -> Callable[[], torch.Tensor]:
    # Compute-bound and long enough that event resolution and host overhead are negligible.
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    return lambda: torch.matmul(a, b)


@pytest.mark.gpu
def test_result_is_a_distribution_with_metadata() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = benchmark(_large_fp16_matmul(), warmup=5, iterations=50)
    stats = result.stats
    assert len(result.samples_us) == 50
    assert 0 < stats.min_us <= stats.median_us <= stats.p95_us <= stats.p99_us <= stats.max_us
    assert result.environment.gpu == torch.cuda.get_device_name()
    l2_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    assert result.l2_flush_bytes >= 2 * l2_bytes


@pytest.mark.gpu
def test_agrees_with_triton_do_bench() -> None:
    # Same method (per-call CUDA events, L2 zeroed before each call): medians should agree to
    # within run-to-run noise; a larger gap means one of the two harnesses mis-measures.
    import triton.testing

    fn = _large_fp16_matmul()
    ours = benchmark(fn, warmup=25, iterations=100).stats.median_us
    triton_us = triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3
    assert ours == pytest.approx(triton_us, rel=0.10)


@pytest.mark.gpu
def test_agrees_with_torch_utils_benchmark() -> None:
    # torch.utils.benchmark times on the host with a warm L2. For a compute-bound kernel that
    # runs for milliseconds, GPU execution dominates both, so the medians should be close.
    from torch.utils import benchmark as torch_benchmark

    fn = _large_fp16_matmul()
    ours = benchmark(fn, warmup=25, iterations=100).stats.median_us
    timer = torch_benchmark.Timer(stmt="fn()", globals={"fn": fn})
    torch_us = timer.blocked_autorange(min_run_time=1.0).median * 1e6
    assert ours == pytest.approx(torch_us, rel=0.15)


@pytest.mark.gpu
def test_warns_when_the_gpu_waits_on_the_host() -> None:
    x = torch.zeros(1, device="cuda")

    def synchronizing_fn() -> None:
        x.add_(1)
        torch.cuda.synchronize()

    with pytest.warns(RuntimeWarning, match="host launch latency"):
        benchmark(synchronizing_fn, warmup=1, iterations=5)
