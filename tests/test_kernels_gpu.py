"""GPU-only checks that cannot be made on a CPU.

The ahead-of-time compile tests cover what the Triton compiler will accept.
These cover what only a real launch reveals: the shared-memory figure the
pipeliner actually allocates, and that the tuner picks a configuration no
worse than the untuned default.
"""

from __future__ import annotations

import pytest
import torch

from kernelforge.tuning.config import KernelConfig, Problem

pytest.importorskip("triton")

pytestmark = pytest.mark.gpu


def _kernel_caches(kernel) -> list[dict]:
    """The per-device compiled-kernel dictionaries of a ``@triton.jit`` kernel.

    This reaches into Triton's internals, which have moved between releases
    (``JITFunction.cache`` became ``device_caches`` holding a tuple). The
    shape is checked rather than assumed, so a future rename skips this test
    instead of failing it.
    """
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton version does not expose a compiled-kernel cache")
    out = []
    for entry in caches.values():
        candidate = entry[0] if isinstance(entry, tuple) else entry
        if isinstance(candidate, dict):
            out.append(candidate)
    if not out:
        pytest.skip("could not locate Triton's compiled-kernel cache")
    return out


def _launch_metadata(config: KernelConfig, device, dtype=torch.float16):
    """Launch the GEMM once and return the compiled kernel's metadata.

    The in-process cache is emptied first so that exactly one entry remains
    afterwards and it is unambiguously the kernel just launched. Clearing only
    forces a recompile.
    """
    from kernelforge.kernels.matmul import matmul, matmul_kernel

    a = torch.randn(512, 512, device=device, dtype=dtype)
    b = torch.randn(512, 512, device=device, dtype=dtype)
    matmul(a, b, config=config)  # ensure the device cache exists
    for cache in _kernel_caches(matmul_kernel):
        cache.clear()

    matmul(a, b, config=config)
    compiled = [kernel for cache in _kernel_caches(matmul_kernel) for kernel in cache.values()]
    assert len(compiled) == 1, f"expected one compiled kernel, found {len(compiled)}"
    return compiled[0].metadata


def test_shared_memory_model_matches_the_pipeliner(device):
    """The search space's shared-memory estimate against a real launch.

    The estimate is ``num_stages * BLOCK_K * (BLOCK_M + BLOCK_N) * itemsize``,
    which is what the pipeliner needs to keep ``num_stages`` operand tiles in
    flight. The ahead-of-time compile test cannot check the ``num_stages``
    factor because the standalone compiler does not run the pipeliner; this
    can.

    Allowed to come out lower than the estimate: Triton may decide a loop is
    not worth pipelining. Allowed a 25% overshoot too, because the operand
    layout is padded for swizzling -- the CPU compile test measures that
    padding directly and sees up to 2x on the single-buffer footprint. Beyond
    that margin the filter would be admitting configurations whose real
    shared-memory demand it has underestimated, which costs tuning time in
    OutOfResources failures.
    """
    from kernelforge.tuning.search import MatmulSearchSpace

    space = MatmulSearchSpace()
    problem = Problem.create("matmul", "fp16", M=512, N=512, K=512)
    for block_m, block_n, block_k, warps, stages in [
        (64, 64, 32, 4, 2),
        (64, 64, 32, 4, 4),
        (128, 128, 32, 8, 3),
    ]:
        config = KernelConfig(
            "matmul",
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=warps,
            num_stages=stages,
        )
        estimate = space.shared_memory_bytes(config, problem)
        actual = int(_launch_metadata(config, device).shared)
        assert actual <= estimate * 1.25, (
            f"{config!r}: pipeliner allocated {actual} B, filter estimated {estimate} B"
        )


def test_tuning_beats_or_matches_the_untuned_default(device):
    """Tuning must not make things worse.

    A weak claim on purpose: which configuration wins depends on the GPU, so
    asserting a particular speedup would be asserting a property of the
    hardware. What must hold on any device is that searching a space that
    contains the default never returns something slower than it, beyond
    measurement noise.
    """
    from kernelforge.kernels import get_operator
    from kernelforge.tuning.tuner import Tuner

    operator = get_operator("matmul")
    problem = Problem.create("matmul", "fp16", M=1024, N=1024, K=1024)
    result = Tuner(warmup=25, iterations=50, max_candidates=12).tune(operator, problem)

    assert result.best is not None, "nothing passed verification"
    assert result.correct > 0
    baseline = result.baselines.get("triton_baseline")
    if baseline is not None:
        assert result.best.median_ms <= baseline.median_ms * 1.05


def test_tuner_records_compile_failures_rather_than_raising(device):
    """A configuration the device rejects must be a row, not a crash.

    Asking for far more shared memory than any current GPU has is the
    reliable way to provoke Triton's OutOfResources, bypassing the filters by
    calling the tuner's verification stage directly.
    """
    from kernelforge.kernels import get_operator
    from kernelforge.tuning.tuner import STATUS_OK, Tuner

    operator = get_operator("matmul")
    problem = Problem.create("matmul", "fp16", M=256, N=256, K=256)
    impossible = KernelConfig(
        "matmul",
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=128,
        GROUP_M=8,
        num_warps=4,
        num_stages=8,
    )
    inputs = operator.make_inputs(problem, device)
    reference = operator.reference(*inputs)
    outcome = Tuner()._verify_candidate(operator, impossible, inputs, reference, problem)
    assert outcome.status != STATUS_OK
    assert outcome.error


def test_cross_process_cache_reuse(device, tmp_path):
    """Tuning writes a configuration that a later process reads back."""
    from kernelforge.kernels import get_operator
    from kernelforge.runtime.dispatch import SOURCE_CACHE, SOURCE_DEFAULT, select_config
    from kernelforge.tuning.cache import ConfigCache
    from kernelforge.tuning.tuner import Tuner

    operator = get_operator("rmsnorm")
    problem = Problem.create("rmsnorm", "fp16", rows=512, cols=1024)
    path = tmp_path / "configs.json"

    assert select_config(operator, problem, cache=ConfigCache(path)).source == SOURCE_DEFAULT

    cache = ConfigCache(path)
    result = Tuner(warmup=5, iterations=20, cache=cache, measure_baselines=False).tune(
        operator, problem
    )
    assert result.best is not None

    fresh = select_config(operator, problem, cache=ConfigCache(path))
    assert fresh.source == SOURCE_CACHE
    assert fresh.config == result.best_config
