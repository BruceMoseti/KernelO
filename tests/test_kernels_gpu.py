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


#: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_SHARED_MEMORY_CASES = (
    (64, 64, 32, 4, 2),
    (64, 64, 32, 4, 4),
    (128, 128, 32, 8, 3),
)


def _gemm_config(block_m, block_n, block_k, warps, stages) -> KernelConfig:
    return KernelConfig(
        "matmul",
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=warps,
        num_stages=stages,
    )


def test_shared_memory_estimate_bounds_real_usage(device):
    """The filter's estimate must be an upper bound on what a launch allocates.

    This is the property the filter actually needs, and it is one-sided on
    purpose. The estimate is used to reject configurations *before* compiling
    them: if it over-estimates, the filter is conservative and costs a few
    candidates; if it under-estimates, the filter admits configurations the
    device cannot run and the budget is spent on `OutOfResources` failures.

    The 25% headroom is for small allocations the model does not count, such
    as the 16 bytes of barriers sm_100 adds.

    Note what this does *not* check: that the allocation scales with
    ``num_stages``. A single-buffer allocation satisfies an upper bound
    trivially. That factor is checked separately below.
    """
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import MatmulSearchSpace

    space = MatmulSearchSpace()
    problem = Problem.create("matmul", "fp16", M=512, N=512, K=512)
    for case in _SHARED_MEMORY_CASES:
        config = _gemm_config(*case)
        estimate = space.shared_memory_bytes(config, problem, device_caps(device))
        actual = int(_launch_metadata(config, device).shared)
        assert actual <= estimate * 1.25, (
            f"{config!r}: pipeliner allocated {actual} B, filter estimated {estimate} B"
        )


def test_pipeliner_multi_buffers_so_the_num_stages_factor_is_real(device):
    """At least one multi-stage configuration must allocate past one buffer.

    The ``num_stages`` factor in the filter's estimate exists because Triton's
    pipeliner keeps operand tiles in flight per stage. The ahead-of-time
    compile test confirms that for kernels specialized the way the JIT
    specializes a launch; this confirms it at a real launch.

    Phrased as "at least one" rather than "every" because Triton may
    legitimately decline to pipeline a particular loop. If *none* of these
    configurations multi-buffers, the factor is fiction on this Triton version
    and the filter over-estimates every GEMM candidate by up to 4x -- which
    would be worth knowing, and is what this test would report.
    """
    multi_buffered = []
    for case in _SHARED_MEMORY_CASES:
        block_m, block_n, block_k, _, stages = case
        if stages < 2:
            continue
        single_buffer = block_k * (block_m + block_n) * 2  # fp16
        actual = int(_launch_metadata(_gemm_config(*case), device).shared)
        if actual > single_buffer * 1.25:
            multi_buffered.append((case, actual, single_buffer))

    assert multi_buffered, (
        "no multi-stage configuration allocated more than a single operand "
        "buffer; the num_stages factor in the shared-memory model does not "
        "correspond to this Triton version's behaviour"
    )


def test_tuning_beats_or_matches_the_untuned_default(device):
    """The tuned winner must not be slower than the untuned default.

    A weak claim on purpose: which configuration wins depends on the GPU, so
    asserting a particular speedup would be asserting a property of the
    hardware rather than of this code.

    Note what is being compared. `DEFAULT_CONFIG` (32x32x32) is *not* in the
    searched budget -- the priority function ranks it below larger tiles for a
    1024-cubed problem -- so this is not "the search returns its own input or
    better". It is the independently measured `triton_baseline`, which runs
    `DEFAULT_CONFIG`, against the best of 12 searched candidates. The 5%
    tolerance absorbs measurement noise; this is the most likely test in the
    suite to need widening on noisy hardware.
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
