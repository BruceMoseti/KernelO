from __future__ import annotations

import os

import pytest
import torch

#: ``TRITON_INTERPRET=1 pytest ...`` runs the Triton kernels on CPU tensors
#: through Triton's interpreter. It has to be set before the test process
#: starts rather than here: Triton reads it when kernels are decorated at
#: import time, and the ahead-of-time compile tests need the real compiler.
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def pytest_collection_modifyitems(config, items):
    """Skip GPU-marked tests when no CUDA device is present.

    The entire CPU-side framework is covered without a GPU; the kernel
    execution tests that need real hardware are marked ``gpu`` and skip with a
    visible reason rather than failing, so the same suite runs in ordinary CI
    and on a GPU runner. Under the interpreter the kernels execute on CPU, but
    GPU-only tests still skip, and so do the compile checks.
    """
    if INTERPRET:
        no_gpu = pytest.mark.skip(reason="TRITON_INTERPRET=1 runs kernels on CPU, not a GPU")
        no_compiler = pytest.mark.skip(reason="needs Triton's compiler; unset TRITON_INTERPRET")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(no_gpu)
            elif item.path.name == "test_triton_compile.py":
                item.add_marker(no_compiler)
        return
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device() -> torch.device:
    if INTERPRET:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available; TRITON_INTERPRET=1 runs the kernels on CPU")
    return torch.device("cuda")


@pytest.fixture
def a100_caps():
    """Properties of an A100-80GB, for testing filters without the hardware."""
    from kernelforge.runtime.env import DeviceCaps

    return DeviceCaps(
        name="NVIDIA A100-SXM4-80GB",
        compute_capability="8.0",
        total_memory_bytes=80 * 1024**3,
        sm_count=108,
        max_shared_memory_per_block=166912,
        shared_memory_per_sm=167936,
        registers_per_sm=65536,
        max_threads_per_sm=2048,
        warp_size=32,
        l2_cache_bytes=40 * 1024 * 1024,
    )


@pytest.fixture
def rtx4090_caps():
    """A consumer card: same sm89 family as a 4080 but different resources."""
    from kernelforge.runtime.env import DeviceCaps

    return DeviceCaps(
        name="NVIDIA GeForce RTX 4090",
        compute_capability="8.9",
        total_memory_bytes=24 * 1024**3,
        sm_count=128,
        max_shared_memory_per_block=101376,
        shared_memory_per_sm=102400,
        registers_per_sm=65536,
        max_threads_per_sm=1536,
        warp_size=32,
        l2_cache_bytes=72 * 1024 * 1024,
    )
