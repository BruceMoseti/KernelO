"""Test configuration.

Without a CUDA GPU, Triton kernels run in Triton's CPU interpreter (TRITON_INTERPRET=1) on CPU
tensors. The interpreter checks kernel logic (indexing, masking, arithmetic), not GPU behavior
(warps, pipelining, shared memory, tensor cores, timing); tests marked `gpu` cover that and are
skipped when no GPU is available.
"""

import os

import pytest
import torch
import triton

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

GPU_AVAILABLE = torch.cuda.is_available() and not triton.knobs.runtime.interpret


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if GPU_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU (CPU-only run or TRITON_INTERPRET=1)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def device() -> torch.device:
    """Device that Triton kernels run on: CUDA, or CPU under the interpreter."""
    return torch.device("cuda" if GPU_AVAILABLE else "cpu")
