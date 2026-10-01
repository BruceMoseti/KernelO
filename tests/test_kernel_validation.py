"""Input validation in the kernel wrappers, and device-key consistency.

These run without a GPU: every guard rejects its input before any launch, so
the checks are reachable with CPU tensors. That is deliberate — a wrapper that
only validates after touching the device cannot be tested here, and would
report a confusing CUDA error instead of a clear one.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from kernelforge.kernels.fused_linear import fused_linear_gelu  # noqa: E402
from kernelforge.kernels.matmul import matmul  # noqa: E402
from kernelforge.kernels.rmsnorm import rmsnorm  # noqa: E402
from kernelforge.kernels.softmax import softmax  # noqa: E402
from kernelforge.kernels.vector_add import vector_add  # noqa: E402
from kernelforge.tuning.config import KernelConfig  # noqa: E402


def test_row_kernels_reject_a_non_contiguous_last_dimension():
    """The latent bug these guards close.

    Both row kernels index columns directly off the row pointer, passing only
    ``stride(0)``. A transposed or last-dim-strided input would read the wrong
    elements and return a confidently wrong answer. Rejecting beats copying
    silently: a hidden ``.contiguous()`` inside a benchmarked function would
    show up as latency attributed to the kernel.
    """
    strided = torch.zeros(8, 16)[:, ::2]
    gamma = torch.ones(8)
    assert strided.stride(1) != 1

    with pytest.raises(ValueError, match="contiguous last dimension"):
        rmsnorm(strided, gamma)
    with pytest.raises(ValueError, match="contiguous last dimension"):
        softmax(strided)

    transposed = torch.zeros(16, 8).t()
    assert transposed.shape == (8, 16) and transposed.stride(1) != 1
    with pytest.raises(ValueError, match="contiguous last dimension"):
        rmsnorm(transposed, torch.ones(16))


def test_rmsnorm_rejects_a_strided_gamma():
    with pytest.raises(ValueError, match="gamma must be contiguous"):
        rmsnorm(torch.zeros(4, 8), torch.ones(16)[::2])


def test_row_kernels_reject_non_2d_input():
    for fn in (lambda t: softmax(t), lambda t: rmsnorm(t, torch.ones(4))):
        with pytest.raises(ValueError, match="2D"):
            fn(torch.zeros(2, 3, 4))


def test_row_kernels_reject_a_block_too_small_for_the_row():
    config = KernelConfig("rmsnorm", BLOCK_SIZE=16, ROWS_PER_PROGRAM=1, num_warps=1)
    with pytest.raises(ValueError, match="cannot hold a row"):
        rmsnorm(torch.zeros(2, 64), torch.ones(64), config=config)


def test_row_kernels_reject_rows_beyond_the_single_pass_limit():
    from kernelforge.tuning.search import RMSNormSearchSpace

    cols = RMSNormSearchSpace.MAX_BLOCK_SIZE * 2
    with pytest.raises(ValueError, match="single-pass limit"):
        rmsnorm(torch.zeros(1, cols), torch.ones(cols))
    with pytest.raises(ValueError, match="single-pass limit"):
        softmax(torch.zeros(1, cols))


def test_gemm_rejects_mismatched_shapes_and_dtypes():
    a = torch.zeros(4, 8)
    with pytest.raises(ValueError, match="shape mismatch"):
        matmul(a, torch.zeros(9, 4))
    with pytest.raises(ValueError, match="dtype mismatch"):
        matmul(a, torch.zeros(8, 4, dtype=torch.float16))
    with pytest.raises(ValueError, match="2D"):
        matmul(a, torch.zeros(8))


def test_fused_linear_rejects_a_mismatched_bias():
    with pytest.raises(ValueError, match="bias must have shape"):
        fused_linear_gelu(torch.zeros(4, 8), torch.zeros(8, 16), torch.zeros(8))


def test_fused_linear_rejects_a_strided_bias():
    """The kernel reads ``bias_ptr + offs_n``, so a strided bias is read wrongly."""
    with pytest.raises(ValueError, match="bias must be contiguous"):
        fused_linear_gelu(torch.zeros(4, 8), torch.zeros(8, 16), torch.ones(32)[::2])


def test_vector_add_rejects_mismatched_inputs():
    a = torch.zeros(8)
    with pytest.raises(ValueError, match="shape mismatch"):
        vector_add(a, torch.zeros(9))
    with pytest.raises(ValueError, match="dtype mismatch"):
        vector_add(a, torch.zeros(8, dtype=torch.float16))


def test_recorded_device_key_matches_the_one_dispatch_looks_up(tmp_path):
    """Both must come from the same derivation.

    The tuner writes cache entries under the environment's ``device_key`` and
    dispatch reads them back under its own lookup. Deriving the key twice let
    them disagree on a CPU-only host, where the capability fallback's key named
    an A100 that was not present, so a written entry could never be read.
    """
    from kernelforge.kernels import get_operator
    from kernelforge.runtime.dispatch import select_config
    from kernelforge.runtime.env import capture_environment, device_key
    from kernelforge.tuning.cache import ConfigCache
    from kernelforge.tuning.config import Problem

    problem = Problem.create("matmul", "fp16", M=64, N=64, K=64)
    recorded = capture_environment().device_key
    assert recorded == device_key()

    cache = ConfigCache(tmp_path / "configs.json")
    assert select_config(get_operator("matmul"), problem, cache=cache).device_key == recorded


def test_device_key_is_cpu_without_a_device():
    from kernelforge.runtime.env import device_key

    if torch.cuda.is_available():
        assert device_key() != "cpu"
    else:
        # Must not inherit the documented A100 fallback's name.
        assert device_key() == "cpu"
