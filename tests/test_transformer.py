"""Transformer-block integration tests.

The parity test is the important one: both backends run on one module with one
set of weights, so a difference in output can only come from the swapped
kernels. Config validation runs anywhere; the forward pass needs a GPU.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from kernelforge.integration.transformer import (  # noqa: E402
    BACKEND_KERNELFORGE,
    BACKEND_TORCH,
    BlockConfig,
    TransformerBlock,
)

SMALL = BlockConfig(hidden=256, heads=8, intermediate=512)


def test_block_config_rejects_an_indivisible_head_count():
    with pytest.raises(ValueError, match="not divisible"):
        BlockConfig(hidden=100, heads=8, intermediate=256)


def test_head_dim_follows_from_the_geometry():
    assert BlockConfig(hidden=4096, heads=32, intermediate=11008).head_dim == 128


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be one of"):
        TransformerBlock(SMALL, device="cpu", dtype=torch.float32, backend="cutlass")


@pytest.mark.gpu
def test_backends_agree_on_one_set_of_weights(device):
    """Swapping the kernels must not change the block's output.

    ``backend`` is an attribute of a single module, so the two forward passes
    share weights exactly and any difference is attributable to the kernels.
    With weights this small the block's output is almost all residual stream,
    so the swapped operators' own outputs are compared as well: a kernel that
    returned zeros would otherwise pass.
    """
    from kernelforge.testing import assert_verified

    torch.manual_seed(0)
    block = TransformerBlock(SMALL, device=device, dtype=torch.float16)
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    x = torch.randn(2, 128, SMALL.hidden, device=device, dtype=torch.float16)
    flat = x.reshape(-1, SMALL.hidden)

    with torch.inference_mode():
        block.backend = BACKEND_TORCH
        expected = block(x)
        expected_norm = block._norm(flat, block.attn_norm_weight)
        expected_mlp = block._mlp_activation(flat)
        block.backend = BACKEND_KERNELFORGE
        actual = block(x)
        actual_norm = block._norm(flat, block.attn_norm_weight)
        actual_mlp = block._mlp_activation(flat)

    assert_verified(expected, actual, dtype=torch.float16, context="transformer block parity")
    assert_verified(expected_norm, actual_norm, dtype=torch.float16, context="RMSNorm parity")
    assert_verified(expected_mlp, actual_mlp, dtype=torch.float16, context="fused MLP parity")


@pytest.mark.gpu
@pytest.mark.parametrize("seq", [1, 7, 128])
def test_parity_holds_for_awkward_sequence_lengths(seq, device):
    """Decode (seq=1) and ragged prefill reach different kernel code paths."""
    from kernelforge.testing import assert_verified

    torch.manual_seed(0)
    block = TransformerBlock(SMALL, device=device, dtype=torch.float16)
    x = torch.randn(1, seq, SMALL.hidden, device=device, dtype=torch.float16)
    with torch.inference_mode():
        block.backend = BACKEND_TORCH
        expected = block(x)
        block.backend = BACKEND_KERNELFORGE
        actual = block(x)
    assert_verified(expected, actual, dtype=torch.float16, context=f"seq={seq}")


@pytest.mark.gpu
def test_input_shape_is_validated(device):
    block = TransformerBlock(SMALL, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match=r"\(batch, seq, hidden\)"):
        block(torch.zeros(4, SMALL.hidden, device=device, dtype=torch.float16))
    with pytest.raises(ValueError, match="expected hidden"):
        block(torch.zeros(1, 4, SMALL.hidden * 2, device=device, dtype=torch.float16))


@pytest.mark.gpu
def test_configurations_are_resolved_once_per_shape(device):
    """Config lookup touches the filesystem, so it must not run per forward."""
    block = TransformerBlock(SMALL, device=device, dtype=torch.float16, backend=BACKEND_KERNELFORGE)
    x = torch.randn(1, 64, SMALL.hidden, device=device, dtype=torch.float16)
    with torch.inference_mode():
        block(x)
        resolved = dict(block._selected)
        block(x)
    assert block._selected == resolved
    assert {operation for operation, _, _ in block._selected} == {"rmsnorm", "fused_linear"}


@pytest.mark.gpu
def test_benchmark_block_times_both_backends(device):
    from kernelforge.integration.transformer import benchmark_block

    results = benchmark_block(SMALL, batch=1, seq=128, device=device, warmup=3, iterations=5)
    assert set(results) == {BACKEND_TORCH, BACKEND_KERNELFORGE}
    for timing in results.values():
        assert timing.median_ms > 0
        assert timing.timer == "cuda_event"
