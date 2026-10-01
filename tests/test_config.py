from __future__ import annotations

import pytest
import torch

from kernelforge.dtypes import dtype_name, itemsize, parse_dtype
from kernelforge.tuning.config import KernelConfig, Problem


def test_config_is_hashable_and_order_independent():
    a = KernelConfig("matmul", BLOCK_M=64, BLOCK_N=128)
    b = KernelConfig("matmul", BLOCK_N=128, BLOCK_M=64)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_config_separates_launch_from_meta_parameters():
    """num_warps and num_stages go to the launcher, not to the kernel body."""
    config = KernelConfig("matmul", BLOCK_M=64, BLOCK_K=32, num_warps=8, num_stages=4)
    assert config.meta == {"BLOCK_K": 32, "BLOCK_M": 64}
    assert config.launch == {"num_stages": 4, "num_warps": 8}


def test_config_round_trips_through_json():
    config = KernelConfig("rmsnorm", BLOCK_SIZE=4096, ROWS_PER_PROGRAM=2, num_warps=8)
    assert KernelConfig.from_json("rmsnorm", config.to_json()) == config


def test_config_digest_is_stable_and_discriminating():
    a = KernelConfig("matmul", BLOCK_M=64, BLOCK_N=128)
    assert a.digest == KernelConfig("matmul", BLOCK_M=64, BLOCK_N=128).digest
    assert a.digest != KernelConfig("matmul", BLOCK_M=64, BLOCK_N=64).digest
    # The operation is part of the digest: identical parameters for different
    # operators must not collide in the database.
    assert a.digest != KernelConfig("fused_linear", BLOCK_M=64, BLOCK_N=128).digest


def test_config_rejects_non_integer_parameters():
    with pytest.raises(TypeError):
        KernelConfig("matmul", BLOCK_M=64.0)
    with pytest.raises(TypeError):
        KernelConfig("matmul", BLOCK_M=True)


def test_config_renders_in_requested_order():
    config = KernelConfig("matmul", BLOCK_M=64, BLOCK_N=128, num_warps=8)
    rendered = config.render(("BLOCK_M", "BLOCK_N", "num_warps"))
    assert [line.split(":")[0] for line in rendered.splitlines()] == [
        "BLOCK_M",
        "BLOCK_N",
        "num_warps",
    ]


def test_problem_shape_key_follows_declaration_order():
    problem = Problem.create("matmul", "fp16", M=2048, N=4096, K=1024)
    assert problem.shape_key == "2048x4096x1024"
    assert problem.cache_key == "matmul/fp16/2048x4096x1024"
    assert problem.dims_dict == {"M": 2048, "N": 4096, "K": 1024}
    assert problem["K"] == 1024


def test_problem_rejects_degenerate_dimensions():
    with pytest.raises(ValueError, match="positive"):
        Problem.create("matmul", "fp16", M=0, N=4, K=4)
    with pytest.raises(ValueError, match="positive"):
        Problem.create("matmul", "fp16", M=-1, N=4, K=4)


def test_problem_is_hashable():
    a = Problem.create("matmul", "fp16", M=1, N=2, K=3)
    b = Problem.create("matmul", torch.float16, M=1, N=2, K=3)
    assert a == b
    assert len({a, b}) == 1


@pytest.mark.parametrize(
    "name,dtype,size",
    [("fp16", torch.float16, 2), ("bf16", torch.bfloat16, 2), ("fp32", torch.float32, 4)],
)
def test_dtype_names_round_trip(name, dtype, size):
    assert parse_dtype(name) is dtype
    assert dtype_name(dtype) == name
    assert itemsize(dtype) == size


def test_unsupported_dtype_is_rejected_by_name():
    with pytest.raises(ValueError, match="unsupported dtype"):
        parse_dtype("fp8")
    with pytest.raises(ValueError, match="unsupported dtype"):
        parse_dtype(torch.int8)
