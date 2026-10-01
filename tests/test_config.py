import pytest
import torch

from kernelforge.tuning.config import KernelConfig, Problem


def test_config_params_are_read_only_and_order_independent() -> None:
    config = KernelConfig("matmul", {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, 8, 4)
    reordered = KernelConfig("matmul", {"BLOCK_K": 32, "BLOCK_N": 128, "BLOCK_M": 64}, 8, 4)
    with pytest.raises(TypeError):
        config.params["BLOCK_M"] = 16  # type: ignore[index]
    assert config == reordered
    assert config.params_json() == reordered.params_json()
    assert (
        str(config)
        == str(reordered)
        == "BLOCK_K=32 BLOCK_M=64 BLOCK_N=128 num_warps=8 num_stages=4"
    )


def test_equal_configs_hash_equal() -> None:
    config = KernelConfig("matmul", {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, 8, 4)
    reordered = KernelConfig("matmul", {"BLOCK_K": 32, "BLOCK_N": 128, "BLOCK_M": 64}, 8, 4)
    assert {config: "best"}[reordered] == "best"
    assert len({config, reordered, KernelConfig("matmul", dict(config.params), 4, 4)}) == 2


def test_config_is_not_matmul_specific() -> None:
    config = KernelConfig("rmsnorm", {"BLOCK_SIZE": 1024, "ROWS_PER_PROGRAM": 2}, 4, 1)
    assert config.params_json() == '{"BLOCK_SIZE": 1024, "ROWS_PER_PROGRAM": 2}'


def test_problem_identity() -> None:
    problem = Problem("matmul", {"M": 2048, "N": 4096, "K": 4096}, torch.float16)
    assert problem.dtype_name == "fp16"
    assert problem.shape_json() == '{"K": 4096, "M": 2048, "N": 4096}'
    assert str(problem) == "matmul 2048 x 4096 x 4096 fp16"
