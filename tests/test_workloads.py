import inspect

import torch

from kernelforge.benchmark.workloads import MATMUL, MATMUL_BASELINE, matmul_problem
from kernelforge.kernels.matmul import matmul
from kernelforge.testing import verify


def test_baseline_config_is_the_kernels_default() -> None:
    defaults = inspect.signature(matmul).parameters
    assert MATMUL_BASELINE.params == {
        "BLOCK_M": defaults["block_m"].default,
        "BLOCK_N": defaults["block_n"].default,
        "BLOCK_K": defaults["block_k"].default,
    }
    assert MATMUL_BASELINE.num_warps == defaults["num_warps"].default
    assert MATMUL_BASELINE.num_stages == defaults["num_stages"].default


def test_matmul_workload_cost_model() -> None:
    problem = matmul_problem(2, 3, 5, torch.float16)
    assert MATMUL.flops(problem) == 2 * 2 * 3 * 5
    assert MATMUL.bytes_moved(problem) == (2 * 5 + 5 * 3 + 2 * 3) * 2


def test_matmul_workload_runs_a_config_correctly(device: torch.device) -> None:
    problem = matmul_problem(33, 47, 61, torch.float32)
    inputs = MATMUL.make_inputs(problem, device)
    assert [tuple(t.shape) for t in inputs] == [(33, 61), (61, 47)]
    result = verify(MATMUL.reference(inputs), MATMUL.run(MATMUL_BASELINE, inputs), torch.float32)
    assert result.passed, result.describe()
