"""Compile the kernels for real NVIDIA targets on the CPU; no GPU or driver is needed.

The interpreter checks what a kernel computes. These tests check what Triton's compiler makes of
it for Ampere (sm_80) and Hopper (sm_90):
- every kernel compiles;
- 16-bit MatMul runs on tensor cores and IEEE fp32 does not;
- the K loop is software-pipelined only when operands are 16-element aligned;
- the compiler never allocates more shared memory than the search space's bound assumes.
Compilation runs in subprocesses (tests/compile_check.py) with TRITON_INTERPRET unset.
"""

import dataclasses
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch

from kernelforge.benchmark.workloads import MATMUL_BASELINE, matmul_problem
from kernelforge.kernels.softmax import num_warps_for
from kernelforge.tuning.config import KernelConfig
from kernelforge.tuning.search import MATMUL_SEARCH_SPACE, DeviceLimits, matmul_shared_memory_bound

ARCHS = (80, 90)
ALIGNED = {"M": 2048, "N": 4096, "K": 4096}
TENSOR_CORE_CONFIG = KernelConfig("matmul", {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, 4, 3)
WORKERS = 4


def _compile(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    env = {name: value for name, value in os.environ.items() if name != "TRITON_INTERPRET"}
    script = Path(__file__).with_name("compile_check.py")

    def run(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not chunk:
            return []
        process = subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(chunk),
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        results: list[dict[str, Any]] = json.loads(process.stdout)
        return results

    with ThreadPoolExecutor(WORKERS) as pool:
        chunks = list(pool.map(run, [requests[i::WORKERS] for i in range(WORKERS)]))
    ordered: list[dict[str, Any]] = [{}] * len(requests)
    for worker, results in enumerate(chunks):
        for position, result in enumerate(results):
            ordered[worker + position * WORKERS] = result
    return ordered


def _matmul(
    dtype: str, config: KernelConfig, arch: int, shape: dict[str, int] = ALIGNED
) -> dict[str, Any]:
    return {
        "kernel": "matmul",
        "dtype": dtype,
        "shape": shape,
        "params": dict(config.params),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "arch": arch,
    }


def test_every_kernel_compiles_for_ampere_and_hopper() -> None:
    requests = []
    for arch in ARCHS:
        for dtype in ("fp32", "fp16", "bf16"):
            requests += [
                {
                    "kernel": "vector_add",
                    "dtype": dtype,
                    "shape": {"n": 1_000_003},
                    "num_warps": 4,
                    "num_stages": 3,
                    "arch": arch,
                },
                {
                    "kernel": "softmax",
                    "dtype": dtype,
                    "shape": {"cols": 4096},
                    "num_warps": num_warps_for(4096),
                    "num_stages": 3,
                    "arch": arch,
                },
                _matmul(dtype, MATMUL_BASELINE, arch),
            ]
    failures = [(r, q) for r, q in zip(_compile(requests), requests, strict=True) if "error" in r]
    assert failures == []


def test_16_bit_matmul_uses_tensor_cores_and_ieee_fp32_does_not() -> None:
    requests = [_matmul(d, TENSOR_CORE_CONFIG, a) for a in ARCHS for d in ("fp16", "bf16", "fp32")]
    results = {
        (q["arch"], q["dtype"]): r for q, r in zip(requests, _compile(requests), strict=True)
    }
    for dtype in ("fp16", "bf16"):
        assert results[(80, dtype)]["mma_sync"]
        assert results[(90, dtype)]["wgmma"]
    for arch in ARCHS:
        assert not results[(arch, "fp32")]["mma_sync"]
        assert not results[(arch, "fp32")]["wgmma"]


def test_k_loop_is_pipelined_only_for_16_element_aligned_operands() -> None:
    stages = (2, 3, 4)
    aligned = [
        _matmul("fp16", dataclasses.replace(TENSOR_CORE_CONFIG, num_stages=s), arch)
        for arch in ARCHS
        for s in stages
    ]
    # Multiples of 8 but not 16: Triton cannot prove 16-byte alignment, so no cp.async.
    misaligned = [
        _matmul("fp16", TENSOR_CORE_CONFIG, arch, shape={"M": 2048, "N": 4104, "K": 4104})
        for arch in ARCHS
    ]
    results = _compile(aligned + misaligned)
    tile_bytes = (64 + 128) * 32 * 2
    for index in range(len(ARCHS)):
        per_stage = results[index * len(stages) : (index + 1) * len(stages)]
        assert all(r["async_copy"] for r in per_stage)
        shared = [r["shared"] for r in per_stage]
        assert [b - a for a, b in itertools.pairwise(shared)] == [tile_bytes, tile_bytes]
    assert not any(r["async_copy"] for r in results[len(aligned) :])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("arch", "device"),
    [(80, DeviceLimits(108, 166_912)), (90, DeviceLimits(132, 232_448))],
    ids=["sm80-A100", "sm90-H100"],
)
def test_compiler_stays_within_the_shared_memory_bound(
    arch: int, device: DeviceLimits, dtype: torch.dtype
) -> None:
    survivors = MATMUL_SEARCH_SPACE.candidates(matmul_problem(2048, 4096, 4096, dtype), device)
    name = "fp16" if dtype == torch.float16 else "fp32"
    results = _compile([_matmul(name, config, arch) for config in survivors.configs])
    for config, result in zip(survivors.configs, results, strict=True):
        assert "error" not in result, (str(config), result)
        bound = matmul_shared_memory_bound(config, dtype)
        assert result["shared"] <= bound <= device.max_shared_memory_bytes, str(config)
