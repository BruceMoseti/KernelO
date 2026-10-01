"""GEMM correctness.

Every test here is marked ``gpu`` and skips without a device.

The shape lists are the point of this file. Powers of two divide evenly into
every block size in the search space, so a kernel with a broken boundary mask
passes 1024x1024x1024 and fails 1023x1023x1023. The awkward sizes -- primes,
one-off-a-tile, degenerate single rows and columns -- are where masking bugs
live, and they are run against the extreme tile shapes rather than only the
default configuration, because a mask that is right for a 32x32 tile can still
be wrong for 128x128.
"""

from __future__ import annotations

import pytest
import torch

from kernelforge.benchmark import workloads
from kernelforge.testing import assert_verified, exact_fp32_matmul
from kernelforge.tuning.config import KernelConfig, Problem

pytest.importorskip("triton")

pytestmark = pytest.mark.gpu

DTYPES = (torch.float16, torch.bfloat16, torch.float32)

#: The awkward shapes from the public workload suite: primes, one-off-a-tile
#: sizes, and degenerate single rows and columns.
CORRECTNESS_SHAPES = [
    tuple(p.dims_dict.values()) for p in workloads.problems("matmul", "correctness", "fp16")
]

#: Tile shapes chosen to stress masking from both ends: the smallest tile the
#: search space allows, the largest, and asymmetric ones.
TILE_CONFIGS = (
    (32, 32, 32, 4, 3),
    (64, 128, 32, 8, 3),
    (128, 64, 64, 8, 3),
    (128, 128, 32, 8, 4),
    (32, 128, 64, 4, 2),
)


def config(block_m, block_n, block_k, warps, stages, *, group_m=8) -> KernelConfig:
    return KernelConfig(
        "matmul",
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=warps,
        num_stages=stages,
    )


def operands(m, k, n, dtype, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(m, k, device=device, dtype=dtype, generator=gen)
    b = torch.randn(k, n, device=device, dtype=dtype, generator=gen)
    return a, b


def reference(a, b):
    with exact_fp32_matmul():
        return torch.matmul(a, b)


@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_default_config_across_awkward_shapes(shape, dtype, device):
    from kernelforge.kernels.matmul import DEFAULT_CONFIG, matmul

    m, n, k = shape
    a, b = operands(m, k, n, dtype, device)
    assert_verified(
        reference(a, b),
        matmul(a, b, config=DEFAULT_CONFIG),
        dtype=dtype,
        context=f"matmul {m}x{n}x{k} {dtype}",
    )


@pytest.mark.parametrize("tile", TILE_CONFIGS)
@pytest.mark.parametrize("shape", [(127, 129, 65), (257, 255, 511), (1, 1, 4096), (4096, 1, 1)])
def test_boundary_masking_across_tile_shapes(tile, shape, device):
    """Shapes indivisible by any block size, against five different tilings."""
    from kernelforge.kernels.matmul import matmul

    m, n, k = shape
    a, b = operands(m, k, n, torch.float16, device)
    assert_verified(
        reference(a, b),
        matmul(a, b, config=config(*tile)),
        dtype=torch.float16,
        context=f"matmul {m}x{n}x{k} tile={tile}",
    )


@pytest.mark.parametrize("seed", range(24))
def test_randomised_shapes(seed, device):
    """Random shapes, random tilings. Catches what a fixed list does not."""
    import random

    from kernelforge.kernels.matmul import matmul
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import MatmulSearchSpace

    rng = random.Random(seed)
    m, n, k = (rng.randint(1, 600) for _ in range(3))
    dtype = rng.choice(DTYPES)
    problem = Problem.create("matmul", dtype, M=m, N=n, K=k)
    candidates = MatmulSearchSpace().candidates(problem, device_caps(device))
    chosen = rng.choice(candidates)

    a, b = operands(m, k, n, dtype, device, seed=seed)
    assert_verified(
        reference(a, b),
        matmul(a, b, config=chosen),
        dtype=dtype,
        context=f"matmul {m}x{n}x{k} {dtype} {chosen!r}",
    )


def test_every_budgeted_candidate_agrees_with_the_reference(device):
    """All 48 configurations the tuner would try produce the same answer.

    If this fails, the tuner's correctness gate is the only thing standing
    between the ranking and a wrong kernel -- which is exactly why the gate
    exists, but the kernel should not need it.
    """
    from kernelforge.kernels.matmul import matmul
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import MatmulSearchSpace

    problem = Problem.create("matmul", "fp16", M=257, N=513, K=129)
    a, b = operands(257, 129, 513, torch.float16, device)
    expected = reference(a, b)
    for candidate in MatmulSearchSpace().candidates(problem, device_caps(device)):
        assert_verified(
            expected,
            matmul(a, b, config=candidate),
            dtype=torch.float16,
            context=f"candidate {candidate!r}",
        )


def test_non_contiguous_operands(device):
    """Transposed views must work without a copy: the kernel takes strides."""
    from kernelforge.kernels.matmul import DEFAULT_CONFIG, matmul

    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(96, 128, device=device, dtype=torch.float16, generator=gen).t()
    b = torch.randn(192, 96, device=device, dtype=torch.float16, generator=gen).t()
    assert not a.is_contiguous() and not b.is_contiguous()
    assert_verified(reference(a, b), matmul(a, b, config=DEFAULT_CONFIG), dtype=torch.float16)


def test_fp32_is_not_silently_computed_in_tf32(device):
    """The fp32 path must hold to the fp32 tolerance.

    ``tl.dot`` would use TF32 by default on Ampere and later, whose 10-bit
    mantissa carries ~1e-3 relative error. The kernel pins ``ieee``, so this
    passes at the fp32 threshold of 1e-5; it would fail by two orders of
    magnitude if the pin were removed.
    """
    from kernelforge.kernels.matmul import DEFAULT_CONFIG, matmul

    a, b = operands(512, 1024, 256, torch.float32, device)
    result = assert_verified(
        reference(a, b), matmul(a, b, config=DEFAULT_CONFIG), dtype=torch.float32
    )
    assert result.error < 1e-5


def test_shape_and_dtype_mismatches_are_rejected(device):
    from kernelforge.kernels.matmul import matmul

    a = torch.randn(4, 8, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="shape mismatch"):
        matmul(a, torch.randn(9, 4, device=device, dtype=torch.float16))
    with pytest.raises(ValueError, match="dtype mismatch"):
        matmul(a, torch.randn(8, 4, device=device, dtype=torch.float32))
    with pytest.raises(ValueError, match="2D"):
        matmul(a, torch.randn(8, device=device, dtype=torch.float16))


def test_triton_autotune_baseline_agrees(device):
    """The comparison baseline has to be correct to be a baseline."""
    from kernelforge.kernels.matmul import matmul_triton_autotune

    a, b = operands(512, 512, 512, torch.float16, device)
    assert_verified(reference(a, b), matmul_triton_autotune(a, b), dtype=torch.float16)
