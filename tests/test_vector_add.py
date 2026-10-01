"""Vector-add correctness.

Tests marked ``gpu`` need a CUDA device. The rest also run on CPU through
Triton's interpreter (``TRITON_INTERPRET=1``), which is how CI executes them.
"""

from __future__ import annotations

import pytest
import torch

from kernelforge.benchmark import workloads
from kernelforge.testing import assert_verified
from kernelforge.tuning.config import Problem

pytest.importorskip("triton")

DTYPES = (torch.float16, torch.bfloat16, torch.float32)

#: Vector add does its arithmetic in the input dtype, and Triton's interpreter
#: does bf16 arithmetic on the raw storage bits, so bf16 needs a GPU.
VECTOR_ADD_DTYPES = [
    pytest.param(dtype, marks=pytest.mark.gpu) if dtype == torch.bfloat16 else dtype
    for dtype in DTYPES
]

CORRECTNESS_ELEMENT_COUNTS = [
    p["n"] for p in workloads.problems("vector_add", "correctness", "fp16")
]


@pytest.mark.parametrize("n", CORRECTNESS_ELEMENT_COUNTS)
@pytest.mark.parametrize("dtype", VECTOR_ADD_DTYPES)
def test_vector_add_across_sizes(n, dtype, device):
    from kernelforge.kernels.vector_add import DEFAULT_CONFIG, vector_add

    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(n, device=device, dtype=dtype, generator=gen)
    b = torch.randn(n, device=device, dtype=dtype, generator=gen)
    assert_verified(
        a.double() + b.double(),
        vector_add(a, b, config=DEFAULT_CONFIG),
        dtype=dtype,
        context=f"n={n}",
    )


def test_vector_add_candidates_all_agree(device):
    from kernelforge.kernels.vector_add import vector_add
    from kernelforge.runtime.env import device_caps
    from kernelforge.tuning.search import VectorAddSearchSpace

    n = 1_000_003
    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(n, device=device, dtype=torch.float16, generator=gen)
    b = torch.randn(n, device=device, dtype=torch.float16, generator=gen)
    expected = a.double() + b.double()
    problem = Problem.create("vector_add", "fp16", n=n)
    for candidate in VectorAddSearchSpace().candidates(problem, device_caps(device)):
        assert_verified(expected, vector_add(a, b, config=candidate), dtype=torch.float16)


def test_vector_add_validates_inputs(device):
    from kernelforge.kernels.vector_add import vector_add

    a = torch.zeros(8, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="shape mismatch"):
        vector_add(a, torch.zeros(9, device=device, dtype=torch.float16))
    with pytest.raises(ValueError, match="dtype mismatch"):
        vector_add(a, torch.zeros(8, device=device, dtype=torch.float32))
