"""Named shape suites.

Benchmarks are only comparable if everyone runs the same shapes, so the shapes
live here rather than in a shell script. Four suites per operator:

``smoke``
    Two or three shapes. For checking that a change did not break anything.
``sweep``
    The representative grid used for the headline results. Not the full
    Cartesian product of M, N and K -- 4^3 shapes at ~30 s of tuning each is
    an hour for a sweep whose extra points mostly repeat the same conclusion.
``transformer``
    The shapes a 7B-class decoder actually issues, at prefill and at decode
    batch sizes. These are the ones that justify the project: a GEMM tuned at
    2048x2048x2048 tells you little about the 1x4096x11008 skinny GEMM that
    dominates single-stream decoding.
``correctness``
    Deliberately awkward sizes for the test suite -- primes, odd numbers just
    over and just under tile boundaries, and degenerate single-row cases.
    Masking bugs hide behind clean powers of two.
"""

from __future__ import annotations

import torch

from kernelforge.tuning.config import Problem

# Llama-2-7B geometry: hidden 4096, SwiGLU intermediate 11008. The MLP here is
# GELU rather than SwiGLU (see kernels/fused_linear.py), but the matrix shapes
# are what the benchmark cares about.
HIDDEN = 4096
INTERMEDIATE = 11008

#: Tokens per forward pass: a single decode step, a small decode batch, and
#: two prefill lengths.
TOKEN_COUNTS = (1, 32, 512, 2048)

_MATMUL_SHAPES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "smoke": (
        (512, 512, 512),
        (1024, 1024, 1024),
    ),
    "sweep": (
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (512, 4096, 4096),
        (4096, 512, 4096),
        (4096, 4096, 512),
        (2048, 4096, 4096),
        (1024, 4096, 1024),
        (4096, 1024, 4096),
    ),
    "transformer": tuple(
        shape
        for tokens in TOKEN_COUNTS
        for shape in (
            (tokens, HIDDEN, HIDDEN),  # attention projections
            (tokens, INTERMEDIATE, HIDDEN),  # MLP up
            (tokens, HIDDEN, INTERMEDIATE),  # MLP down
        )
    ),
    "correctness": (
        (1, 1, 1),
        (1, 128, 128),
        (128, 1, 128),
        (128, 128, 1),
        (127, 127, 127),
        (128, 128, 128),
        (129, 129, 129),
        (257, 129, 65),
        (511, 769, 1025),
        (1023, 1023, 1023),
        (1024, 1024, 1024),
        (2057, 127, 513),
        (768, 2048, 4096),
    ),
}

_ROW_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "smoke": (
        (2048, 1024),
        (4096, 4096),
    ),
    "sweep": tuple(
        (rows, cols)
        for rows in (512, 2048, 8192)
        for cols in (128, 256, 512, 1024, 2048, 4096, 8192)
    ),
    "transformer": tuple((tokens, HIDDEN) for tokens in TOKEN_COUNTS)
    + tuple((tokens, 768) for tokens in TOKEN_COUNTS),
    "correctness": (
        (1, 1),
        (1, 4096),
        (3, 127),
        (17, 257),
        (128, 511),
        (333, 769),
        (1024, 1023),
        (2048, 1024),
        (4096, 4096),
        (129, 8192),
    ),
}

_VECTOR_SHAPES: dict[str, tuple[int, ...]] = {
    "smoke": (1_000_003, 16_777_216),
    "sweep": (1 << 16, 1 << 20, 1 << 24, 1 << 26),
    "transformer": (TOKEN_COUNTS[-1] * HIDDEN,),
    "correctness": (1, 2, 13, 127, 128, 129, 1024, 12_345, 1_000_003, 1 << 20),
}

SUITES: tuple[str, ...] = ("smoke", "sweep", "transformer", "correctness")


def problems(
    operation: str, suite: str = "sweep", dtype: str | torch.dtype = "fp16"
) -> list[Problem]:
    """Every problem in ``suite`` for ``operation``."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; available: {', '.join(SUITES)}")

    if operation in ("matmul", "fused_linear"):
        return [
            Problem.create(operation, dtype, M=m, N=n, K=k) for m, n, k in _MATMUL_SHAPES[suite]
        ]
    if operation in ("rmsnorm", "softmax"):
        return [
            Problem.create(operation, dtype, rows=rows, cols=cols)
            for rows, cols in _ROW_SHAPES[suite]
        ]
    if operation == "vector_add":
        return [Problem.create(operation, dtype, n=n) for n in _VECTOR_SHAPES[suite]]
    raise ValueError(f"no workload suite defined for {operation!r}")
