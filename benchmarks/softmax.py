"""Benchmark softmax: torch.softmax vs torch.compile vs the Triton kernel, on the current GPU.

Usage: python benchmarks/softmax.py [--output results/softmax.csv]

Shapes are the spec's grid (rows 128..8192 x columns 128..4096). Effective bandwidth counts one
read and one write of the tensor. torch.compile runs in its default mode, with dynamic=False so
it specializes on each shape. Its caches are reset per shape: otherwise Dynamo's recompile limit
would silently fall back to eager after a few shapes.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from common import Row, context_columns, measure, require_cuda, write_csv

from kernelforge.kernels.softmax import softmax

ROWS = [128, 512, 2048, 8192]
COLS = [128, 256, 512, 1024, 2048, 4096]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("results/softmax.csv"))
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    require_cuda()

    context = context_columns()
    rows: list[Row] = []
    for dtype in DTYPES:
        for n_rows in ROWS:
            for n_cols in COLS:
                generator = torch.Generator(device="cuda").manual_seed(n_rows * 10_000 + n_cols)
                x = torch.randn(n_rows, n_cols, device="cuda", generator=generator).to(dtype)
                reference = torch.softmax(x.double(), dim=-1)
                torch.compiler.reset()
                compiled = torch.compile(_torch_softmax, dynamic=False)
                implementations = {
                    "torch": partial(_torch_softmax, x),
                    "torch.compile": partial(compiled, x),
                    "triton": partial(softmax, x),
                }
                for name, fn in implementations.items():
                    row = measure(
                        fn,
                        reference,
                        dtype,
                        num_bytes=2 * x.numel() * x.element_size(),
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    labels = {
                        "implementation": name,
                        "dtype": str(dtype),
                        "shape": f"{n_rows}x{n_cols}",
                    }
                    rows.append({**labels, **row, **context})
                    print(rows[-1])
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
