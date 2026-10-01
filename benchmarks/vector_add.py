"""Benchmark Triton vector add against PyTorch eager on the current GPU.

Usage: python benchmarks/vector_add.py [--output results/vector_add.csv]

Sizes are the spec's test sizes plus two large ones, where the kernel is bandwidth-bound rather
than launch-bound. Effective bandwidth counts reading x and y and writing out once each.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from common import Row, context_columns, measure, require_cuda, write_csv

from kernelforge.kernels.vector_add import vector_add

SIZES = [1, 13, 127, 1024, 12345, 1_000_003, 2**24, 2**26]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("results/vector_add.csv"))
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    require_cuda()

    context = context_columns()
    rows: list[Row] = []
    for dtype in DTYPES:
        for n in SIZES:
            generator = torch.Generator(device="cuda").manual_seed(n)
            x = torch.randn(n, device="cuda", generator=generator).to(dtype)
            y = torch.randn(n, device="cuda", generator=generator).to(dtype)
            reference = x.double() + y.double()
            implementations = {
                "torch": partial(torch.add, x, y),
                "triton": partial(vector_add, x, y),
            }
            for name, fn in implementations.items():
                row = measure(
                    fn,
                    reference,
                    dtype,
                    num_bytes=3 * n * x.element_size(),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                rows.append({"implementation": name, "dtype": str(dtype), "n": n, **row, **context})
                print(rows[-1])
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
