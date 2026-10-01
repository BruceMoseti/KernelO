"""Command-line interface.

kernelforge tune matmul --m M --n N --k K [--dtype fp16] [--retune] [--db PATH]
kernelforge cache list [--db PATH]
kernelforge cache clear [--db PATH]
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from kernelforge.benchmark.runner import benchmark
from kernelforge.benchmark.workloads import MATMUL, MATMUL_BASELINE, matmul_problem
from kernelforge.runtime.environment import Environment, collect_environment
from kernelforge.tuning.config import DTYPE_NAMES, KernelConfig, Problem
from kernelforge.tuning.database import TuningDatabase, default_path
from kernelforge.tuning.search import device_limits
from kernelforge.tuning.tuner import Measurement, Timer, TuneResult, measure, tune

DTYPES = {name: dtype for dtype, name in DTYPE_NAMES.items()}
LABELS = {
    "torch": "PyTorch eager",
    "torch.compile": "torch.compile",
    "triton-baseline": "Triton baseline",
    "kernelforge": "KernelForge",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


def _parser() -> argparse.ArgumentParser:
    database = argparse.ArgumentParser(add_help=False)
    database.add_argument(
        "--db", type=Path, default=default_path(), help="tuning database (default: %(default)s)"
    )

    parser = argparse.ArgumentParser(
        prog="kernelforge", description="Hardware-aware GPU kernel autotuning."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    tune_parser = commands.add_parser("tune", help="tune a kernel for one problem on this GPU")
    operations = tune_parser.add_subparsers(dest="operation", required=True)
    matmul = operations.add_parser(
        "matmul", parents=[database], help="C = A @ B, with A: M x K and B: K x N"
    )
    for dim in ("m", "n", "k"):
        matmul.add_argument(f"--{dim}", type=int, required=True)
    matmul.add_argument("--dtype", choices=DTYPES, default="fp16")
    matmul.add_argument("--warmup", type=int, default=25, help="untimed calls per measurement")
    matmul.add_argument("--iterations", type=int, default=200, help="timed calls per measurement")
    matmul.add_argument(
        "--retune", action="store_true", help="ignore a cached config and tune again"
    )
    matmul.set_defaults(handler=_tune_matmul)

    cache = commands.add_parser("cache", help="inspect or clear tuned configurations")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_commands.add_parser(
        "list", parents=[database], help="show cached configurations"
    ).set_defaults(handler=_cache_list)
    cache_commands.add_parser(
        "clear", parents=[database], help="delete cached configurations (keeps measurements)"
    ).set_defaults(handler=_cache_clear)
    return parser


def _tune_matmul(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print(
            "error: kernelforge tune needs a CUDA GPU (kernels are timed with CUDA events)",
            file=sys.stderr,
        )
        return 1
    progress = logging.StreamHandler(sys.stdout)
    progress.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("kernelforge")
    logger.addHandler(progress)
    logger.setLevel(logging.INFO)
    try:
        return _run_tune_matmul(args, torch.device("cuda"))
    finally:
        logger.removeHandler(progress)


def _run_tune_matmul(args: argparse.Namespace, device: torch.device) -> int:
    environment = collect_environment()
    problem = matmul_problem(args.m, args.n, args.k, DTYPES[args.dtype])
    timer = functools.partial(benchmark, warmup=args.warmup, iterations=args.iterations)
    _print_header(environment, args)

    with TuningDatabase(args.db) as database:
        result = tune(
            MATMUL,
            problem,
            database,
            device=device,
            limits=device_limits(),
            environment=environment,
            timer=timer,
            retune=args.retune,
        )
        _print_tuning(result)
        comparison = _compare_matmul(problem, result.best, device, timer)
        run_id = database.start_run(environment)
        for measurement in comparison:
            database.record(run_id, problem, measurement)

    _print_performance(comparison, environment, args)
    print(f"\nRecorded in {args.db}: tuning run {result.run_id}, comparison run {run_id}")
    return 0


def _torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def _compare_matmul(
    problem: Problem, best: KernelConfig, device: torch.device, timer: Timer
) -> list[Measurement]:
    """Verify and time every implementation on the same inputs, back to back."""
    inputs = MATMUL.make_inputs(problem, device)
    reference = MATMUL.reference(inputs)
    torch.compiler.reset()
    compiled = torch.compile(_torch_matmul, dynamic=False)
    implementations: list[tuple[str, KernelConfig | None, functools.partial[torch.Tensor]]] = [
        ("torch", None, functools.partial(_torch_matmul, *inputs)),
        ("torch.compile", None, functools.partial(compiled, *inputs)),
        (
            "triton-baseline",
            MATMUL_BASELINE,
            functools.partial(MATMUL.run, MATMUL_BASELINE, inputs),
        ),
        ("kernelforge", best, functools.partial(MATMUL.run, best, inputs)),
    ]
    flops, num_bytes = MATMUL.flops(problem), MATMUL.bytes_moved(problem)
    return [
        measure(
            name,
            fn,
            reference,
            problem.dtype,
            timer,
            config=config,
            flops=flops,
            num_bytes=num_bytes,
        )
        for name, config, fn in implementations
    ]


def _print_header(environment: Environment, args: argparse.Namespace) -> None:
    print(
        f"GPU: {environment.gpu} (compute capability {environment.compute_capability}, "
        f"driver {environment.driver}, CUDA {environment.cuda})"
    )
    print(f"Software: PyTorch {environment.torch}, Triton {environment.triton}")
    print(f"Shape: {args.m} x {args.n} x {args.k}")
    print(f"dtype: {args.dtype.upper()}")


def _print_tuning(result: TuneResult) -> None:
    if result.from_cache:
        print(f"Cache hit: using the configuration tuned in run {result.run_id} (--retune to redo)")
    for measurement in result.measurements:
        if not measurement.correct:
            print(f"  rejected {measurement.config}: {measurement.error}")
    print("\nBest configuration\n------------------")
    rows = [(name, result.best.params[name]) for name in MATMUL.search_space.params]
    rows += [("num_warps", result.best.num_warps), ("num_stages", result.best.num_stages)]
    for name, value in rows:
        label = f"{name}:"
        print(f"{label}{value:>{14 - len(label)}}")


def _print_performance(
    comparison: list[Measurement], environment: Environment, args: argparse.Namespace
) -> None:
    print(
        f"\nPerformance (median of {args.iterations} timed calls after {args.warmup} warmup, "
        "L2 flushed before each call)\n-----------"
    )
    for measurement in comparison:
        label = f"{LABELS[measurement.implementation]}:"
        if measurement.benchmark is None:
            print(f"{label:<20}rejected: {measurement.error}")
            continue
        stats = measurement.benchmark.stats
        print(f"{label:<20}{stats.median_us / 1e3:9.4f} ms   (p95 {stats.p95_us / 1e3:.4f} ms)")
    by_name = {m.implementation: m for m in comparison}
    kernelforge = by_name["kernelforge"]
    for baseline in ("triton-baseline", "torch"):
        other = by_name[baseline]
        if kernelforge.correct and other.correct:
            speedup = other.median_us / kernelforge.median_us
            print(f"{'Speedup vs ' + LABELS[baseline] + ':':<28}{speedup:.2f}x")
    if kernelforge.tflops is not None:
        print(f"{'Throughput:':<28}{kernelforge.tflops:.1f} TFLOP/s (KernelForge)")
    tf32 = "on" if environment.torch_matmul_allow_tf32 else "off"
    fp16 = "allowed" if environment.torch_matmul_fp16_reduced_precision else "off"
    bf16 = "allowed" if environment.torch_matmul_bf16_reduced_precision else "off"
    print(
        f"torch.compile ran in its default mode. PyTorch matmul settings: TF32 {tf32}, "
        f"reduced-precision reductions fp16 {fp16}, bf16 {bf16}."
    )


def _cache_list(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"No tuning database at {args.db}")
        return 0
    with TuningDatabase(args.db) as database:
        entries = database.cache_entries()
    print(f"{len(entries)} cached configuration(s) in {args.db}")
    for entry in entries:
        key = entry.key
        shape = " ".join(f"{dim}={size}" for dim, size in json.loads(key.shape).items())
        print(
            f"  {key.gpu} (sm {key.compute_capability}, Triton {key.triton}) | {key.operation} "
            f"{key.dtype} {shape} | {entry.config} | median {entry.median_us:.1f} us | "
            f"run {entry.run_id}"
        )
    return 0


def _cache_clear(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"No tuning database at {args.db}")
        return 0
    with TuningDatabase(args.db) as database:
        removed = database.clear_cache()
    print(f"Removed {removed} cached configuration(s) from {args.db}; measurements are kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
