"""Command line interface.

Built last, deliberately. Every subcommand is a thin wrapper over something
that already worked and was already tested: ``tune`` over
:class:`~kernelforge.tuning.tuner.Tuner`, ``benchmark`` over the CUDA-event
harness, ``report`` over the results database. Nothing is implemented here
that is not available as a library call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kernelforge import __version__
from kernelforge.dtypes import SUPPORTED_DTYPES, parse_dtype
from kernelforge.tuning.config import Problem

#: Dimension names and defaults per operation, so one set of flags serves all.
_GEMM_OPERATIONS = ("matmul", "fused_linear")
_ROW_OPERATIONS = ("rmsnorm", "softmax")
_DEFAULT_GEMM = {"m": 2048, "n": 4096, "k": 4096}
_DEFAULT_ROW = {"rows": 4096, "cols": 4096}
_DEFAULT_ELEMENTS = 1 << 24

OPERATIONS = (*_GEMM_OPERATIONS, *_ROW_OPERATIONS, "vector_add")


def _rule(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def build_problem(operation: str, args: argparse.Namespace) -> Problem:
    dtype = parse_dtype(args.dtype)
    if operation in _GEMM_OPERATIONS:
        return Problem.create(
            operation,
            dtype,
            M=args.m or _DEFAULT_GEMM["m"],
            N=args.n or _DEFAULT_GEMM["n"],
            K=args.k or _DEFAULT_GEMM["k"],
        )
    if operation in _ROW_OPERATIONS:
        return Problem.create(
            operation,
            dtype,
            rows=args.rows or _DEFAULT_ROW["rows"],
            cols=args.cols or _DEFAULT_ROW["cols"],
        )
    if operation == "vector_add":
        return Problem.create(operation, dtype, n=args.elements or _DEFAULT_ELEMENTS)
    raise ValueError(f"unknown operation {operation!r}")


def _describe_problem(problem: Problem) -> str:
    from kernelforge.runtime.env import capture_environment

    environment = capture_environment()
    shape = " x ".join(str(v) for _, v in problem.dims)
    return "\n".join(
        [
            f"GPU:    {environment.gpu_name or 'none (CPU only)'}",
            f"Shape:  {shape}  ({'/'.join(k for k, _ in problem.dims)})",
            f"dtype:  {problem.dtype_name.upper()}",
        ]
    )


def _open_db(args: argparse.Namespace):
    from kernelforge.db import ResultsDB

    if getattr(args, "no_db", False):
        return None
    return ResultsDB(args.db)


def _open_cache(args: argparse.Namespace):
    from kernelforge.tuning.cache import ConfigCache

    if getattr(args, "no_cache", False):
        return None
    return ConfigCache(args.cache)


# --- env -----------------------------------------------------------------
def command_env(args: argparse.Namespace) -> int:
    from kernelforge.runtime.env import capture_environment, device_caps

    environment = capture_environment()
    print(environment.render())
    if environment.gpu_name is None:
        print("\nNo CUDA device: kernels cannot run here, but the tuning core can be imported.")
        return 0
    caps = device_caps()
    print(_rule("Device limits"))
    for title, value in (
        ("Shared memory per block (opt-in)", f"{caps.max_shared_memory_per_block / 1024:.0f} KiB"),
        ("Shared memory per SM", f"{caps.shared_memory_per_sm / 1024:.0f} KiB"),
        ("Registers per SM", f"{caps.registers_per_sm}"),
        ("Max threads per SM", f"{caps.max_threads_per_sm}"),
        ("L2 cache", f"{caps.l2_cache_bytes / 1024**2:.0f} MiB"),
        ("Cache key", caps.key),
    ):
        print(f"{title:<34}{value}")
    return 0


# --- tune ----------------------------------------------------------------
def _render_tuning(result, operator) -> str:
    from kernelforge.benchmark import metrics

    lines = [_rule("Best configuration")]
    best = result.best
    if best is None:
        lines.append("No configuration passed verification.")
        counts = result.status_counts()
        lines += [f"  {status}: {count}" for status, count in sorted(counts.items())]
        return "\n".join(lines)

    lines.append(best.config.render(operator.config_order))

    lines.append(_rule("Performance"))
    latencies = sorted(result.baselines.items(), key=lambda kv: kv[1].median_ms)
    latencies.append(("kernelforge", best.timing))
    width = max(len(label) for label, _ in latencies) + 2
    lines += [f"{label + ':':<{width}}{timing.median_ms:>9.3f} ms" for label, timing in latencies]

    derived = [
        (f"speedup vs {label}", f"{result.speedup_over(label):.2f}x")
        for label in ("torch_eager", "torch_compile", "triton_baseline", "triton_autotune", "cuda")
        if result.speedup_over(label) is not None
    ]
    if result.memory_bound:
        derived.append(("bandwidth", f"{result.gbps(best.median_ms):.1f} GB/s"))
    else:
        derived.append(("throughput", f"{result.tflops(best.median_ms):.1f} TFLOP/s"))
    # Arithmetic intensity against the device's own FLOP/byte ratio is the
    # first-order answer to "what stops this kernel going faster": below the
    # ridge point, no tiling gets past the memory system.
    derived.append(
        (
            "arithmetic intensity",
            f"{metrics.arithmetic_intensity(result.flops, result.bytes_moved):.1f} FLOP/byte",
        )
    )
    derived_width = max(len(label) for label, _ in derived) + 2
    lines.append("")
    lines += [f"{label + ':':<{derived_width}}{value:>16}" for label, value in derived]

    if best.timing is not None and best.timing.at_timer_resolution:
        lines.append(
            "\nWarning: median latency is at CUDA event resolution; "
            "use a larger problem for a meaningful comparison."
        )
    return "\n".join(lines)


def command_tune(args: argparse.Namespace) -> int:
    from kernelforge.kernels import get_operator
    from kernelforge.tuning.search import search_space
    from kernelforge.tuning.tuner import Tuner

    problem = build_problem(args.operation, args)

    if args.explain:
        from kernelforge.runtime.env import device_caps

        space = search_space(args.operation)
        if args.max_candidates is not None:
            space.max_candidates = args.max_candidates
        print(space.explain(problem, device_caps()))
        return 0

    operator = get_operator(args.operation)
    print(_describe_problem(problem))
    print()

    db = _open_db(args)
    cache = _open_cache(args)
    try:
        result = Tuner(
            warmup=args.warmup,
            iterations=args.iterations,
            max_candidates=args.max_candidates,
            measure_baselines=not args.no_baselines,
            flush_l2=not args.no_flush_l2,
            db=db,
            cache=cache,
            log=print,
        ).tune(operator, problem)
    finally:
        if db is not None:
            db.close()

    print(
        f"\nSearch space: {result.generated} grid points, {result.feasible} feasible, "
        f"{result.tested} measured in {result.elapsed_s:.1f}s"
    )
    print(_render_tuning(result, operator))
    return 0 if result.best is not None else 1


# --- benchmark -----------------------------------------------------------
def command_benchmark(args: argparse.Namespace) -> int:
    from kernelforge.benchmark import workloads
    from kernelforge.benchmark.runner import benchmark
    from kernelforge.db import Measurement
    from kernelforge.kernels import get_operator
    from kernelforge.runtime.dispatch import select_config
    from kernelforge.runtime.env import capture_environment, require_cuda
    from kernelforge.testing import exact_fp32_matmul, verify

    device = require_cuda()
    operator = get_operator(args.operation)
    problems = workloads.problems(args.operation, args.suite, args.dtype)
    environment = capture_environment(device)
    cache = _open_cache(args)
    db = _open_db(args)
    run_id = (
        db.start_run(environment, notes=f"benchmark:{args.suite}", version=__version__)
        if db
        else None
    )

    print(environment.render())
    print(_rule(f"{args.operation} / {args.suite} / {args.dtype}"))
    header = f"{'shape':<24}{'impl':<18}{'median ms':>11}{'p99 ms':>10}{'metric':>14}"
    print(header)
    print("-" * len(header))

    try:
        for problem in problems:
            inputs = operator.make_inputs(problem, device)
            with exact_fp32_matmul():
                reference = operator.reference(*inputs)
                selection = select_config(operator, problem, cache=cache, device=device)
                runnable = dict(operator.baselines(problem, inputs))
                runnable["kernelforge"] = lambda c=selection.config, i=inputs: operator.run(c, *i)

                for label, fn in runnable.items():
                    try:
                        check = verify(reference, fn(), dtype=problem.dtype)
                    except Exception as exc:
                        print(f"{problem.shape_key:<24}{label:<18}  failed: {type(exc).__name__}")
                        continue
                    if not check.passed:
                        print(f"{problem.shape_key:<24}{label:<18}  INCORRECT: {check.reason}")
                        continue
                    timing = benchmark(
                        fn,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        device=device,
                        flush_l2=not args.no_flush_l2,
                    )
                    if operator.is_memory_bound():
                        from kernelforge.benchmark.metrics import gbps

                        value = gbps(operator.bytes_moved(problem), timing.median_ms)
                        metric = f"{value:.1f} GB/s"
                        tflops_value, gbps_value = None, value
                    else:
                        from kernelforge.benchmark.metrics import tflops

                        value = tflops(operator.flops(problem), timing.median_ms)
                        metric = f"{value:.1f} TFLOP/s"
                        tflops_value, gbps_value = value, None
                    print(
                        f"{problem.shape_key:<24}{label:<18}"
                        f"{timing.median_ms:>11.4f}{timing.p99_ms:>10.4f}{metric:>14}"
                    )
                    if db is not None:
                        db.record(
                            run_id,
                            problem,
                            Measurement(
                                label=label,
                                status="ok",
                                config=selection.config if label == "kernelforge" else None,
                                verification=check,
                                timing=timing,
                                tflops=tflops_value,
                                gbps=gbps_value,
                            ),
                        )
    finally:
        if db is not None:
            db.close()
    return 0


# --- compare -------------------------------------------------------------
def command_compare(args: argparse.Namespace) -> int:
    if args.operation == "transformer":
        return _compare_transformer(args)

    from kernelforge.benchmark.runner import benchmark
    from kernelforge.kernels import get_operator
    from kernelforge.runtime.dispatch import select_config
    from kernelforge.runtime.env import require_cuda
    from kernelforge.testing import exact_fp32_matmul, verify

    device = require_cuda()
    operator = get_operator(args.operation)
    problem = build_problem(args.operation, args)
    print(_describe_problem(problem))

    inputs = operator.make_inputs(problem, device)
    cache = _open_cache(args)
    with exact_fp32_matmul():
        reference = operator.reference(*inputs)
        selection = select_config(operator, problem, cache=cache, device=device)
        print(f"config: {selection.describe()}")
        print(f"        {', '.join(f'{k}={v}' for k, v in selection.config.as_dict().items())}")

        runnable = dict(operator.baselines(problem, inputs))
        runnable["kernelforge"] = lambda c=selection.config: operator.run(c, *inputs)

        print(_rule("Comparison"))
        rows = []
        for label, fn in runnable.items():
            check = verify(reference, fn(), dtype=problem.dtype)
            if not check.passed:
                print(f"{label}: INCORRECT ({check.reason})")
                continue
            timing = benchmark(
                fn,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
                flush_l2=not args.no_flush_l2,
            )
            rows.append((label, timing, check))

    rows.sort(key=lambda row: row[1].median_ms)
    eager = next((t.median_ms for label, t, _ in rows if label == "torch_eager"), None)
    header = (
        f"{'impl':<18}{'median ms':>11}{'p99 ms':>10}{'vs eager':>10}{'metric':>14}{'max err':>11}"
    )
    print(header)
    print("-" * len(header))
    for label, timing, check in rows:
        if operator.is_memory_bound():
            from kernelforge.benchmark.metrics import gbps

            metric = f"{gbps(operator.bytes_moved(problem), timing.median_ms):.1f} GB/s"
        else:
            from kernelforge.benchmark.metrics import tflops

            metric = f"{tflops(operator.flops(problem), timing.median_ms):.1f} TFLOP/s"
        relative = f"{eager / timing.median_ms:.2f}x" if eager else "-"
        print(
            f"{label:<18}{timing.median_ms:>11.4f}{timing.p99_ms:>10.4f}"
            f"{relative:>10}{metric:>14}{check.error:>11.2e}"
        )
    return 0


def _compare_transformer(args: argparse.Namespace) -> int:
    from kernelforge.integration import BlockConfig, benchmark_block
    from kernelforge.runtime.env import require_cuda

    device = require_cuda()
    config = BlockConfig(hidden=args.hidden, heads=args.heads, intermediate=args.intermediate)
    print(
        f"Transformer block: hidden={config.hidden} heads={config.heads} "
        f"intermediate={config.intermediate} batch={args.batch} seq={args.seq} "
        f"dtype={args.dtype}"
    )
    results = benchmark_block(
        config,
        batch=args.batch,
        seq=args.seq,
        dtype=parse_dtype(args.dtype),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        cache=_open_cache(args),
    )
    print(_rule("Block latency"))
    baseline = results["torch"].median_ms
    for backend, timing in results.items():
        print(
            f"{backend:<14}{timing.median_ms:>10.3f} ms"
            f"{'  (' + format(baseline / timing.median_ms, '.3f') + 'x)' if backend != 'torch' else ''}"
        )
    print(
        "\nOnly the two RMSNorms and the MLP activation are swapped; attention and the\n"
        "projections are PyTorch in both. The block-level speedup is therefore bounded\n"
        "by the share of block time those operators held."
    )
    return 0


# --- profile -------------------------------------------------------------
def nsight_child_argv(args: argparse.Namespace) -> list[str]:
    """CLI arguments for the single-launch process that ``ncu`` profiles.

    Rebuilt from the parsed arguments rather than sliced out of ``sys.argv``:
    slicing drops any shape flag written after ``--backend``, which would
    silently profile the default shape instead of the requested one. The cache
    is passed through so the child picks the same configuration the parent
    would, and the database is switched off because one serialised launch under
    a profiler is not a measurement worth recording.
    """
    argv = ["profile", args.operation, "--dtype", args.dtype]
    for flag, value in (
        ("-m", args.m),
        ("-n", args.n),
        ("-k", args.k),
        ("--rows", args.rows),
        ("--cols", args.cols),
        ("--elements", args.elements),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if args.no_cache:
        argv.append("--no-cache")
    else:
        argv += ["--cache", args.cache]
    return argv + ["--no-db", "--backend", "launch-once"]


def command_profile(args: argparse.Namespace) -> int:
    # Handled before any device work: the parent only has to assemble a command
    # line, and allocating this problem's inputs in both processes would double
    # its memory footprint for nothing.
    if args.backend == "nsight":
        from kernelforge.profiling import nsight

        if not nsight.ncu_available():
            print("ncu not found on PATH; install NVIDIA Nsight Compute.", file=sys.stderr)
            return 2
        run = nsight.run(
            nsight.self_command(nsight_child_argv(args)),
            sections=tuple(args.sections),
            report_path=args.report,
            launch_count=1,
        )
        print(run.render())
        return 0 if run.ok else 1

    from kernelforge.kernels import get_operator
    from kernelforge.runtime.dispatch import select_config
    from kernelforge.runtime.env import require_cuda

    device = require_cuda()
    operator = get_operator(args.operation)
    problem = build_problem(args.operation, args)
    inputs = operator.make_inputs(problem, device)
    selection = select_config(operator, problem, cache=_open_cache(args), device=device)

    if args.backend == "launch-once":
        import torch

        operator.run(selection.config, *inputs)
        torch.cuda.synchronize()
        return 0

    from kernelforge.profiling import compare_launch_counts

    print(_describe_problem(problem))
    candidates = dict(operator.baselines(problem, inputs))
    candidates["kernelforge"] = lambda: operator.run(selection.config, *inputs)
    print(_rule("Kernel attribution"))
    for result in compare_launch_counts(candidates, warmup=args.warmup, iterations=args.iterations):
        print(result.render())
        print()
    return 0


# --- report --------------------------------------------------------------
def command_report(args: argparse.Namespace) -> int:
    from kernelforge.benchmark import report
    from kernelforge.db import ResultsDB

    path = Path(args.db)
    if not path.exists():
        print(f"no results database at {path}; run `kernelforge tune` first.", file=sys.stderr)
        return 2
    with ResultsDB(path) as db:
        counts = db.counts()
        print(
            f"{counts['results']} measurements across {counts['problems']} problems "
            f"and {counts['runs']} runs"
        )
        print(report.generate(db, args.out).render())
    return 0


# --- cache ---------------------------------------------------------------
def command_cache(args: argparse.Namespace) -> int:
    from kernelforge.tuning.cache import ConfigCache

    cache = ConfigCache(args.cache)
    if args.cache_command == "clear":
        removed = cache.clear()
        print(f"removed {removed} cached configuration(s) from {cache.path}")
        return 0
    entries = cache.entries()
    print(f"{cache.path}: {len(entries)} entry(ies)")
    for entry in entries:
        print(f"  {entry.describe()}")
    return 0


# --- parser --------------------------------------------------------------
def _add_shape_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "shape", "dimensions for the selected operation; sensible defaults apply"
    )
    group.add_argument("-m", "--m", type=int, help="GEMM rows of A (matmul, fused_linear)")
    group.add_argument("-n", "--n", type=int, help="GEMM columns of B (matmul, fused_linear)")
    group.add_argument("-k", "--k", type=int, help="GEMM reduction size (matmul, fused_linear)")
    group.add_argument("--rows", type=int, help="rows (rmsnorm, softmax)")
    group.add_argument("--cols", type=int, help="row width (rmsnorm, softmax)")
    group.add_argument("--elements", type=int, help="element count (vector_add)")


def _add_measurement_arguments(parser: argparse.ArgumentParser, *, warmup=25, iterations=200):
    parser.add_argument("--dtype", default="fp16", choices=SUPPORTED_DTYPES)
    parser.add_argument("--warmup", type=int, default=warmup)
    parser.add_argument("--iterations", type=int, default=iterations)
    parser.add_argument(
        "--no-flush-l2",
        action="store_true",
        help="do not flush L2 between iterations (inflates cache-resident problems)",
    )


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    from kernelforge.db import default_db_path
    from kernelforge.tuning.cache import default_cache_path

    parser.add_argument("--db", default=str(default_db_path()), help="results database path")
    parser.add_argument("--no-db", action="store_true", help="do not record results")
    parser.add_argument("--cache", default=str(default_cache_path()), help="config cache path")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernelforge",
        description="Hardware-aware GPU kernel autotuning for transformer inference.",
    )
    parser.add_argument("--version", action="version", version=f"kernelforge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("env", help="print GPU, driver and library versions")

    tune = subparsers.add_parser("tune", help="search configurations for one problem")
    tune.add_argument("operation", choices=OPERATIONS)
    _add_shape_arguments(tune)
    _add_measurement_arguments(tune)
    _add_storage_arguments(tune)
    tune.add_argument("--max-candidates", type=int, help="override the candidate budget")
    tune.add_argument("--no-baselines", action="store_true", help="skip comparison baselines")
    tune.add_argument(
        "--explain",
        action="store_true",
        help="print the candidate-generation breakdown and exit without tuning",
    )

    bench = subparsers.add_parser("benchmark", help="run a named workload suite")
    bench.add_argument("operation", choices=OPERATIONS)
    bench.add_argument("--suite", default="sweep", choices=("smoke", "sweep", "transformer"))
    _add_measurement_arguments(bench, iterations=100)
    _add_storage_arguments(bench)

    compare = subparsers.add_parser(
        "compare", help="compare implementations at one shape, or a whole transformer block"
    )
    compare.add_argument("operation", choices=(*OPERATIONS, "transformer"))
    _add_shape_arguments(compare)
    _add_measurement_arguments(compare, iterations=100)
    _add_storage_arguments(compare)
    compare.add_argument("--batch", type=int, default=1, help="transformer: batch size")
    compare.add_argument("--seq", type=int, default=2048, help="transformer: sequence length")
    compare.add_argument("--hidden", type=int, default=4096, help="transformer: hidden size")
    compare.add_argument("--heads", type=int, default=32, help="transformer: attention heads")
    compare.add_argument(
        "--intermediate", type=int, default=11008, help="transformer: MLP intermediate size"
    )

    profile = subparsers.add_parser("profile", help="attribute time to kernels and counters")
    profile.add_argument("operation", choices=OPERATIONS)
    _add_shape_arguments(profile)
    _add_measurement_arguments(profile, warmup=10, iterations=20)
    _add_storage_arguments(profile)
    profile.add_argument(
        "--backend",
        default="torch",
        choices=("torch", "nsight", "launch-once"),
        help="torch: kernel attribution; nsight: hardware counters via ncu; "
        "launch-once: single launch, used as the ncu target",
    )
    profile.add_argument(
        "--sections",
        nargs="+",
        default=list(_nsight_default_sections()),
        help="Nsight Compute sections to collect",
    )
    profile.add_argument("--report", help="write an .ncu-rep report to this path")

    report = subparsers.add_parser("report", help="render tables and figures from the database")
    _add_storage_arguments(report)
    report.add_argument("--out", default="reports", help="output directory")

    cache = subparsers.add_parser("cache", help="inspect or clear the tuned-config cache")
    cache.add_argument("cache_command", choices=("list", "clear"))
    _add_storage_arguments(cache)

    return parser


def _nsight_default_sections() -> tuple[str, ...]:
    from kernelforge.profiling.nsight import DEFAULT_SECTIONS

    return DEFAULT_SECTIONS


_COMMANDS = {
    "env": command_env,
    "tune": command_tune,
    "benchmark": command_benchmark,
    "compare": command_compare,
    "profile": command_profile,
    "report": command_report,
    "cache": command_cache,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
