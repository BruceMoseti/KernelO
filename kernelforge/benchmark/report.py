"""Reports generated from the results database.

Everything a reader sees is derived from rows in SQLite, so no number in a
report was transcribed by hand and every one of them carries the GPU, driver
and library versions it was measured on. Re-running the benchmarks and
regenerating gives a new report; there is no step where a figure is copied
into prose and then goes stale.

Figures are emitted only where there is data for them. A report produced from
a database with no fused-linear results says so rather than writing an empty
axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernelforge.benchmark import metrics
from kernelforge.db import ENVIRONMENT_FIELDS, ResultsDB

if TYPE_CHECKING:  # pandas is an optional extra; see `_require_dependencies`
    from pandas import DataFrame

#: Order in which implementations appear in tables and legends.
LABEL_ORDER = (
    "torch_eager",
    "torch_compile",
    "triton_baseline",
    "triton_autotune",
    "cuda",
    "kernelforge",
)

LABEL_TITLES = {
    "torch_eager": "PyTorch eager",
    "torch_compile": "torch.compile",
    "triton_baseline": "Triton (untuned)",
    "triton_autotune": "Triton autotune",
    "cuda": "CUDA (handwritten)",
    "kernelforge": "KernelForge",
}

MEMORY_BOUND_OPERATIONS = frozenset({"rmsnorm", "softmax", "vector_add"})


@dataclass
class ReportArtifacts:
    directory: Path
    summary: Path
    figures: list[Path] = field(default_factory=list)
    tables: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Report written to {self.directory}", f"  {self.summary.name}"]
        lines += [f"  {p.name}" for p in self.tables]
        lines += [f"  {p.name}" for p in self.figures]
        lines += [f"  note: {note}" for note in self.notes]
        return "\n".join(lines)


def _require_dependencies() -> tuple[Any, Any]:
    try:
        import matplotlib
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise RuntimeError(
            "report generation needs pandas and matplotlib: pip install 'kernelforge[report]'"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return pandas, plt


def _environments(db: ResultsDB) -> tuple[list[tuple], dict[int, int]]:
    """Every distinct environment, and the number of the one each run used.

    Numbered in order of first appearance, so that table rows can name the
    environment table row they were measured in.
    """
    numbers: dict[tuple, int] = {}
    of_run = {}
    for run in db.runs():
        key = tuple(run[field] for _, field in ENVIRONMENT_FIELDS)
        of_run[run["run_id"]] = numbers.setdefault(key, len(numbers) + 1)
    return list(numbers), of_run


def _frame(pandas: Any, db: ResultsDB, environment_of_run: dict[int, int]) -> DataFrame:
    """Correct, timed rows as a DataFrame, with config parameters expanded."""
    rows = [r for r in db.rows() if r["median_us"] is not None and r["correct"] != 0]
    frame = pandas.DataFrame(rows)
    if frame.empty:
        return frame
    frame["environment"] = frame["run_id"].map(environment_of_run)
    frame["size"] = frame["dims_json"].map(lambda s: prod(json.loads(s).values()))
    params = frame["params_json"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    for key in sorted({k for d in params for k in d}):
        frame[f"cfg_{key}"] = params.map(lambda d, k=key: d.get(k))
    return frame


def _best_per_label(frame: DataFrame) -> DataFrame:
    """Fastest row for each (environment, operation, dtype, shape, label).

    Ordered by problem size rather than by the shape string: sorting
    ``shape_key`` lexicographically puts ``512x512x512`` after
    ``2048x4096x4096``, which reads as noise in a table.
    """
    keys = ["environment", "operation", "dtype", "shape_key", "label"]
    index = frame.groupby(keys)["median_us"].idxmin()
    return frame.loc[index].sort_values(["operation", "dtype", "size", "environment"]).copy()


def _ordered_labels(labels: Any) -> list[str]:
    present = set(labels)
    ranked = [label for label in LABEL_ORDER if label in present]
    return ranked + sorted(present - set(ranked))


def _with_axis_labels(frame: DataFrame) -> DataFrame:
    """Add the x-axis key, which has to distinguish dtypes and environments.

    A shape alone is not a unique series: the same ``1024x1024x1024`` may have
    been measured in fp16 and bf16, or on two GPUs, and indexing on the shape
    would both collapse two bars into one label and hand matplotlib a
    two-element Series where it expects a scalar.
    """
    key = frame["shape_key"]
    if frame["dtype"].nunique() > 1:
        key = key + " " + frame["dtype"]
    if frame["environment"].nunique() > 1:
        key = key + " env " + frame["environment"].astype(str)
    return frame.assign(axis_key=key)


def _axis_order(frame: DataFrame) -> list[str]:
    pairs = frame[["axis_key", "size", "dtype"]].drop_duplicates().sort_values(["size", "dtype"])
    return list(pairs["axis_key"])


def _grouped_bars(
    plt: Any, frame: DataFrame, *, value: str, ylabel: str, title: str, path: Path
) -> Path:
    frame = _with_axis_labels(frame)
    labels = _ordered_labels(frame["label"])
    shapes = _axis_order(frame)
    width = 0.8 / max(len(labels), 1)
    figure, axis = plt.subplots(figsize=(max(7.0, 1.1 * len(shapes) + 2.5), 4.2))
    for offset, label in enumerate(labels):
        subset = frame[frame["label"] == label].set_index("axis_key")
        heights = [subset[value].get(shape, float("nan")) for shape in shapes]
        positions = [i + offset * width for i in range(len(shapes))]
        axis.bar(positions, heights, width=width, label=LABEL_TITLES.get(label, label))
    axis.set_xticks([i + 0.4 - width / 2 for i in range(len(shapes))])
    axis.set_xticklabels(shapes, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _tuning_heatmap(plt: Any, frame: DataFrame, path: Path) -> Path | None:
    """Median latency over the BLOCK_M x BLOCK_N plane for one GEMM shape.

    Uses the (environment, shape, dtype) combination with the most measured
    candidates, which is the one tuning explored most thoroughly, and takes the
    *best* latency over the remaining parameters at each point -- so the
    picture is "what is the best this tile can do", not an average over
    configurations that were never going to win. The environment and dtype are
    part of the selection because tilings measured on two GPUs, or in fp16 and
    bf16, must not be combined.
    """
    # Databases holding only baselines, or only non-GEMM operators, have no
    # tile columns at all.
    if frame.empty or "cfg_BLOCK_M" not in frame.columns:
        return None
    candidates = frame[(frame["label"] == "kernelforge") & frame["cfg_BLOCK_M"].notna()]
    if candidates.empty:
        return None
    keys = ["environment", "shape_key", "dtype"]
    environment, shape, dtype = candidates.groupby(keys)["median_us"].count().idxmax()
    subset = candidates[
        (candidates["environment"] == environment)
        & (candidates["shape_key"] == shape)
        & (candidates["dtype"] == dtype)
    ]
    table = subset.pivot_table(
        index="cfg_BLOCK_M", columns="cfg_BLOCK_N", values="median_us", aggfunc="min"
    )
    if table.empty:
        return None

    figure, axis = plt.subplots(figsize=(5.6, 4.4))
    image = axis.imshow(table.values, cmap="viridis_r", aspect="auto")
    axis.set_xticks(range(len(table.columns)))
    axis.set_xticklabels([int(c) for c in table.columns])
    axis.set_yticks(range(len(table.index)))
    axis.set_yticklabels([int(i) for i in table.index])
    axis.set_xlabel("BLOCK_N")
    axis.set_ylabel("BLOCK_M")
    axis.set_title(f"Best median latency by tile ({shape} {dtype}, env {environment})")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            if value == value:  # not NaN
                axis.text(j, i, f"{value:.0f}", ha="center", va="center", color="white", fontsize=7)
    figure.colorbar(image, ax=axis, label="median latency (us)")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _environment_section(environments: list[tuple]) -> list[str]:
    if not environments:
        return ["No runs recorded."]
    lines = [
        "| env | " + " | ".join(title for title, _ in ENVIRONMENT_FIELDS) + " |",
        "| " + " | ".join(["---"] * (len(ENVIRONMENT_FIELDS) + 1)) + " |",
    ]
    lines += [
        f"| {number} | " + " | ".join(str(value or "-") for value in values) + " |"
        for number, values in enumerate(environments, start=1)
    ]
    return lines


def _peaks_section(gpu_names: Any) -> list[str]:
    """The published roofs that "of published peak" divides by, with their sources."""
    known = [name for name in gpu_names if name in metrics.PUBLISHED_PEAKS]
    unknown = [name for name in gpu_names if name not in metrics.PUBLISHED_PEAKS]
    lines = []
    if known:
        lines += [
            '"of published peak" divides by NVIDIA\'s published dense peak for the GPU: '
            "a spec-sheet figure at boost clock, not a measurement.",
            "",
            "| GPU | FP16/BF16 Tensor Core TFLOP/s (FP32 accumulate) | FP32 TFLOP/s "
            "| DRAM GB/s | Source |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in known:
            peak = metrics.PUBLISHED_PEAKS[name]
            lines.append(
                f"| {name} | {peak.tensor_tflops:g} | {peak.fp32_tflops:g} "
                f"| {peak.dram_gbps:g} | {peak.source} |"
            )
        lines.append("")
    if unknown:
        lines += [
            f"No published peak on file for {', '.join(unknown)}; add one to "
            "`PUBLISHED_PEAKS` in `kernelforge/benchmark/metrics.py` from NVIDIA's "
            "datasheet for that board.",
            "",
        ]
    return lines


def _methodology_warnings(frame: DataFrame) -> list[str]:
    """Settings that make rows in this database incomparable with each other."""
    warnings = []
    if frame["flushed_l2"].nunique() > 1:
        flushed = int((frame["flushed_l2"] == 1).sum())
        warnings.append(
            f"this database mixes {flushed} L2-flushed measurement(s) with "
            f"{len(frame) - flushed} unflushed; the unflushed rows report a bandwidth "
            "the kernel would not see on cold inputs, so the two are not comparable"
        )
    if frame["timer"].nunique() > 1:
        warnings.append(
            f"this database mixes timers ({', '.join(sorted(frame['timer'].unique()))}); "
            "only cuda_event rows are GPU measurements"
        )
    return warnings


def _summary_table(frame: DataFrame, operation: str) -> list[str]:
    best = _best_per_label(frame[frame["operation"] == operation])
    if best.empty:
        return []
    labels = _ordered_labels(best["label"])
    memory_bound = operation in MEMORY_BOUND_OPERATIONS
    metric_title = "GB/s" if memory_bound else "TFLOP/s"
    metric_column = "gbps" if memory_bound else "tflops"

    header = ["shape", "dtype", "env", *(LABEL_TITLES.get(x, x) + " (us)" for x in labels)]
    header += [f"KernelForge {metric_title}", "of published peak", "speedup vs eager"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    groups = best.groupby(["shape_key", "dtype", "environment"], sort=False)
    for (shape, dtype, environment), group in groups:
        times = group.set_index("label")["median_us"]
        row = [shape, dtype, str(environment)] + [
            f"{times[label]:.1f}" if label in times else "-" for label in labels
        ]
        mine = group[group["label"] == "kernelforge"]
        throughput = mine[metric_column].max() if not mine.empty else None
        row.append(f"{throughput:.1f}" if throughput and throughput == throughput else "-")
        peak = metrics.PUBLISHED_PEAKS.get(group["gpu_name"].iloc[0])
        if peak is not None and throughput and throughput == throughput:
            row.append(f"{throughput / peak.rate(dtype, memory_bound=memory_bound):.1%}")
        else:
            row.append("-")
        if "kernelforge" in times and "torch_eager" in times:
            row.append(f"{times['torch_eager'] / times['kernelforge']:.2f}x")
        else:
            row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def generate(
    db: ResultsDB, directory: str | Path = "reports", *, operations: list[str] | None = None
) -> ReportArtifacts:
    """Write ``summary.md``, CSV exports and figures into ``directory``."""
    pandas, plt = _require_dependencies()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = ReportArtifacts(directory=out, summary=out / "summary.md")

    environments, environment_of_run = _environments(db)
    frame = _frame(pandas, db, environment_of_run)
    if frame.empty:
        artifacts.summary.write_text(
            "# KernelForge results\n\nNo correct, timed measurements in the database yet. "
            "Run `kernelforge tune` or `kernelforge benchmark` first.\n"
        )
        artifacts.notes.append("database contained no usable measurements")
        return artifacts

    present = operations or sorted(frame["operation"].unique())
    lines = [
        "# KernelForge results",
        "",
        "Generated from the results database by `kernelforge report`. "
        "Latencies are medians over the benchmark iterations recorded with each row; "
        "see `docs/BENCHMARKING.md` for the methodology.",
        "",
        "## Environment",
        "",
        *_environment_section(environments),
        "",
        *_peaks_section(frame["gpu_name"].dropna().unique()),
    ]

    # Flushed and unflushed timings are not comparable: leaving the inputs
    # resident in L2 inflates throughput for any problem whose working set
    # fits. Ranking by median across both would quietly prefer the unflushed
    # row, so mixing is called out rather than left to be discovered.
    for warning in _methodology_warnings(frame):
        artifacts.notes.append(warning)
        lines += [f"> **Warning:** {warning}", ""]

    for operation in present:
        table = _summary_table(frame, operation)
        if not table:
            continue
        lines += [f"## {operation}", "", *table, ""]

        subset = frame[frame["operation"] == operation]
        export = out / f"{operation}.csv"
        _best_per_label(subset).to_csv(export, index=False)
        artifacts.tables.append(export)

        best = _best_per_label(subset)
        artifacts.figures.append(
            _grouped_bars(
                plt,
                best,
                value="median_us",
                ylabel="median latency (us)",
                title=f"{operation}: latency by shape",
                path=out / f"{operation}_latency.png",
            )
        )
        if operation in MEMORY_BOUND_OPERATIONS:
            if best["gbps"].notna().any():
                artifacts.figures.append(
                    _grouped_bars(
                        plt,
                        best,
                        value="gbps",
                        ylabel="GB/s",
                        title=f"{operation}: effective bandwidth",
                        path=out / f"{operation}_bandwidth.png",
                    )
                )
        elif best["tflops"].notna().any():
            artifacts.figures.append(
                _grouped_bars(
                    plt,
                    best,
                    value="tflops",
                    ylabel="TFLOP/s",
                    title=f"{operation}: throughput by shape",
                    path=out / f"{operation}_tflops.png",
                )
            )

    heatmap = _tuning_heatmap(
        plt, frame[frame["operation"] == "matmul"], out / "tuning_heatmap.png"
    )
    if heatmap is not None:
        artifacts.figures.append(heatmap)
    else:
        artifacts.notes.append("no tuned GEMM candidates recorded, skipped tuning_heatmap.png")

    lines += [
        "## Files",
        "",
        *(f"- `{p.name}`" for p in artifacts.tables + artifacts.figures),
        "",
    ]
    artifacts.summary.write_text("\n".join(lines))
    return artifacts
