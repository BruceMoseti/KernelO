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

from kernelforge.db import ResultsDB

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


def _require_dependencies():
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


def _frame(pandas, db: ResultsDB):
    """Correct, timed rows as a DataFrame, with config parameters expanded."""
    rows = [r for r in db.rows() if r["median_us"] is not None and r["correct"] != 0]
    frame = pandas.DataFrame(rows)
    if frame.empty:
        return frame
    frame["size"] = frame["dims_json"].map(lambda s: prod(json.loads(s).values()))
    params = frame["params_json"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    for key in sorted({k for d in params for k in d}):
        frame[f"cfg_{key}"] = params.map(lambda d, k=key: d.get(k))
    return frame


def _best_per_label(frame):
    """Fastest row for each (operation, dtype, shape, label)."""
    index = frame.groupby(["operation", "dtype", "shape_key", "label"])["median_us"].idxmin()
    return frame.loc[index].copy()


def _ordered_labels(labels) -> list[str]:
    present = set(labels)
    ranked = [label for label in LABEL_ORDER if label in present]
    return ranked + sorted(present - set(ranked))


def _shape_order(frame) -> list[str]:
    pairs = frame[["shape_key", "size"]].drop_duplicates().sort_values("size")
    return list(pairs["shape_key"])


def _grouped_bars(plt, frame, *, value, ylabel, title, path: Path) -> Path:
    labels = _ordered_labels(frame["label"])
    shapes = _shape_order(frame)
    width = 0.8 / max(len(labels), 1)
    figure, axis = plt.subplots(figsize=(max(7.0, 1.1 * len(shapes) + 2.5), 4.2))
    for offset, label in enumerate(labels):
        subset = frame[frame["label"] == label].set_index("shape_key")
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


def _tuning_heatmap(plt, frame, path: Path) -> Path | None:
    """Median latency over the BLOCK_M x BLOCK_N plane for one GEMM shape.

    Uses the largest shape with the most measured candidates, and takes the
    best latency over the remaining parameters at each point, so the picture is
    "what is the best this tile can do" rather than an average over
    configurations that were never going to win.
    """
    # Databases holding only baselines, or only non-GEMM operators, have no
    # tile columns at all.
    if frame.empty or "cfg_BLOCK_M" not in frame.columns:
        return None
    candidates = frame[(frame["label"] == "kernelforge") & frame["cfg_BLOCK_M"].notna()]
    if candidates.empty:
        return None
    shape = candidates.groupby("shape_key")["median_us"].count().idxmax()
    subset = candidates[candidates["shape_key"] == shape]
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
    axis.set_title(f"Best median latency by tile ({shape})")
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


def _environment_section(db: ResultsDB) -> list[str]:
    runs = db.rows()
    if not runs:
        return ["No runs recorded."]
    latest = runs[-1]
    fields = (
        ("GPU", "gpu_name"),
        ("Compute capability", "gpu_arch"),
        ("CUDA", "cuda_version"),
        ("Driver", "driver_version"),
        ("PyTorch", "torch_version"),
        ("Triton", "triton_version"),
    )
    lines = ["| Field | Value |", "| --- | --- |"]
    lines += [f"| {title} | {latest.get(key) or '-'} |" for title, key in fields]
    return lines


def _summary_table(frame, operation: str) -> list[str]:
    best = _best_per_label(frame[frame["operation"] == operation])
    if best.empty:
        return []
    labels = _ordered_labels(best["label"])
    memory_bound = operation in MEMORY_BOUND_OPERATIONS
    metric_title = "GB/s" if memory_bound else "TFLOP/s"
    metric_column = "gbps" if memory_bound else "tflops"

    header = ["shape", "dtype", *(LABEL_TITLES.get(x, x) + " (us)" for x in labels)]
    header += [f"KernelForge {metric_title}", "speedup vs eager"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    for (shape, dtype), group in best.groupby(["shape_key", "dtype"], sort=False):
        times = group.set_index("label")["median_us"]
        row = [shape, dtype] + [
            f"{times[label]:.1f}" if label in times else "-" for label in labels
        ]
        mine = group[group["label"] == "kernelforge"]
        throughput = mine[metric_column].max() if not mine.empty else None
        row.append(f"{throughput:.1f}" if throughput and throughput == throughput else "-")
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

    frame = _frame(pandas, db)
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
        *_environment_section(db),
        "",
    ]

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
