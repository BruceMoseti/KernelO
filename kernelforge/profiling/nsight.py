"""Nsight Compute integration.

The latency harness answers "which is faster". This answers "why", by pulling
the hardware counters that say where a kernel's time went: achieved occupancy,
SM and DRAM throughput against peak, L2 hit rate, registers per thread, shared
memory per block, and the reasons warps were not issuing.

Nsight Compute is driven by *section* name rather than by individual metric
name. Sections are stable across ncu versions and map directly onto the
questions being asked, whereas metric names move between releases.

A warning worth repeating, because Nsight's own documentation makes it:
**higher occupancy does not mean faster**. A large-tile GEMM deliberately
trades occupancy for data reuse and register residency, and will often beat a
higher-occupancy configuration. Occupancy is one input to an explanation, not
the explanation.

Profiling serialises kernel launches and replays them to collect counters, so
an ncu run is far slower than the kernel and its wall-clock time means nothing.
Collect counters for the handful of configurations a case study compares, not
for a benchmark sweep.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Nsight Compute sections, keyed by the question each one answers.
SECTIONS: dict[str, str] = {
    "launch": "LaunchStats",
    "occupancy": "Occupancy",
    "memory": "MemoryWorkloadAnalysis",
    "throughput": "SpeedOfLight",
    "stalls": "WarpStateStats",
    "instructions": "InstructionStats",
}

DEFAULT_SECTIONS = ("launch", "occupancy", "throughput", "memory", "stalls")


def ncu_available() -> bool:
    return shutil.which("ncu") is not None


@dataclass(frozen=True)
class NsightRun:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    report_path: Path | None = None
    metrics: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def render(self) -> str:
        lines = [f"$ {' '.join(self.command)}"]
        if not self.ok:
            lines.append(f"ncu exited {self.returncode}")
            if self.stderr.strip():
                lines.append(self.stderr.strip())
            return "\n".join(lines)
        if self.metrics:
            for kernel, values in self.metrics.items():
                lines.append(f"\n{kernel}")
                width = max(len(k) for k in values)
                lines.extend(f"  {k:<{width}}  {v}" for k, v in values.items())
        else:
            lines.append(self.stdout.strip())
        if self.report_path is not None:
            lines.append(f"\nreport: {self.report_path} (open with ncu-ui)")
        return "\n".join(lines)


def build_command(
    target: list[str],
    *,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
    kernel_filter: str | None = None,
    launch_count: int = 1,
    launch_skip: int = 0,
    report_path: str | Path | None = None,
    csv: bool = True,
) -> list[str]:
    """Assemble the ``ncu`` command line that profiles ``target``."""
    unknown = sorted(set(sections) - set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown sections {unknown}; available: {sorted(SECTIONS)}")

    command = ["ncu", "--target-processes", "all"]
    for name in sections:
        command += ["--section", SECTIONS[name]]
    if kernel_filter:
        command += ["--kernel-name", kernel_filter]
    command += ["--launch-count", str(launch_count), "--launch-skip", str(launch_skip)]
    if report_path is not None:
        command += ["--export", str(report_path), "--force-overwrite"]
    if csv:
        command += ["--csv", "--page", "raw"]
    return command + target


def self_command(argv: list[str]) -> list[str]:
    """The command that re-runs this CLI with ``argv`` inside ncu."""
    return [sys.executable, "-m", "kernelforge.cli.main", *argv]


#: The per-launch columns that precede the metrics on the raw page.
_LAUNCH_COLUMNS = frozenset(
    {
        "ID",
        "Process ID",
        "Process Name",
        "Host Name",
        "Kernel Name",
        "Kernel Time",
        "Context",
        "Stream",
    }
)


def _parse_csv(stdout: str) -> dict[str, dict[str, str]]:
    """Parse ``ncu --csv --page raw`` into {launch: {metric: value}}.

    The raw page is a header row naming the launch columns and then every
    collected metric, a row of units, and one row per profiled launch. ncu
    writes its own ``==PROF==`` lines to the same stream. A launch is keyed by
    its ID as well as its kernel's name, because one kernel can run twice.

    Best effort by design: if a future version changes the layout, the caller
    still has the raw text, so a parse failure must not lose the measurement.
    """
    import csv

    lines = [line for line in stdout.splitlines() if line and not line.startswith("==")]
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if start is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        header, units, *launches = csv.reader(lines[start:])
        for launch in launches:
            fields = dict(zip(header, launch, strict=True))
            out[f"{fields['ID']}: {fields['Kernel Name']}"] = {
                column: f"{value} {unit}".strip()
                for column, unit, value in zip(header, units, launch, strict=True)
                if column not in _LAUNCH_COLUMNS
            }
    except (csv.Error, KeyError, ValueError):
        return {}
    return out


def run(
    target: list[str],
    *,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
    kernel_filter: str | None = None,
    launch_count: int = 1,
    launch_skip: int = 0,
    report_path: str | Path | None = None,
    timeout: float = 900.0,
) -> NsightRun:
    """Profile ``target`` under Nsight Compute."""
    if not ncu_available():
        raise RuntimeError(
            "ncu was not found on PATH; install NVIDIA Nsight Compute to collect hardware counters"
        )
    command = build_command(
        target,
        sections=sections,
        kernel_filter=kernel_filter,
        launch_count=launch_count,
        launch_skip=launch_skip,
        report_path=report_path,
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return NsightRun(
        command=tuple(command),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        report_path=Path(report_path) if report_path else None,
        metrics=_parse_csv(result.stdout) if result.returncode == 0 else {},
    )
