"""Guard the CPU-only import path.

The package promises that importing it and running the CPU-side tools needs
neither Triton nor a CUDA context. That promise is one careless top-level
``import triton`` away from breaking, and the breakage is invisible on a
developer machine that has both. These tests make it visible.

Checked in a subprocess, because by the time this file runs the test session
has almost certainly imported Triton already. The whole module list is probed
in one interpreter, since the question is whether *any* of them reaches for a
kernel; the per-module bisect only runs when the aggregate answer is bad.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Modules that must not reach for a kernel at import time.
CPU_SAFE_MODULES = (
    "kernelforge",
    "kernelforge.cli.main",
    "kernelforge.tuning",
    "kernelforge.tuning.search",
    "kernelforge.tuning.tuner",
    "kernelforge.benchmark",
    "kernelforge.benchmark.report",
    "kernelforge.benchmark.workloads",
    "kernelforge.db",
    "kernelforge.testing",
    "kernelforge.runtime.env",
    "kernelforge.runtime.dispatch",
    "kernelforge.integration",
)

_SCRIPT = """
import importlib, sys
for name in {modules!r}:
    importlib.import_module(name)
import torch
print(any(m == "triton" or m.startswith("triton.") for m in sys.modules),
      torch.cuda.is_initialized())
"""


def _probe(modules: tuple[str, ...]) -> tuple[bool, bool]:
    """Import ``modules`` in a fresh interpreter; report Triton and CUDA state."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(modules=list(modules))],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"importing {modules} failed:\n{result.stderr}"
    triton_loaded, cuda_initialised = result.stdout.split()
    return triton_loaded == "True", cuda_initialised == "True"


def _offenders(index: int) -> list[str]:
    """Name the modules responsible, probed one at a time."""
    return [module for module in CPU_SAFE_MODULES if _probe((module,))[index]]


@pytest.fixture(scope="module")
def aggregate() -> tuple[bool, bool]:
    return _probe(CPU_SAFE_MODULES)


def test_cpu_safe_modules_do_not_import_triton(aggregate):
    """The kernel registry is a table of module paths for exactly this reason."""
    triton_loaded, _ = aggregate
    assert not triton_loaded, f"modules that pulled in Triton: {_offenders(0)}"


def test_cpu_safe_modules_do_not_initialise_cuda(aggregate):
    """Importing must not create a CUDA context.

    A context costs hundreds of megabytes of device memory, so a plain
    ``import kernelforge`` would fail on a machine whose GPU is busy.
    """
    _, cuda_initialised = aggregate
    assert not cuda_initialised, f"modules that initialised CUDA: {_offenders(1)}"


def test_console_entry_point_resolves():
    """``kernelforge --version`` must work from the declared entry point."""
    result = subprocess.run(
        [sys.executable, "-m", "kernelforge.cli.main", "--version"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "kernelforge" in result.stdout
    # A runpy warning about the module already being in sys.modules would
    # appear here if kernelforge/cli/__init__.py imported main; the Nsight
    # integration re-invokes the CLI this way.
    assert "RuntimeWarning" not in result.stderr
