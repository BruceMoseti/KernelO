"""Persistent tuned-configuration cache.

Tuning a GEMM shape costs tens of seconds of compiling and benchmarking. A
serving process cannot pay that on every startup, so the winning configuration
is written out keyed by

    device -> Triton version -> operation -> dtype -> shape

and a later run with an identical problem on an identical device reuses it
instead of re-tuning. This is the piece that turns the tuner into something a
runtime can actually sit on top of.

**Why the device is part of the key, not just the architecture.** An RTX 4090
and an RTX 4080 are both ``sm89`` but differ in SM count, L2 size and memory
bandwidth, so the best tile for one is not the best tile for the other. The
key uses the full board name plus the architecture, which means a cache file
copied to a different machine misses rather than silently serving a
configuration tuned for other hardware.

**Why the Triton version is part of the key.** The compiler generates the code
that was measured, so a ranking made under one Triton release does not carry
over to another. After an upgrade the cache misses: dispatch falls back to
the operator default and ``kernelforge tune`` searches again, instead of either
reusing a configuration that was never verified with the new compiler.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.tuning.config import KernelConfig, Problem

CACHE_VERSION = 2


def default_cache_path() -> Path:
    if env := os.environ.get("KERNELFORGE_CACHE"):
        return Path(env)
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "kernelforge" / "configs.json"


@dataclass(frozen=True)
class CacheEntry:
    device_key: str
    triton_version: str
    operation: str
    dtype: str
    shape_key: str
    config: KernelConfig
    median_us: float | None
    timestamp: str
    candidates_tested: int | None = None

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.config.as_dict().items())
        latency = f"{self.median_us:.1f} us" if self.median_us is not None else "unmeasured"
        return (
            f"{self.device_key}  Triton {self.triton_version}  {self.operation:<13} "
            f"{self.dtype:<5} {self.shape_key:<22} {latency:>12}  {params}"
        )


class ConfigCache:
    """A JSON file of best-known configurations, nested device/op/dtype/shape."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_cache_path()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": CACHE_VERSION, "entries": {}}
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"cache file {self.path} is not valid JSON: {exc}") from exc
        version = data.get("version")
        if version != CACHE_VERSION:
            raise RuntimeError(
                f"cache file {self.path} has version {version}, expected {CACHE_VERSION}; "
                "run `kernelforge cache clear` to discard it"
            )
        return data

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written through a temporary file in the same directory so that a
        # crash mid-write cannot leave a truncated cache behind.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(self._data, handle, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(
        self, problem: Problem, device_key: str, triton_version: str | None
    ) -> KernelConfig | None:
        # str(): JSON keys are strings, and None (no Triton) must survive a reload.
        node = (
            self._data["entries"]
            .get(device_key, {})
            .get(str(triton_version), {})
            .get(problem.operation, {})
            .get(problem.dtype_name, {})
            .get(problem.shape_key)
        )
        if node is None:
            return None
        return KernelConfig(problem.operation, **node["config"])

    def put(
        self,
        problem: Problem,
        device_key: str,
        triton_version: str | None,
        config: KernelConfig,
        *,
        median_us: float | None = None,
        timestamp: str = "",
        candidates_tested: int | None = None,
    ) -> None:
        entries = self._data["entries"]
        node = entries.setdefault(device_key, {})
        node = node.setdefault(str(triton_version), {})
        node = node.setdefault(problem.operation, {})
        node = node.setdefault(problem.dtype_name, {})
        node[problem.shape_key] = {
            "config": config.as_dict(),
            "median_us": median_us,
            "timestamp": timestamp,
            "candidates_tested": candidates_tested,
        }
        self._flush()

    def entries(self) -> list[CacheEntry]:
        out: list[CacheEntry] = []
        for device_key, versions in self._data["entries"].items():
            for triton_version, operations in versions.items():
                for operation, dtypes in operations.items():
                    for dtype, shapes in dtypes.items():
                        for shape_key, payload in shapes.items():
                            out.append(
                                CacheEntry(
                                    device_key=device_key,
                                    triton_version=triton_version,
                                    operation=operation,
                                    dtype=dtype,
                                    shape_key=shape_key,
                                    config=KernelConfig(operation, **payload["config"]),
                                    median_us=payload.get("median_us"),
                                    timestamp=payload.get("timestamp", ""),
                                    candidates_tested=payload.get("candidates_tested"),
                                )
                            )
        return sorted(
            out,
            key=lambda e: (e.device_key, e.triton_version, e.operation, e.dtype, e.shape_key),
        )

    def clear(self) -> int:
        removed = len(self.entries())
        self._data = {"version": CACHE_VERSION, "entries": {}}
        if self.path.exists():
            self.path.unlink()
        return removed

    def __len__(self) -> int:
        return len(self.entries())
