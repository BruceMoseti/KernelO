"""Operator-level profiling with ``torch.profiler``.

This answers the question the latency harness cannot: *how many kernels ran,
and where did the device time go*. For the fusion comparison that is the whole
measurement -- the unfused sequence launches three kernels and writes the
``M x N`` intermediate twice more than the fused one, and the launch count is
the direct evidence.

Two notes on methodology, both of which the PyTorch profiler documentation
calls out:

* CUDA is warmed up before the profiled region. The first launch of a kernel
  pays for context setup and module loading, which otherwise lands entirely on
  the first measured iteration.
* Tracing is not free. Latency numbers come from the CUDA-event harness in
  ``kernelforge.benchmark.runner``; the figures here are for attribution --
  counts, and device time split by kernel -- not for headline latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.autograd import DeviceType


def _device_time_us(event) -> float:
    """Total device time for an event, across PyTorch naming generations."""
    for attribute in ("device_time_total", "cuda_time_total"):
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value)
    return 0.0


@dataclass(frozen=True)
class KernelRecord:
    name: str
    launches: int
    device_time_us: float


@dataclass(frozen=True)
class ProfileResult:
    label: str
    iterations: int
    kernels: tuple[KernelRecord, ...]
    trace_path: Path | None = None

    @property
    def launches_per_iteration(self) -> float:
        """Kernel launches per call of the profiled function.

        The number that distinguishes a fused implementation from an unfused
        one, independently of how fast either is.
        """
        total = sum(k.launches for k in self.kernels)
        return total / self.iterations if self.iterations else 0.0

    @property
    def device_time_per_iteration_us(self) -> float:
        total = sum(k.device_time_us for k in self.kernels)
        return total / self.iterations if self.iterations else 0.0

    def render(self, limit: int = 10) -> str:
        lines = [
            f"{self.label}",
            f"  kernel launches per call : {self.launches_per_iteration:.2f}",
            f"  device time per call     : {self.device_time_per_iteration_us:.1f} us",
        ]
        if self.kernels:
            lines.append("  kernels by device time:")
            width = min(62, max(len(k.name) for k in self.kernels))
            for kernel in self.kernels[:limit]:
                share = kernel.device_time_us / max(
                    sum(k.device_time_us for k in self.kernels), 1e-9
                )
                lines.append(
                    f"    {kernel.name[:width]:<{width}}  "
                    f"{kernel.launches / max(self.iterations, 1):>5.1f} launches  "
                    f"{kernel.device_time_us / max(self.iterations, 1):>9.1f} us  "
                    f"{share * 100:>5.1f}%"
                )
        if self.trace_path is not None:
            lines.append(f"  chrome trace: {self.trace_path}")
        return "\n".join(lines)


def profile_kernels(
    fn,
    *,
    label: str,
    warmup: int = 10,
    iterations: int = 20,
    trace_path: str | Path | None = None,
) -> ProfileResult:
    """Profile ``fn`` and attribute device time to individual CUDA kernels."""
    if not torch.cuda.is_available():
        raise RuntimeError("profiling KernelForge kernels requires a CUDA device")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()

    kernels = [
        KernelRecord(
            name=event.key,
            launches=int(event.count),
            device_time_us=_device_time_us(event),
        )
        for event in profiler.key_averages()
        if event.device_type == DeviceType.CUDA and _device_time_us(event) > 0
    ]
    kernels.sort(key=lambda k: -k.device_time_us)

    resolved_trace = None
    if trace_path is not None:
        resolved_trace = Path(trace_path)
        resolved_trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(resolved_trace))

    return ProfileResult(
        label=label,
        iterations=iterations,
        kernels=tuple(kernels),
        trace_path=resolved_trace,
    )


def compare_launch_counts(
    candidates: dict[str, object], *, warmup: int = 10, iterations: int = 20
) -> list[ProfileResult]:
    """Profile several implementations of the same operation.

    Used by ``kernelforge profile fused_linear``, where the interesting output
    is the launch count of the unfused sequence against the fused kernel.
    """
    return [
        profile_kernels(fn, label=label, warmup=warmup, iterations=iterations)
        for label, fn in candidates.items()
    ]
