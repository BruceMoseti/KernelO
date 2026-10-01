"""The generic tuner: generate, prune, verify, benchmark, record, rank, cache.

For a problem, the tuner takes the candidates that survive the kernel's search-space rules, runs
each one, and checks its output against a float64 reference. Candidates that fail to compile,
fail at run time, or produce wrong results are recorded with the reason and never benchmarked or
ranked. Correct candidates are benchmarked. Every outcome is stored in the database, and the
correct candidate with the lowest median latency is cached under a hardware-aware key. A later
call for the same problem on the same GPU and Triton version returns the cached config without
retuning.

The tuner never reads parameter names. It works for any kernel that provides a `Tunable`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import torch

from kernelforge.benchmark.metrics import gbps, tflops
from kernelforge.benchmark.runner import BenchmarkResult
from kernelforge.runtime.environment import Environment
from kernelforge.testing import verify
from kernelforge.tuning.cache import cache_key
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.database import TuningDatabase
from kernelforge.tuning.search import Candidates, DeviceLimits, SearchSpace

Timer = Callable[[Callable[[], object]], BenchmarkResult]


class Tunable(Protocol):
    """What the tuner needs from a kernel."""

    @property
    def search_space(self) -> SearchSpace: ...

    def make_inputs(self, problem: Problem, device: torch.device) -> tuple[torch.Tensor, ...]: ...

    def reference(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """The float64 result for these inputs."""
        ...

    def run(self, config: KernelConfig, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor: ...

    def flops(self, problem: Problem) -> int | None: ...

    def bytes_moved(self, problem: Problem) -> int: ...


@dataclass(frozen=True)
class Measurement:
    """Outcome of one implementation on one problem: rejected with a reason, or timed."""

    implementation: str
    config: KernelConfig | None
    correct: bool
    error: str | None = None
    max_error_ratio: float | None = None
    benchmark: BenchmarkResult | None = None
    tflops: float | None = None
    gbps: float | None = None

    @property
    def median_us(self) -> float:
        if self.benchmark is None:
            raise ValueError(f"{self.implementation} was not benchmarked: {self.error}")
        return self.benchmark.stats.median_us


def measure(
    implementation: str,
    fn: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    dtype: torch.dtype,
    timer: Timer,
    *,
    config: KernelConfig | None = None,
    flops: int | None = None,
    num_bytes: int | None = None,
) -> Measurement:
    """Run `fn` once and verify its output; benchmark it only if the output is correct."""
    try:
        verification = verify(reference, fn(), dtype)
    except Exception as error:  # compile errors, launch failures, resource limits
        return Measurement(implementation, config, False, error=f"{type(error).__name__}: {error}")
    if not verification.passed:
        return Measurement(
            implementation,
            config,
            False,
            error=verification.describe(),
            max_error_ratio=verification.max_error_ratio,
        )
    result = timer(fn)
    median_us = result.stats.median_us
    return Measurement(
        implementation,
        config,
        True,
        max_error_ratio=verification.max_error_ratio,
        benchmark=result,
        tflops=None if flops is None else tflops(flops, median_us),
        gbps=None if num_bytes is None else gbps(num_bytes, median_us),
    )


@dataclass(frozen=True)
class TuneResult:
    problem: Problem
    best: KernelConfig
    best_median_us: float
    run_id: int  # the run that measured `best`
    from_cache: bool
    candidates: Candidates | None = None  # None on a cache hit
    measurements: tuple[Measurement, ...] = ()  # empty on a cache hit


class NoCorrectCandidateError(RuntimeError):
    pass


def tune(
    tunable: Tunable,
    problem: Problem,
    database: TuningDatabase,
    *,
    device: torch.device,
    limits: DeviceLimits,
    environment: Environment,
    timer: Timer,
    retune: bool = False,
) -> TuneResult:
    """Find the fastest correct config for `problem`, or return the cached one."""
    key = cache_key(problem, environment)
    entry = None if retune else database.cached(key)
    if entry is not None:
        return TuneResult(problem, entry.config, entry.median_us, entry.run_id, from_cache=True)

    candidates = tunable.search_space.candidates(problem, limits)
    inputs = tunable.make_inputs(problem, device)
    reference = tunable.reference(inputs)
    flops, num_bytes = tunable.flops(problem), tunable.bytes_moved(problem)
    run_id = database.start_run(environment)
    measurements = []
    for config in candidates.configs:
        measurement = measure(
            "kernelforge",
            partial(tunable.run, config, inputs),
            reference,
            problem.dtype,
            timer,
            config=config,
            flops=flops,
            num_bytes=num_bytes,
        )
        database.record(run_id, problem, measurement)
        measurements.append(measurement)

    correct = [m for m in measurements if m.correct]
    if not correct:
        raise NoCorrectCandidateError(
            f"none of the {len(measurements)} candidates for {problem} passed correctness"
        )
    best = min(correct, key=lambda m: m.median_us)
    assert best.config is not None
    database.store_cache(key, best.config, best.median_us, run_id)
    return TuneResult(
        problem,
        best.config,
        best.median_us,
        run_id,
        from_cache=False,
        candidates=candidates,
        measurements=tuple(measurements),
    )
