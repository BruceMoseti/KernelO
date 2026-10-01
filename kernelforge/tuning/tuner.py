"""The tuner.

    problem -> candidates -> filters -> compile -> verify -> benchmark -> rank

Two properties matter more than anything else here.

**An incorrect configuration is never ranked.** Verification happens before
benchmarking, and a candidate that fails is recorded with its error and
dropped. A tuner that ranks on latency alone will happily pick a kernel whose
masking is broken, because skipping work is fast.

**Verification and benchmarking are separate passes.** All candidates are
compiled and checked first, then the survivors are timed. Interleaving them
would put Triton's compilation of candidate *i+1* in the middle of the
measurement of candidate *i*.

What this is not: a wrapper around ``@triton.autotune``. Triton's autotuner
takes a hand-written config list, times each one once, keeps the winner in
process memory and never checks that any of them is correct. KernelForge
generates the space from hardware properties, filters it with stated reasons,
gates on correctness, measures a distribution rather than a single sample,
persists results for later comparison and caches the winner across processes.
Triton's autotuner is available as a *baseline* so the difference is a
measurement rather than a claim -- see ``triton_autotune`` in
``kernelforge/kernels/matmul.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from kernelforge.benchmark import metrics
from kernelforge.benchmark.runner import (
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUP,
    TimingResult,
    benchmark,
)
from kernelforge.db import Measurement, ResultsDB
from kernelforge.kernels.base import Operator
from kernelforge.runtime.env import Environment, capture_environment, device_caps, require_cuda
from kernelforge.testing import VerificationResult, exact_fp32_matmul, verify
from kernelforge.tuning.cache import ConfigCache
from kernelforge.tuning.config import KernelConfig, Problem

#: Candidate outcomes. Only `ok` candidates are ranked.
STATUS_OK = "ok"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_RUNTIME_ERROR = "runtime_error"
STATUS_INCORRECT = "incorrect"
STATUS_OOM = "out_of_memory"

_MAX_ERROR_CHARS = 400


def _classify(exc: BaseException) -> tuple[str, str]:
    """Map an exception from a candidate onto a status and a short message."""
    name = type(exc).__name__
    message = " ".join(str(exc).split())[:_MAX_ERROR_CHARS]
    if "OutOfMemory" in name:
        return STATUS_OOM, message
    # Triton raises OutOfResources when a config asks for more shared memory or
    # more registers than the device allows -- a compile-time rejection, not a
    # bug. The filters catch most of these; the rest land here.
    if "OutOfResources" in name or "Compil" in name or "Resource" in name:
        return STATUS_COMPILE_ERROR, f"{name}: {message}"
    return STATUS_RUNTIME_ERROR, f"{name}: {message}"


@dataclass(frozen=True)
class CandidateOutcome:
    config: KernelConfig
    status: str
    verification: VerificationResult | None = None
    timing: TimingResult | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def median_ms(self) -> float:
        if self.timing is None:
            raise ValueError(f"candidate {self.config!r} has no timing ({self.status})")
        return self.timing.median_ms


@dataclass
class TuningResult:
    problem: Problem
    environment: Environment
    generated: int
    feasible: int
    outcomes: tuple[CandidateOutcome, ...]
    baselines: dict[str, TimingResult]
    flops: int
    bytes_moved: int
    elapsed_s: float
    memory_bound: bool = False
    rejections: tuple[tuple[KernelConfig, str], ...] = field(default=(), repr=False)

    @property
    def tested(self) -> int:
        """Candidates that were compiled and run."""
        return len(self.outcomes)

    @property
    def correct(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    def ranked(self) -> list[CandidateOutcome]:
        """Correct candidates, fastest first.

        Ranked on the median rather than the minimum: the minimum is the single
        luckiest sample, and picking a configuration by its best-case latency
        rewards variance.
        """
        return sorted((o for o in self.outcomes if o.ok), key=lambda o: o.median_ms)

    @property
    def best(self) -> CandidateOutcome | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    @property
    def best_config(self) -> KernelConfig | None:
        best = self.best
        return best.config if best else None

    def tflops(self, ms: float) -> float:
        return metrics.tflops(self.flops, ms)

    def gbps(self, ms: float) -> float:
        return metrics.gbps(self.bytes_moved, ms)

    def speedup_over(self, label: str) -> float | None:
        best = self.best
        if best is None or label not in self.baselines:
            return None
        return metrics.speedup(self.baselines[label].median_ms, best.median_ms)

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        return counts


class Tuner:
    """Drives the tuning pipeline for any :class:`Operator`."""

    def __init__(
        self,
        *,
        warmup: int = DEFAULT_WARMUP,
        iterations: int = DEFAULT_ITERATIONS,
        flush_l2: bool = True,
        max_candidates: int | None = None,
        measure_baselines: bool = True,
        seed: int = 0,
        db: ResultsDB | None = None,
        cache: ConfigCache | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.warmup = warmup
        self.iterations = iterations
        self.flush_l2 = flush_l2
        self.max_candidates = max_candidates
        self.measure_baselines = measure_baselines
        self.seed = seed
        self.db = db
        self.cache = cache
        self._log = log

    def log(self, message: str) -> None:
        if self._log is not None:
            self._log(message)

    def tune(
        self,
        operator: Operator,
        problem: Problem,
        *,
        device: torch.device | str | None = None,
    ) -> TuningResult:
        started = time.perf_counter()
        resolved = torch.device(device) if device is not None else require_cuda()
        env = capture_environment(resolved)
        caps = device_caps(resolved)

        candidates = operator.search_space().generate(
            problem, caps, max_candidates=self.max_candidates
        )
        inputs = operator.make_inputs(problem, resolved, seed=self.seed)

        # One numerical policy for the whole session. TF32 would otherwise
        # apply to the PyTorch baselines but not to the Triton kernels, which
        # pin fp32 to IEEE, and the comparison would be between precisions
        # instead of between kernels. For fp16 and bf16 this changes nothing.
        with exact_fp32_matmul():
            # Candidates are held elementwise to a float64 reference. The
            # baselines are other implementations, compared with PyTorch's own.
            reference = operator.exact_reference(*inputs)

            self.log(f"Verifying candidates... ({len(candidates)} to check)")
            outcomes = [
                self._verify_candidate(operator, config, inputs, reference, problem)
                for config in candidates
            ]
            passed = [o for o in outcomes if o.ok]
            self.log(f"{len(passed)} / {len(candidates)} configurations passed correctness")

            self.log("Benchmarking...")
            timed = [self._time_candidate(operator, o, inputs, resolved) for o in outcomes]

            baselines = (
                self._measure_baselines(
                    operator, problem, inputs, operator.reference(*inputs), resolved
                )
                if self.measure_baselines
                else {}
            )

        result = TuningResult(
            problem=problem,
            environment=env,
            generated=candidates.generated,
            feasible=candidates.feasible,
            outcomes=tuple(timed),
            baselines=baselines,
            flops=operator.flops(problem),
            bytes_moved=operator.bytes_moved(problem),
            elapsed_s=time.perf_counter() - started,
            memory_bound=operator.is_memory_bound(),
            rejections=candidates.rejected,
        )
        self._persist(result, env)
        return result

    # --- pipeline stages ------------------------------------------------
    def _verify_candidate(
        self,
        operator: Operator,
        config: KernelConfig,
        inputs: tuple[torch.Tensor, ...],
        reference: torch.Tensor,
        problem: Problem,
    ) -> CandidateOutcome:
        try:
            output = operator.run(config, *inputs)
        except Exception as exc:  # a failing candidate must not stop the sweep
            status, message = _classify(exc)
            return CandidateOutcome(config=config, status=status, error=message)
        check = verify(reference, output, dtype=problem.dtype)
        return CandidateOutcome(
            config=config,
            status=STATUS_OK if check.passed else STATUS_INCORRECT,
            verification=check,
            error="" if check.passed else check.reason,
        )

    def _time_candidate(
        self,
        operator: Operator,
        outcome: CandidateOutcome,
        inputs: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> CandidateOutcome:
        if not outcome.ok:
            return outcome
        try:
            timing = benchmark(
                lambda: operator.run(outcome.config, *inputs),
                warmup=self.warmup,
                iterations=self.iterations,
                device=device,
                flush_l2=self.flush_l2,
            )
        except Exception as exc:
            status, message = _classify(exc)
            return CandidateOutcome(
                config=outcome.config,
                status=status,
                verification=outcome.verification,
                error=message,
            )
        return CandidateOutcome(
            config=outcome.config,
            status=STATUS_OK,
            verification=outcome.verification,
            timing=timing,
        )

    def _measure_baselines(
        self,
        operator: Operator,
        problem: Problem,
        inputs: tuple[torch.Tensor, ...],
        reference: torch.Tensor,
        device: torch.device,
    ) -> dict[str, TimingResult]:
        """Time every comparison implementation under identical settings.

        Same warmup, same iteration count, same L2 flush as the candidates;
        otherwise the speedup reported against them is an artefact of the
        harness. Each baseline is also verified, which is how a
        ``torch.compile`` result that silently disagrees would be caught.
        """
        out: dict[str, TimingResult] = {}
        for label, fn in operator.baselines(problem, inputs).items():
            try:
                check = verify(reference, fn(), dtype=problem.dtype)
                if not check.passed:
                    self.log(f"  baseline {label} disagrees with the reference: {check.reason}")
                    continue
                out[label] = benchmark(
                    fn,
                    warmup=self.warmup,
                    iterations=self.iterations,
                    device=device,
                    flush_l2=self.flush_l2,
                )
            except Exception as exc:
                _, message = _classify(exc)
                self.log(f"  baseline {label} unavailable: {message}")
        return out

    def _persist(self, result: TuningResult, env: Environment) -> None:
        if self.cache is not None and result.best is not None:
            self.cache.put(
                result.problem,
                env.device_key,
                result.best.config,
                median_us=result.best.timing.median_us if result.best.timing else None,
                timestamp=env.timestamp,
                candidates_tested=result.tested,
            )
        if self.db is None:
            return

        from kernelforge import __version__

        run_id = self.db.start_run(env, notes="tune", version=__version__)
        measurements = []
        for outcome in result.outcomes:
            ms = outcome.timing.median_ms if outcome.timing else None
            measurements.append(
                Measurement(
                    label="kernelforge",
                    status=outcome.status,
                    config=outcome.config,
                    verification=outcome.verification,
                    timing=outcome.timing,
                    tflops=result.tflops(ms) if ms else None,
                    gbps=result.gbps(ms) if ms else None,
                    error=outcome.error,
                )
            )
        for label, timing in result.baselines.items():
            measurements.append(
                Measurement(
                    label=label,
                    status=STATUS_OK,
                    timing=timing,
                    tflops=result.tflops(timing.median_ms),
                    gbps=result.gbps(timing.median_ms),
                )
            )
        self.db.record_many(run_id, result.problem, measurements)
