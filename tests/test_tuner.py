"""Tuner pipeline tests.

On CPU, candidates really run (Triton interpreter) and are verified, but there is no GPU timer.
A stand-in timer returns synthetic latencies chosen per config, so that ranking, caching and
rejection can be checked deterministically. Synthetic latencies exist only inside these tests
and their temporary databases.
"""

import dataclasses
import functools
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from kernelforge.benchmark.metrics import summarize
from kernelforge.benchmark.runner import BenchmarkResult, benchmark
from kernelforge.benchmark.workloads import MATMUL, matmul_problem
from kernelforge.runtime.environment import Environment, collect_environment
from kernelforge.tuning.config import KernelConfig, Problem
from kernelforge.tuning.database import TuningDatabase
from kernelforge.tuning.search import DeviceLimits, Rule, SearchSpace, device_limits
from kernelforge.tuning.tuner import NoCorrectCandidateError, Timer, TuneResult, tune

SYNTHETIC_GPU = Environment(
    gpu="Synthetic test GPU",
    compute_capability="0.0",
    sm_count=4,
    gpu_memory_bytes=None,
    l2_cache_bytes=None,
    driver=None,
    cuda=None,
    torch=torch.__version__,
    triton=triton.__version__,
    python="test",
    platform="test",
    cpu="test",
    torch_matmul_allow_tf32=False,
    torch_matmul_fp16_reduced_precision=True,
    torch_matmul_bf16_reduced_precision=True,
)
LIMITS = DeviceLimits(sm_count=4, max_shared_memory_bytes=1 << 20)
PROBLEM = matmul_problem(48, 40, 80, torch.float32)


def _config(block_m: int, block_n: int, block_k: int) -> KernelConfig:
    return KernelConfig(
        "matmul", {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}, 2, 2
    )


class SmallMatmul:
    """The real MatMul workload on an 8-config space, with optional sabotage of chosen configs."""

    search_space = SearchSpace(
        kernel="matmul",
        params={"BLOCK_M": (16, 32), "BLOCK_N": (16, 32), "BLOCK_K": (16, 32)},
        num_warps=(2,),
        num_stages=(2,),
        dtypes=(torch.float32,),
        rules=(),
    )

    def __init__(self, sabotage: dict[KernelConfig, str] | None = None) -> None:
        self.sabotage = sabotage or {}
        self.last_config: KernelConfig | None = None
        self.timed: list[KernelConfig] = []

    def make_inputs(self, problem: Problem, device: torch.device) -> tuple[torch.Tensor, ...]:
        return MATMUL.make_inputs(problem, device)

    def reference(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return MATMUL.reference(inputs)

    def run(self, config: KernelConfig, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.last_config = config
        if self.sabotage.get(config) == "raise":
            raise RuntimeError("simulated compilation failure")
        out = MATMUL.run(config, inputs)
        if self.sabotage.get(config) == "wrong":
            out[0, 0] += 1
        return out

    def flops(self, problem: Problem) -> int:
        return MATMUL.flops(problem)

    def bytes_moved(self, problem: Problem) -> int:
        return MATMUL.bytes_moved(problem)

    def timer(self, latencies_us: dict[KernelConfig, float]) -> Timer:
        def synthetic_timer(fn: object) -> BenchmarkResult:
            assert callable(fn)
            fn()
            assert self.last_config is not None
            self.timed.append(self.last_config)
            latency = latencies_us.get(self.last_config, 100.0)
            return BenchmarkResult(
                stats=summarize([latency, latency]),
                samples_us=(latency, latency),
                warmup=1,
                iterations=2,
                l2_flush_bytes=0,
                environment=SYNTHETIC_GPU,
            )

        return synthetic_timer


def _tune(
    workload: SmallMatmul,
    database: TuningDatabase,
    device: torch.device,
    latencies_us: dict[KernelConfig, float] | None = None,
    environment: Environment = SYNTHETIC_GPU,
    retune: bool = False,
) -> TuneResult:
    return tune(
        workload,
        PROBLEM,
        database,
        device=device,
        limits=LIMITS,
        environment=environment,
        timer=workload.timer(latencies_us or {}),
        retune=retune,
    )


def test_wrong_candidate_is_rejected_and_never_ranked(tmp_path: Path, device: torch.device) -> None:
    wrong = _config(16, 16, 16)
    workload = SmallMatmul(sabotage={wrong: "wrong"})
    with TuningDatabase(tmp_path / "tuning.db") as database:
        # The wrong candidate would win if it were ever timed and ranked.
        result = _tune(workload, database, device, latencies_us={wrong: 1.0})
        assert result.best != wrong
        assert wrong not in workload.timed
        rejected = next(m for m in result.measurements if m.config == wrong)
        assert not rejected.correct
        assert rejected.benchmark is None
        assert rejected.error is not None and "outside tolerance" in rejected.error
        rows = database.execute(
            "SELECT correct, median_us FROM results JOIN configs USING (config_id)"
            " WHERE params_json = ?",
            (wrong.params_json(),),
        )
        assert rows == [(0, None)]


def test_failing_candidate_is_recorded_with_its_error(tmp_path: Path, device: torch.device) -> None:
    broken = _config(32, 32, 32)
    workload = SmallMatmul(sabotage={broken: "raise"})
    with TuningDatabase(tmp_path / "tuning.db") as database:
        result = _tune(workload, database, device, latencies_us={broken: 1.0})
        assert result.best != broken
        failed = next(m for m in result.measurements if m.config == broken)
        assert failed.error == "RuntimeError: simulated compilation failure"
        assert database.execute(
            "SELECT count(*) FROM results WHERE error LIKE 'RuntimeError%'"
        ) == [(1,)]


def test_every_candidate_is_verified_before_any_is_timed(
    tmp_path: Path, device: torch.device, caplog: pytest.LogCaptureFixture
) -> None:
    order: list[str] = []
    workload = SmallMatmul(sabotage={_config(16, 16, 16): "wrong"})
    run, timer = workload.run, workload.timer({})

    def recording_run(config: KernelConfig, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        order.append("run")
        return run(config, inputs)

    def recording_timer(fn: object) -> BenchmarkResult:
        order.append("time")
        return timer(fn)

    workload.run = recording_run  # type: ignore[method-assign]
    caplog.set_level("INFO", logger="kernelforge.tuning.tuner")
    with TuningDatabase(tmp_path / "tuning.db") as database:
        tune(
            workload,
            PROBLEM,
            database,
            device=device,
            limits=LIMITS,
            environment=SYNTHETIC_GPU,
            timer=recording_timer,
        )
    first_timing = order.index("time")
    assert order[:first_timing] == ["run"] * 8
    assert "7 / 8 configurations passed correctness" in caplog.messages


def test_best_is_the_fastest_correct_candidate(tmp_path: Path, device: torch.device) -> None:
    fastest = _config(32, 16, 32)
    workload = SmallMatmul()
    with TuningDatabase(tmp_path / "tuning.db") as database:
        result = _tune(workload, database, device, latencies_us={fastest: 5.0})
        assert result.best == fastest
        assert result.best_median_us == 5.0
        assert not result.from_cache
        assert len(result.measurements) == 8 and all(m.correct for m in result.measurements)
        assert database.execute("SELECT count(*), sum(correct) FROM results") == [(8, 8)]


def test_second_tune_is_a_cache_hit(tmp_path: Path, device: torch.device) -> None:
    fastest = _config(16, 32, 16)
    workload = SmallMatmul()
    with TuningDatabase(tmp_path / "tuning.db") as database:
        first = _tune(workload, database, device, latencies_us={fastest: 5.0})
        timed_after_first = len(workload.timed)
        second = _tune(workload, database, device)
        assert second.from_cache
        assert second.best == first.best == fastest
        assert second.run_id == first.run_id
        assert len(workload.timed) == timed_after_first
        assert database.execute("SELECT count(*) FROM runs") == [(1,)]


def test_retune_measures_again_and_replaces_the_cache_entry(
    tmp_path: Path, device: torch.device
) -> None:
    workload = SmallMatmul()
    with TuningDatabase(tmp_path / "tuning.db") as database:
        _tune(workload, database, device, latencies_us={_config(16, 16, 16): 5.0})
        result = _tune(
            workload, database, device, latencies_us={_config(32, 32, 32): 5.0}, retune=True
        )
        assert not result.from_cache
        assert result.best == _config(32, 32, 32)
        assert [entry.config for entry in database.cache_entries()] == [_config(32, 32, 32)]


@pytest.mark.parametrize(
    "change", [{"gpu": "Another GPU"}, {"compute_capability": "1.0"}, {"triton": "0.0.0"}]
)
def test_cache_key_is_hardware_and_compiler_aware(
    tmp_path: Path, device: torch.device, change: dict[str, str]
) -> None:
    workload = SmallMatmul()
    with TuningDatabase(tmp_path / "tuning.db") as database:
        _tune(workload, database, device)
        other = _tune(
            workload, database, device, environment=dataclasses.replace(SYNTHETIC_GPU, **change)
        )
        assert not other.from_cache
        assert len(database.cache_entries()) == 2


def test_nothing_is_cached_when_every_candidate_is_wrong(
    tmp_path: Path, device: torch.device
) -> None:
    workload = SmallMatmul(sabotage={c: "wrong" for c in SmallMatmul.search_space.grid()})
    with TuningDatabase(tmp_path / "tuning.db") as database:
        with pytest.raises(NoCorrectCandidateError, match="none of the 8 candidates"):
            _tune(workload, database, device)
        assert workload.timed == []
        assert database.cache_entries() == []
        assert database.execute("SELECT count(*), sum(correct) FROM results") == [(8, 0)]


@triton.jit
def _double_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) * 2, mask=mask)


class DoubleWorkload:
    """A non-MatMul kernel with a single BLOCK_SIZE parameter and its own pruning rule."""

    search_space = SearchSpace(
        kernel="double",
        params={"BLOCK_SIZE": (64, 128, 256)},
        num_warps=(1, 2),
        num_stages=(1,),
        dtypes=(torch.float32,),
        rules=(
            Rule(
                "block_exceeds_input",
                "a block larger than the input only adds masked lanes",
                lambda config, problem, device: config.params["BLOCK_SIZE"] > problem.shape["n"],
            ),
        ),
    )

    def make_inputs(self, problem: Problem, device: torch.device) -> tuple[torch.Tensor, ...]:
        return (torch.randn(problem.shape["n"], device=device),)

    def reference(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return inputs[0].double() * 2

    def run(self, config: KernelConfig, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        (x,) = inputs
        out = torch.empty_like(x)
        block = config.params["BLOCK_SIZE"]
        grid = (triton.cdiv(x.numel(), block),)
        _double_kernel[grid](x, out, x.numel(), BLOCK_SIZE=block, num_warps=config.num_warps)
        return out

    def flops(self, problem: Problem) -> int:
        return problem.shape["n"]

    def bytes_moved(self, problem: Problem) -> int:
        return 2 * 4 * problem.shape["n"]


def test_the_same_tuner_handles_a_kernel_without_matmul_parameters(
    tmp_path: Path, device: torch.device
) -> None:
    workload = DoubleWorkload()

    def synthetic_timer(fn: object) -> BenchmarkResult:
        return BenchmarkResult(summarize([3.0, 3.0]), (3.0, 3.0), 1, 2, 0, SYNTHETIC_GPU)

    problem = Problem("double", {"n": 200}, torch.float32)
    with TuningDatabase(tmp_path / "tuning.db") as database:
        result = tune(
            workload,
            problem,
            database,
            device=device,
            limits=LIMITS,
            environment=SYNTHETIC_GPU,
            timer=synthetic_timer,
        )
        assert result.candidates is not None
        assert result.candidates.pruned == {"block_exceeds_input": 2}
        assert all(m.correct for m in result.measurements)
        assert result.best.params["BLOCK_SIZE"] in (64, 128)
        stored = database.execute(
            "SELECT DISTINCT json_extract(params_json, '$.BLOCK_SIZE') FROM configs"
        )
        assert sorted(value for (value,) in stored) == [64, 128]


@pytest.mark.gpu
def test_tunes_matmul_on_the_gpu(tmp_path: Path) -> None:
    problem = matmul_problem(1024, 1024, 1024, torch.float16)
    gpu_tune = functools.partial(
        tune,
        MATMUL,
        problem,
        device=torch.device("cuda"),
        limits=device_limits(),
        environment=collect_environment(),
        timer=functools.partial(benchmark, warmup=5, iterations=20),
    )
    with TuningDatabase(tmp_path / "tuning.db") as database:
        first = gpu_tune(database)
        assert not first.from_cache
        assert first.candidates is not None and first.best in first.candidates.configs
        assert all(m.correct for m in first.measurements), [m.error for m in first.measurements]
        second = gpu_tune(database)
        assert second.from_cache and second.best == first.best
