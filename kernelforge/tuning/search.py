"""Candidate generation.

A naive GEMM grid over the parameters KernelForge tunes is 432 points, and
compiling plus verifying plus benchmarking all of them costs minutes per
shape for configurations that cannot win. Candidate generation therefore has
three stages:

1. **Enumerate** the full grid.
2. **Reject** configurations that are infeasible on this device or
   indefensible for this problem, each with a reason that the CLI can print.
   These rules are derived from hardware limits (shared memory, the 255
   architectural registers per thread, 128-byte cache lines), not from
   measurements, so they transfer across GPUs.
3. **Budget** the survivors: sort by a documented priority and keep the top
   ``max_candidates``. The budget bounds tuning cost without narrowing the
   rules to the point where they might exclude the true optimum.

Every stage count is reported, so a shrinking search space is visible rather
than silent.

This module imports neither Triton nor CUDA: the filters read a
:class:`~kernelforge.runtime.env.DeviceCaps` value object, which makes them
unit testable against the properties of a GPU that is not attached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from itertools import product

from kernelforge.dtypes import itemsize
from kernelforge.runtime.env import DeviceCaps, device_caps
from kernelforge.tuning.config import KernelConfig, Problem

#: Default candidate budget per problem.
DEFAULT_MAX_CANDIDATES = 48

#: Hardware limit: a thread can address 255 registers before spilling to
#: local memory. The accumulator is only part of a thread's register demand
#: (addresses, pipeline buffers and the epilogue need their share), so the
#: accumulator alone is capped well below the architectural limit.
MAX_ACC_REGS_PER_THREAD = 128

#: Below this, the prologue, index arithmetic and epilogue dominate the inner
#: loop and the configuration cannot be competitive.
MIN_ACC_REGS_PER_THREAD = 4

#: A global load is issued in 128-byte transactions. A tile row shorter than
#: half a transaction wastes most of every line it touches, because successive
#: tile rows are strided by the full matrix dimension and cannot be coalesced
#: into the same line.
MIN_CONTIGUOUS_TILE_BYTES = 64


def next_power_of_2(n: int) -> int:
    if n < 1:
        raise ValueError(f"expected a positive integer, got {n}")
    return 1 << (n - 1).bit_length()


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class SearchSpace(ABC):
    """The set of configurations worth trying for one operation."""

    operation: str

    def __init__(self, max_candidates: int = DEFAULT_MAX_CANDIDATES) -> None:
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be positive, got {max_candidates}")
        self.max_candidates = max_candidates

    @abstractmethod
    def grid(self, problem: Problem) -> Iterator[KernelConfig]:
        """Every point in the unfiltered parameter grid."""

    @abstractmethod
    def reject_reason(self, config: KernelConfig, problem: Problem, caps: DeviceCaps) -> str | None:
        """Why ``config`` is not worth trying, or ``None`` to keep it."""

    def num_programs(self, config: KernelConfig, problem: Problem) -> int:
        """Size of the launch grid, used by the utilisation prune."""
        raise NotImplementedError

    def priority(
        self, config: KernelConfig, problem: Problem, caps: DeviceCaps
    ) -> tuple[float, ...]:
        """Sort key for the budget; lower sorts first.

        The final element is always the serialised config so that ties resolve
        deterministically and a given problem always yields the same
        candidates.
        """
        return (0.0,)

    def candidates(
        self,
        problem: Problem,
        caps: DeviceCaps | None = None,
        *,
        max_candidates: int | None = None,
    ) -> list[KernelConfig]:
        return self.generate(problem, caps, max_candidates=max_candidates).candidates

    def generate(
        self,
        problem: Problem,
        caps: DeviceCaps | None = None,
        *,
        max_candidates: int | None = None,
    ) -> CandidateSet:
        if problem.operation != self.operation:
            raise ValueError(
                f"{type(self).__name__} handles {self.operation!r}, got {problem.operation!r}"
            )
        caps = caps if caps is not None else device_caps()
        budget = max_candidates if max_candidates is not None else self.max_candidates

        rejected: list[tuple[KernelConfig, str]] = []
        kept: list[KernelConfig] = []
        generated = 0
        for config in self.grid(problem):
            generated += 1
            reason = self.reject_reason(config, problem, caps)
            if reason is None:
                kept.append(config)
            else:
                rejected.append((config, reason))

        feasible = len(kept)
        kept, pruned = self._prune_underutilised(kept, problem, caps)
        rejected.extend(pruned)

        kept.sort(key=lambda c: (*self.priority(c, problem, caps), c.to_json()))
        return CandidateSet(
            problem=problem,
            generated=generated,
            feasible=feasible,
            after_prune=len(kept),
            budget=budget,
            candidates=kept[:budget],
            rejected=tuple(rejected),
        )

    def _prune_underutilised(
        self, configs: list[KernelConfig], problem: Problem, caps: DeviceCaps
    ) -> tuple[list[KernelConfig], list[tuple[KernelConfig, str]]]:
        """Drop configurations that cannot fill the GPU -- unless none can.

        A launch grid smaller than the SM count leaves multiprocessors idle for
        the whole kernel. The rule is conditional because for a small enough
        problem *no* tiling fills the machine, and in that case the least-bad
        option still has to be measured.
        """
        try:
            programs = {c: self.num_programs(c, problem) for c in configs}
        except NotImplementedError:
            return configs, []
        if not programs or max(programs.values()) < caps.sm_count:
            return configs, []
        floor = caps.sm_count // 2
        kept = [c for c in configs if programs[c] >= floor]
        dropped = [
            (c, f"launch grid of {programs[c]} programs cannot fill {caps.sm_count} SMs")
            for c in configs
            if programs[c] < floor
        ]
        return kept, dropped

    def explain(self, problem: Problem, caps: DeviceCaps | None = None) -> str:
        """Human-readable breakdown of the filtering, for `--explain`."""
        result = self.generate(problem, caps)
        lines = [
            f"{problem.describe()}",
            f"  grid points      : {result.generated}",
            f"  feasible         : {result.feasible}",
            f"  after prune      : {result.after_prune}",
            f"  selected (budget): {len(result.candidates)} of {result.budget}",
        ]
        reasons: dict[str, int] = {}
        for _, reason in result.rejected:
            key = reason.split(":")[0].split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            lines.append("  rejections by rule:")
            lines.extend(
                f"    {count:>4}  {reason}"
                for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])
            )
        return "\n".join(lines)


class CandidateSet:
    """Candidates plus the counts at each stage of generation."""

    __slots__ = (
        "problem",
        "generated",
        "feasible",
        "after_prune",
        "budget",
        "candidates",
        "rejected",
    )

    def __init__(
        self,
        *,
        problem: Problem,
        generated: int,
        feasible: int,
        after_prune: int,
        budget: int,
        candidates: list[KernelConfig],
        rejected: tuple[tuple[KernelConfig, str], ...],
    ) -> None:
        self.problem = problem
        self.generated = generated
        self.feasible = feasible
        self.after_prune = after_prune
        self.budget = budget
        self.candidates = candidates
        self.rejected = rejected

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[KernelConfig]:
        return iter(self.candidates)


class _BlockedGemmSpace(SearchSpace):
    """Shared search space for the blocked-GEMM kernels.

    ``matmul`` and ``fused_linear`` have the same inner loop and therefore the
    same tiling tradeoffs; only the epilogue differs. They share this space
    rather than duplicating the filters.
    """

    BLOCK_M = (16, 32, 64, 128)
    BLOCK_N = (16, 32, 64, 128)
    BLOCK_K = (16, 32, 64)
    NUM_WARPS = (2, 4, 8)
    NUM_STAGES = (2, 3, 4)
    GROUP_M = 8

    #: Smallest output tile worth launching a program for.
    MIN_TILE_ELEMENTS = 1024

    def grid(self, problem: Problem) -> Iterator[KernelConfig]:
        for bm, bn, bk, warps, stages in product(
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K, self.NUM_WARPS, self.NUM_STAGES
        ):
            yield KernelConfig(
                self.operation,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_M=self.GROUP_M,
                num_warps=warps,
                num_stages=stages,
            )

    def shared_memory_bytes(self, config: KernelConfig, problem: Problem) -> int:
        """Shared memory the software pipeline needs for the A and B tiles.

        Triton's pipeliner keeps ``num_stages`` in-flight copies of both
        operand tiles so that the global loads for iteration *i+1* overlap the
        ``tl.dot`` of iteration *i*, hence the ``num_stages`` factor.

        How much of this is checked, and where:

        * ``tests/test_triton_compile.py`` compiles each candidate for sm80 and
          sm90 on a CPU-only machine and checks the single-buffer tile
          footprint, ``BLOCK_K * (BLOCK_M + BLOCK_N) * itemsize``, against what
          the compiler allocates. It cannot check the ``num_stages`` factor:
          the standalone ``triton.compile`` entry point does not run the
          software pipeliner, so its allocation stays at one buffer per operand
          whatever ``num_stages`` says.
        * ``tests/test_kernels_gpu.py::test_shared_memory_model`` checks the
          full figure against a real JIT launch, where the pipeliner does run.

        The estimate is therefore an upper bound used to reject configurations
        before compiling them. Anything it lets through that the compiler then
        rejects comes back as Triton's own ``OutOfResources`` and is recorded
        as a compile failure, so an inaccurate model costs tuning time rather
        than correctness.
        """
        elems = config["num_stages"] * config["BLOCK_K"] * (config["BLOCK_M"] + config["BLOCK_N"])
        return elems * itemsize(problem.dtype)

    def acc_regs_per_thread(self, config: KernelConfig) -> float:
        """fp32 accumulator registers each thread holds."""
        threads = config["num_warps"] * 32
        return config["BLOCK_M"] * config["BLOCK_N"] / threads

    def num_programs(self, config: KernelConfig, problem: Problem) -> int:
        dims = problem.dims_dict
        return ceil_div(dims["M"], config["BLOCK_M"]) * ceil_div(dims["N"], config["BLOCK_N"])

    def reject_reason(self, config: KernelConfig, problem: Problem, caps: DeviceCaps) -> str | None:
        dims = problem.dims_dict
        m, n, k = dims["M"], dims["N"], dims["K"]
        bm, bn, bk = config["BLOCK_M"], config["BLOCK_N"], config["BLOCK_K"]
        width = itemsize(problem.dtype)

        smem = self.shared_memory_bytes(config, problem)
        if smem > caps.max_shared_memory_per_block:
            return (
                f"shared memory: {smem // 1024} KiB of tiles exceeds the "
                f"{caps.max_shared_memory_per_block // 1024} KiB per-block limit"
            )

        if bm * bn < self.MIN_TILE_ELEMENTS:
            return f"tile too small: {bm}x{bn} output elements per program"

        acc = self.acc_regs_per_thread(config)
        if acc > MAX_ACC_REGS_PER_THREAD:
            return f"register pressure: {acc:.0f} accumulator registers per thread"
        if acc < MIN_ACC_REGS_PER_THREAD:
            return f"too little work per thread: {acc:.1f} accumulator registers"

        # Each warp of the MMA pipeline needs at least one 16x16 output tile to
        # own; below that, warps either idle or split the K loop.
        if bm * bn < 256 * config["num_warps"]:
            return f"too little output per warp at num_warps={config['num_warps']}"

        if bk * width < MIN_CONTIGUOUS_TILE_BYTES:
            return f"uncoalesced A tile: BLOCK_K={bk} is {bk * width} contiguous bytes"
        if bn * width < MIN_CONTIGUOUS_TILE_BYTES:
            return f"uncoalesced B tile: BLOCK_N={bn} is {bn * width} contiguous bytes"

        # A tile larger than the problem computes masked-off work that is
        # thrown away. Allowing one doubling keeps useful tilings for shapes
        # that are just over a tile boundary.
        if bm > 2 * m:
            return f"tile overshoot: BLOCK_M={bm} for M={m}"
        if bn > 2 * n:
            return f"tile overshoot: BLOCK_N={bn} for N={n}"

        k_iters = ceil_div(k, bk)
        if config["num_stages"] > 2 and k_iters < config["num_stages"]:
            return (
                f"pipeline cannot fill: {k_iters} K iterations for "
                f"num_stages={config['num_stages']}"
            )
        return None

    def priority(
        self, config: KernelConfig, problem: Problem, caps: DeviceCaps
    ) -> tuple[float, ...]:
        """Rank survivors for the budget.

        1. At least one full wave of programs. A grid smaller than the SM
           count leaves hardware idle for the whole kernel.
        2. Data reuse per byte of tile loaded, ``BM*BN / (BM+BN)``: a tile
           loads ``BK*(BM+BN)`` elements to produce ``BM*BN`` outputs, so this
           ratio is the arithmetic intensity of the inner loop.
        3. Least wasted work from padding the problem up to whole tiles.
        """
        dims = problem.dims_dict
        bm, bn = config["BLOCK_M"], config["BLOCK_N"]
        programs = self.num_programs(config, problem)
        full_wave = 0 if programs >= caps.sm_count else 1
        reuse = (bm * bn) / (bm + bn)
        padded = ceil_div(dims["M"], bm) * bm * ceil_div(dims["N"], bn) * bn
        waste = padded / (dims["M"] * dims["N"])
        return (full_wave, -reuse, waste)


class MatmulSearchSpace(_BlockedGemmSpace):
    operation = "matmul"


class FusedLinearSearchSpace(_BlockedGemmSpace):
    operation = "fused_linear"


class _RowReductionSpace(SearchSpace):
    """Shared search space for the single-pass row-wise kernels.

    ``softmax`` and ``rmsnorm`` both reduce along the row and then rescale it,
    so both are tuned over the same parameters. Note how much smaller this
    space is than the GEMM one: ``BLOCK_SIZE`` is *determined* by the row
    width, because a single-pass kernel has to hold the whole row, which
    leaves only the thread count and the number of rows per program free. The
    tuner does not assume otherwise -- that is the point of letting each
    operator own its space.
    """

    NUM_WARPS = (1, 2, 4, 8, 16)
    ROWS_PER_PROGRAM = (1, 2, 4, 8)

    #: A row wider than this cannot be held in registers by a single-pass
    #: kernel; see the guard in the kernel wrappers.
    MAX_BLOCK_SIZE = 65536

    #: Elements each thread holds in registers during the reduction.
    MIN_ELEMS_PER_THREAD = 1
    MAX_ELEMS_PER_THREAD = 64

    def grid(self, problem: Problem) -> Iterator[KernelConfig]:
        block = next_power_of_2(problem.dims_dict["cols"])
        for warps, rows in product(self.NUM_WARPS, self.ROWS_PER_PROGRAM):
            yield KernelConfig(
                self.operation,
                BLOCK_SIZE=block,
                ROWS_PER_PROGRAM=rows,
                num_warps=warps,
            )

    def num_programs(self, config: KernelConfig, problem: Problem) -> int:
        return ceil_div(problem.dims_dict["rows"], config["ROWS_PER_PROGRAM"])

    def reject_reason(self, config: KernelConfig, problem: Problem, caps: DeviceCaps) -> str | None:
        block = config["BLOCK_SIZE"]
        if block > self.MAX_BLOCK_SIZE:
            return f"row of {problem.dims_dict['cols']} exceeds the single-pass limit"

        per_thread = block / (config["num_warps"] * caps.warp_size)
        if per_thread < self.MIN_ELEMS_PER_THREAD:
            return (
                f"idle threads: {config['num_warps'] * caps.warp_size} threads for "
                f"BLOCK_SIZE={block}"
            )
        if per_thread > self.MAX_ELEMS_PER_THREAD:
            return f"register pressure: {per_thread:.0f} elements per thread"

        rows = problem.dims_dict["rows"]
        if config["ROWS_PER_PROGRAM"] > rows:
            return f"ROWS_PER_PROGRAM={config['ROWS_PER_PROGRAM']} exceeds {rows} rows"
        return None

    def priority(
        self, config: KernelConfig, problem: Problem, caps: DeviceCaps
    ) -> tuple[float, ...]:
        """Prefer a full wave, then fewer elements per thread.

        These kernels are memory bound, so the goal is enough concurrent
        threads to keep loads in flight rather than data reuse.
        """
        programs = self.num_programs(config, problem)
        full_wave = 0 if programs >= caps.sm_count else 1
        per_thread = config["BLOCK_SIZE"] / (config["num_warps"] * caps.warp_size)
        return (full_wave, per_thread)


class SoftmaxSearchSpace(_RowReductionSpace):
    operation = "softmax"


class RMSNormSearchSpace(_RowReductionSpace):
    operation = "rmsnorm"


class VectorAddSearchSpace(SearchSpace):
    """Trivially elementwise: no tiling, no pipelining, nothing to reuse."""

    operation = "vector_add"
    BLOCK_SIZE = (128, 256, 512, 1024, 2048, 4096)
    NUM_WARPS = (1, 2, 4, 8)

    def grid(self, problem: Problem) -> Iterator[KernelConfig]:
        for block, warps in product(self.BLOCK_SIZE, self.NUM_WARPS):
            yield KernelConfig(self.operation, BLOCK_SIZE=block, num_warps=warps)

    def num_programs(self, config: KernelConfig, problem: Problem) -> int:
        return ceil_div(problem.dims_dict["n"], config["BLOCK_SIZE"])

    def reject_reason(self, config: KernelConfig, problem: Problem, caps: DeviceCaps) -> str | None:
        per_thread = config["BLOCK_SIZE"] / (config["num_warps"] * caps.warp_size)
        if per_thread < 1:
            return f"idle threads at BLOCK_SIZE={config['BLOCK_SIZE']}"
        if per_thread > 64:
            return f"register pressure: {per_thread:.0f} elements per thread"
        return None

    def priority(
        self, config: KernelConfig, problem: Problem, caps: DeviceCaps
    ) -> tuple[float, ...]:
        programs = self.num_programs(config, problem)
        full_wave = 0 if programs >= caps.sm_count else 1
        return (full_wave, -config["BLOCK_SIZE"])


SEARCH_SPACES: dict[str, type[SearchSpace]] = {
    "vector_add": VectorAddSearchSpace,
    "softmax": SoftmaxSearchSpace,
    "matmul": MatmulSearchSpace,
    "rmsnorm": RMSNormSearchSpace,
    "fused_linear": FusedLinearSearchSpace,
}


def search_space(operation: str, **kwargs) -> SearchSpace:
    try:
        return SEARCH_SPACES[operation](**kwargs)
    except KeyError:
        raise ValueError(
            f"no search space for {operation!r}; known: {sorted(SEARCH_SPACES)}"
        ) from None


def dims_signature(problem: Problem) -> Sequence[str]:
    return [name for name, _ in problem.dims]
