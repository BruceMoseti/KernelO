"""Candidate generation.

A naive GEMM grid over the parameters KernelForge tunes is 432 points, and
compiling plus verifying plus benchmarking all of them costs minutes per
shape for configurations that cannot win. Candidate generation therefore has
three stages:

1. **Enumerate** the full grid.
2. **Reject** configurations that are infeasible on this device or
   indefensible for this problem, each with a reason that the CLI can print.
   These rules come in two kinds, and the distinction is worth keeping
   straight rather than blurring:

   * **Hardware limits** — shared memory per block, 128-byte memory
     transactions, a pipeline that cannot be filled. These are device
     properties; a configuration violating one either fails to compile or
     provably wastes bandwidth.
   * **Efficiency heuristics** — ``MIN_TILE_ELEMENTS``, the output-per-warp
     floor, and the register cap below the architectural 255. These are
     judgement calls about what cannot be competitive. They are the reason
     the candidate budget exists as a *separate* mechanism: a heuristic that
     is wrong silently excludes the optimum, so each one is named in its
     rejection message and each is justified in the constant's docstring.

   Neither kind is derived from measurements, so both transfer across GPUs.
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
from collections.abc import Iterator
from itertools import product

from kernelforge.dtypes import itemsize
from kernelforge.runtime.env import DeviceCaps, device_caps
from kernelforge.tuning.config import KernelConfig, Problem

#: Default candidate budget per problem.
DEFAULT_MAX_CANDIDATES = 48

#: Efficiency heuristic, bounded by a hardware limit. A thread can address 255
#: registers before spilling to local memory, so an accumulator above that
#: *must* spill. This cap is lower because the accumulator is only part of a
#: thread's demand -- addresses, pipeline buffers and the epilogue need their
#: share -- and where exactly the real ceiling falls depends on what the
#: compiler allocates. 128 is a judgement, not a device property.
MAX_ACC_REGS_PER_THREAD = 128

#: Efficiency heuristic. Below this, the prologue, index arithmetic and
#: epilogue dominate the inner loop.
MIN_ACC_REGS_PER_THREAD = 4

#: Hardware limit. A global load is issued in 128-byte transactions. A tile row
#: shorter than half a transaction wastes most of every line it touches,
#: because successive tile rows are strided by the full matrix dimension and
#: cannot be coalesced into the same line.
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
        """Drop configurations that cannot fill half the GPU -- unless none can.

        A launch grid smaller than the SM count leaves multiprocessors idle for
        the whole kernel. The floor is half the SM count rather than all of it,
        because a grid of 0.6 waves can still beat a smaller tile that reaches
        1.0, and the budget would rather spend a slot measuring it than rule it
        out. The rule is conditional for a stronger reason: for a small enough
        problem *no* tiling fills the machine, and the least-bad option still
        has to be measured.
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
            (c, f"launch grid of {programs[c]} programs fills under half of {caps.sm_count} SMs")
            for c in configs
            if programs[c] < floor
        ]
        return kept, dropped

    def explain(self, problem: Problem, caps: DeviceCaps | None = None) -> str:
        """Human-readable breakdown of the filtering, for `--explain`."""
        resolved = caps if caps is not None else device_caps()
        result = self.generate(problem, resolved)
        # The counts below are a function of the device's properties, so the
        # device has to be named -- and labelled when its properties were
        # assumed rather than read from attached hardware.
        suffix = "" if resolved.measured else "  <- no device attached"
        lines = [
            f"{problem.describe()}",
            f"  device           : {resolved.name}, {resolved.sm_count} SMs, "
            f"{resolved.max_shared_memory_per_block // 1024} KiB shared/block{suffix}",
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

    #: Efficiency heuristic, and the largest single rejection bucket (81 of 252
    #: for the flagship shape), so it deserves justification. A program's
    #: fixed cost is the group-ordering arithmetic, the operand pointer setup
    #: and the epilogue's masked store. Below roughly a thousand output
    #: elements that fixed cost stops being amortised: a 16x32 tile does 512
    #: multiply-accumulates per K-step against the same prologue a 128x128
    #: tile pays once for 16384. The exact threshold is a judgement; it is set
    #: at the smallest power-of-two tile area that keeps a 4-warp program's
    #: accumulator above `MIN_ACC_REGS_PER_THREAD`.
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

    def shared_memory_bytes(self, config: KernelConfig, problem: Problem, caps: DeviceCaps) -> int:
        """Shared memory for the pipelined operand tiles, or for the output tile.

        Triton's pipeliner keeps several ``BLOCK_K * (BLOCK_M + BLOCK_N)``
        operand tiles in shared memory so that the global loads for later
        iterations overlap the ``tl.dot`` of the current one. On Ampere, Ada
        and consumer Blackwell (sm_8x, sm_12x) it copies with ``cp.async`` and
        keeps ``num_stages - 1`` tiles; on Hopper's wgmma path it keeps
        ``num_stages``, which the estimate assumes for every other
        architecture. The epilogue stages the ``BLOCK_M x BLOCK_N`` output tile
        through shared memory for the store, reusing the pipeline buffers, so
        the estimate is the larger of the two.

        How much of this is checked, and where:

        * ``tests/test_triton_compile.py`` compiles kernels on a CPU-only
          machine, specialized the way the JIT specializes a launch on
          contiguous operands, which is what lets the pipeliner run. It checks
          that each stage adds one operand tile and that the estimate bounds
          the compiler's allocation for sm80, sm89, sm90 and sm120.
        * ``tests/test_kernels_gpu.py`` checks two things against a real JIT
          launch: that this estimate is an *upper bound* on what a launch
          allocates, which is the property the filter needs; and separately
          that at least one multi-stage configuration allocates past a single
          operand buffer, so the ``num_stages`` factor corresponds to the
          pipeliner's actual behaviour rather than to an assumption.

        On those architectures the estimate is an upper bound, used to reject
        configurations before compiling them. Anything it lets through that
        the compiler then rejects comes back as Triton's own
        ``OutOfResources`` and is recorded as a compile failure, so an
        inaccurate model costs tuning time rather than correctness.
        """
        major = int(caps.compute_capability.split(".")[0])
        buffers = config["num_stages"] - 1 if major in (8, 12) else config["num_stages"]
        pipeline = buffers * config["BLOCK_K"] * (config["BLOCK_M"] + config["BLOCK_N"])
        output = config["BLOCK_M"] * config["BLOCK_N"]
        return max(pipeline, output) * itemsize(problem.dtype)

    def acc_regs_per_thread(self, config: KernelConfig) -> float:
        """fp32 accumulator registers each thread holds."""
        threads = config["num_warps"] * 32
        return config["BLOCK_M"] * config["BLOCK_N"] / threads

    def num_programs(self, config: KernelConfig, problem: Problem) -> int:
        dims = problem.dims_dict
        return ceil_div(dims["M"], config["BLOCK_M"]) * ceil_div(dims["N"], config["BLOCK_N"])

    @staticmethod
    def _overshoots(block: int, sizes: tuple[int, ...], dim: int) -> bool:
        return block > 2 * dim and block > min(sizes)

    def reject_reason(self, config: KernelConfig, problem: Problem, caps: DeviceCaps) -> str | None:
        dims = problem.dims_dict
        m, n, k = dims["M"], dims["N"], dims["K"]
        bm, bn, bk = config["BLOCK_M"], config["BLOCK_N"], config["BLOCK_K"]
        width = itemsize(problem.dtype)

        smem = self.shared_memory_bytes(config, problem, caps)
        if smem > caps.max_shared_memory_per_block:
            return (
                f"shared memory: {smem // 1024} KiB of tiles exceeds the "
                f"{caps.max_shared_memory_per_block // 1024} KiB per-block limit"
            )

        # The size heuristics below prefer a bigger tile or fewer warps. When the
        # overshoot rule admits no bigger tile and no fewer warps exist, there is
        # nothing to prefer, and applying them would leave a small problem with
        # no candidates at all.
        largest_m = max(b for b in self.BLOCK_M if not self._overshoots(b, self.BLOCK_M, m))
        largest_n = max(b for b in self.BLOCK_N if not self._overshoots(b, self.BLOCK_N, n))
        least_bad = (
            bm == largest_m and bn == largest_n and config["num_warps"] == min(self.NUM_WARPS)
        )

        if bm * bn < self.MIN_TILE_ELEMENTS and not least_bad:
            return f"tile too small: {bm}x{bn} output elements per program"

        acc = self.acc_regs_per_thread(config)
        if acc > MAX_ACC_REGS_PER_THREAD:
            return f"register pressure: {acc:.0f} accumulator registers per thread"
        if acc < MIN_ACC_REGS_PER_THREAD and not least_bad:
            return f"too little work per thread: {acc:.1f} accumulator registers"

        # Efficiency heuristic. Each warp of the MMA pipeline should own at
        # least one 16x16 output tile. Triton *can* decompose otherwise by
        # splitting the K loop within the block, so this excludes
        # configurations that are inefficient rather than impossible.
        if bm * bn < 256 * config["num_warps"] and not least_bad:
            return f"too little output per warp at num_warps={config['num_warps']}"

        if bk * width < MIN_CONTIGUOUS_TILE_BYTES:
            return f"uncoalesced A tile: BLOCK_K={bk} is {bk * width} contiguous bytes"
        # A tile as wide as the matrix row is contiguous with the next row.
        if bn * width < MIN_CONTIGUOUS_TILE_BYTES and bn < n:
            return f"uncoalesced B tile: BLOCK_N={bn} is {bn * width} contiguous bytes"

        # A tile larger than the problem computes masked-off work that is
        # thrown away. Allowing one doubling keeps useful tilings for shapes
        # that are just over a tile boundary.
        #
        # The rule only applies while a smaller tile is still available. The
        # smallest BLOCK_M in the grid is 16, so for a decode-shaped GEMM with
        # M=1 every tile "overshoots" and an unconditional rule would reject
        # the entire grid -- which is exactly the shape that dominates
        # single-stream decoding and most needs tuning.
        if self._overshoots(bm, self.BLOCK_M, m):
            return f"tile overshoot: BLOCK_M={bm} for M={m}"
        if self._overshoots(bn, self.BLOCK_N, n):
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

        1. Programs, saturating at one full wave. Clamping at ``sm_count`` is
           what makes this work in both regimes: for a large problem every
           tiling fills the GPU, the term ties, and reuse decides; for a small
           one it cannot, and the term prefers the tiling that keeps more
           multiprocessors busy. Without the clamp, a 256x256 GEMM would
           spend its budget on 128x128 tiles that occupy four SMs out of a
           hundred.
        2. Data reuse per byte of tile loaded, ``BM*BN / (BM+BN)``: a tile
           loads ``BK*(BM+BN)`` elements to produce ``BM*BN`` outputs, so this
           ratio is the arithmetic intensity of the inner loop.
        3. Least wasted work from padding the problem up to whole tiles.
        """
        dims = problem.dims_dict
        bm, bn = config["BLOCK_M"], config["BLOCK_N"]
        programs = min(self.num_programs(config, problem), caps.sm_count)
        reuse = (bm * bn) / (bm + bn)
        padded = ceil_div(dims["M"], bm) * bm * ceil_div(dims["N"], bn) * bn
        waste = padded / (dims["M"] * dims["N"])
        return (-programs, -reuse, waste)


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

    #: Elements each thread holds in registers during the reduction.
    MIN_ELEMS_PER_THREAD = 1
    MAX_ELEMS_PER_THREAD = 64

    #: Widest row a single-pass kernel will accept; the wrappers raise above
    #: it rather than silently producing a wrong answer.
    #:
    #: Tied to ``MAX_ELEMS_PER_THREAD`` on purpose. The default configuration
    #: allocates one warp per 256 columns up to 8 warps, so at 16384 columns a
    #: thread already holds ``16384 / 256 = 64`` elements -- exactly the limit
    #: above. A wider row would hand the default configuration a register
    #: demand the search space itself rejects. Real transformers top out at a
    #: hidden size of 8192, so this is not a constraint in practice; a wider
    #: row needs a two-pass kernel, which is not implemented.
    MAX_BLOCK_SIZE = MAX_ELEMS_PER_THREAD * 8 * 32

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
        """Prefer parallelism up to a full wave, then fewer elements per thread.

        These kernels are memory bound, so the goal is enough concurrent
        threads to keep loads in flight rather than data reuse.
        """
        programs = min(self.num_programs(config, problem), caps.sm_count)
        per_thread = config["BLOCK_SIZE"] / (config["num_warps"] * caps.warp_size)
        return (-programs, per_thread)


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
        programs = min(self.num_programs(config, problem), caps.sm_count)
        return (-programs, -config["BLOCK_SIZE"])


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
