# Design

Why the pieces are shaped the way they are, including the decisions that went
the other way.

## The harness came before the kernels

The first component built was `benchmark/runner.py`, not `matmul`. Writing the
kernel first means the first latency number arrives before there is anything
trustworthy to produce it, and a number you half-believe is worse than no
number: it anchors every later judgement. The same argument put `testing.py`
before the kernels. By the time the GEMM existed, there was already a timing
loop that handled asynchronous launches and an error metric that held across
shapes.

## The tuner knows nothing about GEMMs

`Tuner.tune` takes an `Operator`. It asks for a search space, inputs, a
reference, and a way to run one configuration; everything else — filtering,
verification, timing, ranking, persistence — is generic. Adding an operator
means implementing that interface, not touching the tuner.

This mattered immediately at RMSNorm. Its parameters are `BLOCK_SIZE`,
`ROWS_PER_PROGRAM` and `num_warps`; it has no `BLOCK_M`. A tuner with GEMM
tiling baked into it would have needed a second code path, and the two would
have drifted. The RMSNorm space is also *intrinsically smaller*: a single-pass
row kernel must hold the whole row, so `BLOCK_SIZE` is determined by the
problem and only two parameters are free. Sixteen candidates against the GEMM's
hundreds is the right answer, not a gap to be filled.

## Configurations are open key/value sets

`KernelConfig` holds a mapping, not named fields. Named GEMM fields would force
every other operator to carry meaningless ones. The consequence reaches the
database: configs are stored as JSON plus a content digest rather than as
`block_m/block_n/block_k/warps/stages` columns.

That is a deliberate departure from the obvious schema. Fixed columns cannot
represent an RMSNorm config, and adding a nullable column per operator
parameter turns the table into a sparse matrix that every query must
special-case. The digest gives configs a stable identity for joins; SQLite's
SQLite's `json_extract` remains available should a query ever need a predicate
on one parameter, though nothing here uses it yet; the report code expands the
JSON into DataFrame columns, which is where that shape is actually convenient.

## Filters are hardware rules; the budget bounds cost

Candidate generation separates two different things that are easy to conflate:

- **Filters** encode what the hardware cannot do or what cannot be defended:
  shared memory over the per-block limit, more accumulator registers per thread
  than a thread can address, tile rows shorter than half a memory transaction.
  These are derived from device properties, so they transfer across GPUs and
  each one can state its reason.
- **The budget** bounds how long tuning takes. 432 grid points become 180
  feasible, and measuring 180 configurations per shape is minutes of compiling.

Keeping them separate is what stops the filters from being quietly tightened
until the count looks nice. Both can exclude the true optimum. The difference
is that a rule claims a configuration cannot win, while the budget only defers
it: the cut follows the stated priority, never splits configurations that
priority ranks equal, and `--min-candidates` raises it. All three counts are
reported for the same reason.

The filters read a `DeviceCaps` value object rather than `torch.cuda` directly.
That is not indirection for its own sake: it makes the filters unit-testable
against the properties of a GPU that is not attached, which is how
`test_shared_memory_filter_is_device_specific` can assert that a tile feasible
on an A100 is rejected for an RTX 4090 on a machine with neither.

## Verification before benchmarking, in separate passes

Candidates are compiled and checked first; the survivors are then timed. Two
reasons. Interleaving would put Triton's compilation of candidate *i+1* in the
middle of the measurement of candidate *i*. And the ordering makes the central
invariant structural rather than a matter of care: ranking only ever sees
verified candidates. A tuner that ranks on latency alone prefers kernels that
skip work, and a broken boundary mask is exactly a kernel that skips work.

### Why a scale-invariant error metric

The gate is `max|out - ref| / max|ref|` against a per-dtype threshold, not
`torch.allclose`. A GEMM's output magnitude grows like `sqrt(K)`, so an
absolute tolerance tuned at one `K` is wrong at another: too strict at
`K=8192`, too loose at `K=128`. Elementwise mismatch counts are still computed
and reported, because "0.4% of elements differ" and "every element differs
slightly" are different bugs, but they are diagnostics rather than the gate.

### Why fp32 means IEEE on both sides

`tl.dot` would use TF32 for fp32 operands by default, and so would
`torch.matmul`. Comparing a TF32 kernel against an fp32 reference measures the
precision gap, not the kernel; comparing TF32 against TF32 hides a real
precision choice behind the word "fp32". The kernels pin `input_precision`, and
the tuner wraps the whole session with TF32 disabled so the PyTorch baselines
match. Session-level rather than per-call, because a context manager inside the
timed closure would add Python overhead to every measured iteration.

## Ranked on the median

The minimum is the single luckiest sample; ranking by it rewards variance. The
mean is skewed by clock throttling and the occasional preempted launch — one
outlier in two hundred moves it by an order of magnitude and leaves the median
untouched, which `test_median_is_robust_to_an_outlier_the_mean_is_not` asserts
directly. p95 and p99 are recorded alongside, because a configuration with a
good median and a bad tail is worth knowing about even if it wins.

## Two stores, because they answer different questions

The SQLite database is the experiment log: every candidate, including the ones
that failed and why, with the hardware and library versions attached to the
run. It is append-only in practice and is what makes a number from last month
comparable, or knowably incomparable, to one from today.

The config cache is an operational artefact: one JSON file holding the best
known configuration per device, operation, dtype and shape, which a process can
read in microseconds. Its key includes the full board name and not just the
compute capability, because an RTX 4090 and an RTX 4080 are both `sm89` and
differ in SM count, L2 size and memory bandwidth. A cache file copied to
another machine therefore misses rather than silently serving a configuration
tuned for other hardware.

Selecting a configuration never triggers tuning. A thirty-second compile sweep
inside someone's inference loop would be a bug, not a feature, so
`runtime/dispatch.py` reads the cache and falls back to the operator default,
reporting which it used.

## Triton's autotuner is a baseline, not a dependency

`@triton.autotune` over a hand-written config list is in `kernels/matmul.py` as
`matmul_triton_autotune`, using the same kernel. If KernelForge were a wrapper
around it there would be no project, and the honest way to say so is to measure
both. The differences are in candidate generation from hardware properties,
the correctness gate, a distribution instead of a single sample, cross-process
persistence, and a recorded history — not in the kernel.

## What the CPU-only verification can and cannot see

`triton.compile` lowers a kernel to PTX and cubin for a named target without a
device. That turns a real class of kernel bugs into ordinary CI failures, and
it caught one during development: `_GELU_COEFF` as a plain Python global, which
a `@triton.jit` kernel cannot read.

It also has a specific limit worth recording, because it changes what a test
can claim. The standalone compile entry point does not run the software
pipeliner, so its shared-memory allocation stays at one buffer per operand
however high `num_stages` is — confirmed by inspecting the TTGIR, and unchanged
by supplying full pointer-divisibility hints. So the CPU test checks the
single-buffer tile footprint against the compiler's own figure, and the
`num_stages` factor is checked against a real launch in
`tests/test_kernels_gpu.py`, which asserts both that the estimate upper-bounds
a real allocation and that multi-buffering actually happens for at least one
multi-stage configuration. Describing the CPU test as validating the whole
model would have been wrong, and an upper-bound assertion alone would not have
checked the `num_stages` factor either -- a single-buffer allocation satisfies
it trivially.

The same approach covers the handwritten CUDA kernel. `clang++` in CUDA mode
compiles `__global__` code to PTX given only CUDA's headers and libdevice,
both available from PyPI, so the device code is checked for sm80 and sm90 in
CPU-only CI. The ATen launch site cannot go through that path: ATen requires
C++20, and CUDA's `crt/host_defines.h` collides with libstdc++'s use of
`__attribute__((__noinline__))` when the CUDA headers are processed first — a
collision that header's own comments describe. nvcc avoids it; clang does not.
That side is boilerplate and is compiled by `cpp_extension.load` on first use.
Putting the templated kernel in a `.cuh` is idiomatic for CUDA templates
anyway; the testability is a bonus rather than the justification.

## Decisions that went the other way

**Subprocess isolation for candidates.** A kernel that triggers an illegal
memory access poisons the CUDA context, and every later candidate in the
process then fails. Isolating each candidate in a subprocess would fix that at
the cost of a process launch and a CUDA context per candidate — seconds each,
against a 48-candidate budget. Triton's own autotuner runs in-process for the
same reason. The kernels here keep every load in bounds (the `% M` wrap in the
GEMM is exactly that), so the exposure is to kernels added later, and the
failure is loud rather than silent.

**Fixed-column config table.** Covered above: cannot represent RMSNorm.

**A `--tune-on-miss` dispatch mode.** Tempting, and wrong: it would make the
first inference request after a deploy take thirty seconds.

**Batched timing for very fast kernels.** CUDA events resolve to roughly half a
microsecond, so a kernel below a few microseconds is at the edge of the timer.
Timing a batch of iterations between one event pair would fix the resolution
but destroy the distribution, which is the thing the harness exists to produce.
Instead `TimingResult.at_timer_resolution` flags it and the CLI prints a
warning, because the real fix is a larger problem, not a cleverer timer.

**Replacing the attention and projection GEMMs in the transformer block.**
Swapping in four more GEMMs would make the block-level speedup look larger
while measuring this kernel against cuBLAS four more times, which the operator
benchmarks already do in isolation. Leaving them alone keeps the integration a
measurement of the fused epilogue and the normalisations.

**Numbers in the README.** The strongest temptation. Plausible figures would
make this document look finished, and they would be fabrications. The apparatus
is committed; the numbers come from your GPU.
