# Project notes

Engineering narrative for KernelForge: what the problem was, how it was built,
what went wrong, and what I would change. Candid about limits rather than
promotional — the failures are the useful part.

Everything here is traceable to code or git history in this repository. Where
something could not be verified, it says so.

---

## 1. Problem statement

A Triton or CUDA kernel's performance is dominated by a handful of integer
execution parameters: the output tile shape (`BLOCK_M`, `BLOCK_N`), the
reduction block (`BLOCK_K`), the thread count (`num_warps`), and the software
pipeline depth (`num_stages`). The best values depend on *both* the problem
shape and the specific GPU, and the dependency is not weak — a 128×128 tile
over four pipeline stages needs 128 KiB of shared memory, which fits an A100's
163 KiB opt-in budget and does not fit an RTX 4090's 99 KiB. There is no
defensible default.

The standard answer is `@triton.autotune` over a hand-written candidate list.
It works, and I kept it as a measured baseline rather than dismissing it. But it
leaves out everything that makes a tuning result *trustworthy*:

1. The candidate list is hand-written, so it neither adapts to the device nor
   explains why a configuration is absent.
2. Candidates are never checked for correctness. Skipping work is fast, and a
   kernel with a broken boundary mask is precisely a kernel that skips work.
3. Each candidate is timed once, so the ranking inherits single-sample variance.
4. The winner lives in process memory and dies with the process.
5. Nothing is recorded, so last month's numbers cannot be compared with today's,
   and cannot be known to be incomparable either.

KernelForge addresses those five points. It is not a faster kernel library; it
is the infrastructure that decides *which* kernel to run and proves the answer
is right.

### The constraint that shaped everything

I built this on a machine with **no GPU, no CUDA toolkit, and no `nvidia-smi`**.
That makes it impossible to measure a single latency or execute a single Triton
kernel. Two consequences ran through every decision:

- **No performance numbers anywhere.** Not in the README, not in `results/`,
  not in the case studies. Plausible invented figures would be undetectable to
  a casual reader and disqualifying to a careful one.
- **I had to find verification that works without a device.** This turned out
  to be the most interesting engineering in the project (§4).

---

## 2. Architecture

The core abstraction is that **the tuner knows nothing about GEMMs**.

```
Operator (interface)                Tuner (generic)
├── search_space()      ──────────► generate → filter → prune → budget
├── make_inputs(seed)   ──────────► reproducible inputs
├── reference()         ──────────► PyTorch ground truth
├── run(config, *in)    ──────────► PASS A: compile + verify
├── flops() / bytes()   ──────────► PASS B: benchmark survivors
└── baselines()         ──────────► comparison set, identical settings
                                    rank on median → SQLite + config cache
```

Five operators implement that interface: `matmul`, `rmsnorm`, `fused_linear`,
`softmax`, `vector_add`. Adding one means implementing the interface, not
editing the pipeline.

This abstraction earned its keep immediately at RMSNorm. Its parameters are
`BLOCK_SIZE`, `ROWS_PER_PROGRAM`, `num_warps` — it has no `BLOCK_M` at all. A
tuner with GEMM tiling baked in would have needed a second code path, and the
two would have drifted. The RMSNorm space is also *intrinsically smaller*: a
single-pass row kernel must hold the whole row, so `BLOCK_SIZE` is determined by
the problem and only two parameters are free. Sixteen candidates against the
GEMM's hundreds is the correct answer, not a gap to fill.

Supporting layers: a CUDA-event benchmark harness, a scale-invariant correctness
gate, a SQLite experiment log, a JSON config cache for cross-process reuse, a
profiling layer (`torch.profiler` plus Nsight Compute), a transformer block with
swappable kernels, and a CLI.

---

## 3. What I implemented

All of it; this is a from-scratch project (initial commit was a 10-byte README).
Specifically:

**GPU kernels** (Triton) — blocked GEMM with L2-aware grouped program ordering
and fp32 accumulation; single-pass RMSNorm and softmax with fp32 reductions;
fused linear + bias + GELU with the epilogue applied to the accumulator while
it is still in registers; vector add. **CUDA C++** — a handwritten RMSNorm using
`__shfl_down_sync` warp reductions and a two-stage shared-memory block
reduction, exposed through an ATen extension built on demand.

**Framework** — the search-space abstraction and its hardware filters; the
generic tuner; the CUDA-event timing harness with L2 flushing and distribution
statistics; the correctness gate; the SQLite schema and queries; the config
cache; report generation (SQLite → markdown, CSV, matplotlib figures); the CLI;
the transformer-block integration.

**Verification** — the ahead-of-time compilation harnesses that lower Triton and
CUDA kernels to PTX with no GPU, 482 tests, and two CI workflows.

---

## 4. Hardest technical problem

**Verifying GPU kernels on a machine with no GPU.**

The naive position is that kernel code is simply unverifiable without hardware,
and the honest alternative is to ship it untested and say so. I did not accept
that, and the way out was realising that *compilation* needs no device.

`triton.compile` lowers a kernel to PTX **and cubin** for a named target
(`GPUTarget("cuda", 80, 32)`) with no driver present. That immediately catches a
real class of bugs — and it caught one within minutes of being wired up (§9).
But the more valuable realisation was that the generated PTX is *inspectable*,
which turns vague intentions into exact assertions:

```python
# The fp16 GEMM must reach the tensor cores, not a scalar fallback.
assert "mma.sync.aligned.m16n8k16" in compiled.ptx

# The GELU rewrite must cost exactly one hardware exponential per element.
per_thread = BLOCK_M * BLOCK_N // (num_warps * 32)
assert compiled.ptx.count("ex2.approx.f32") == per_thread
```

That second one is a genuinely strong test: it confirms the algebraic rewrite
(§6) produced the instruction sequence I intended, not merely a correct answer.

For the handwritten CUDA kernel I needed a device compiler without `nvcc` (not
installable in that environment). `clang++` in CUDA mode can compile
`__global__` code to PTX given only CUDA's headers and `libdevice`, both
available as PyPI wheels, so I assembled a synthetic CUDA root by symlinking
pieces of four `nvidia-*-cu12` packages and pointed clang at it. The device code
is now checked for `sm80` and `sm90` in CPU-only CI, with assertions on its
instruction profile: five unrolled warp shuffles (log₂ 32), two barriers, one
`rsqrt` per instantiation, and no out-of-line device calls.

### The part I got wrong, and how I found out

I initially documented that the CPU test validates the shared-memory model
`num_stages × BLOCK_K × (BLOCK_M + BLOCK_N) × itemsize`. Then I noticed the
compiler reported the *same* allocation for `num_stages` 1, 2, 3 and 4 — the
model predicted a 4× spread. Rather than assume a measurement artefact, I dumped
the TTGIR:

```
%a_73 = ttg.local_alloc %a_72 : (tensor<128x64xf16>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
```

A single-buffer `local_alloc` inside `scf.for`, and no `async_copy` — the
software pipeliner had not run at all. I tried supplying full pointer
divisibility hints (`tt.divisibility: 16`) on every pointer and stride argument;
no change. So the standalone compile entry point does not pipeline, which means
the CPU test **cannot** check the `num_stages` factor.

I corrected the claim rather than the test: the CPU test now checks the
single-buffer tile footprint against the compiler's own figure (within a
factor-of-two band, because the operand layout is padded for swizzling).

There is a second-order version of the same mistake that I also had to fix.
The GPU-gated replacement asserted `actual <= estimate * 1.25` — an upper
bound, which is the property the *filter* needs, but which a single-buffer
allocation satisfies trivially. So it did not check the `num_stages` factor
either, while the docs said it did. There are now two tests: one for the upper
bound, and one asserting that at least one multi-stage configuration allocates
past a single operand buffer, which is the only thing that confirms the factor
corresponds to the pipeliner's behaviour. "At least one" rather than "every",
because Triton may legitimately decline to pipeline a given loop.

The lesson worth recording: a test is only worth what its docstring claims, and
I had written a claim my test did not support. Finding it required being
suspicious of a result that *looked* fine — the model predicted a 4x spread and
the compiler reported none, and the temptation was to treat that as a
measurement artefact rather than read the IR.

---

## 5. The important algorithm

**Blocked GEMM with L2-aware program scheduling.**

Computing one output element reads a row of A and a column of B: 2K elements for
K multiply-adds, an arithmetic intensity of 1 FLOP per element loaded. No GPU
can feed that. A `BLOCK_M × BLOCK_N` tile instead loads
`BLOCK_K × (BLOCK_M + BLOCK_N)` elements to produce `BLOCK_M × BLOCK_N` results,
so intensity rises to `BM·BN/(BM+BN)` per K-step — 64 for a 128×128 tile against
1 for the naive version. That ratio is literally the `reuse` term in the
search-space priority function; the tuner's ranking is derived from the same
arithmetic that motivates tiling in the first place.

The scheduling layer on top is the part worth explaining. With the obvious
row-major program order, the programs resident on the GPU at any moment span one
horizontal strip of C, which touches `BLOCK_M` rows of A and *all* of B.
Launching in `GROUP_M`-row groups makes the concurrent working set a square-ish
block of C, so both operand strips stay small enough to live in L2 and get
reused by neighbouring programs. Identical arithmetic, identical number of
global loads *issued*, far more of them served by cache.

```python
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)  # last group is short
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

**The subtlety I expect to be asked about** is the operand pointer computation:

```python
offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
```

Taking the row index modulo `M` looks like a correctness bug. It is not. Rows
past the end wrap to some other *valid* row, so every load is in bounds;
`tl.dot` output lanes depend only on their own A row and B column, so wrapped
lanes cannot contaminate valid ones; and the epilogue's store mask discards
them. Only the K axis needs a real mask, because a short final K step must
contribute *zero* rather than wrapped data. The payoff is no M/N masking in the
inner loop at all. I reasoned it through by hand for `M=1`, `N=1` and `K=1`,
and the 88 tests in `test_matmul.py` exercise it over primes,
one-off-a-tile sizes and degenerate rows. CI runs 56 of them on CPU through
Triton's interpreter; the bf16 cases and the largest shapes need a GPU and have
not run.

---

## 6. Major engineering decision

**Separating hardware filters from the candidate budget.**

The GEMM grid is 432 points. Measuring 180 feasible ones costs minutes of
compilation per shape, so the space has to shrink. The tempting move is to
tighten the rules until roughly 50 candidates survive.

I rejected that, because the thresholds required had no hardware justification —
and an unjustified filter is exactly how a search space silently loses its
optimum. Instead there are two distinct mechanisms:

- **Filters** encode what the hardware cannot do or what cannot be defended:
  shared memory over the per-block limit, more fp32 accumulators per thread than
  a thread can address, tile rows shorter than half a 128-byte transaction.
  These read a `DeviceCaps` value object rather than `torch.cuda` directly,
  which is why `test_shared_memory_filter_is_device_specific` can assert that a
  tile feasible on an A100 is rejected for an RTX 4090 on a machine with neither.
- **A budget** bounds search cost by keeping the top 48 under a documented
  priority ordering.

The asymmetry is the whole argument: a wrong filter can exclude the true
optimum; a budget can only cost the chance of finding it. All three counts
(generated / feasible / budgeted) are reported so a shrinking space is visible.

### A related subtlety I got wrong first

The priority key started as `(full_wave_flag, −reuse, waste)` with a binary
"does this fill the GPU" flag. Testing a 256×256 GEMM showed the selected
candidates were 128×128 tiles producing **four programs on a 108-SM device** —
reuse had won in a regime where it is irrelevant, because no tiling can fill
that machine and parallelism is the only thing that matters.

The fix is one clamp:

```python
programs = min(self.num_programs(config, problem), caps.sm_count)
return (-programs, -reuse, waste)
```

Saturating at the SM count makes a single key correct in both regimes. For a
large problem every tiling exceeds `sm_count`, the term ties, and reuse decides;
for a small one it cannot tie, and parallelism decides. I like this one because
the bug was invisible in the large-shape tests that existed, and the fix made
the function simpler rather than adding a special case.

---

## 7. Alternative design considered

**Subprocess isolation per candidate.** A kernel that triggers an illegal memory
access poisons the CUDA context, and every subsequent candidate in that process
then fails — so one bad configuration can corrupt a whole sweep.

I ran candidates in-process anyway. A process launch plus a fresh CUDA context
per candidate is on the order of seconds each against a 48-candidate budget,
which would dominate the tuning time it is protecting. Triton's own autotuner
makes the same call. And these kernels keep every load in bounds by construction
(§5), so the real exposure is to kernels added later, not to the ones shipped.

The tradeoff is documented in `DESIGN.md` rather than hidden, and "subprocess
isolation behind a flag" is the fourth item in the README's future work: the
right default is in-process, but a framework that is unsafe for kernels it did
not ship with has a real gap.

**Also rejected:** a fixed-column `configs` table (cannot represent an RMSNorm
configuration at all); tune-on-cache-miss in the dispatch path (would make the
first inference request after a deploy take thirty seconds); and batched timing
for sub-microsecond kernels (would fix timer resolution but destroy the latency
distribution, which is the thing the harness exists to produce — so the result
is flagged via `at_timer_resolution` instead, because the real fix is a larger
problem, not a cleverer timer).

---

## 8. Performance bottleneck

I could not measure one, so what I can defend is the *analytic* bottleneck
identification the framework performs — which is the more useful skill anyway.

`kernelforge tune` reports arithmetic intensity next to throughput, because the
ratio of a kernel's FLOP/byte to the device's peak-FLOP/s-over-peak-GB/s ridge
point decides whether tuning can help at all. An A100 at 312 TFLOP/s fp16 and
2039 GB/s has a ridge point near **153 FLOP/byte**:

| Workload | Arithmetic intensity | Verdict |
| --- | --- | --- |
| `2048×4096×4096` prefill GEMM | 1024 FLOP/byte | 6.7× above the ridge — genuinely compute-bound, tiling and tensor-core utilisation are the levers |
| `4096×4096×4096` GEMM | 1365 FLOP/byte | compute-bound |
| `1×11008×4096` decode GEMM | **1.0 FLOP/byte** | 153× *below* the ridge — hopelessly memory-bound |

That last row is the interesting one and it is computed by code in the repo
(`metrics.arithmetic_intensity`). A single-token decode GEMM reads the entire
weight matrix to produce one output row. No tile shape changes that: the kernel
is bandwidth-bound by a factor of 153, and the search space reflects it —
only 30 of 432 candidates are feasible, all at the smallest tile the grid
offers, because there is no parallelism in M to exploit.

So the honest conclusion for that shape is that **tuning is the wrong tool**.
The levers are algorithmic: split-K to manufacture parallelism across the
reduction (README future work #3), or batching more tokens to amortise the
weight read. Being able to say "this kernel is already finished, stop tuning it"
is a result.

RMSNorm and softmax are memory-bound by construction — O(1) arithmetic per
element — which is exactly why they are in the project next to a GEMM. They are
reported in GB/s rather than TFLOP/s (`Operator.is_memory_bound()`), because
quoting TFLOP/s for RMSNorm would be meaningless.

---

## 9. How I tested it

482 tests, organised by what they *need* rather than by layer: 189 run on a
CPU-only machine, 293 are GPU-gated and skip with a stated reason.

**The structural guarantee.** The property that matters most — an incorrect
configuration is never ranked — is tested without a GPU, by driving the tuner
end-to-end with a purpose-built CPU operator whose five configurations fail in
each way a real one does: one raises at launch, one returns the wrong answer,
one is correct but slow, one is correct and fast, one is rejected by a filter.
The assertion is that the ranked set contains exactly the two correct ones.

**Shape selection is the substance of the kernel tests.** Powers of two divide
evenly into every block size in the search space, so a kernel with a broken
boundary mask passes `1024×1024×1024` and fails `1023×1023×1023`. The suites run
primes, sizes one off a tile boundary, and degenerate single rows and columns,
against five tile shapes rather than only the default — a mask that is right for
a 32×32 tile can still be wrong for 128×128. Every budgeted candidate is checked
against the reference, not just the shipped default.

**Tests designed to fail if a specific decision were reverted:**

- the fp32 GEMM holds its `1e-5` threshold only because `tl.dot` is pinned to
  IEEE; TF32's 10-bit mantissa carries ~1e-3 relative error against a 1e-5
  threshold, so removing the pin should fail it by about two orders of
  magnitude (predicted from the formats, not yet observed);
- the fp16 RMSNorm needs its fp32 reduction for an 8192-wide row — and the test
  also asserts the naive fp16 reduction is measurably *worse*, so the test has
  teeth rather than merely passing;
- softmax survives logits of 60, which overflow fp16 `exp` and produce NaN
  without the max subtraction;
- `gelu(xw + b)` is pinned apart from `gelu(xw) + b` by a zero-weight case,
  which fixes the operation order;
- `test_cpu_only_imports.py` imports the 13 CPU-side modules in a fresh
  subprocess and asserts neither Triton nor a CUDA context was pulled in — a
  claim that was true and one careless top-level import from being silently
  false. The aggregate answer is what matters, so it is one subprocess; the
  per-module bisect runs only to name the offender on failure.

**What is honestly untested:** every GPU execution path. 293 tests, all Triton
launch behaviour, the ATen launch site of the CUDA extension, the Nsight
integration and all profiling were written but never run. The compile checks
substantially de-risk the kernel *bodies*; they say nothing about launch-time
behaviour, and nothing in this repository claims otherwise.

---

## 10. A bug I encountered

Several, all in git history. The most instructive is the CI failure, because it
had a second-order consequence I nearly missed.

### The CI failure: a toolchain bug, and the fix that broke the guard

Both `tests` matrix jobs failed with 18 errors before reaching any of my code:

```
/usr/include/c++/14/limits:2089:27: error: __float128 is not supported on this target
 2089 |     struct numeric_limits<__float128>
```

My CUDA compile harness told clang where libstdc++ lives by taking the **newest**
`/usr/include/c++/<ver>` on the machine. Locally that was GCC 13 and it worked;
the GitHub runner also has GCC 14, and that combination cannot work. Clang's
CUDA wrapper header includes `<cmath>` for *every* CUDA translation unit, so the
standard library enters the NVPTX device pass whether the kernel touches it or
not — and libstdc++ 14's `<limits>` declares `numeric_limits<__float128>`
unconditionally, because the guard macro is baked into the installed headers
when GCC is configured for x86. NVPTX has no such type.

What I did rather than guess: installed `libstdc++-14-dev` locally to get the
identical 18 errors, which gave me a reproduction to test fixes against. Then I
tested candidates systematically — `-std=c++14`, `-std=gnu++17`, `-nostdinc++`,
`--gcc-toolchain=/usr`, and letting clang choose its own default all fail the
same way, and `-stdlib=libc++` is refused outright by CUDA's own
`crt/host_defines.h` with "libc++ is not supported on x86 system". No flag
avoids it; selecting a compatible version is the only option.

So discovery now **probes instead of assuming**: each candidate libstdc++ is
verified by compiling an empty `__global__` kernel — decisive precisely because
clang drags in the standard library regardless of content, and ~0.2 s — and the
first that works is used. The deeper defect was that discovery had returned a
toolchain that *cannot compile*, turning a missing dependency into ten confusing
errors inside the kernel tests instead of one honest skip.

**The part I nearly missed.** My CI had a step asserting the device-code checks
actually *ran*, because a skipped check plus a green build means the kernels were
never compiled. That step grepped for the text of the old skip message — which
my fix reworded. The guard would have stopped firing while still looking green.
I rewrote it to assert "nothing was skipped in these two files" (precise, because
under `-m "not gpu"` the device-dependent tests there are *deselected* rather
than skipped), then verified all three paths by hiding dependencies: compatible
libstdc++ present → 28 passed, guard green; only GCC 14 → 11 skipped, guard
fires; no clang at all → 11 skipped, guard fires.

### Others worth mentioning

- **Tuning was completely dead for decode shapes.** The tile-overshoot rule
  rejected all 432 grid points whenever `M < 8`, because the smallest `BLOCK_M`
  is 16 and the rule was unconditional. `kernelforge tune matmul --m 1` reported
  `feasible: 0` and exited 1 — for exactly the single-token GEMM that dominates
  decoding, and the shape the workload suite, the experiment script and one case
  study all depend on. The rule now stops applying once a config is already at
  the smallest tile in the grid. Notably, those 30 newly-admitted candidates had
  *never been compiled*, because the budget never selects `BLOCK_M=16` for a
  large shape — so I added a decode shape to the exhaustive compile sweep.
- **`_GELU_COEFF` as a plain Python global.** Caught by the AOT compile check
  within minutes of writing it: a `@triton.jit` kernel cannot read an ordinary
  Python global; it must be `tl.constexpr(...)`. This is the bug that justified
  the whole compile-check approach.
- **The report crashed on a two-dtype database.** Bar charts indexed on
  `shape_key` alone, so the same shape measured in fp16 and bf16 handed
  matplotlib a two-element Series where it expected a scalar. An ordinary path —
  both the GPU workflow and the experiment script take a dtype parameter and
  write the same database.
- **An internal inconsistency ruff found.** The benchmark loop captured a loop
  variable in a closure (`B023`); it happened to work because the closure was
  called in the same iteration, but it was one refactor from being wrong.

---

## 11. What I would improve with more time

**First, and above everything: run it on a GPU.** Every performance claim this
framework is designed to produce is currently unproduced, and the case studies
have predictions written down waiting to be falsified. That is the one thing
that would change the project's character.

Then, in order of engineering value:

1. **A two-pass RMSNorm** to lift the single-pass width cap, adding a genuine
   algorithmic axis to that search space rather than only a tuning one.
2. **Model-based candidate ordering.** The budget currently keeps the top 48 by a
   hand-written key. The SQLite history already stores every candidate's latency
   against shape and device; fitting a cost model on it would let the budget
   spend its slots where the model is most uncertain, turning the database from a
   log into training data.
3. **Split-K for the decode regime**, which is the only thing that addresses the
   1.0 FLOP/byte analysis in §8.
4. **Subprocess isolation behind a flag**, closing the gap in §7.
5. **Autotune `GROUP_M`** instead of fixing it at 8. The L2 reuse argument
   depends on L2 size and problem shape, both of which the framework already
   knows, so it should be a searched parameter with a feasibility rule.

---

## 12. Design FAQ

Questions a reader of this code tends to ask, and the short answers. Each one
points at a decision that is argued at more length above or in `DESIGN.md`.

**1. Why does the correctness gate run before benchmarking rather than after?**
A tuner ranking on latency alone prefers kernels that skip work, and a broken
boundary mask is a kernel that skips work. Running verification as a separate
earlier pass makes that structurally impossible — no code path reaches ranking
without passing the gate. Secondary benefit: it keeps Triton's compilation of
candidate *i+1* out of the measurement of candidate *i*.

**2. Why not `torch.allclose`?** It needs an absolute tolerance that depends on
data magnitude, and a GEMM's output grows like `√K`. A tolerance tuned at
`K=512` either rejects correct kernels at `K=8192` or passes broken ones at
`K=128`. The scale-invariant `max|out−ref|/max|ref|` lets one threshold per
dtype hold across every shape. Thresholds come from each format's output
rounding (fp16 2⁻¹¹ ≈ 4.9e-4, bf16 2⁻⁸ ≈ 3.9e-3) with headroom for summation
order.

**3. What is TF32 and why did you disable it?** A 19-bit tensor-core format with
a 10-bit mantissa, which PyTorch may use for fp32 matmuls on Ampere+ by default.
Its ~1e-3 relative error is a hundred times my fp32 threshold. If the kernel used
TF32 and the reference used fp32, the comparison would measure the precision gap
rather than the kernel; if both used it, a real precision choice would be hidden
behind the word "fp32". So `tl.dot` is pinned to `ieee` and the tuner disables
TF32 session-wide — and I state the consequence, that my fp32 PyTorch baseline is
slower than a user's default.

**4. Walk me through the GEMM's L2 optimisation.** Row-major program order makes
the resident working set one strip of C, touching `BLOCK_M` rows of A and all of
B. `GROUP_M` grouping makes it a square-ish block, so both operand strips fit in
L2 and are reused by neighbouring programs. Same arithmetic, same loads issued,
more of them cache-served. Be ready to derive the index arithmetic and explain
why the last group is short.

**5. Why is `offs_am` computed modulo `M`?** To avoid masking the M/N axes in the
inner loop. Out-of-range rows wrap to valid rows so all loads are in bounds;
`tl.dot` output lanes depend only on their own A row and B column so wrapped
lanes cannot contaminate valid ones; the epilogue store mask discards them. The
K axis still needs a real mask because a short final step must contribute zero,
not wrapped data.

**6. How did you choose `num_warps`, and why does the best value change with
shape?** It sets threads per program, which fixes both accumulator registers per
thread (`BM·BN/(32·warps)` — more registers means more ILP but closer to the
255-register ceiling) and the number of warps available to hide memory latency.
Large square shapes have K iterations to pipeline and favour the middle of the
range; a skinny `M=1` shape is bound by loading the weight matrix and favours
more warps. The distinguishing evidence is the warp stall reason in Nsight's
`WarpStateStats` — memory-dependency stalls mean latency hiding, whereas a
register count at the ceiling with local-memory traffic means spilling.

**7. How do you verify a GPU kernel without a GPU?** `triton.compile` lowers to
PTX and cubin for a named target with no driver. That catches undefined names,
illegal tile shapes and bad `tl.*` calls, and the PTX is inspectable, so I assert
tensor-core MMA selection and exact per-element instruction counts. `clang++` in
CUDA mode does the same for the CUDA kernel given headers and libdevice from
PyPI. Know the limit: the standalone entry point does not run the software
pipeliner, so shared-memory multi-buffering must be checked on a device.

**8. What is the shared-memory model, and how do you know it is right?**
`num_stages × BLOCK_K × (BLOCK_M + BLOCK_N) × itemsize`, because the pipeliner
keeps `num_stages` operand tiles in flight. It is an upper bound used to reject
configurations before compiling; anything it lets through that the compiler then
rejects returns as `OutOfResources` and is recorded as a compile failure, so an
inaccurate model costs tuning time rather than correctness. The tile half is
checked against the compiler on CPU; the `num_stages` factor against a real
launch on GPU.

**9. Why do you rank on the median rather than the minimum or mean?** The
minimum is the single luckiest sample, so ranking by it rewards variance. The
mean is skewed by clock throttling and preempted launches — one outlier in two
hundred moves it by an order of magnitude and leaves the median untouched, which
a test asserts directly. p95/p99 are recorded because a good median with a bad
tail is worth knowing about.

**10. Why flush L2 between iterations, and when shouldn't you?** Re-running on
the same tensors leaves inputs resident in L2, so a problem whose working set
fits reports a bandwidth it would never see in a real model. The zeroing is
enqueued before `start_event` on an in-order stream, so it is outside the
measured interval. Don't flush when the working set already greatly exceeds L2 —
the transformer block sets `flush_l2=False` for that reason, since flushing only
adds noise to an already-cold measurement.

**11. How does the fused kernel save traffic, and why won't latency improve
proportionally?** Unfused, the bias add and activation each read and rewrite the
full `M×N` intermediate: `5·M·N` of output traffic and three launches, against
`M·N` and one. At 4096×11008 fp16 total DRAM traffic falls 548 → 204 MiB. But the
GEMM reads `M·K + K·N` of operands regardless, so the saving applies to a
fraction of total traffic — the prediction written down in the case study is
explicitly that latency improves by *less* than the traffic ratio. The real
competition is `torch.compile`, whose Inductor already fuses elementwise
epilogues onto a matmul.

**12. Why is the cache keyed on the board name rather than the compute
capability?** An RTX 4090 and an RTX 4080 are both `sm89` but differ in SM count,
L2 size and memory bandwidth, so the best tile for one is not best for the
other. Keying on the full name means a cache file copied to another machine
*misses* rather than silently serving a configuration tuned for different
hardware. Related: selecting a configuration never triggers tuning, because a
thirty-second compile sweep inside an inference loop would be a bug.

**13. Why store configurations as JSON instead of typed columns?** A GEMM config
is `BLOCK_M/N/K/GROUP_M`; an RMSNorm config is `BLOCK_SIZE/ROWS_PER_PROGRAM`.
Fixed columns cannot represent the second, and a nullable column per operator
parameter turns the table into a sparse matrix every query must special-case. A
content digest gives configs a stable identity for joins; `json_extract` recovers
single-parameter predicates where needed.

**14. How is this different from `@triton.autotune`?** Triton's autotuner takes a
hand-written config list, times each once, keeps the winner in process memory and
never checks correctness. KernelForge derives the space from hardware properties
with stated rejection reasons, gates on correctness, measures a distribution,
persists results with provenance, and caches across processes. I kept Triton's
autotuner as a *measured baseline* rather than asserting the difference.

**15. What is the biggest weakness of this project?** No measured numbers, and
every GPU execution path untested — 293 tests written and never run. I would not
soften that. What I would add is that the constraint produced a verification
strategy I would now use even with hardware available, because PTX-level
assertions catch things a passing numerical test does not: that a kernel reached
the tensor cores at all, and that an algebraic rewrite produced the instruction
sequence intended.

**16. If I gave you an A100 for an hour, what would you measure first?** The
three case-study predictions, in order of falsifiability: the fusion traffic
saving (quantitative and exact — 344 MiB, checkable directly in Nsight's DRAM
counters); the tile-size ordering at 4096³ *and* at 256², where I predict the
ordering reverses; and the block-level transformer comparison, where I expect the
end-to-end gain to be much smaller than the operator gains, because only two
normalisations and one activation are swapped. That last one is the measurement
most likely to be humbling, which is why it is in the repository.
