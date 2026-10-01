# KernelForge

Hardware-aware GPU kernel autotuning and inference optimisation for transformer
workloads.

KernelForge generates execution configurations for Triton kernels from the
properties of the GPU it is running on, rejects the ones the hardware cannot
run with a stated reason, verifies every survivor against a PyTorch reference,
benchmarks the survivors with CUDA events, ranks them, records the result in
SQLite with full hardware provenance, and caches the winner so a later process
does not pay to tune again.

```
kernelforge tune matmul --m 2048 --n 4096 --k 4096 --dtype fp16
```

---

## A note on the numbers in this repository

**There are none, and that is deliberate.**

A latency figure without the GPU, driver, CUDA version and measurement
methodology attached is not a result. So this repository ships no benchmark
tables, no `results/*.csv`, and no README claims of the form "1.4x faster".
Every number is produced by running the tools below on your own hardware, and
every number that comes out carries the GPU, driver, PyTorch and Triton
versions it was measured on, in the same database row.

What *is* committed here is the measurement apparatus, the kernels, and the
tests — and a surprising amount of that is verified without a GPU at all:

| Verified in CPU-only CI | How |
| --- | --- |
| Every kernel compiles for Ampere and Hopper | `triton.compile` lowers to PTX and cubin for a named target without a device |
| All 48 budgeted GEMM candidates compile | the same, across the real search space |
| The fp16 GEMM reaches the tensor cores | `mma.sync.aligned.m16n8k16` asserted in the generated PTX |
| The fused epilogue costs one exponential per element | `ex2.approx.f32` count asserted against accumulators per thread |
| The shared-memory model matches the compiler | single-buffer tile footprint checked against the compiler's own allocation |
| The handwritten CUDA kernel compiles for sm80 and sm90 | clang in CUDA mode, with headers and libdevice from PyPI |
| Its instruction profile is as intended | five unrolled warp shuffles, two barriers, one `rsqrt`, no out-of-line calls |
| An incorrect configuration is never ranked | the tuner driven end to end on a CPU operator whose configs fail in each real way |
| Search-space filters are device-specific | a tile feasible on an A100 is rejected for an RTX 4090, both without the hardware |

On a CPU-only machine, `pytest -m "not gpu"` gives **167 passed, 1 skipped**.
A further **292** tests need a CUDA device and skip with a stated reason
rather than failing, so the same suite runs in ordinary CI and on a GPU runner.

---

## 1. Motivation

A Triton kernel's performance is decided mostly by five integers — the tile
shape, the warp count and the pipeline depth. The right values depend on the
problem shape *and* on the GPU: a 128x128 tile over four pipeline stages needs
128 KiB of shared memory, which fits an A100 and does not fit an RTX 4090.
There is no single good default.

The usual answer is `@triton.autotune` over a hand-written list of candidates.
That works, and KernelForge measures itself against it. But it leaves out
everything that makes tuning trustworthy:

- The candidate list is hand-written, so it neither adapts to the device nor
  explains why a configuration is absent.
- Candidates are never checked for correctness. Skipping work is fast, and a
  kernel whose boundary mask is broken skips work.
- Each candidate is timed once, so the ranking inherits the variance of a
  single sample.
- The result lives in process memory and dies with the process.
- Nothing is recorded, so yesterday's numbers cannot be compared with today's.

KernelForge addresses those five points, and keeps Triton's autotuner available
as a baseline so the difference is a measurement rather than a claim.

## 2. Architecture

```
                       KernelForge
                    ┌─────────────┐
                    │  Workload   │   operation + dims + dtype
                    └──────┬──────┘
                           │
                           ▼
                   ┌──────────────┐   per-operator; GEMM tiling and
                   │ Search Space │   row-reduction parameters are
                   └──────┬───────┘   not the same space
                          │
                  candidate configs
                          │
                          ▼
                   ┌────────────┐     hardware filters: shared memory,
                   │  Filters   │     registers per thread, coalescing,
                   └─────┬──────┘     parallelism, pipeline depth
                          │
                          ▼
                   ┌────────────┐
                   │ Compile /  │
                   │  Runtime   │
                   └─────┬──────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Correctness             Failure  →  recorded with its reason
              │
              ▼
          Benchmark      CUDA events, L2 flushed, distribution
              │
              ▼
          Profiler       launch counts, device time, ncu counters
              │
              ▼
      Results Database   SQLite, with hardware provenance per row
              │
              ▼
      Best Configuration →  config cache  →  runtime dispatch
```

Code layout:

```
kernelforge/
├── kernels/        Triton kernels + the Operator the tuner drives
├── tuning/         config, search spaces, tuner, config cache
├── benchmark/      CUDA-event timing, metrics, workloads, reports
├── profiling/      torch.profiler and Nsight Compute
├── runtime/        environment capture, cache-aware dispatch
├── integration/    transformer block with swappable kernels
├── csrc/           handwritten CUDA RMSNorm
├── cli/            command line interface
├── db.py           SQLite results store
├── testing.py      the correctness gate
└── dtypes.py
```

## 3. Supported operators

Three flagship operators, chosen because they fail in different ways, plus two
that exist to exercise the pipeline end to end.

| Operator | Shape | Bound by | What it demonstrates |
| --- | --- | --- | --- |
| `matmul` | `M,N,K` | compute | tiling, tensor cores, L2-aware scheduling, autotuning |
| `rmsnorm` | `rows,cols` | memory | reductions, numerical stability, a second search space |
| `fused_linear` | `M,N,K` | mixed | operator fusion, reduced intermediate traffic, one launch instead of three |
| `softmax` | `rows,cols` | memory | row-wise reduction, stability under fp16 overflow |
| `vector_add` | `n` | memory | the simplest full pass through the pipeline |

Deliberately **not** here: fused attention, a serving layer, a compiler, a web
dashboard, twenty more operators. Each would dilute the point, which is GPU
performance engineering on operators whose behaviour can be explained.

## 4. The autotuner

Candidate generation runs in three stages, and all three counts are reported,
so a shrinking search space is visible rather than silent:

```
$ kernelforge tune matmul --m 2048 --n 4096 --k 4096 --explain
matmul M/N/K=2048 x 4096 x 4096 fp16
  grid points      : 432
  feasible         : 180
  after prune      : 180
  selected (budget): 48 of 48
  rejections by rule:
     105  uncoalesced A tile
      81  tile too small
      30  uncoalesced B tile
      27  too little output per warp at num_warps=8
       9  register pressure
```

The rules come from hardware limits rather than from measurements, so they
transfer across GPUs:

| Rule | Reasoning |
| --- | --- |
| Shared memory | `num_stages * BLOCK_K * (BLOCK_M + BLOCK_N) * itemsize` must fit the per-block opt-in limit |
| Register pressure | `BLOCK_M*BLOCK_N / threads` fp32 accumulators per thread, against the 255 a thread can address |
| Coalescing | a tile row shorter than 64 B wastes most of every 128 B transaction it touches, since successive rows are strided |
| Tile overshoot | a tile more than twice the problem dimension computes masked-off work |
| Pipeline depth | `num_stages > 2` needs at least that many K iterations to overlap |
| Parallelism | a launch grid below the SM count leaves multiprocessors idle — applied only when some tiling can fill the GPU |

Survivors are sorted by a documented priority and truncated to a budget
(default 48). The budget bounds tuning cost without tightening the rules to the
point where they might exclude the true optimum. Priority is programs
(saturating at one full wave), then tile reuse `BM*BN/(BM+BN)`, then least
padding waste — the clamp is what makes a small problem prefer parallelism and
a large one prefer reuse.

Each operator owns its space. An RMSNorm config is `BLOCK_SIZE`,
`ROWS_PER_PROGRAM`, `num_warps`; a single-pass row kernel has to hold the whole
row, so `BLOCK_SIZE` is *determined* by the row width and the space is an order
of magnitude smaller than the GEMM's. The tuner does not assume otherwise.

## 5. Correctness

Verification happens before benchmarking, in a separate pass, and a
configuration that fails is recorded with its error and dropped. **An incorrect
configuration is never ranked** — a tuner that ranks on latency alone will
happily pick a kernel whose masking is broken.

The gate is a scale-invariant error, `max|out - ref| / max|ref|`, against a
per-dtype threshold. `torch.allclose` needs an absolute tolerance that depends
on the magnitude of the data, which for a GEMM grows like `sqrt(K)`: a
tolerance tuned for `K=512` either rejects correct kernels at `K=8192` or waves
through broken ones at `K=128`.

Two details that are easy to get wrong and are handled explicitly:

- **TF32.** On Ampere and later, PyTorch may run an fp32 matmul through TF32
  tensor cores, whose 10-bit mantissa carries ~1e-3 relative error — a hundred
  times the fp32 threshold. Every fp32 comparison here runs with TF32 disabled
  on *both* sides, and `tl.dot` is pinned to `ieee`. Otherwise "fp32" would
  mean TF32 on one side and fp32 on the other, and the comparison would measure
  the precision gap rather than the kernel.
- **Shapes.** Tests run primes, one-off-a-tile sizes, single rows and single
  columns against the extreme tile shapes, because masking bugs hide behind
  clean powers of two.

## 6. Benchmark methodology

See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) for the full statement. In
brief: CUDA events on the active stream, one synchronisation after the whole
measurement loop, an L2-sized buffer zeroed before each timed iteration, 25
warmup and 200 measured iterations by default, ranked on the median. Baselines
are measured with identical settings — otherwise the speedup reported against
them is an artefact of the harness.

The timing loop is cross-checked against `torch.utils.benchmark` on a GPU, and
asserted to agree within 15%.

## 7. Results

Run them yourself:

```bash
pip install -e '.[dev,report]'

kernelforge env                                  # hardware and library versions
kernelforge tune matmul --m 2048 --n 4096 --k 4096 --dtype fp16
kernelforge benchmark matmul --suite transformer --dtype fp16
kernelforge benchmark rmsnorm --suite sweep      --dtype fp16
kernelforge compare   fused_linear --m 4096 --n 11008 --k 4096
kernelforge compare   transformer --seq 2048
kernelforge report                               # reports/summary.md + figures
```

`kernelforge tune` prints the following. The latencies are shown as dots
because they are not measurements — only the shape-derived quantities, which
do not depend on the hardware, are filled in:

```
GPU:    <your GPU>
Shape:  2048 x 4096 x 4096  (M/N/K)
dtype:  FP16

Verifying candidates... (48 to check)
.. / 48 configurations passed correctness
Benchmarking...

Search space: 432 grid points, 180 feasible, 48 measured in ..s

Best configuration
------------------
BLOCK_M:      ..
BLOCK_N:      ..
BLOCK_K:      ..
GROUP_M:       8
num_warps:    ..
num_stages:   ..

Performance
-----------
triton_autotune:     ..... ms
torch_compile:       ..... ms
torch_eager:         ..... ms
triton_baseline:     ..... ms
kernelforge:         ..... ms

speedup vs torch_eager:               ...x
speedup vs torch_compile:             ...x
speedup vs triton_baseline:           ...x
speedup vs triton_autotune:           ...x
throughput:                     ... TFLOP/s
arithmetic intensity:       1024.0 FLOP/byte
```

That last line is the one to read first. At 1024 FLOP/byte this shape is far
above an A100's ridge point of ~153, so it genuinely can be compute bound and
tuning the tiling is worth doing. A shape below the ridge point cannot be, and
the honest conclusion there is that the kernel is already finished.

`kernelforge report` reads the database and writes `reports/summary.md`,
per-operator CSV exports, latency and throughput charts, a bandwidth chart for
the memory-bound operators, and a `BLOCK_M x BLOCK_N` latency heatmap. Nothing
is transcribed by hand, so regenerating cannot go stale.

## 8. Performance analysis

Latency says which is faster; counters say why.

```bash
kernelforge profile fused_linear --m 4096 --n 11008 --k 4096      # launches + device time
kernelforge profile matmul --backend nsight --sections occupancy memory throughput stalls
```

The `torch` backend attributes device time to individual kernels and reports
launches per call — which is the direct evidence for fusion, independently of
any latency. The `nsight` backend re-invokes the CLI under `ncu` and collects
occupancy, SM and DRAM throughput against peak, L2 behaviour, register and
shared-memory usage, and warp stall reasons.

[`docs/CASE_STUDIES.md`](docs/CASE_STUDIES.md) sets out three analyses — tile
size, warp count, and fusion — each with a stated prediction, the command that
tests it, and the counters that decide it. The predictions are written down
first on purpose: a case study that reports whatever happened is not an
analysis.

One warning repeated from Nsight's own documentation: **higher occupancy does
not mean faster.** A large-tile GEMM trades occupancy for data reuse and
register residency, and often wins.

## 9. Transformer integration

```bash
kernelforge compare transformer --seq 2048 --hidden 4096 --intermediate 11008
```

A pre-norm decoder block in which both RMSNorms and the MLP activation are
swappable. `backend` is an attribute of a single module, so both paths share one
set of weights and a parity check cannot be fooled by differing initialisation.

Attention stays on `scaled_dot_product_attention` and the QKV, output and down
projections stay on `nn.Linear`, in both backends. Writing a competitive fused
attention kernel is a separate project; borrowing PyTorch's and saying so is
better than shipping a slow one.

This is the measurement most likely to be humbling, and that is why it is here.
Block latency is dominated by the attention and projection GEMMs, so the
end-to-end gain is bounded by the share of block time the swapped operators
held. A kernel twice as fast does not make a model twice as fast.

## 10. Reproducing

```bash
git clone <this repository> && cd kernelforge
pip install -e '.[dev,report]'

pytest -m "not gpu"      # the whole CPU-side framework, incl. kernel compilation
pytest                   # everything, on a machine with a CUDA device
pytest -m slow           # compile every budgeted candidate
ruff check . && ruff format --check .
```

Requires Linux, Python 3.10+, PyTorch 2.2+, and an NVIDIA GPU to *run* kernels.
Triton ships with PyTorch's CUDA builds. Nsight Compute is needed only for
`--backend nsight`, and a CUDA toolkit only for the handwritten CUDA RMSNorm,
which is built on demand and skipped when absent.

Keep the GPU fixed when comparing. A number from one card and a number from
another are not a comparison, and the database records which was which so that
mistake is at least detectable.

## Documents

- [`DESIGN.md`](DESIGN.md) — why the pieces are shaped the way they are, and
  the decisions that went the other way.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — measurement methodology in
  full, including what the numbers do not mean.
- [`docs/CASE_STUDIES.md`](docs/CASE_STUDIES.md) — the three performance
  analyses, with predictions stated up front.

## Licence

MIT. See [`LICENSE`](LICENSE).
