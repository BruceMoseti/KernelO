# Benchmark methodology

The credibility of every number KernelForge produces rests on this document.
It states exactly how latency is measured, what is held fixed, and what the
numbers do not mean.

## What is recorded with every measurement

Each row in the results database carries the run it belongs to, and each run
carries:

| Field | Source |
| --- | --- |
| GPU name, compute capability, SM count, VRAM | `torch.cuda.get_device_properties` |
| CUDA version | `torch.version.cuda` |
| Driver version | `torch._C._cuda_getDriverVersion`, falling back to `nvidia-smi` |
| PyTorch, Triton, Python versions | the installed packages |
| Platform, hostname, UTC timestamp | the host |
| dtype, tensor shapes, kernel configuration | the problem and config descriptors |
| warmup count, iteration count, timer, L2-flush flag | the harness |

`kernelforge env` prints the same snapshot. A latency without this context is
not comparable to anything, which is why it is attached at the row level rather
than written in a commit message.

## Timing

CUDA launches are asynchronous, so

```python
t0 = time.perf_counter()
kernel()
t1 = time.perf_counter()
```

measures the time to *enqueue* work, not to execute it.

The harness (`kernelforge/benchmark/runner.py`) instead:

1. Runs `warmup` iterations and synchronises. The first launch of a kernel pays
   for module loading and context setup; Triton's first call also compiles.
2. For each of `iterations` measured runs, zeroes an L2-sized buffer, records a
   start event, calls the function, records an end event.
3. Synchronises **once**, after the whole loop, then reads the elapsed time of
   each event pair.

Synchronising once rather than per iteration keeps the cost of synchronisation
out of the samples. The events are recorded on the active stream, which is
in-order, so the cache-flush kernel enqueued before `start_event` has completed
by the time the event is recorded and is not inside the measured interval.

Defaults: `warmup=25`, `iterations=200`. Both are CLI flags.

### Cross-check

`cross_check_median_ms` times the same callable with
`torch.utils.benchmark.Timer.blocked_autorange`, which handles accelerator
synchronisation and adaptive replication independently.
`test_harness_agrees_with_pytorch_benchmark` asserts the two agree within 15%
on a 2048³ fp16 GEMM. Two independent implementations agreeing is the evidence
that the loop measures the kernel rather than something else.

### Timer resolution

`torch.cuda.Event.elapsed_time` resolves to roughly half a microsecond.
`TimingResult.at_timer_resolution` flags a median at or below 2 µs and the CLI
prints a warning. Such a figure is reported but should not be compared between
implementations — the fix is a larger problem, not more iterations.

## L2 cache flushing

Re-running a kernel on the same tensors leaves the inputs resident in L2, and
for a problem whose working set fits, the reported throughput is one the kernel
would never reach in a model where the inputs are not already cached. An
L2-sized buffer (from `DeviceCaps.l2_cache_bytes`) is zeroed before each timed
iteration.

This makes small problems look slower than a naive harness reports, which is
the point. `--no-flush-l2` disables it; `TimingResult.flushed_l2` records which
was used and the flag is stored with every database row, so `kernelforge
report` detects a database holding both and prints a warning into
`summary.md` rather than quietly ranking an unflushed row above a flushed
one.

The exception is the transformer block, where the working set greatly exceeds
L2 and flushing adds noise to an already-cold measurement. That is set
explicitly in `benchmark_block` with the reason in a comment.

## Statistics

All samples are summarised as median, mean, standard deviation, min, max, p95
and p99. Percentiles use NumPy's default linear interpolation between order
statistics.

**Ranking uses the median.** The minimum is the single luckiest sample and
rewards variance. The mean is skewed by clock throttling and the occasional
preempted launch: one outlier in two hundred moves it by an order of magnitude
and leaves the median untouched.

p95 and p99 are recorded because a configuration with a good median and a bad
tail is worth knowing about even when it wins.

## Fairness of comparisons

- **Identical settings.** Baselines are timed with the same warmup, iteration
  count and L2-flush setting as the candidates. Otherwise the reported speedup
  is partly an artefact of the harness.
- **Identical inputs.** Inputs are generated once per problem from a seeded
  `torch.Generator` and shared by every implementation.
- **Identical precision.** See below.
- **`torch.compile` is warmed outside the measurement.** Compilation is
  triggered eagerly when the baseline is constructed, so Dynamo tracing and
  Inductor codegen do not land on the first measured iteration.
- **Allocation is inside the timed region for everyone.** Each implementation
  allocates its own output, as `torch.matmul` does. Excluding allocation for
  the Triton kernels and not for PyTorch would be the unfair choice.
- **One GPU.** Never compare a number from one card with a number from
  another. The database records the device per run so the mistake is at least
  detectable.

### Precision

For fp32, TF32 is disabled on both sides. On Ampere and later, PyTorch may run
an fp32 matmul through TF32 tensor cores, whose 10-bit mantissa carries ~1e-3
relative error, and `tl.dot` would do the same. Comparing a TF32 kernel against
an fp32 reference measures the precision gap rather than the kernel. The
kernels pin `input_precision="ieee"` and the tuner disables TF32 for the whole
session so the PyTorch baselines match.

Consequence worth stating plainly: the fp32 PyTorch numbers here are **slower
than what a PyTorch user gets by default**, because the default is TF32. For a
tensor-core comparison use `--dtype fp16` or `--dtype bf16`, where the question
does not arise.

## Correctness tolerances

The gate is a scale-invariant error:

```
err = max|out - ref| / max|ref|
```

against a per-dtype threshold:

| dtype | Threshold | Reasoning |
| --- | --- | --- |
| fp32 | 1e-5 | with TF32 disabled, a correct kernel differs only by summation order |
| fp16 | 5e-3 | output rounding is 2⁻¹¹ ≈ 4.9e-4; the margin covers a differing accumulation order |
| bf16 | 2e-2 | output rounding is 2⁻⁸ ≈ 3.9e-3 |

A single threshold per dtype holds across every shape because the metric is
scale invariant; an absolute tolerance would not, since a GEMM's output
magnitude grows like `sqrt(K)`.

Non-finite output fails before any arithmetic. Shape mismatch fails
immediately. An all-zero reference falls back to absolute error so the division
stays meaningful. Elementwise mismatch counts are reported as diagnostics but
are not the gate.

## Derived metrics

| Quantity | Formula |
| --- | --- |
| GEMM FLOPs | `2*M*N*K` |
| TFLOP/s | `flops / (ms * 1e-3) / 1e12` |
| GB/s | `bytes / (ms * 1e-3) / 1e9` |
| GEMM bytes | `(M*K + K*N + M*N) * itemsize` |
| RMSNorm bytes | `(2*rows*cols + cols) * itemsize` |
| Fused linear bytes | `(M*K + K*N + N + M*N) * itemsize`, against `5*M*N` of output traffic unfused |
| Arithmetic intensity | `flops / bytes` |
| Fraction of published peak | TFLOP/s or GB/s divided by the GPU's entry in `PUBLISHED_PEAKS` (`benchmark/metrics.py`) |

Byte counts are **minimum** traffic: each input read once, each output written
once. A tiled GEMM necessarily re-reads its operands, so measured bandwidth can
come out below the achievable peak. That gap is a result, not an error in the
model — it is what the memory analysis in the case studies examines.

`kernelforge report` gives KernelForge's throughput as a fraction of NVIDIA's
published dense peak for the GPU, and cites the document each peak comes from.
The roof is Tensor Core FP16/BF16 math with FP32 accumulate for fp16 and bf16
(half the FP16-accumulate figure on GeForce cards), FP32 without Tensor Cores
for fp32, since the kernels pin `tl.dot` to IEEE, and DRAM bandwidth for the
memory-bound operators. These are spec-sheet figures at boost clock, not
measurements. A GPU with no entry gets no fraction rather than an estimate.

Compute-bound operators are reported in TFLOP/s and memory-bound ones in GB/s,
selected by `Operator.is_memory_bound()`. Reporting TFLOP/s for RMSNorm would
be meaningless; it does O(1) arithmetic per element and its ceiling is the
memory system.

Arithmetic intensity is worth comparing against the GPU's ratio of peak FLOP/s
to peak GB/s. An A100 at 312 TFLOP/s fp16 and 2039 GB/s has a ridge point near
153 FLOP/byte: below that, no amount of kernel tuning gets past the memory
system, and the honest conclusion is that the kernel is already done.

## Profiling

Latency comes from the CUDA-event harness. The profiler is for attribution.

- `torch.profiler` reports kernel launches per call and device time per kernel.
  CUDA is warmed before the profiled region. Tracing adds overhead, so these
  figures are not used as headline latency.
- Nsight Compute serialises and replays launches to collect counters, so an
  `ncu` run's wall-clock time means nothing. Counters are collected for the few
  configurations a case study compares, never for a sweep.

## What these numbers do not mean

- **Not a model-level speedup.** Operator latency is not block latency, and
  block latency is not tokens per second. `kernelforge compare transformer`
  exists to show the difference.
- **Not transferable across GPUs.** A configuration tuned on one board can be
  wrong on another of the same architecture. That is why the cache key includes
  the board name.
- **Not a statement about your shapes.** A GEMM tuned at 2048×4096×4096 says
  little about the 1×4096×11008 skinny GEMM that dominates single-stream
  decoding. The `transformer` workload suite exists for that reason.
- **Not stable under thermal throttling.** A long sweep on a card that throttles
  measures the thermal envelope as much as the kernels. p99 against the median
  is the signal to look at.
