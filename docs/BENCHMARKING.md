# Benchmarking methodology

> **Status:** implemented, not yet validated on GPU hardware. No measurements have been taken,
> so this document describes the method only; it contains no results.

The harness lives in `kernelforge/benchmark/runner.py` (`benchmark()`), with statistics and
throughput formulas in `kernelforge/benchmark/metrics.py`.

## What every measurement records

| Category  | Fields                                                                                   |
|-----------|------------------------------------------------------------------------------------------|
| Hardware  | GPU model, compute capability, SM count, GPU memory, L2 size, NVIDIA driver version      |
| Software  | CUDA version PyTorch was built with, PyTorch, Triton, Python, OS, CPU model               |
| Harness   | warmup iterations, timed iterations, L2 flush buffer size, every raw sample               |

`benchmark()` returns these with the latency statistics. The caller records what the harness
cannot know about the callable: operation, dtype, tensor shapes, and kernel configuration.

The GPU must stay fixed for any comparison. Results measured on different GPUs, driver versions,
or software stacks are not compared with each other.

## Timing

**Device time from CUDA events.** CUDA kernels launch asynchronously, so a host timer around a
call measures launch overhead, not execution. Each timed call is bracketed by two CUDA events
recorded on the current stream:

```
synchronize
repeat warmup times:      zero flush buffer; fn()
repeat iterations times:  zero flush buffer; start.record(); fn(); end.record()
synchronize
sample_i = start_i.elapsed_time(end_i)
```

There is no synchronization inside the loop. Elapsed times are read once, after the final
synchronize. The callable must launch its work on the current stream and must not synchronize
(see the queue check below).

**Warmup.** Warmup calls run the same loop body as timed calls. The first call JIT-compiles the
Triton kernel (or triggers cuBLAS heuristics and allocator growth), so `warmup >= 1` is enforced
to keep compilation out of the samples. The default is 25 warmup and 200 timed iterations.

**L2 cache state: cold.** Before every call, outside the timed region, the harness zeroes a
buffer of `max(256 MiB, 2 × L2 size)`. Every call therefore starts with inputs evicted from L2,
which is what an inference workload sees when weights stream from DRAM. Without this flush, any
input smaller than L2 would be served from cache after the first iteration. That would overstate
bandwidth for memory-bound kernels. The buffer is twice the L2 size because L2 replacement is not strict LRU. The 256 MiB
floor matches `triton.testing.do_bench`. One side effect: the flush leaves dirty lines in L2,
which are written back while the timed kernel runs. PyTorch's and Triton's benchmarkers flush
the same way.

**Host launch latency is kept out of the samples, and this is checked on every run.** If the GPU
is idle when `start.record()` executes, the interval up to `end` includes the time the host spends
launching `fn`. The flush preceding each call keeps the GPU busy while the host enqueues the next
call. Because the loop never synchronizes, the GPU stays behind the host. After enqueuing the
last call, the harness asks whether the last start event has already executed
(`Event.query()`). If it has, the GPU may have waited on the host inside a timed region, and the
harness emits a `RuntimeWarning`. A callable that synchronizes internally triggers this warning,
and a test covers that case.

**Variance and statistics.** All samples are kept. The harness reports the median, mean, p95,
p99, minimum, maximum, and sample standard deviation. Percentiles use linear interpolation, which
is NumPy's default. The **median** is the ranking statistic because it is robust to occasional
outliers such as clock changes, interrupts, or the host being descheduled. The tail statistics
show how noisy a measurement was. With 200 samples, p99 is set by the two or three largest
samples, so treat it as a rough tail indicator.

**Resolution.** CUDA documents event timing resolution as about 0.5 µs. For kernels that take a
few microseconds, that is a visible fraction of each sample. The full distribution is reported for
this reason.

**Clocks.** The harness does not lock GPU clocks, because that requires root. Boost clocks
depend on temperature and power, which adds run-to-run variance. For low-variance comparisons,
lock the clocks with `nvidia-smi --lock-gpu-clocks`, and benchmark every implementation in a
comparison in the same session on the same GPU.

**Cross-checks** run as GPU-only tests in `tests/test_runner.py`; they are not yet run on
hardware:

- `triton.testing.do_bench` uses the same method, so medians must agree within 10%.
- `torch.utils.benchmark.Timer` uses host timing with a warm L2. On a compute-bound,
  multi-millisecond MatMul, GPU execution dominates both, so medians must agree within 15%.

## Correctness

A kernel output is only timed or ranked after `kernelforge.testing.verify` accepts it. The
reference is computed in **float64 from the same low-precision inputs** the kernel received.
Upcasting is exact, and float64's own error is negligible, so the measured error belongs to the
kernel. Each element must satisfy

```
|actual − reference| ≤ rtol(dtype) · |reference| + atol · rms(reference) + spacing(dtype)
```

| dtype | rtol            | atol (× rms of reference) |
|-------|-----------------|---------------------------|
| fp32  | 2⁻¹⁶ (128 ulps) | 2⁻¹²                      |
| fp16  | 2⁻¹⁰ (1 ulp)    | 2⁻¹²                      |
| bf16  | 2⁻⁷ (1 ulp)     | 2⁻¹²                      |

- **rtol** covers rounding to the output dtype, which is at most half an ulp, plus fp32
  arithmetic inside the kernel. For fp32 outputs there is no rounding step, so the budget covers
  approximate `exp` and division, softmax argument reduction, and reduction-order error.
- **atol × rms(reference)** covers outputs that are close to zero because of cancellation in a
  reduction. Their absolute error comes from the fp32 accumulator and scales with the typical
  output size, not with the element itself. Scaling by rms makes the check scale-invariant.
- **spacing** is the subnormal spacing of the output dtype: the smallest normal number times
  eps. fp16 cannot represent values below its smallest normal number more finely than this.
  Long fp16 softmax rows hit this, and the 32768-column test fails without it.

`tests/test_testing.py` checks both directions on CPU:

- Correct results, accumulated in fp32 and rounded to the output dtype, pass for reduction
  lengths up to 4096. They use less than 0.6 of the budget; rounding to fp16 or bf16 alone uses
  about half.
- These injected bugs fail: a dropped K term, fp16 or bf16 accumulation, TF32-rounded inputs
  checked as IEEE fp32, a single zeroed element, a NaN, softmax padding loaded as 0 instead of
  −∞, and an unwritten tail element.

Some bugs cannot be detected from a low-precision output at all. For example, an error smaller
than fp16 resolution in a long softmax row looks identical to a correct result.

Tensor cores truncate inside each MMA instead of rounding, so GPU accumulation error can exceed
the CPU-simulated error used in these tests. The GPU kernel tests use the same tolerances and
will show whether the headroom holds. They have not been run yet.

## Metrics

| Kernel class                          | Reported                                                     |
|---------------------------------------|--------------------------------------------------------------|
| Compute-bound (MatMul)                | latency; TFLOP/s = 2·M·N·K / t / 10¹²                        |
| Memory-bound (vector add, softmax)    | latency; effective bandwidth = minimum bytes moved / t / 10⁹ |

"Minimum bytes moved" counts each input read once and each output written once. That is the
traffic an ideal kernel would generate, so effective bandwidth can be compared with the GPU's peak
DRAM bandwidth. Bandwidth uses GB = 10⁹ bytes, the unit GPU memory bandwidth is quoted in.
