# Performance case studies

Three analyses: tile size, warp count, and fusion.

**The predictions below are written before the measurements, deliberately.** A
case study that reports whatever happened and calls it the expected result is
not an analysis, it is a changelog. Each study states what should happen and
why, the command that tests it, and the specific counter that would show the
prediction to be wrong. Being wrong is a result — the most interesting ones
usually are.

> **Status: the result tables in this document are empty.** They are filled by
> running the commands on a specific GPU. The repository ships no measurements,
> for the reason set out in the README: a latency without its hardware is not a
> result. Record the GPU, driver and library versions from `kernelforge env`
> alongside any numbers added here.

A note that applies to all three, from Nsight Compute's own documentation:
**higher occupancy does not imply higher performance.** Occupancy is one input
to an explanation. If a configuration wins at lower occupancy, that is the
finding, not an anomaly to be explained away.

---

## A. Tile size

### Question

How does GEMM latency vary with the output tile, and what stops the largest
tile from always winning?

### Prediction

Compare 32×32, 64×64 and 128×128 at a large shape (`M=N=K=4096`, fp16).

A `BLOCK_M × BLOCK_N` tile loads `BLOCK_K*(BLOCK_M + BLOCK_N)` elements per
K-step to produce `BLOCK_M*BLOCK_N` results, so data reuse scales as
`BM*BN/(BM+BN)`: 16 for a 32×32 tile, 32 for 64×64, 64 for 128×128. Four times
the reuse of the smallest means a quarter of the global traffic for the same
arithmetic. So latency should fall as the tile grows — until one of three
things stops it:

1. **Shared memory.** `num_stages * BLOCK_K * (BM + BN) * itemsize`. At 128×128,
   `BLOCK_K=64`, four stages, fp16, that is 128 KiB — within an A100's 163 KiB
   opt-in budget and beyond an RTX 4090's 99 KiB. On a 4090 the search space
   rejects it before compiling.
2. **Registers.** The fp32 accumulator is `BM*BN/threads` values per thread: 64
   at 128×128 with 8 warps, 256 with 2 warps. The architecture gives a thread
   255 addressable registers, so the latter spills to local memory.
3. **Parallelism.** `ceil(M/BM) * ceil(N/BN)` programs. At `M=N=4096` a
   128×128 tile gives 1024 programs, which fills any current GPU. At
   `M=N=256` it gives 4, and an A100 has 108 SMs — which is the regime the
   search-space priority function clamps for.

Expected shape of the result: at 4096³ the largest tile that fits wins, with
diminishing returns from 64×64 to 128×128 as the kernel approaches the compute
bound and reuse stops being the binding constraint. At 256² the ordering should
**reverse**, with small tiles winning on parallelism alone.

### Method

```bash
kernelforge tune matmul --m 4096 --n 4096 --k 4096 --dtype fp16
kernelforge tune matmul --m 256  --n 256  --k 1024 --dtype fp16
kernelforge report     # reports/tuning_heatmap.png is the BLOCK_M x BLOCK_N plane
```

Both runs record every candidate, so the heatmap is the whole tile plane rather
than three hand-picked points.

### Counters that decide it

| Nsight section | What to read |
| --- | --- |
| `SpeedOfLight` | SM vs DRAM throughput as a percentage of peak. The large tile should be nearer the compute roof and the small tile nearer the memory roof. |
| `MemoryWorkloadAnalysis` | DRAM bytes. Should fall roughly as reuse rises; if it does not, the tiling is not achieving the reuse the arithmetic predicts. |
| `LaunchStats` | Registers per thread and shared memory per block. A register count at the ceiling with local-memory traffic present means a spill. |
| `Occupancy` | Achieved occupancy. Expected to *fall* as the tile grows. If the large tile also wins, that is the occupancy point, measured. |

```bash
kernelforge profile matmul --m 4096 --n 4096 --k 4096 --backend nsight \
    --sections throughput memory launch occupancy
```

### Results

Results table to be filled from `results/matmul_fp16.csv` and the `ncu` output, one
row per tile at `4096³` and one per tile at `256²`, with columns: median
latency, TFLOP/s, achieved occupancy, SM and DRAM throughput as a percentage of
peak, registers per thread, and shared memory per block.

**Which prediction held, and which did not:** to be written from the
measurements, stating explicitly where the outcome differed from the prediction
above.

---

## B. Warp count

### Question

Why does the best `num_warps` change with tensor shape, when the tile is held
fixed?

### Prediction

Compare `num_warps` of 2, 4 and 8 at a fixed 64×128 tile across a large square
shape (`4096×4096×4096`), a skinny decode-shaped one (`1×11008×4096`), and a
short-K one (`4096×4096×256`).

`num_warps` sets the threads per program, which fixes two things at once:

- **Accumulator registers per thread**, `BM*BN/(32*num_warps)`: 128 at 2 warps,
  64 at 4, 32 at 8 for a 64×128 tile. Fewer warps means more values per thread,
  which is more instruction-level parallelism in the inner loop but closer to
  the register ceiling.
- **Warps available to hide latency.** More warps give the scheduler more to
  issue while a global load or an MMA is in flight.

So the prediction is shape-dependent:

- **Large square.** Abundant K iterations to pipeline and plenty of programs.
  Expect the middle of the range to win — enough warps to hide latency, enough
  registers per thread for ILP.
- **Skinny (`M=1`).** One row of output. Most of a 64-row tile is masked off, so
  a thread's useful work is tiny and the kernel is bound by loading the weight
  matrix. Expect more warps to win, because the only thing that matters is
  having loads in flight.
- **Short K.** Few K iterations means the prologue and epilogue are a large
  fraction of the kernel. Expect the ordering to be flatter and the result to
  be decided by epilogue cost rather than by the inner loop.

The counter that distinguishes the explanations is the warp stall reason. If
more warps win because of latency hiding, `WarpStateStats` should show stalls
dominated by memory dependency at low warp counts. If low warp counts lose to
spilling instead, `LaunchStats` will show it in registers per thread and local
memory traffic — a different cause with the same symptom.

### Method

```bash
for shape in "4096 4096 4096" "1 11008 4096" "4096 4096 256"; do
  set -- $shape
  kernelforge tune matmul --m $1 --n $2 --k $3 --dtype fp16
done
kernelforge report
```

The per-operator CSV export carries `cfg_num_warps` alongside the latency for
every candidate, so the comparison at fixed tile is a filter on the export
rather than a separate experiment.

### Counters that decide it

```bash
kernelforge profile matmul --m 1 --n 11008 --k 4096 --backend nsight \
    --sections stalls occupancy launch throughput
```

| Nsight section | What to read |
| --- | --- |
| `WarpStateStats` | Dominant stall reason. Memory-dependency stalls point to latency hiding; `long_scoreboard` or short-scoreboard stalls point elsewhere. |
| `LaunchStats` | Registers per thread, and whether local memory is being touched at all — the signature of a spill. |
| `Occupancy` | Theoretical against achieved, to separate "not enough warps resident" from "enough warps, all stalled". |

### Results

Results table to be filled from the `cfg_num_warps` column of
`results/matmul_fp16.csv` at fixed tile, one row per (shape, `num_warps`) pair, with
the dominant warp stall reason from `WarpStateStats` alongside.

**Which prediction held, and which did not:** to be written from the
measurements, stating explicitly where the outcome differed from the prediction
above.

---

## C. Fusion

### Question

What does folding bias and GELU into the GEMM epilogue actually save, and does
the saving match the predicted reduction in memory traffic?

### Prediction

This study has a quantitative prediction, which makes it the easiest of the
three to falsify.

The unfused sequence materialises two full `M × N` intermediates:

```
x @ w      -> tmp    write M*N
tmp + bias -> tmp2   read M*N, write M*N
gelu(tmp2) -> y      read M*N, write M*N
```

Three launches and `5*M*N` elements of output traffic, against one launch and
`M*N` fused. For `M=4096`, `N=11008`, fp16, the `4*M*N` difference is 344 MiB
of avoided round trips.

So:

1. **The fused path is one launch; the unfused path is at least three.**
   Structural rather than statistical, and asserted in
   `test_fusion_reduces_the_kernel_launch_count`. Not pinned to exactly three,
   because cuBLAS may split a GEMM across kernels.
2. **DRAM traffic drops by about `4*M*N*itemsize`.** Measurable directly in
   `MemoryWorkloadAnalysis`.
3. **Latency improves by less than the traffic ratio suggests.** The GEMM reads
   `M*K + K*N` of operands regardless, and at these shapes that is comparable
   to the output traffic, so the saving applies to a fraction of total traffic,
   not to all of it. A naive reading of "5× less output traffic" predicting a
   5× speedup would be wrong, and saying so up front is the point.
4. **`torch.compile` should capture part of this on its own.** Inductor fuses
   elementwise epilogues onto a matmul. The interesting comparison is therefore
   not fused-against-eager, which is easy, but fused-against-`torch.compile`,
   which is the real competition. If `torch.compile` matches the fused kernel,
   that is a legitimate and useful finding.

The shape dependence is predictable and worth checking: fusion should matter
most where `M*N` is large relative to `M*K + K*N`, i.e. where the output is
large compared with the operands.

### Method

```bash
kernelforge compare fused_linear --m 4096 --n 11008 --k 4096 --dtype fp16
kernelforge profile fused_linear --m 4096 --n 11008 --k 4096
kernelforge benchmark fused_linear --suite transformer --dtype fp16
kernelforge report     # reports/fusion_speedup.png: the fused kernel against each baseline, by shape
```

`kernelforge profile` with the default `torch` backend prints launches per call
and device time per kernel for the unfused sequence, `torch.compile` and the
fused kernel side by side.

### Counters that decide it

```bash
kernelforge profile fused_linear --m 4096 --n 11008 --k 4096 --backend nsight \
    --sections memory throughput launch
```

| Nsight section | What to read |
| --- | --- |
| `MemoryWorkloadAnalysis` | DRAM read and write bytes. Compare the measured difference against the predicted `4*M*N*itemsize`. |
| `SpeedOfLight` | Whether the fused kernel moved the bottleneck from memory towards compute. |
| `LaunchStats` | Register and shared-memory cost of the epilogue: it runs on the accumulator in registers, so it should be close to free. |

### Results

Results table to be filled from `kernelforge profile fused_linear` and the
`ncu` memory section: launches per call, median latency, and DRAM read/write
bytes for the unfused sequence, `torch.compile`, and the fused kernel.

Predicted output-traffic saving at 4096×11008 fp16: `4 × 4096 × 11008 × 2 B`
= 344 MiB, against total traffic of 548 MiB unfused and 204 MiB fused. This is
the one number in this document that is exact and hardware-independent, so it
is the easiest of the three predictions to falsify.

**Which prediction held, and which did not:** to be written from the
measurements, stating explicitly where the outcome differed from the prediction
above.

---

## Transformer-block context

Whatever the three studies conclude at the operator level, the block-level
measurement is the one that bounds their significance:

```bash
kernelforge compare transformer --seq 2048 --hidden 4096 --intermediate 11008
```

Only the two RMSNorms and the MLP activation are swapped; attention and the
projections are PyTorch in both backends. The end-to-end gain is therefore
bounded by the share of block time those operators held, and the expected
finding is that it is much smaller than the operator-level speedups. That is
Amdahl's law, and it is the correct conclusion rather than a disappointing one.

Results to be filled from `kernelforge compare transformer`: block median
latency for each backend, against the operator-level speedups measured above.

The gap between those two numbers is the point of the measurement.
