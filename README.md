# KernelForge

Hardware-aware GPU kernel autotuning and inference optimization for transformer workloads.

> **Status:** milestone M1 (foundation) is implemented. Kernel logic is verified on CPU with
> Triton's interpreter, and every kernel is compiled for Ampere (sm_80) and Hopper (sm_90) to
> check what the compiler generates. Nothing has run on a GPU yet, so this repository contains no
> performance numbers. `scripts/gpu_validate.sh` runs the full GPU validation once hardware is
> available.

## What M1 contains

| Component | Code | Verified on CPU | Still needs a GPU |
|---|---|---|---|
| Benchmark harness: CUDA events, cold L2, latency distribution, hardware metadata | `kernelforge/benchmark/runner.py` | statistics, input checks | the timings themselves, cross-checked against `triton.testing.do_bench` and `torch.utils.benchmark` |
| Correctness harness: float64 reference, per-dtype tolerances | `kernelforge/testing.py` | yes | headroom under tensor-core accumulation |
| Triton vector add and fused softmax | `kernelforge/kernels/` | fp32 and fp16; bf16 softmax | bf16 vector add, large shapes, performance |
| Tiled MatMul | `kernelforge/kernels/matmul.py` | fp32 and fp16, all search-space tile shapes, out-of-bounds guard bands | bf16, large shapes, warps and stages, performance |
| `KernelConfig`, search space, pruning rules | `kernelforge/tuning/config.py`, `search.py` | yes | whether pruning ever drops the true best |
| Tuner, SQLite results, config cache | `kernelforge/tuning/` | yes, with a stand-in timer | tuning with real timings |
| CLI | `kernelforge/cli/main.py` | `cache` commands; `tune` flow with stand-ins | `tune matmul` on hardware |
| Compiled code for real GPUs | `tests/test_compile.py` | every kernel compiles for sm_80 and sm_90; 16-bit MatMul lowers to tensor-core MMA (`mma.sync`, `wgmma`) and IEEE fp32 does not; the K loop is pipelined for 16-element-aligned operands; compiled shared memory stays within the pruning bound | running the compiled code |

bf16 vector add and bf16 MatMul are GPU-only because Triton 3.8's interpreter does bf16
arithmetic on the raw storage bits.

## Usage (needs an NVIDIA GPU)

```bash
kernelforge tune matmul --m 2048 --n 4096 --k 4096 --dtype fp16
kernelforge cache list
kernelforge cache clear
```

`tune matmul` does the following:

1. Prunes the search space.
2. Verifies every candidate against a float64 reference.
3. Benchmarks the correct candidates.
4. Records every result in SQLite (`~/.cache/kernelforge/tuning.db`; change it with `--db`).
5. Caches the fastest configuration for this GPU, Triton version, dtype and shape.
6. Times PyTorch eager, `torch.compile`, the fixed-config Triton baseline and the tuned kernel
   back to back.

Running the same problem again on the same GPU reuses the cached configuration; pass
`--retune` to tune again. `cache clear` forgets tuned configurations but keeps the measurement
history.

## Development setup

Linux, Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate

# GPU machine: the default PyPI torch wheel bundles CUDA and a matching Triton.
pip install -e ".[dev]"

# CPU-only machine: install the CPU torch wheel first.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

Without a CUDA GPU, the test suite runs Triton kernels in Triton's CPU interpreter
(`TRITON_INTERPRET=1`, set automatically) and skips GPU-only tests with a reason. The interpreter
checks kernel logic (indexing, masking, arithmetic), not GPU behavior or performance.

## GPU validation

`scripts/gpu_validate.sh` runs the following on one GPU and writes everything to
`results/<timestamp>_<gpu>/`:

- The whole test suite with compiled kernels.
- The vector add and softmax benchmarks.
- `tune matmul` for 2048×4096×4096 in fp16, bf16 and fp32, plus a repeat that must hit the
  cache.

## Documentation

- [docs/BENCHMARKING.md](docs/BENCHMARKING.md): timing methodology, L2 handling, correctness
  tolerances and their justification, and metrics.
