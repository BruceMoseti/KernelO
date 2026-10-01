# KernelForge

Hardware-aware GPU kernel autotuning and inference optimization for transformer workloads.

> **Status:** under construction (milestone M1, foundation). No GPU measurements have been taken
> yet, so this repository contains no performance numbers.

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
