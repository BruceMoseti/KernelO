#!/usr/bin/env bash
# Full GPU validation and benchmark run.
#
# Run from a checkout with the package installed (pip install -e ".[dev]") on the GPU whose
# results will be reported. Every comparison must come from one GPU. Output goes to
# results/<UTC timestamp>_<gpu>/:
#
#   nvidia-smi.txt    GPU, driver, clocks, power and temperature at the start of the run
#   pip-freeze.txt    exact Python package versions
#   pytest.log        the whole test suite on the GPU: compiled kernels, GPU-only tests enabled
#   vector_add.csv    benchmarks/vector_add.py
#   softmax.csv       benchmarks/softmax.py
#   tune_*.txt        kernelforge tune matmul reports, then a repeat that must hit the cache
#   tuning.db         every tuning candidate and comparison, with full metadata
#
# The script stops at the first failure, so no benchmark runs on kernels that failed their tests.
# GPU clocks are not locked, because that needs root. For lower variance, run
# `sudo nvidia-smi --lock-gpu-clocks=<MHz>` first and `sudo nvidia-smi --reset-gpu-clocks`
# afterwards.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${TRITON_INTERPRET:-0}" == "1" ]]; then
    echo "error: unset TRITON_INTERPRET; this run must test compiled GPU kernels" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 \
    || ! python -c "import sys, torch; sys.exit(not torch.cuda.is_available())"; then
    echo "error: needs an NVIDIA GPU visible to PyTorch" >&2
    exit 1
fi

gpu=$(python -c "import torch; print(torch.cuda.get_device_name())")
slug=$(echo "$gpu" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
out="results/$(date -u +%Y%m%dT%H%M%SZ)_${slug}"
mkdir -p "$out"
echo "Writing results for ${gpu} to ${out}"

nvidia-smi -q > "$out/nvidia-smi.txt"
pip freeze > "$out/pip-freeze.txt"

pytest -ra 2>&1 | tee "$out/pytest.log"

python benchmarks/vector_add.py --output "$out/vector_add.csv"
python benchmarks/softmax.py --output "$out/softmax.csv"

shape=(--m 2048 --n 4096 --k 4096)
for dtype in fp16 bf16 fp32; do
    kernelforge tune matmul "${shape[@]}" --dtype "$dtype" --db "$out/tuning.db" \
        | tee "$out/tune_matmul_2048x4096x4096_${dtype}.txt"
done
kernelforge tune matmul "${shape[@]}" --dtype fp16 --db "$out/tuning.db" \
    | tee "$out/tune_matmul_2048x4096x4096_fp16_repeat.txt"
grep -q "Cache hit" "$out/tune_matmul_2048x4096x4096_fp16_repeat.txt"
kernelforge cache list --db "$out/tuning.db" | tee "$out/cache_list.txt"

echo "Done. Results for ${gpu} are in ${out}"
