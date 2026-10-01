#!/usr/bin/env bash
# Every GPU check plus one tuning run, in one command, for a GPU machine without
# the self-hosted runner that .github/workflows/gpu-validation.yml needs.
#
# Run from a checkout with the package installed, on the GPU whose results will
# be reported. Everything goes to results/<UTC timestamp>_<gpu>/, which has its
# own results database and config cache:
#
#   nvidia-smi.txt    GPU, driver, clocks, power and temperature at the start
#   pip-freeze.txt    exact Python package versions
#   environment.txt   kernelforge env
#   pytest.log        the whole suite, GPU-only and slow tests included
#   tune*.txt         kernelforge tune matmul, then a repeat that must hit the cache
#   kernelforge.db    every candidate and baseline, with hardware provenance
#   configs.json      the tuned configuration
#
# Stops at the first failure, so nothing is tuned on kernels that failed their
# tests. GPU clocks are not locked, because that needs root.
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

kf=(python -m kernelforge.cli.main)
nvidia-smi -q > "$out/nvidia-smi.txt"
python -m pip freeze > "$out/pip-freeze.txt"
"${kf[@]}" env > "$out/environment.txt"

python -m pytest -ra 2>&1 | tee "$out/pytest.log"

tune=(tune matmul --m 2048 --n 4096 --k 4096 --dtype fp16
      --db "$out/kernelforge.db" --cache "$out/configs.json")
"${kf[@]}" "${tune[@]}" | tee "$out/tune.txt"
"${kf[@]}" "${tune[@]}" | tee "$out/tune-repeat.txt"
if ! grep -q "Cache hit" "$out/tune-repeat.txt"; then
    echo "error: the repeated tune did not hit the cache" >&2
    exit 1
fi

echo "Done. Results for ${gpu} are in ${out}"
