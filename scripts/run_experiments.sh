#!/usr/bin/env bash
# Produce the full result set from a clean database.
#
# One script so that a result set is reproducible by name rather than by
# remembering which commands were run in which order. Everything lands in the
# results database and is rendered by the final report step; nothing is
# transcribed by hand.
#
# Runs 12 tuning sweeps, 5 benchmark suites, a profile pass and two transformer
# comparisons, so it is long -- most of it compiling tuning candidates. The
# runtime has not been measured on real hardware, so no estimate is quoted
# here; use SUITE=smoke for a quick pass.

set -euo pipefail

DTYPE="${DTYPE:-fp16}"
SUITE="${SUITE:-sweep}"
KF="${KF:-python -m kernelforge.cli.main}"

echo "=== Environment ==="
$KF env

echo
echo "=== Tuning the flagship shapes ==="
# Transformer-shaped GEMMs: attention projection, MLP up, MLP down, at prefill
# and at a single decode step. The skinny M=1 cases are the ones a square-shape
# sweep would miss, and they are what single-stream decoding actually issues.
for shape in "2048 4096 4096" "4096 4096 4096" "2048 11008 4096" "2048 4096 11008" "1 4096 4096" "1 11008 4096"; do
  read -r m n k <<<"$shape"
  $KF tune matmul --m "$m" --n "$n" --k "$k" --dtype "$DTYPE"
done

for shape in "2048 11008 4096" "1 11008 4096"; do
  read -r m n k <<<"$shape"
  $KF tune fused_linear --m "$m" --n "$n" --k "$k" --dtype "$DTYPE"
done

for cols in 768 2048 4096 8192; do
  $KF tune rmsnorm --rows 4096 --cols "$cols" --dtype "$DTYPE"
done

echo
echo "=== Benchmark suites ==="
for op in matmul rmsnorm fused_linear softmax vector_add; do
  $KF benchmark "$op" --suite "$SUITE" --dtype "$DTYPE"
done

echo
echo "=== Fusion attribution ==="
$KF profile fused_linear --m 4096 --n 11008 --k 4096

echo
echo "=== Transformer block ==="
$KF compare transformer --seq 2048
$KF compare transformer --seq 1

echo
echo "=== Tuned configurations ==="
$KF cache list

echo
echo "=== Report ==="
$KF report

echo
echo "Done. See reports/summary.md."
