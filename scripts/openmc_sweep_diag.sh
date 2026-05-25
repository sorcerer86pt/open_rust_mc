#!/bin/bash
# Run openmc_scene_runner on 4 representative ICSBEP cases for the
# engine-vs-OpenMC residual broad-check.

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate openmc

REPO=/mnt/c/Users/fog/madman_svd_experiment
XS=$REPO/data/endfb-viii.1-hdf5/cross_sections.xml
OUT=$REPO/outputs/openmc_diag
mkdir -p "$OUT"

CASES=(
  heu-met-fast-019_case-1
  heu-met-fast-069_case-1
  heu-met-fast-027
  heu-sol-therm-004_case-1
)

for case in "${CASES[@]}"; do
  echo "==================================="
  echo "=== $case ==="
  echo "==================================="
  WORK=/tmp/openmc_${case//[^a-zA-Z0-9_]/_}
  rm -rf "$WORK"
  mkdir -p "$WORK"
  cd "$WORK"
  timeout 600 python "$REPO/scripts/openmc_scene_runner.py" \
    "$REPO/bench/icsbep/${case}.json" \
    "$OUT/${case}_omc.json" \
    --particles 5000 --batches 50 --inactive 15 --seeds 1 \
    --cross-sections "$XS" 2>&1 | grep -E "k_ref|seed 0|mean k|Delta|Δ" | head -5 || true
done
