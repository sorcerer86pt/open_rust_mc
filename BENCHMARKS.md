# Benchmarks

How to reproduce all benchmark results from the paper.

## Prerequisites

```bash
cd rust_prototype
cargo build --release

# For GPU benchmarks:
cargo build --release --features cuda

# Nuclear data: ENDF/B-VII.1 HDF5 from https://openmc.org/data/
# Extract to data/endfb-vii.1-hdf5/
DATA=../data/endfb-vii.1-hdf5/neutron
```

## 1. Quick Smoke Test (~10 seconds)

Verify everything works before running long benchmarks.

```bash
cargo run --release --bin godiva -- $DATA --mode both --rank 5 \
  --batches 15 --inactive 5 --particles 2000
```

Expected: SVD k_eff ~1.0, Table k_eff ~1.0, SVD speedup ~1.3x.

## 2. Statistical Benchmark — Godiva (~40 minutes)

The proper benchmark: 10 independent seeds, 1M particles, 150 batches.
Reports mean +/- stddev of ns/particle — the number that matters.

```bash
cargo run --release --bin godiva -- $DATA --mode both --rank 5 \
  --batches 150 --inactive 20 --particles 1000000 --seeds 10
```

Expected output:
- SVD: k_eff = 1.00017 +/- 0.00005, ~850 ns/particle
- Table: k_eff = 0.99905 +/- 0.00013, ~1170 ns/particle
- SVD speedup: ~1.4x
- Fidelity cost: 111 pcm

For a quick version (~5 minutes):
```bash
cargo run --release --bin godiva -- $DATA --mode both --rank 5 \
  --batches 30 --inactive 10 --particles 10000 --seeds 3
```

## 3. Rank Sweep — Accuracy vs Speed Tradeoff

Test different SVD ranks to see the fidelity-speed curve.

```bash
for RANK in 2 3 4 5 6; do
  echo "=== Rank $RANK ==="
  cargo run --release --bin godiva -- $DATA --mode svd --rank $RANK \
    --batches 50 --inactive 10 --particles 20000 --seeds 3
done
```

## 4. GPU Reconstruction Benchmark

Compares CPU vs GPU for the SVD dot product kernel only (not full transport).
Requires NVIDIA GPU + CUDA toolkit.

```bash
# 1M particles, rank 5, U-235 fission
cargo run --release --features cuda --bin gpu_bench -- $DATA \
  --rank 5 --particles 1000000

# Vary particle count
for N in 100000 500000 1000000 5000000; do
  cargo run --release --features cuda --bin gpu_bench -- $DATA \
    --rank 5 --particles $N
done

# Vary rank
for RANK in 2 3 4 5 6; do
  cargo run --release --features cuda --bin gpu_bench -- $DATA \
    --rank $RANK --particles 1000000
done
```

Expected (RTX A1000, small-L3 system):
- CPU: ~37 ns/particle
- GPU: ~4 ns/particle
- Speedup: ~9x
- Error: machine epsilon (~1e-16)

On better GPUs (RTX 3080, A100, H100) expect 20-50x.

## 5. PWR Pin Cell — Multi-Material (~15 minutes)

Tests 8 nuclides across 3 materials with heterogeneous geometry.
This is where XS lookup fraction of runtime increases.

```bash
cargo run --release --bin pwr_pincell -- $DATA --mode both --rank 5 \
  --batches 100 --inactive 20 --particles 50000
```

## 6. Memory Comparison

The bench_mem binary reports SVD vs table memory at various ranks.

```bash
cargo run --release --bin bench_mem -- $DATA/U235.h5 $DATA/U238.h5
```

## 7. Resonance Integral Validation (requires OpenMC in WSL)

Computes integral sigma(E) dE/E over standard energy groups.
More physically meaningful than pointwise max error.

```bash
# In WSL with conda:
conda activate openmc
cd scripts
python resonance_integral_validation.py
```

Expected: <0.2% error on all groups at rank 5.

## 8. Three-Way Comparison — SVD vs Table vs OpenMC (requires WSL)

Full honesty test including OpenMC reference.

```bash
conda activate openmc
python scripts/honesty_test.py --particles 1000000 --batches 150 --seeds 10
```

## 9. Unit Tests

```bash
cargo test --lib          # 32 tests
cargo test --lib -- -q    # quiet mode
```

Key tests:
- `godiva_eigenvalue_smoke_test` — basic k_eff sanity
- `void_streaming_pincell_geometry` — particles cross void gaps
- `tracking_mode_single_material_is_surface` — auto-detect picks surface for Godiva
- `tracking_mode_high_contrast_falls_back` — high XS contrast falls back to surface
- `different_seeds_produce_different_results` — seed independence

## Hardware Used for Paper Results

Two systems, labelled by CPU L3 capacity and GPU class:

- **Large-L3**: AMD Ryzen 7 9800X3D (8-core, 96 MB L3 3D V-cache)
  + NVIDIA RTX 3080 (68 SMs, 10 GB, 5 MB L2), 64 GB DDR5-6000,
  Windows 11
- **Small-L3** (Dell Precision 5570): Intel Core i7-12800H (12th-gen
  Alder Lake, 6 P + 8 E cores, 24 MB L3 Smart Cache) + NVIDIA RTX
  A1000 (Ampere, 16 SMs, 4 GB, 1.5 MB L2), 32 GB DDR4
- Nuclear data: ENDF/B-VII.1 HDF5 (5.8 GB)
- OpenMC: v0.15.3 (WSL Ubuntu 24.04)

Note: CPU timing has high variance on consumer hardware. For
publication-quality numbers, run on a dedicated HPC node with CPU
pinning and no competing processes.

## Key Numbers Summary

| Benchmark | SVD (rank=5) | Table | OpenMC |
|-----------|-------------|-------|--------|
| Godiva k_eff | 1.00017 +/- 0.00005 | 0.99905 +/- 0.00013 | 0.99844 +/- 0.00006 |
| Delta from experiment | 17 pcm | 95 pcm | 156 pcm |
| ns/particle (10 seeds) | 859 +/- 237 | 1172 +/- 115 | — |
| SVD speedup vs table | 1.36x | — | — |
| GPU kernel speedup | 8.7x vs CPU | — | — |
| XS memory (3 nuclides) | 127 MB | 108 MB | 817 MB (process) |
| Resonance integral err | <0.2% | exact | exact |
| SVD fidelity cost | 111 pcm vs table | — | — |
