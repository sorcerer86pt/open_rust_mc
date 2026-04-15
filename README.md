# open_rust_mc — Cache-Resident Cross-Section Reconstruction via SVD

A pure-Rust Monte Carlo neutronics engine that replaces multi-gigabyte
pointwise cross-section tables with SVD-compressed basis vectors.
Includes a head-to-head "honesty test" against pointwise table lookup
and OpenMC, with multi-seed statistical benchmarking.

## Key Results (10 independent seeds, 1M particles/batch)

| Metric | SVD (rank=5) | Pointwise Table | OpenMC |
|--------|-------------|-----------------|--------|
| Godiva k_eff | 1.00012 +/- 0.00007 | 0.99905 +/- 0.00013 | 0.99844 +/- 0.00006 |
| Delta from experiment | **12 pcm** | 95 pcm | 156 pcm |
| ns/particle (quiet) | 700 +/- 8 | 832 +/- 11 | — |
| SVD speedup | 1.19x | baseline | — |
| SVD fidelity cost | 107 pcm vs table | — | — |
| GPU kernel speedup | 8.7x vs CPU | — | — |
| Resonance integral error | < 0.2% all groups | exact | exact |

Note: timing was on a shared desktop workstation. k_eff is stable across
all seeds; timing variance is from system load. On a dedicated HPC node
the timing standard deviations would be substantially smaller.

| Metric | Value |
|--------|-------|
| CPU kernel speedup | 8-13x (isolated reconstruction benchmark) |
| Memory reduction | 530x hybrid SVD+WMP (15 KB vs 7.8 MB) |
| Rank-1 reactions | 47 of 52 U-235 channels |

## Physics Implemented

The engine reads OpenMC HDF5 nuclear data files directly and implements:

- **SVD-compressed cross-sections** for all reaction channels (MT=2, 4, 16-18, 51-91, 102)
- **Ducru temperature interpolation** (kernel reconstruction, Ducru et al. 2017)
- **Energy-dependent nu-bar** (total = prompt + delayed neutron yields from HDF5)
- **Anisotropic scattering** (tabular mu/CDF angular distributions with stochastic interpolation)
- **Data-driven fission spectrum** (continuous tabulated outgoing energy distributions)
- **Discrete inelastic levels** (MT=51-91 with real Q-values, two-body kinematics)
- **Continuum inelastic** (MT=91 evaporation spectrum)
- **URR probability tables** (20-band sampling, multiply_smooth + absolute modes)
- **Free gas thermal scattering** (Maxwell-Boltzmann target velocity below 400*kT)
- **(n,2n) and (n,3n) reactions** (MT=16, MT=17)
- **Void streaming** (particles free-stream through void gaps)
- **Auto-detect tracking mode** (surface vs delta tracking based on geometry)
- **Rayon parallel transport** (8.7x speedup over single-threaded)

## Optimisations

- **f32 SVD basis** with f64 accumulator (halves memory, zero accuracy loss)
- **Arc shared energy grids** across reactions per nuclide (saves 111 MB)
- **exp2(x * LOG2_10)** replacing powf(10, x) (3-5x faster transcendental)
- **Stack-allocated collision buffers** (eliminates 20M heap allocs per simulation)
- **Single binary search** per nuclide per collision (5 of 6 eliminated)
- **Hash-based O(1) energy index** (Brown 2014, 8192 bins) for SVD path
- **CUDA GPU reconstruction** via cudarc (8.7x on RTX A1000, zero error)

## Structure

```
rust_prototype/    Pure-Rust engine (reads HDF5 natively via hdf5-pure)
  src/
    physics/       Collision processing, scattering kinematics
    transport/     Particle tracking, k-eigenvalue solver, XS providers
    geometry/      CSG geometry, BVH acceleration
    hdf5_reader.rs Pure-Rust HDF5 reader with single-pass caching
    kernel.rs      SVD reconstruction hot path (FMA + hash lookup + Ducru)
    gpu.rs         CUDA GPU reconstruction (feature-gated)
    table.rs       Pointwise table lookup (OpenMC-style baseline)
  src/bin/
    godiva.rs      Godiva eigenvalue benchmark (--mode svd|table|both --seeds N)
    pwr_pincell.rs PWR pin cell benchmark (8 nuclides, 3 materials)
    gpu_bench.rs   GPU reconstruction benchmark (--features cuda)
    bench_mem.rs   Memory/speed comparison tool
scripts/           Python analysis pipeline
  honesty_test.py  Three-way comparison: SVD vs Table vs OpenMC
  resonance_integral_validation.py  Resonance integral accuracy
cuda_bench/        Standalone CUDA benchmark kernel
paper/             LaTeX manuscript (18 pages)
```

## Quick Start

### Honesty test (SVD vs Table, head-to-head)
```bash
cd rust_prototype
cargo run --release --bin godiva -- ../data/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 50 --inactive 10 --particles 20000
```

### Multi-seed statistical benchmark
```bash
cargo run --release --bin godiva -- ../data/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 150 --inactive 20 --particles 1000000 --seeds 10
```

### GPU benchmark (requires CUDA toolkit)
```bash
cargo run --release --features cuda --bin gpu_bench -- \
  ../data/endfb-vii.1-hdf5/neutron --rank 5 --particles 1000000
```

### Run tests
```bash
cargo test --lib    # 32 tests
```

### PWR pin cell (8 nuclides, 3 materials)
```bash
cargo run --release --bin pwr_pincell -- ../data/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 100 --inactive 20 --particles 50000
```

### Rank sweep (accuracy vs speed tradeoff)
```bash
for RANK in 2 3 4 5 6; do
  echo "=== Rank $RANK ==="
  cargo run --release --bin godiva -- ../data/endfb-vii.1-hdf5/neutron \
    --mode svd --rank $RANK --batches 50 --inactive 10 --particles 20000 --seeds 3
done
```

### Resonance integral validation (requires Linux; WSL on Windows)
```bash
conda activate openmc
python scripts/resonance_integral_validation.py
```

### Three-way comparison: SVD vs Table vs OpenMC (requires Linux; WSL on Windows)
```bash
conda activate openmc
python scripts/honesty_test.py --particles 1000000 --batches 150
```

### Nuclear data
Download ENDF/B-VII.1 HDF5 from https://openmc.org/data/ and extract to `data/`.

For detailed benchmark instructions and expected results, see [BENCHMARKS.md](BENCHMARKS.md).

## Godiva k_eff Progression

| Physics | k_eff | Delta from expt |
|---------|-------|----------------|
| Constant nu-bar, isotropic, Watt spectrum | 0.994 | 600 pcm |
| + Energy-dependent nu-bar from HDF5 | 1.059 | 5900 pcm |
| + Anisotropic scattering (tabular CDF) | 0.965 | 3500 pcm |
| + Data-driven fission spectrum | 1.006 | 600 pcm |
| + URR probability tables | 1.007 | 700 pcm |
| + Correlated CDF interpolation | 1.000 | 37 pcm |
| + Phase 1 optimisations (10 seeds) | **1.00012** | **12 pcm** |

## References

- Ducru et al., "Kernel reconstruction methods for Doppler broadening," J. Comput. Phys. 335 (2017) 535-557
- Tramm et al., "Performance Portable MC Particle Transport on Intel, NVIDIA, and AMD GPUs," EPJ Web Conf. 302 (2024) 04010
- Brown, "New Hash-Based Energy Lookup Algorithm for MC Codes," Trans. ANS 111 (2014) 659-662

## Paper

See `paper/svd_cross_section_compression.tex` — "Cache-Resident Cross-Section
Reconstruction via Singular Value Decomposition for Monte Carlo Neutron Transport"

Full reproducible benchmarks: see [BENCHMARKS.md](BENCHMARKS.md)

## License

MIT
