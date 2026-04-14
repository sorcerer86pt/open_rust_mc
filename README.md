# open_rust_mc — Cache-Resident Cross-Section Reconstruction via SVD

A pure-Rust Monte Carlo neutronics engine that replaces multi-gigabyte
pointwise cross-section tables with cache-resident SVD basis vectors,
achieving 8-13x CPU speedup and 2.6-2.8x GPU speedup over traditional
table lookup.

## Key Results

| Metric | Value |
|--------|-------|
| Godiva k_eff | 0.99963 +/- 0.00091 (37 pcm from experiment) |
| OpenMC reference | 0.99857 |
| CPU XS speedup | 8-13x (3-5 ns/point vs 40 ns/point table lookup) |
| GPU XS speedup | 2.6-2.8x (RTX A1000) |
| Memory reduction | 530x hybrid SVD+WMP (15 KB vs 7.8 MB) |
| Rank-1 reactions | 47 of 52 U-235 channels |
| Simulation speed | 633 ms for 80 batches x 10k particles (rayon parallel) |
| Data loading | 2.8 s for 3 nuclides (single-pass HDF5) |

## Physics Implemented

The engine reads OpenMC HDF5 nuclear data files directly and implements:

- **SVD-compressed cross-sections** for all reaction channels (MT=2, 4, 16-18, 51-91, 102)
- **Energy-dependent nu-bar** (total = prompt + delayed neutron yields from HDF5)
- **Anisotropic scattering** (tabular mu/CDF angular distributions with stochastic interpolation)
- **Data-driven fission spectrum** (continuous tabulated outgoing energy distributions)
- **Discrete inelastic levels** (MT=51-91 with real Q-values, two-body kinematics)
- **Continuum inelastic** (MT=91 evaporation spectrum)
- **URR probability tables** (20-band sampling, multiply_smooth + absolute modes)
- **Free gas thermal scattering** (Maxwell-Boltzmann target velocity below 400*kT)
- **(n,2n) and (n,3n) reactions** (MT=16, MT=17)
- **Rayon parallel transport** (8.7x speedup over single-threaded)

## Structure

```
rust_prototype/    Pure-Rust engine (reads HDF5 natively via hdf5-pure)
  src/
    physics/       Collision processing, scattering kinematics
    transport/     Particle tracking, k-eigenvalue solver, SVD XS provider
    geometry/      CSG geometry, BVH acceleration
    hdf5_reader.rs Pure-Rust HDF5 reader with single-pass caching
    kernel.rs      SVD reconstruction hot path (FMA)
  src/bin/
    godiva.rs      End-to-end Godiva eigenvalue benchmark
scripts/           Python analysis pipeline (Phases 1-5)
cuda_bench/        CUDA GPU benchmark kernel
paper/             LaTeX manuscript
```

## Quick Start

### Run the Godiva benchmark
```bash
cd rust_prototype

# Download ENDF/B-VII.1 HDF5 from https://openmc.org/data/
# Extract to ../data/endfb-vii.1-hdf5/

cargo run --release --bin godiva -- ../data/endfb-vii.1-hdf5/neutron \
  --rank 5 --batches 80 --inactive 15 --particles 10000
# Expected: k_eff ~ 1.000, delta ~ 80 pcm, time ~ 3.4 s
```

### Run tests
```bash
cd rust_prototype && cargo test --lib
# 28 tests pass
```

### SVD analysis (Python)
```bash
python -m venv .venv && source .venv/bin/activate
pip install openmc h5py numpy scipy matplotlib
python scripts/phase1_extraction.py data/endfb-vii.1-hdf5/neutron/U235.h5
python scripts/phase2_svd_analysis.py
```

### Nuclear data
Download ENDF/B-VII.1 HDF5 from https://openmc.org/data/ and extract to `data/`.

## Godiva k_eff Progression

The following table shows how each physics improvement affected the eigenvalue:

| Physics | k_eff | Delta from expt |
|---------|-------|----------------|
| Constant nu-bar, isotropic, Watt spectrum | 0.994 | 600 pcm |
| + Energy-dependent nu-bar from HDF5 | 1.059 | 5900 pcm |
| + Anisotropic scattering (tabular CDF) | 0.965 | 3500 pcm |
| + Data-driven fission spectrum | 1.006 | 600 pcm |
| + URR probability tables | 1.007 | 700 pcm |
| + Correlated CDF interpolation | **1.000** | **37 pcm** |

## Paper

See `paper/svd_cross_section_compression.tex` -- "Cache-Resident Cross-Section
Reconstruction via Singular Value Decomposition for Monte Carlo Neutron Transport"

## License

MIT
