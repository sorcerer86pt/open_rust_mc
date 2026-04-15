# How to Resume This Project

## Quick Start (new Claude Code session)

Paste this as your first message:

```
Read C:\Users\fog\madman_svd_experiment\CLAUDE.md then continue working on open_rust_mc.

Current state: pure Rust Monte Carlo engine with SVD-compressed cross-sections.
Repo: https://github.com/sorcerer86pt/open_rust_mc

Validated results (10 seeds, 1M particles/batch, 150 batches):
- SVD k_eff = 1.00012 +/- 0.00007 (12 pcm from experiment)
- Table k_eff = 0.99905 +/- 0.00013 (95 pcm from experiment)
- OpenMC k_eff = 0.99844 +/- 0.00006 (156 pcm from experiment)
- SVD speedup vs table: 1.19x (700 vs 832 ns/particle, quiet periods)
- SVD fidelity cost: 107 pcm vs table
- GPU kernel speedup: 8.7x on RTX A1000 (zero error)

Completed physics:
- Energy-dependent nu-bar (prompt + delayed) from HDF5
- Anisotropic scattering (tabular CDF angular distributions)
- Data-driven fission spectrum (continuous tabulated from HDF5)
- Discrete inelastic levels (MT=51-91) with real Q-values
- Continuum inelastic (MT=91) evaporation spectrum
- URR probability tables (20-band sampling)
- Free gas thermal scattering (Maxwell-Boltzmann target velocity)
- (n,2n) and (n,3n) reactions
- Void cell free-streaming
- Auto-detect tracking mode (surface vs delta)
- Rayon parallel transport

Completed optimisations:
- f32 SVD basis with f64 accumulator (halved memory)
- Arc<[f64]> shared energy grids (eliminated 111 MB duplication)
- exp2(x * LOG2_10) replacing powf (3-5x faster)
- Stack-allocated collision buffers (no heap alloc in hot path)
- Single binary search per nuclide per collision
- Hash-based O(1) energy index (Brown 2014, 8192 bins)
- Ducru kernel reconstruction for temperature interpolation (2017)
- CUDA GPU reconstruction via cudarc (feature-gated)

Multi-seed benchmarking: --seeds N flag, reports mean +/- stddev ns/particle
Honesty test: --mode svd|table|both (same engine, only XS lookup differs)

Paper: 18 pages, all numbers consistent, Ducru/Tramm/Brown refs, resonance
integral validation (<0.2% all groups), honest GPU comparison vs OpenMC.

Benoit Forget (MIT, co-author of WMP) reviewed the paper and suggested:
1. Energy index search clarification (done)
2. Ducru temperature interpolation (done)  
3. Resonance integral validation (done)
All three addressed and committed.

Next steps (not yet done):
- PyO3 Python bindings (research done, ready to implement)
- Event-based GPU transport (cudarc integration done, kernel works)
- PWR pin cell at scale (binary exists, void streaming works, needs validation)
- S(alpha,beta) thermal scattering (needed for thermal reactors)
- HPC benchmarking on dedicated cluster (desktop noise too high)

Working directory: C:\Users\fog\madman_svd_experiment
Rust project: C:\Users\fog\madman_svd_experiment\rust_prototype
Nuclear data: C:\Users\fog\madman_svd_experiment\data\endfb-vii.1-hdf5\neutron
Paper: C:\Users\fog\madman_svd_experiment\paper\svd_cross_section_compression.tex

Git: sorcerer86pt with GPG signing. NEVER bypass signing. Create new commits, don't amend.
OpenMC: wsl -d Ubuntu-24.04, conda activate openmc.
CUDA: nvcc 12.9 available. Build with --features cuda.
```

## Environment Setup (already done, just verify)

```bash
# Rust
cd ~/madman_svd_experiment/rust_prototype && cargo test --lib
# Should show 32 passing tests

# GPU build (optional, requires CUDA toolkit)
cargo build --release --features cuda --bin gpu_bench

# OpenMC in WSL
wsl -d Ubuntu-24.04 -- bash -c 'source ~/miniforge3/bin/activate openmc && openmc --version'

# Nuclear data
ls ~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron/U235.h5

# Git
cd ~/madman_svd_experiment && git status && git log --oneline -5
```

## Key commands

```bash
cd ~/madman_svd_experiment/rust_prototype

# Run all tests (32 tests)
cargo test --lib

# Quick honesty test
cargo run --release --bin godiva -- ../data/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 20 --inactive 5 --particles 5000

# Full 10-seed benchmark (~30 min)
cargo run --release --bin godiva -- ../data/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 150 --inactive 20 --particles 1000000 --seeds 10

# SVD-only or table-only (faster, one mode)
cargo run --release --bin godiva -- ../data/endfb-vii.1-hdf5/neutron \
  --mode svd --rank 5 --batches 150 --inactive 20 --particles 1000000 --seeds 10

# GPU benchmark (requires --features cuda)
cargo run --release --features cuda --bin gpu_bench -- \
  ../data/endfb-vii.1-hdf5/neutron --rank 5 --particles 1000000

# PWR pin cell (8 nuclides, 3 materials)
cargo run --release --bin pwr_pincell -- ../data/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 100 --inactive 20 --particles 50000

# Resonance integral validation (WSL)
wsl -d Ubuntu-24.04 -- bash -c 'source ~/miniforge3/bin/activate openmc && \
  cd /mnt/c/Users/fog/madman_svd_experiment/scripts && \
  python resonance_integral_validation.py'

# Three-way comparison: SVD vs Table vs OpenMC (WSL)
wsl -d Ubuntu-24.04 -- bash -c 'source ~/miniforge3/bin/activate openmc && \
  cd /mnt/c/Users/fog/madman_svd_experiment/scripts && \
  python honesty_test.py --particles 1000000 --batches 150'

# Compile paper
cd ~/madman_svd_experiment/paper && pdflatex svd_cross_section_compression.tex
```

## File layout

```
rust_prototype/src/
  kernel.rs          SVD kernel (f32 basis, hash lookup, Ducru temp interp)
  table.rs           Pointwise table (OpenMC-style baseline)
  gpu.rs             CUDA GPU reconstruction (feature-gated)
  transport/
    simulate.rs      Eigenvalue solver (multi-seed, auto-detect tracking)
    xs_provider.rs   SvdXsProvider + TableXsProvider (honesty test toggle)
    material.rs      Material composition
    particle.rs      Particle state
    rng.rs           PCG-64 RNG
  physics/
    collision.rs     Reaction sampling (MicroXs with Default for stack alloc)
    scatter.rs       Elastic + inelastic kinematics
  geometry/          CSG geometry, BVH, ray tracing
  hdf5_reader.rs     Pure-Rust HDF5 reader
  bin/
    godiva.rs        Godiva benchmark (--mode --seeds)
    pwr_pincell.rs   PWR pin cell (8 nuclides)
    gpu_bench.rs     GPU reconstruction benchmark
    bench_mem.rs     Memory comparison tool
scripts/
  honesty_test.py              Three-way SVD/Table/OpenMC comparison
  resonance_integral_validation.py  Resonance integral accuracy
paper/
  svd_cross_section_compression.tex  18-page manuscript
BENCHMARKS.md                  Full reproducible benchmark instructions
OPTIMIZATION_PLAN.md           Phase 1/2/3 optimization roadmap
```

## Nuclear data layout

```
data/endfb-vii.1-hdf5/           5.9 GB total, 444 nuclides
  cross_sections.xml             Master index
  neutron/                       Neutron interaction data
    U234.h5, U235.h5, U238.h5   Godiva (3 nuclides)
    H1.h5, O16.h5               Water moderator
    Zr90.h5-Zr96.h5             Zircaloy cladding
    ... (444 files total)
  photon/                        Photon data (not used yet)
  wmp/                           Windowed Multipole data (not used yet)
```

HDF5 structure per nuclide (e.g. U235.h5):
```
/{nuclide}/
  energy/{temp}/                 Energy grid per temperature (e.g. 294K)
  reactions/
    reaction_002/{temp}/xs       Elastic (MT=2)
    reaction_004/{temp}/xs       Inelastic (MT=4)
    reaction_016/{temp}/xs       (n,2n) (MT=16)
    reaction_018/{temp}/xs       Fission (MT=18)
    reaction_018/product_0/      Fission neutron yield + spectrum
    reaction_051-091/{temp}/xs   Discrete inelastic levels
    reaction_102/{temp}/xs       Capture (MT=102)
  urr/{temp}/                    URR probability tables
```

Key nuclide sizes (unionised grid, 6 temps):
- U235: 83,114 energy points, 6 reactions + 41 discrete levels
- U238: 185,903 energy points, 5 reactions + 41 discrete levels
- H1: 590 energy points, 2 reactions
- O16: 3,063 energy points, 4 reactions + 8 levels

## What NOT to do

- Don't bypass GPG signing (configured for sorcerer86pt)
- Don't delete `data/endfb-vii.1-hdf5/` (5.8 GB, slow to re-download)
- Don't amend commits (create new ones)
- Don't push to main without tests passing (32 tests)
- Don't claim GPU speedup without comparing to OpenMC's GPU results (Tramm 2024)
- Don't report single-run timing as definitive (use --seeds 10)
