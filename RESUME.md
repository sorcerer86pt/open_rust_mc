# How to Resume This Project

## Quick Start (new Claude Code session)

Paste this as your first message:

```
Read C:\Users\fog\madman_svd_experiment\CLAUDE.md then C:\Users\fog\madman_svd_experiment\resume.md and continue working on open_rust_mc.

Current state: pure Rust Monte Carlo engine with SVD-compressed cross-sections + event-based GPU transport.
Repo: https://github.com/sorcerer86pt/open_rust_mc

Validated results (10 seeds, 50k particles/batch, 150 batches):
  CPU:
  - Godiva Table k=0.99923 ± 0.00048, 1009 ± 59 ns/p
  - Godiva SVD   k=1.00019 ± 0.00035,  699 ± 47 ns/p (1.44x speedup)
  - PWR Table    k=1.35471 ± 0.00045, 22157 ± 804 ns/p
  - PWR SVD      k=1.35675 ± 0.00042, 15417 ± 1607 ns/p (1.44x speedup)
  GPU (RTX A1000 laptop):
  - PWR GPU SVD  k=1.37534 ± 0.00019, ~15k ns/p (physics gap from missing URR effect)
  - Godiva GPU   k=0.99160 ± 0.00060 (84 pcm gap from approx continuum inelastic)
  - GPU scaling: 32k→9k ns/p from 5k→200k particles

GPU architecture:
  - CUDA kernel in gpu/cuda/transport.cu (separate file, include_str!)
  - Packed TransportParams: 66 u64 fields in one device buffer (no 50+ kernel args)
  - Persistent kernel with warp-level reductions, energy-sorted compaction
  - Full physics: SVD XS, S(α,β), discrete levels, angular dist, nu-bar table, fission CDF, URR
  - Supports PWR pin cell (8 nuclides) and Godiva (3 nuclides) geometries

Physics gaps remaining on GPU (vs CPU):
  - Godiva: ~84 pcm from approximate continuum inelastic (MT=91) evaporation model
  - PWR: ~200 pcm from same + subtle URR offset indexing
  - Need to match CPU's exact evaporation model for <10 pcm

Paper: 22 pages, svd_cross_section_compression.tex
  - New section on GPU event-based transport with three-way comparison tables
  - Appendix with per-seed data for Godiva and PWR
  - Needs updating with final 10-seed GPU results once physics gap closed

Working directory: C:\Users\fog\madman_svd_experiment
Rust project: C:\Users\fog\madman_svd_experiment\rust_prototype
Nuclear data: C:\Users\fog\madman_svd_experiment\data\endfb-vii.1-hdf5\neutron
Paper: C:\Users\fog\madman_svd_experiment\paper\svd_cross_section_compression.tex
CUDA kernel: C:\Users\fog\madman_svd_experiment\rust_prototype\gpu\cuda\transport.cu

Git: sorcerer86pt with GPG signing. NEVER bypass signing. Create new commits, don't amend.
OpenMC: wsl -d Ubuntu-24.04, conda activate openmc (v0.15.3).
CUDA: nvcc 12.9 available. Build with --features cuda.
```

## Environment Setup (already done, just verify)

```bash
# Rust
cd ~/madman_svd_experiment/rust_prototype && cargo test --lib
# Should show 36 passing tests

# GPU build
cargo build --release --features cuda --bin gpu_pwr_bench

# OpenMC in WSL
wsl -d Ubuntu-24.04 -- bash -c 'source ~/miniforge3/bin/activate openmc && openmc --version'

# Nuclear data
ls ~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron/U235.h5

# Git
cd ~/madman_svd_experiment && git status && git log --oneline -5
```

## Key Commands

```bash
cd ~/madman_svd_experiment/rust_prototype

# Run all tests (36 tests)
cargo test --lib

# Godiva honesty test (CPU SVD vs Table)
cargo run --release --bin godiva -- ../data/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 150 --inactive 20 --particles 50000 --seeds 10

# PWR pin cell honesty test
cargo run --release --bin pwr_pincell -- ../data/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 150 --inactive 20 --particles 50000 --seeds 10

# GPU PWR benchmark
cargo run --release --features cuda --bin gpu_pwr_bench -- \
  ../data/endfb-vii.1-hdf5/neutron --rank 5 -B 100 --inactive 20 \
  --particles 50000 --seeds 5 --geometry pwr

# GPU Godiva benchmark
cargo run --release --features cuda --bin gpu_pwr_bench -- \
  ../data/endfb-vii.1-hdf5/neutron --rank 5 -B 100 --inactive 20 \
  --particles 50000 --seeds 5 --geometry godiva

# Full paper benchmark (all modes, 10 seeds)
cd .. && powershell -ExecutionPolicy Bypass -File run_paper_full.ps1 -Seeds 10 -Particles 50000

# OpenMC reference (WSL)
wsl -d Ubuntu-24.04 -- bash -c 'source ~/miniforge3/bin/activate openmc && \
  cd /mnt/c/Users/fog/madman_svd_experiment/scripts && \
  python paper_openmc_benchmark.py --seeds 10 --particles 50000 --batches 150'

# Compile paper
cd ~/madman_svd_experiment/paper && pdflatex svd_cross_section_compression.tex && \
  pdflatex svd_cross_section_compression.tex
```

## File Layout

```
rust_prototype/
  gpu/
    cuda/
      transport.cu              — ALL CUDA kernels (persistent transport, utility)
    opencl/                     — (future) OpenCL kernels
  src/
    lib.rs                      — crate root
    gpu.rs                      — SVD reconstruction kernel (isolated benchmark)
    gpu_transport.rs            — Rust orchestration for GPU transport
    geometry/
      mod.rs, surface.rs, aabb.rs, cell.rs, bvh.rs, ray.rs, universe.rs, lattice.rs
    physics/
      collision.rs              — reaction sampling, fission yield, inelastic levels
      scatter.rs                — elastic + inelastic kinematics, free-gas thermal
    transport/
      particle.rs, rng.rs, material.rs, simulate.rs, xs_provider.rs
    kernel.rs                   — SVD reconstruction (FMA + faer)
    decompose.rs                — faer SVD computation
    hdf5_reader.rs              — pure-Rust HDF5 reader + thermal + URR + angular dist
    thermal.rs                  — S(α,β) data structures + sampling
    table.rs                    — pointwise table (baseline)
    compare.rs, loader.rs, nuclide.rs, error.rs
  src/bin/
    godiva.rs                   — Godiva eigenvalue (--mode svd|table|both --seeds N)
    pwr_pincell.rs              — PWR pin cell (--mode svd|table|both --seeds N)
    gpu_pwr_bench.rs            — GPU benchmark (--geometry pwr|godiva --seeds N)
    bench_mem.rs                — memory comparison (all 423 nuclides)
    gpu_bench.rs                — isolated GPU SVD reconstruction benchmark
    validate_vs_openmc.rs       — bit-exact validation
  benches/
    reconstruction.rs           — criterion benchmarks

scripts/
  paper_openmc_benchmark.py     — multi-seed OpenMC runner for Godiva + PWR
  phase4_pwr_pincell.py         — OpenMC PWR pin cell benchmark
  honesty_test.py               — three-way SVD/Table/OpenMC comparison
  resonance_integral_validation.py

paper/
  svd_cross_section_compression.tex  — 22-page manuscript
  svd_cross_section_compression.pdf

run_paper_full.ps1              — full 10-seed benchmark (CPU + GPU + OpenMC)
run_pwr_tests.ps1               — PWR validation suite
run_tests.ps1                   — general test runner
```

## GPU Architecture (transport.cu)

### Packed TransportParams

All read-only physics data packed as 66 `unsigned long long` values:
- Device pointers stored as u64 addresses
- Scalars cast to u64
- Doubles stored as bit patterns (`f64::to_bits()` / `__longlong_as_double`)

Access via macros:
```cuda
#define PTR_F(p, idx)   ((const float*)  (p)[(idx)])
#define PTR_D(p, idx)   ((const double*) (p)[(idx)])
#define PTR_I(p, idx)   ((const int*)    (p)[(idx)])
#define SCALAR_I(p, idx) ((int)(p)[(idx)])
#define SCALAR_D(p, idx) __longlong_as_double((long long)(p)[(idx)])
```

Field indices defined as `#define P_BASIS 0`, `P_COEFFS 1`, etc. (66 fields total).
Rust side packs a `Vec<u64>` with matching order, uploads via `clone_htod`.

### Kernel Architecture

1. **init_source** — initialize particles from source bank, set cell/direction
2. **compact_alive** — atomic compaction of alive particle indices
3. **energy_bin_count/scatter** — 256-bin counting sort for energy-sorted access
4. **transport_persistent** — main kernel, N steps per launch:
   - Particle state in registers across steps (no per-step global memory)
   - XS lookup: SVD reconstruct (rank-k FMA, `__ldg` for read-only cache)
   - S(α,β): CDF sampling for H1 below 3.75 eV
   - URR: band sampling with multiplicative/absolute factors
   - Angular distributions: CDF-sampled mu from tabular data
   - Discrete levels: SVD reconstruct per-level XS, sample proportional, use Q-value
   - Continuum inelastic (MT=91): evaporation spectrum
   - Free-gas thermal: Box-Muller target velocity for light nuclides below 400kT
   - Fission: energy-dependent nu-bar + tabulated CDF spectrum
   - Warp-level counter reduction (`__shfl_down_sync`)
   - `__launch_bounds__(256, 2)` for occupancy tuning

### Data Upload (Rust → GPU)

All data uploaded as flat arrays with per-nuclide offset tables:
- SVD basis (f32): ~32 MB for 8 PWR nuclides, ~100 MB discrete levels
- Energy grids (f64): ~2.5 MB shared across reactions
- S(α,β) CDF tables: ~8 MB per temperature (106 E_in × 48k E_out × 771k mu)
- Angular distributions: ~200 KB
- URR tables: ~50 KB
- Nu-bar + fission spectrum CDFs: ~250 KB

Total GPU memory: ~150 MB for PWR, ~200 MB for Godiva (more level basis data)

## Nuclear Data (HDF5 Format)

ENDF/B-VII.1 from https://openmc.org/data/, extracted to `data/endfb-vii.1-hdf5/`.

### Per-nuclide HDF5 structure (e.g., U235.h5)

```
/{nuclide}/
  energy/{temp}/                 Energy grid per temperature (e.g., 294K)
    [N_E float64 values, sorted ascending, in eV]

  reactions/
    reaction_002/{temp}/xs       Elastic (MT=2)     [N_E float64]
    reaction_004/{temp}/xs       Inelastic (MT=4)   [N_E float64]
    reaction_016/{temp}/xs       (n,2n) (MT=16)     [N_E float64]
    reaction_017/{temp}/xs       (n,3n) (MT=17)     [N_E float64]
    reaction_018/{temp}/xs       Fission (MT=18)    [N_E float64]
    reaction_018/product_0/
      yield/                     Nu-bar table
        energy [N_nu float64]    Energy grid (eV)
        yield  [N_nu float64]    Nu-bar values
      energy_distribution/
        energy [N_fis float64]   Incident energy grid
        energy_out/              Per-incident: [5, N_out] packed array
          Row 0: outgoing energies (eV)
          Row 1: PDF
          Row 2: CDF
          Row 3: mu interpolation codes
          Row 4: mu offsets
        mu/                      Per-incident: [3, N_mu] packed array
          Row 0: cosine values
          Row 1: PDF
          Row 2: CDF
    reaction_051-091/{temp}/xs   Discrete inelastic levels
      Attribute: Q_value (f64)   Q-value in eV (negative = excitation)
    reaction_102/{temp}/xs       Capture (MT=102)   [N_E float64]

  reactions/reaction_002/
    product_0/angle/             Angular distribution
      energy [N_ang float64]     Energy grid
      mu/ or distribution/       Tabular mu/CDF per energy
      Attribute: center_of_mass (bool)

  urr/{temp}/                    URR probability tables
    energies [N_urr float64]     Energy grid (eV)
    table [N_urr, 6, N_bands]    Packed: cumprob, total, elastic, fission, (n,gamma), heating
    Attribute: N_bands (int)
    Attribute: multiply_smooth (bool)

c_H_in_H2O.h5                   S(α,β) thermal scattering
  /{material}/
    kTs [N_temp float64]         Boltzmann energies (eV)
    inelastic/{temp}/
      energy_out [5, N_eout]     Packed: E_out, PDF, CDF, mu_interp, mu_offsets
        Attribute: offsets [N_inc int]  → boundaries in e_out array
      mu [3, N_mu]               Packed: mu, PDF, CDF
    elastic/{temp}/              Optional elastic (Bragg edges, Debye-Waller)
```

### Key nuclide sizes (unionised grid, 6 temps)
- U235: 83,114 energy points, 6 reactions + 41 discrete levels
- U238: 185,903 energy points, 5 reactions + 41 discrete levels
- H1: 590 energy points, 2 reactions
- O16: 3,063 energy points, 4 reactions + 8 levels
- c_H_in_H2O: 9 temps, 106 inc energies, ~50k E_out pts, ~770k mu pts per temp

## SVD Reconstruction Mathematics

Cross-section matrix M[N_E × N_T] (energies × temperatures) in log₁₀ space:
```
log₁₀(M) ≈ U_k × Σ_k × V_k^T
```

Pre-multiplied basis: `B = U_k × Σ_k` (stored as f32 for halved memory).
Temperature coefficients: `c = V_k^T[:, t]` (one column of V^T for temp t).

Reconstruction at energy index i:
```
log₁₀(σ) = Σ_j B[i,j] × c[j]    (rank-k FMA loop)
σ = exp2(log₁₀(σ) × log₂(10))     (fast transcendental)
```

GPU advantage: sequential access to B rows (coalesced), c in shared memory,
pure FMA with no branches → zero warp divergence. Replaces O(log N) binary
search on 100k+ point tables with O(k) sequential reads (k=5).

## Current Bugs / Physics Gaps

### GPU Godiva k=0.992 (should be 1.000, ~84 pcm gap)
- Continuum inelastic (MT=91) uses approximate evaporation model
- CPU uses: T = sqrt(E*/a), a = A/8, E_out ~ -T×ln(ξ₁×ξ₂)
- GPU has same formula but energy transfer calculation may differ slightly
- URR tables uploaded but effect may be small for Godiva

### GPU PWR k=1.375 (CPU=1.357, ~200 pcm gap)
- Same continuum inelastic issue affects inelastic in Zr isotopes
- URR offset indexing in apply_urr may be wrong: `base = off*n_b + ie*n_b`
  should be `base = (off + ie) * n_b` (flat array indexing)

### RULE: No shortcuts. Ever.

Every shortcut in this project has cost more time debugging than doing it right:
- "Simplified physics" → wrong k_eff → days of debugging
- Wrong AWR for hydrogen → k=0.87 instead of 1.36 → wasted benchmark runs
- Treating inelastic as capture → k=0.67 → had to redo all GPU results
- Random Q-values instead of SVD level sampling → k=1.10 → still wrong
- Tuning random parameters to match k instead of fixing physics → never converges

The correct approach is ALWAYS: read the real data from HDF5, implement the
exact same algorithm as the CPU, use SVD reconstruction for cross-sections.
If a physics feature exists on CPU, port it to GPU correctly. Don't approximate.
Don't skip. Don't "add it later." Do it right the first time.

### What NOT to do
- Don't bypass GPG signing (configured for sorcerer86pt)
- Don't delete `data/endfb-vii.1-hdf5/` (5.8 GB, slow to re-download)
- Don't amend commits (create new ones)
- Don't push to main without tests passing (36 tests)
- Don't approximate physics — use real data from HDF5
- Don't run benchmarks in parallel (thermal throttling contaminates timing)
- Don't add kernel arguments individually — use packed TransportParams buffer
- Don't take shortcuts on physics — every shortcut creates a bug that takes longer to debug than doing it right
- Don't guess parameter values — look up the real data
- Don't skip uploading data to GPU "for simplicity" — null pointers crash, missing physics gives wrong answers

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Godiva CPU Table k_eff (10 seeds) | 0.99923 ± 0.00048 |
| Godiva CPU SVD k_eff (10 seeds) | 1.00019 ± 0.00035 |
| Godiva GPU SVD k_eff | 0.99160 ± 0.00060 |
| PWR CPU Table k_inf (10 seeds) | 1.35471 ± 0.00045 |
| PWR CPU SVD k_inf (10 seeds) | 1.35675 ± 0.00042 |
| PWR GPU SVD k_inf | 1.37534 ± 0.00019 |
| CPU SVD speedup vs Table (Godiva) | 1.44x |
| CPU SVD speedup vs Table (PWR) | 1.44x |
| GPU SVD reconstruction speedup | 4.2x (RTX A1000) |
| GPU transport at 200k particles | 9,045 ns/p |
| SVD-Table k gap (Godiva) | 96 pcm |
| SVD-Table k gap (PWR) | 204 pcm |
| SVD memory (8 PWR nuclides) | 32 MB basis + 100 MB levels |
| Library-wide compression ratio | 4.8x (423 nuclides, rank=5) |
| Paper pages | 22 |

## Next Steps

1. **Fix GPU physics gap to <10 pcm**: debug continuum inelastic energy transfer
   and URR offset indexing in transport.cu
2. **Rerun 10-seed benchmarks**: Godiva + PWR with corrected GPU physics
3. **Add OpenMC reference**: run paper_openmc_benchmark.py for 10-seed comparison
4. **Update paper**: final three-way tables (OpenMC vs CPU SVD vs GPU SVD)
   with per-seed appendix data
5. **OpenCL port**: translate transport.cu → transport.cl for AMD/Intel GPUs
6. **HPC benchmarking**: run on dedicated cluster node with CPU pinning
