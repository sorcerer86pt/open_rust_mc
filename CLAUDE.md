# open_rust_mc — Project Memory

## What This Is

A pure-Rust Monte Carlo neutron transport engine that replaces OpenMC's
multi-gigabyte pointwise cross-section tables with SVD-compressed
cache-resident reconstruction. The engine reads OpenMC HDF5 nuclear
data files directly (via `hdf5-pure`, no C dependency) and runs
eigenvalue simulations end-to-end.

## What We've Proven

### SVD Compression (validated)
- Singular spectrum decays 5 orders of magnitude across 8 values
- 47/52 U-235 reactions are rank-1 (machine-epsilon compressible)
- Bit-exact match with OpenMC's Python API (relative error = 0)
- Godiva k_eff < 10 pcm deviation at all ranks (fission-only)
- Godiva k_eff 3.7 pcm with all reactions modified (k=4)
- PWR pin cell: 60-93 pcm (confirms need for hybrid SVD+WMP)
- Hybrid SVD(k=2)+WMP: 530x memory reduction (15 KB vs 7.8 MB)

### Performance (validated)
- CPU: 8-13x faster than table lookup (3-5 ns/pt vs 40 ns/pt)
- GPU: 2.6-2.8x on RTX A1000 (laptop), expected better on 3080/A100
- Data loading + SVD decompose: ~6 s for 3 nuclides (with all physics data)
- Full Godiva eigenvalue (80 batches, 10k particles): 633 ms (rayon parallel)

### Cross-isotope sharing (investigated, negative result)
- Subspace angle > 85° at k=2 between U-235/U-238/Pu-239
- Each nuclide needs its own SVD basis — not a problem, bases are small

## Current State: k_eff = 1.00016 +/- 0.00080 (Godiva, real ENDF data)

OpenMC gets 0.99857. We get 1.00016 +/- 0.00080. Gap = ~160 pcm from OpenMC.
Delta from experiment = **16 pcm** (< 0.2 sigma). 150 batches, 20k particles.

### Implemented (Priority 1 from previous round)

**1. Energy-dependent nu-bar (DONE)**
- Reads total ν̄(E) from HDF5 (prompt + delayed neutron yields)
- U-235: 79-point table, 2.435 thermal → 2.530 @ 1 MeV → 3.836 @ 10 MeV
- U-238: 10-point table, 2.483 thermal → 2.554 @ 1 MeV → 3.849 @ 10 MeV
- U-234: falls back to constant (no neutron product in reaction_018)
- Revealed that old k_eff=0.994 was coincidental (compensating errors)

**2. Discrete inelastic levels MT=51-91 (DONE)**
- Loads all 41 levels per nuclide with real Q-values from HDF5 attributes
- SVD-compressed at rank 2, cross-sections used for level sampling
- Proper two-body kinematics with exact Q-value per level
- Impact on k_eff: negligible for Godiva (< 50 pcm)

**3. Continuum inelastic MT=91 (DONE)**
- Evaporation spectrum with nuclear temperature T = sqrt(E*/a)
- Level density parameter a = A/8 MeV⁻¹
- Detected automatically from HDF5 level enumeration

**4. (n,3n) reaction MT=17 (DONE)**
- SVD kernel loaded, banks 2 extra neutrons
- Small cross-section, only at high energies (~10-20 pcm)

**5. Anisotropic scattering angular distributions (DONE)**
- Reads tabular mu/cdf distributions from HDF5 with offsets attribute
- Samples scattering cosine from energy-dependent CDF in CM frame
- U-235: 49 energies, U-234: 53 energies, U-238: 38 energies
- Impact: ~9400 pcm (from 1.059 down to 0.965) — massive

**6. Data-driven fission energy spectrum (DONE)**
- Reads continuous tabulated outgoing energy distributions from HDF5
- Replaces hardcoded Watt spectrum (a=0.988, b=2.249)
- U-235: 20 incident energies, U-238: 25 incident energies
- Impact: ~4000 pcm (from 0.965 up to 1.006) — critical correction

**7. URR probability tables (DONE)**
- Reads probability tables from HDF5: energy grid + [N_E, 6, N_bands] table
- Handles both `multiply_smooth=true` (factors) and `false` (absolute XS)
- U-234: 26 energies, 1.5-100 keV; U-235: 19 energies, 2.25-25 keV; U-238: 18 energies, 20-149 keV
- Impact: ~100-200 pcm improvement

### Honesty Test (implemented)
- `--mode svd|table|both` flag on godiva binary
- `TableXsProvider` — OpenMC-style pointwise table lookup, same physics
- `--mode both` runs both providers back-to-back, prints comparison
- At rank=5, SVD uses ~1.7x MORE memory per reaction than single-temp table
  (basis stores `rank` values per energy point vs table's 1 value)
- SVD memory advantage appears at rank≤1 (single-temp) or multi-temperature
- Discrete levels (41 per nuclide) dominate memory for both approaches
- OpenMC comparison script: `scripts/honesty_test.py` (WSL + conda)

### S(α,β) Thermal Scattering (DONE — needs validation run)

**8. S(α,β) thermal scattering for H in H₂O (DONE)**
- Reads OpenMC thermal scattering HDF5 files (e.g., `c_H_in_H2O.h5`)
- Continuous inelastic (iwt=2): CDF sampling with lin-lin interpolation,
  scaled energy bounds (OpenMC Eq 31-35), discrete cosine smearing
- Discrete inelastic (iwt=0,1): equiprobable/skewed bin sampling
- Coherent elastic: Bragg edge sampling (graphite, Be)
- Incoherent elastic: Debye-Waller angle formula
- Stochastic temperature interpolation (OpenMC method)
- c_H_in_H2O: 9 temperatures (294-800K), 106 energies, ~50K E_out pts,
  ~770K mu pts per temperature, loaded from 186 MB HDF5 file
- Transport integration: replaces free-atom elastic XS below energy_max
  (~3.75 eV) with thermal inelastic+elastic XS, samples S(α,β) for
  collision outcomes
- Impact: expected ~1000-5000 pcm for PWR pin cell (awaiting validation)

**9. PWR pin cell multi-seed benchmarking (DONE)**
- `--seeds N` flag for independent runs with confidence intervals
- ns/particle timing, speedup ratios, memory comparison
- Auto-loads `c_H_in_H2O.h5` for H1 nuclide in water material
- 8 nuclides (U235, U238, O16, H1, Zr90-94), 3 materials

## What Needs Fixing (Physics Gaps vs OpenMC)

### Priority 1 — Validate PWR pin cell with S(α,β) — DONE 2026-04-22
- Rust Table vs OpenMC 0.15.3: **12 pcm** (within 1σ)
- Rust SVD k=5 vs OpenMC: **−67 pcm** (within combined σ)
- Rust SVD vs Rust Table: 59 pcm (compression cost)
- S(α,β) impact on k_inf: ~300 pcm (disables → k_inf up)
- Geometry/stats: 3.1% UO₂, 1.26 cm pitch, 100b × 20k × 3 seeds
- Artifacts: `outputs/pwr_sab_on.txt`, `pwr_sab_off.txt`,
  `openmc_pwr_ref.json`; writeup in `resume.md`
- Caveats documented: SVD at rank 5 is 5× larger and 0.95× as
  fast as Table for this all-reactions 9-nuclide problem; memory
  win is rank ≤ 1 or hybrid SVD+WMP, not all-reactions rank-5.

### Priority 2 — Event-based GPU transport
- Current: history-based (one particle birth-to-death per thread)
- Target: event-based (batch operations: sort→XS lookup→collide→compact)
- Tramm et al. 2024: event-based is 6x faster than history-based on GPU
- Sort particles by (material, energy) before XS lookup = critical optimization
- Already have GPU SVD kernel (8.7x on RTX A1000), need event loop

### Priority 3 — Feature parity

**10. Photon transport**
- OpenMC: full coupled neutron-photon transport
- Not needed for k_eff but needed for dose/shielding calculations

**11. Depletion / burnup**
- Separate project — Chebyshev Rational Approximation Method (CRAM)
  for matrix exponential of the transmutation matrix

## Architecture Decisions Made

### What works well (keep)
- **Enum dispatch for surfaces** — zero-cost, no vtable, jump table
- **SvdKernel with pre-multiplied basis** — hot path is pure FMA
- **PCG-64 PRNG** — fast, reproducible, parallel-safe
- **hdf5-pure** — no C dependency, reads OpenMC files correctly
- **faer for SVD** — SIMD-optimised, correct results

### What to improve
- **Particle transport is history-based** — switch to event-based for
  better branch prediction and GPU readiness
- **Cell finding is linear scan** — BVH is built but not used in transport
  loop yet (need to integrate)
- **HDF5 file is re-read per reaction** — should load once and extract
  all reactions in a single pass

### Recently completed architecture improvements
- **Rayon parallel transport** — 8.7x speedup (633 ms vs 5540 ms for Godiva)
- **Free gas thermal scattering** — Maxwell-Boltzmann target velocity
  sampling below 400*kT threshold (important for thermal systems)

## File Layout

```
rust_prototype/
  src/
    lib.rs                      — crate root
    geometry/
      mod.rs                    — Vec3, re-exports
      surface.rs                — 6 surface types (enum dispatch)
      aabb.rs                   — AABB ray intersection (branchless)
      cell.rs                   — CSG boolean regions
      bvh.rs                    — Bounding Volume Hierarchy
      ray.rs                    — ray tracing, cell finding
      universe.rs               — cell grouping
      lattice.rs                — rectangular lattice
    physics/
      collision.rs              — reaction sampling, fission yield
      scatter.rs                — elastic + inelastic kinematics
    transport/
      particle.rs               — particle state, fission bank
      rng.rs                    — PCG-64 generator
      material.rs               — material composition, macro XS
      simulate.rs               — k-eigenvalue power iteration
      xs_provider.rs            — SVD kernel ↔ transport bridge
    kernel.rs                   — SVD reconstruction (FMA + faer)
    decompose.rs                — faer SVD computation
    hdf5_reader.rs              — pure-Rust HDF5 reader + thermal loader
    thermal.rs                  — S(α,β) data structures + sampling
    compare.rs                  — error analysis
    loader.rs                   — numpy .npy loader
    nuclide.rs                  — nuclide data from .npy
    table.rs                    — pointwise table (baseline)
    error.rs                    — error types
  src/bin/
    godiva.rs                   — end-to-end Godiva eigenvalue
    pwr_pincell.rs              — PWR pin cell (8 nuclides, S(α,β))
    bench_mem.rs                — memory/speed comparison
    validate_vs_openmc.rs       — bit-exact validation
  benches/
    reconstruction.rs           — criterion benchmarks

scripts/                        — Python analysis pipeline
  phase1_extraction.py          — extract XS from HDF5
  phase2_svd_analysis.py        — SVD spectrum analysis
  phase3_error_analysis.py      — regional error analysis
  phase4_keff_benchmark.py      — OpenMC Godiva benchmark
  phase4_multi_reaction_godiva.py — all-reaction Godiva
  phase4_pwr_pincell.py         — PWR pin cell benchmark
  phase5_3_windowed_svd.py      — per-window SVD analysis
  phase5_all_reactions.py       — 52-reaction sweep
  phase5_cross_isotope.py       — cross-isotope sharing
  phase5_hybrid_svd_wmp.py      — SVD+WMP hybrid analysis
  phase5_multi_reaction.py      — multi-reaction SVD
  cache_feasibility_analysis.py — cache tier analysis
  export_openmc_reference.py    — export OpenMC reference values

cuda_bench/
  svd_gpu_bench.cu              — GPU reconstruction benchmark

paper/
  svd_cross_section_compression.tex  — manuscript (12 pages)
  svd_cross_section_compression.pdf
```

## Build & Run

```bash
# Build
cd rust_prototype && cargo build --release

# Run all tests (36 tests)
cargo test --lib

# Run Godiva with real nuclear data
cargo run --release --bin godiva -- path/to/endfb-vii.1-hdf5/neutron \
  --rank 5 --batches 80 --inactive 15 --particles 10000

# Honesty test: SVD vs pointwise table head-to-head
cargo run --release --bin godiva -- path/to/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 150 --inactive 20 --particles 20000

# PWR pin cell with S(α,β) thermal scattering
cargo run --release --bin pwr_pincell -- path/to/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 100 --inactive 20 --particles 20000

# PWR pin cell full benchmark (multi-seed)
cargo run --release --bin pwr_pincell -- path/to/endfb-vii.1-hdf5/neutron \
  --mode both --rank 5 --batches 150 --inactive 20 --particles 50000 --seeds 5

# Table-only mode (OpenMC-style baseline)
cargo run --release --bin godiva -- path/to/endfb-vii.1-hdf5/neutron \
  --mode table --batches 150 --inactive 20 --particles 20000

# GPU benchmark (requires --features cuda)
cargo run --release --features cuda --bin gpu_bench -- \
  path/to/endfb-vii.1-hdf5/neutron --rank 5 --particles 1000000

# Full test suite (PowerShell)
cd .. && .\run_pwr_tests.ps1              # all tests
cd .. && .\run_pwr_tests.ps1 -Download    # download data + run tests
```

## Nuclear Data

Download ENDF/B-VII.1 HDF5 from https://openmc.org/data/
Extract to `data/endfb-vii.1-hdf5/`. Key files:
- `neutron/U234.h5`, `U235.h5`, `U238.h5` (Godiva)
- `neutron/H1.h5`, `O16.h5`, `Zr90.h5` (PWR pin cell)
- `neutron/c_H_in_H2O.h5` (S(α,β) thermal scattering for H in water)
- Full library: 444 nuclide files + thermal data, 5.8 GB

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| U-235 fission σ₂/σ₁ | 7.7e-2 |
| Reactions that are rank-1 | 47/52 |
| CPU speedup vs table | 8-13x |
| GPU speedup vs table | 2.6-2.8x |
| Hybrid SVD+WMP memory | 15 KB vs 7.8 MB (530x) |
| Godiva dk (fission SVD k=4) | 6.9 pcm |
| Godiva dk (all rxn SVD k=4) | 3.7 pcm |
| PWR pin cell dk (SVD k=5) | 59.7 pcm |
| Our Rust Godiva k_eff | 1.00016 +/- 0.00080 |
| OpenMC Godiva k_eff | 0.99857 |
| Gap from OpenMC | ~160 pcm |
| Gap from experiment | **16 pcm** |
| History: const nu-bar | 0.994 (coincidental) |
| History: + E-dep nu-bar | 1.059 (+6500 pcm) |
| History: + aniso scatter | 0.965 (-9400 pcm) |
| History: + data fission | 1.006 (+4100 pcm) |
| History: + URR + interp | 1.000 (-600 pcm) |
