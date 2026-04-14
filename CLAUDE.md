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
- Data loading + SVD decompose: ~800 ms for 3 nuclides
- Full Godiva eigenvalue (80 batches, 10k particles): 2.4 seconds

### Cross-isotope sharing (investigated, negative result)
- Subspace angle > 85° at k=2 between U-235/U-238/Pu-239
- Each nuclide needs its own SVD basis — not a problem, bases are small

## Current State: k_eff = 1.059 (Godiva, real ENDF data)

OpenMC gets 0.99857. We get 1.059. Gap = ~6100 pcm (too high).

Previous k_eff was 0.994 with constant nu-bar=2.43. The coincidental agreement
with experiment was because the too-low nu-bar compensated for missing physics
(primarily anisotropic scattering). With correct energy-dependent nu-bar
(~2.53 at 1 MeV), k_eff jumped to 1.059. The ~6000 pcm gap from OpenMC is
from missing angular distributions and other Priority 1 items below.

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

## What Needs Fixing (Physics Gaps vs OpenMC)

### Priority 1 — Closes the ~6000 pcm gap

**5. Energy-dependent scattering angular distributions (BIGGEST IMPACT)**
- Currently: isotropic in CM frame for all energies
- OpenMC: reads angular distribution tables (Legendre coefficients or
  tabular P(mu|E)) from HDF5 for each reaction
- Impact: **~3000-4000 pcm** — forward-peaked scattering increases leakage
  from the small Godiva sphere, dramatically lowering k_eff
- HDF5 path: `{nuclide}/reactions/reaction_{MT:03}/product_0/distribution`
- This is now the single biggest source of error

**6. Watt fission spectrum parameters from data**
- Currently: hardcoded a=0.988 MeV, b=2.249/MeV (U-235 thermal)
- OpenMC: reads parameters from HDF5 per nuclide, energy-dependent
- Impact: ~500-1000 pcm (spectrum shape affects leakage in small systems)
- HDF5 path: energy distributions stored under reaction products

**7. Unresolved Resonance Range (URR) probability tables**
- Currently: ignored (use average cross-sections)
- OpenMC: samples from probability tables in the URR (2.25 keV - 25 keV for U-235)
- Impact: ~100-500 pcm (affects self-shielding in the URR)
- HDF5 path: `{nuclide}/urr/{temp}/table`

### Priority 2 — Refinement

**8. Free gas thermal scattering correction**
- Currently: target nucleus is stationary (cold target approximation)
- OpenMC: samples target velocity from Maxwell-Boltzmann distribution,
  applies relative velocity correction (important at low energies)
- Impact: ~10-30 pcm for thermal systems, negligible for fast Godiva

### Priority 3 — Feature parity (correctness, not pcm)

**9. S(α,β) thermal scattering data**
- Needed for: H in water, graphite, UO2, ZrH
- OpenMC: reads `c_H_in_H2O.h5` etc., applies coherent/incoherent elastic
  and inelastic thermal scattering below ~4 eV
- Impact: critical for thermal reactors (PWR pin cell)
- These files are huge (178 MB for H₂O) but SVD may compress them well

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
- **No rayon parallelism yet** — transport loop is single-threaded

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
    hdf5_reader.rs              — pure-Rust HDF5 reader
    compare.rs                  — error analysis
    loader.rs                   — numpy .npy loader
    nuclide.rs                  — nuclide data from .npy
    table.rs                    — pointwise table (baseline)
    error.rs                    — error types
  src/bin/
    godiva.rs                   — end-to-end Godiva eigenvalue
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

# Run Godiva with real nuclear data
cargo run --release --bin godiva -- path/to/endfb-vii.1-hdf5/neutron \
  --rank 5 --batches 80 --inactive 15 --particles 10000

# Run all tests
cargo test --lib

# HDF5 exploration
cargo run --release -- explore path/to/U235.h5

# SVD benchmark (needs .npy files from Python pipeline)
cargo run --release -- npy --prefix jeff33_
```

## Nuclear Data

Download ENDF/B-VII.1 HDF5 from https://openmc.org/data/
Extract to `data/endfb-vii.1-hdf5/`. Key files:
- `neutron/U234.h5`, `U235.h5`, `U238.h5` (Godiva)
- `neutron/H1.h5`, `O16.h5`, `Zr90.h5` (PWR pin cell)
- Full library: 444 nuclide files, 5.8 GB

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
| Our Rust Godiva k_eff | 1.059 |
| OpenMC Godiva k_eff | 0.99857 |
| Gap to close | ~6100 pcm (from isotropic scattering + missing physics) |
| Previous k_eff (const nu-bar) | 0.994 (coincidental, compensating errors) |
