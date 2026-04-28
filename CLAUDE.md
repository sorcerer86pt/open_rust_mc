# open_rust_mc — Project Memory

## What This Is

A pure-Rust Monte Carlo neutron transport engine that replaces OpenMC's
multi-gigabyte pointwise cross-section tables with SVD-compressed
cache-resident reconstruction. The engine reads OpenMC HDF5 nuclear
data files directly (via `hdf5-pure`, no C dependency) and runs
eigenvalue simulations end-to-end.

## How to read the numbers below

Every row is tagged with one of:

- `[micro]` = isolated kernel / spectrum / one-nuclide-one-reaction
  measurement. Often optimistic; does NOT generalize.
- `[godiva]` = end-to-end Godiva, 3 nuclides, fast spectrum.
- `[pwr]` = end-to-end PWR pin cell, 9 nuclides, thermal spectrum,
  S(α,β) on. This is the realistic deployment case.
- `[projected]` = analytical or extrapolated, never measured at scale.
  Treat as a hypothesis until a `[pwr]` row replaces it.

A number quoted without scope is a bug. The repeated pattern across
this project: `[micro]` headline numbers shrink (or invert sign) when
re-measured under `[pwr]`. Don't quote micro numbers in contexts
where pwr-scope applies.

## What We've Proven

### SVD compression (spectrum + reconstruction)
- `[micro]` Singular spectrum decays 5 orders of magnitude across 8 values
- `[micro]` 47/52 U-235 reactions are rank-1-compressible to machine epsilon.
  **Caveat:** rank 1 is not a deployable rank for k_eff — production runs use
  rank 5. The 47 figure does not translate to a deployment win.
- `[micro]` Bit-exact match with OpenMC's Python API (relative error = 0)
- `[godiva]` Godiva k_eff < 10 pcm deviation at all ranks (fission-only,
  pre-coupling correction)
- `[godiva]` Godiva k_eff 3.7 pcm with all reactions modified (k=4,
  pre-coupling correction)
- `[pwr]` PWR pin cell SVD(k=5) vs ACE+WMP: **5 pcm** at on-library 600 K
  (paper, 100 b × 20 k × 1 seed). Earlier "60–93 pcm" was a partial-physics
  measurement; replaced.
- `[pwr]` Hybrid SVD+WMP in-engine memory: 519 MB at rank 5, vs pure SVD
  518 MB and pointwise Table 103 MB. Hybrid is not a memory win at the
  engine level on PWR; see paper §hybrid for the per-nuclide
  representation-byte accounting where the picture differs.

### Performance — what micro vs full transport actually says
- `[micro]` SVD reconstruction kernel only, hot cache: 8–13× faster than
  pointwise table (3–5 ns/pt vs 40 ns/pt). **Does not survive integration.**
- `[godiva]` SVD rank-5 vs Table, full transport: 1.37×–1.90× CPU throughput
  (paper §godiva, 10 seeds × 150 b × 50 k).
- `[pwr]` SVD rank-5 vs Table, full transport: SVD **0.95× as fast as Table**
  (sim 46.8 s vs 41.7 s, paper §pwr / `outputs/full_test_run/10_pwr_all_rank5.txt`).
  The micro 8–13× does not appear here.
- `[micro]` GPU 2.6–2.8× on RTX A1000 — kernel-only SVD reconstruction vs
  CPU. **vs single-thread CPU.** Against 20-core rayon CPU, the integration
  story is GPU SVD **1.3× slower** on Godiva (paper §gpu).
- `[godiva]` Loading + SVD decompose: ~6 s for 3 nuclides
- `[godiva]` Eigenvalue 80 b × 10 k: 633 ms (rayon parallel, old run; for current
  paper-stat numbers see paper §godiva)

### Cross-isotope sharing (investigated, negative result)
- Subspace angle > 85° at k=2 between U-235/U-238/Pu-239
- Each nuclide needs its own SVD basis — not a problem, bases are small

## Current State: k_eff = 1.00079 +/- 0.00038 (Godiva, real ENDF data)

**Benchmark is ICSBEP HMF-001** (k = 1.0000 ± 100 pcm experimental).
We get 1.00079 ± 0.00038 → **Δ_ICSBEP = +79 pcm, inside σ_exp. Pass.**
OpenMC 0.15.3 on the same HDF5 gets 0.99901 (−99 pcm vs ICSBEP) — also
inside σ_exp. Both codes straddle experiment from opposite sides;
OpenMC is an independent cross-check, not the benchmark.
5 seeds × 150 batches × 50k particles, CPU SVD k=5. See `resume.md`
for the three transport fixes that closed +325 → +79 pcm.

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

## Rank-vs-memory-vs-precision sweep — what we actually measured

### Godiva, 3 nuclides, on-library 294 K (`outputs/sweep_svd_wins.csv`)
80 b × 5 000 p × 3 seeds, `--discrete-rank 1`. Memory is in-engine
working-set including all reactions and discrete levels.

| rank | SVD mem | Table mem | WMP mem | SVD k_eff | Table k_eff | WMP k_eff | SVD ns/p |
|-----:|--------:|----------:|--------:|----------:|------------:|----------:|---------:|
| 1    | 113 MB  | 110 MB    | 107 MB  | 1.00056   | 1.00322     | 1.00372   | 754      |
| 2    | 126 MB  | 110 MB    | 107 MB  | 1.00601   | 1.00322     | 1.00372   | 844      |
| 3    | 138 MB  | 110 MB    | 107 MB  | 1.00563   | 1.00322     | 1.00372   | 845      |
| 5    | 164 MB  | 110 MB    | 107 MB  | 1.00358   | 1.00322     | 1.00372   | 904      |
| 7    | 176 MB  | 110 MB    | 107 MB  | 1.00202   | 1.00322     | 1.00372   | 3 191    |

**Reading:** SVD memory is *strictly larger than the Table baseline at
every rank including rank 1*. Memory is monotone in rank. Precision
gap to ACE+WMP is non-monotone in rank (rank 1 happens to land closest
on Godiva — but rank 2/3 are unstable, rank 5 is the production choice).
ns/p is roughly flat 750–905 from rank 1–5, then jumps at rank 7.

### Godiva stochastic 450 K (off-library)
Same script, `--target-temp 450` (between 294 K and 600 K library cols).

| rank | SVD mem | Table mem | WMP mem | SVD k_eff | Table k_eff | WMP k_eff |
|-----:|--------:|----------:|--------:|----------:|------------:|----------:|
| 1    | 113 MB  | 222 MB    | 213 MB  | 0.99699   | 1.00472     | 1.00501   |
| 2    | 126 MB  | 222 MB    | 213 MB  | 1.00430   | 1.00697     | 1.00501   |
| 3    | 138 MB  | 222 MB    | 213 MB  | 1.00704   | 1.00282     | 1.00501   |
| 5    | 164 MB  | 222 MB    | 213 MB  | 1.00372   | 1.00696     | 1.00501   |
| 7    | 176 MB  | 222 MB    | 213 MB  | 1.00144   | 1.00471     | 1.00501   |

**Reading:** off-library is the only regime where SVD beats Table on
memory (164 MB vs 222 MB at rank 5 = 1.35× smaller). Table doubles
because pseudo-interpolation loads two temperature columns; SVD
reconstructs from one library plus Ducru weights. **This is the
honest "where SVD wins" headline — off-library, not on-library.**

### PWR pin cell, 9 nuclides, on-library (`outputs/sweep_svd_wins_pwr.csv`)
80 b × 5 000 p × 3 seeds, sequential run (no other CPU load),
`--discrete-rank 1`. Same accounting as Godiva: all-reactions,
all-nuclide engine working set.

| rank | SVD mem | Table mem | WMP mem | SVD k_inf | Table k_inf | WMP k_inf |
|-----:|--------:|----------:|--------:|----------:|------------:|----------:|
| 1    | 106 MB  | 103 MB    | 100 MB  | 0.871     | 1.327       | 1.329     |
| 2    | 118 MB  | 103 MB    | 100 MB  | 1.409     | 1.327       | 1.329     |
| 3    | 131 MB  | 103 MB    | 100 MB  | 1.321     | 1.327       | 1.329     |
| 5    | 156 MB  | 103 MB    | 100 MB  | 1.328     | 1.327       | 1.329     |
| 7    | 169 MB  | 103 MB    | 100 MB  | 1.328     | 1.327       | 1.329     |

**Reading:** rank 1 collapses (no resonance self-shielding,
46 000 pcm low); rank 2 overshoots (8 000 pcm high); rank 3
recovers to within 800 pcm of WMP; rank 5 is the deployable
floor at 170 pcm below WMP and 5 pcm above Table; rank 7
buys nothing on precision and adds 13 MB. SVD is larger than
Table at every rank, never wins on memory.

### PWR pin cell, 9 nuclides, off-library +150 K
Same script, `--target-temp-offset 150`. Table doubles
because pseudo-interpolation loads two library columns per nuclide.

| rank | SVD mem | Table mem | WMP mem | SVD k_inf | Table k_inf | WMP k_inf |
|-----:|--------:|----------:|--------:|----------:|------------:|----------:|
| 1    | 106 MB  | 206 MB    | 200 MB  | 1.547     | 1.322       | 1.329     |
| 2    | 118 MB  | 206 MB    | 200 MB  | 1.387     | 1.323       | 1.328     |
| 3    | 131 MB  | 206 MB    | 200 MB  | 1.321     | 1.321       | 1.328     |
| 5    | 156 MB  | 206 MB    | 200 MB  | 1.324     | 1.324       | 1.328     |
| 7    | 169 MB  | 206 MB    | 200 MB  | 1.327     | 1.320       | 1.326     |

**Reading:** off-library is the only PWR regime where SVD wins
on memory (rank 5: 156 MB vs 206 MB = 1.32× smaller). SVD
matches Table at rank 5 (5 pcm gap) and lands ~460 pcm below
WMP. Earlier rank 5 single-point measurement that gave SVD
518 MB was the all-rank build with discrete levels at full
rank — this sweep uses `--discrete-rank 1` (production
default), bringing memory back into the 100-200 MB band where
the Pareto comparison is meaningful.

Plot: `outputs/memory_vs_precision.png`. Paper section: §memprec.

## Key Numbers to Remember (scope-tagged)

| Metric | Scope | Value |
|--------|-------|-------|
| U-235 fission σ₂/σ₁ | `[micro]` | 7.7e-2 |
| Reactions rank-1 to machine ε | `[micro]` | 47/52 (does not = deployable rank) |
| Reconstruction kernel speedup vs table | `[micro]` | 8–13× (does NOT survive integration) |
| GPU SVD reconstruction kernel speedup | `[micro]` | 2.6–2.8× vs single-thread CPU |
| Godiva SVD k=5 vs Table CPU throughput | `[godiva]` | 1.37×–1.90× (paper §godiva) |
| PWR SVD k=5 vs Table CPU throughput | `[pwr]` | **0.95×** (SVD slightly slower) |
| GPU SVD vs GPU pointwise (Godiva) | `[godiva]` | **0.77×** (GPU SVD 1.3× slower; paper §gpu) |
| Hybrid SVD+WMP throughput vs CPU SVD | `[pwr]` | **0.49×** (2.06× slower; paper §hybrid) |
| Hybrid in-engine memory vs Table (PWR) | `[pwr]` | **5.2× larger** (519 MB vs 100.6 MB) |
| Hybrid representation-byte ratio (paper Table 5) | `[micro]` | 132.9× (per-nuclide accounting, NOT engine memory) |
| Hybrid smooth-only rebuild memory reduction | `[pwr]` | 1.06× (519 → 488 MB) |
| Godiva dk (fission SVD k=4, pre-coupling) | `[godiva]` | 6.9 pcm |
| Godiva dk (all rxn SVD k=4, pre-coupling) | `[godiva]` | 3.7 pcm |
| PWR SVD k=5 vs ACE+WMP gap | `[pwr]` | 5 pcm (paper §pwr, on-library) |
| ICSBEP HMF-001 (benchmark) | `[godiva]` | 1.0000 ± 100 pcm (σ_exp) |
| Rust Godiva k_eff (SVD k=5) | `[godiva]` | 1.00079 ± 0.00038 |
| **Δ_ICSBEP (pass criterion)** | `[godiva]` | **+79 pcm, inside σ_exp** |
| OpenMC 0.15.3 Godiva k_eff (same HDF5) | `[godiva]` | 0.99901 ± 0.00038 |
| OpenMC Δ_ICSBEP (cross-check) | `[godiva]` | −99 pcm, inside σ_exp |
| Rust-vs-OpenMC (cross-code) | `[godiva]` | +178 pcm (not a benchmark) |
| History: const nu-bar | `[godiva]` | 0.994 (coincidental) |
| History: + E-dep nu-bar | `[godiva]` | 1.059 (+6500 pcm) |
| History: + aniso scatter | `[godiva]` | 0.965 (-9400 pcm) |
| History: + data fission | `[godiva]` | 1.006 (+4100 pcm) |
| History: + URR + interp | `[godiva]` | 1.000 (-600 pcm) |
