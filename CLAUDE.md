# open_rust_mc — Project Memory

## What This Is

A pure-Rust Monte Carlo radiation transport engine. Originally an SVD
cross-section compression experiment for neutron k-eigenvalue work
(Godiva, PWR pin cell); now also a coupled neutron–photon transport
code with depletion (CRAM-16/-48), variance reduction (weight windows,
random-ray FW-CADIS), recursive universe geometry on CPU + CUDA, and
a Python (PyO3) front-end.

Reads OpenMC HDF5 nuclear data directly via `hdf5-pure` — no C
dependency. The SVD compression line is still in tree (`kernel.rs`,
`decompose.rs`, `XsMode::Svd`) and remains a head-to-head provider
against pointwise tables / WMP / Hybrid; it's no longer the only
thing the engine does.

`origin/main` at `025c486`. Lib tests **287 / 287 green**. `cargo
check` default and `cargo check --features cuda` both clean.

## How to read the numbers below

Every row is tagged with one of:

- `[micro]` = isolated kernel / spectrum / one-nuclide-one-reaction.
  Optimistic; does NOT generalize.
- `[godiva]` = end-to-end Godiva, 3 nuclides, fast spectrum.
- `[pwr]` = end-to-end PWR pin cell, 9 nuclides, thermal, S(α,β) on.
- `[assembly]` = 17×17 PWR assembly (depth-3 recursive geometry).
- `[hex]` = HexLattice mini-core (1- or 2-ring).
- `[shield]` = photon shielding slab (`shield_slab`).
- `[photon]` = γ-heating / pulse-height / coupled n–γ.
- `[depletion]` = CRAM transmutation, fresh-corrector predictor.
- `[projected]` = analytical or extrapolated. Hypothesis until a
  scoped row replaces it.

A number quoted without scope is a bug. Repeated pattern: `[micro]`
headlines shrink (or invert sign) under `[pwr]`/`[assembly]`.

## What We've Proven

### SVD compression (spectrum + reconstruction)
- `[micro]` U-235 singular spectrum decays 5 OoM across 8 values;
  47/52 reactions rank-1 to machine ε. **Rank 1 is not deployable**
  for k_eff; rank 5 is the production choice.
- `[micro]` Bit-exact match with OpenMC's Python XS API.
- `[godiva]` k_eff < 10 pcm at all ranks (fission-only); 3.7 pcm
  with all reactions, k=4 (pre-coupling correction).
- `[pwr]` SVD(k=5) vs ACE+WMP: **5 pcm** on-library 600 K
  (paper, 100 b × 20 k × 1 seed).
- `[pwr]` SVD memory is **larger** than Table at every rank
  on-library. Memory win only appears off-library (Table needs two
  temperature columns; SVD reconstructs from one).

### Coupled neutron–photon physics (validated)
- `[photon]` Compton (KN + S(x,Z) + optional Doppler), Rayleigh,
  photoelectric phase 1, Bethe-Heitler pair, full electron condensed-
  history walk with Bethe-Bloch dE/dx + Highland MS + Seltzer-Berger
  brems. Cs-137 pulse height, NIST water buildup factors, Hubbell
  Compton spectrum all pass.
- `[photon]` PWR γ-heating split (fuel/gap/clad/water) =
  84.1 / 0.00 / 9.8 / 5.7 %; OpenMC 0.15.3 on same geometry
  ≈ 85 / 0 / 9 / 6 % — every region within 1 pp.

### Depletion (CRAM-16 / -48)
- `[depletion]` Xe-135 equilibrium vs analytical: **1e-4 relative**.
- `[depletion]` CRAM-48 matches CRAM-16 to 1e-13 on non-stiff; not
  worse at `λ·Δt = 50`.
- `[depletion]` `deplete_pwr` fresh-corrector (predictor + EOC-flux
  corrector) wired end-to-end; full PWR bench vs OpenMC found a
  chain-calibration issue (commit `8b7f594`), since closed by
  on-the-fly chain-XS spectrum collapse (`fd530d0`).

### Variance reduction
- `[shield]` 100 cm water 1 MeV photons, RR-CADIS (random-ray
  adjoint) FOM = **354/s vs analog 161.7/s** → **2.19×**.
- `[shield]` 200 cm water: RR-CADIS FOM **1.52/s vs analog 0.351/s**
  → **4.32×**. Means unbiased within combined σ.
- `[shield]` Old "lite" detector-backward CADIS proxy: 0.69× / 0.20×
  (worse than analog at both depths). Removed in this round.
- `[godiva]` Survival biasing + Russian roulette: σ_track per seed
  −28 %, FOM_track 354 → 500 (+41 %).
- `[pwr]` Survival biasing on PWR pin cell: cross-seed σ on k_inf
  −55 %, FOM_collision 412 → 1842 (4.5×).

### Recursive geometry (CPU + GPU)
- `[assembly]` 17×17 PWR depth-3 lattice runs end-to-end on CPU,
  k_inf = 1.14958 ± 0.00318. Was inexpressible pre-refactor.
- `[assembly]` Depth-3 is 1.07× slower than depth-1 with matched
  physics — depth penalty is single-digit %, not the 16× earlier
  apples-to-oranges claim.
- `[hex]` 1-ring (7 pins) k_inf = 1.35829 ± 0.00329; 2-ring (19)
  k_inf = 1.36424 ± 0.00399 — agree within 1 σ.
- `[assembly]` GPU recursive transport (`const_xs_transport_persistent`):
  **6.74× CPU speedup** at k within MC noise on RTX A1000. Cell-find
  3.2–4.0×, trace-step 5.2–5.9×, multi-step walk **24×**, all
  bit-exact (≤9.3e-11 max-rel-err) vs CPU on 200k–1M event sweeps.

### Random-ray TRRM (forward + adjoint)
- `[micro]` 1-group infinite reflective box k_inf within 500 pcm of
  analytic 1.25.
- `[micro]` Adjoint-identity (fwd vs adjoint k): agree within 800 pcm.
- `[micro]` MoC integrator τ→0 series vs analytic: 1e-12.
- `[pwr]` `rr_pincell` 2-group UO₂+water: physically correct per-
  region thermal/fast spectra; runs in ~12 s wall.

## Current State

**Neutron k-eigenvalue (the original benchmark):**
- `[godiva]` Rust SVD k=5: **1.00079 ± 0.00038**, ICSBEP HMF-001
  (1.0000 ± 100 pcm) → **Δ_ICSBEP = +79 pcm, inside σ_exp. Pass.**
- `[godiva]` OpenMC 0.15.3 same HDF5: 0.99901 → −99 pcm (cross-check).
- `[pwr]` Rust Table vs OpenMC 0.15.3: 12 pcm. Rust SVD k=5 vs OpenMC:
  −67 pcm.

**Test count progression:** 36 (original) → 148 (coupled n–γ) → 184 →
198 (recursive phase 1) → 227 (track-length k / SB / WW / hex CPU+GPU
/ shapes / dispatch) → 260 (depletion + CADIS-lite) → **287** (random-
ray TRRM + adjoint + FW-CADIS gain delivered).

## Features Implemented

### Neutron physics
- Energy-dependent ν̄(E) from HDF5 (prompt + delayed yields).
- Discrete inelastic MT=51–91 (real Q-values, two-body kinematics,
  per-level SVD rank 2).
- Continuum inelastic MT=91 (evaporation, T = √(E*/a), a = A/8).
- (n,3n) MT=17, banks 2 extra neutrons.
- Anisotropic scattering (tabular μ/CDF in CM frame from HDF5).
- Data-driven fission outgoing-energy spectrum (replaces Watt).
- URR probability tables (`multiply_smooth` true/false handled).
- S(α,β) thermal scattering for H in H₂O (continuous inelastic
  iwt=2, discrete iwt=0/1, coherent Bragg, incoherent Debye-Waller,
  stochastic T interpolation).
- Free-gas thermal scattering below 400·kT.
- Delayed-neutron emission: per-nuclide β(E), soft Watt aggregate
  delayed spectrum (a = 0.4 MeV). `[godiva]` closes Δ_ICSBEP
  196 → 19 pcm.
- URR equivalence theory (Carlvik-Pellaud Dancoff for square lattices,
  `pwr_pincell --urr-equivalence`).

### Photon physics
- Compton (KN + S(x,Z) + optional Doppler broadening from Compton
  profiles), Rayleigh (form factor + Thomson rejection),
  photoelectric phase 1 (subshell sampling), Bethe-Heitler pair.
- Full electron condensed-history transport: Bethe-Bloch dE/dx
  (per-element I from HDF5), Highland multiple scattering with
  per-cell X₀, Seltzer-Berger bremsstrahlung single-event
  emission with secondary γ banked back into the photon loop.
- NEE (next-event estimator) for tallies.
- Adjoint photon CADIS slab walker (CE adjoint Compton kernel).

### Transport / VR / tallies
- History-based + rayon-parallel CPU transport.
- Track-length k_eff estimator alongside collision estimator
  (`[godiva]` 3.9× lower seed-to-seed σ).
- Survival biasing + Russian roulette (w_min=0.25, w_survive=1.0).
- Forward weight windows (Cartesian mesh, split/roulette,
  `max_split=8`, geometric-mean w_survive).
- `WeightWindow::from_flux` (forward CADIS bootstrap from any flux).
- `WeightWindow::from_flux_adaptive` (variable-band, didn't beat
  fixed ratio empirically — flag retained behind `--ww-growth=0.0`).
- Random-ray multigroup TRRM (`random_ray::*`): forward + adjoint,
  cell-based or Cartesian FSRs with analytic or stochastic volume,
  mortal or immortal-ray (Tramm-Siegel 2021), analytic MoC ODE step.
- `rr_cadis_slab` produces the JSON `shield_slab --cadis-load`
  consumes (`{thickness_cm, n_z_bins, counts}`).
- Surface current tallies (J+ / J-), Cartesian mesh flux tallies
  (Amanatides-Woo), HDF5 statepoint with restart (warm source bank).
- Per-MT (n,p) / (n,α) tallies, BatchTallies cleanup, EntropyMesh
  in shared math lib.

### Geometry
- Recursive universes via `CoordStack` (`SmallVec<[Coord; 4]>`).
- Per-universe surface restriction + opt-in BVH (≥8 cells with
  finite AABBs) — 3.0× assembly speedup alone.
- `Mat3` cell rotations propagate through descent; rotation-free
  geometries pay zero overhead.
- `RectLattice.material_overrides` (one pin universe across an
  assembly with different enrichments/burnup tiers).
- `HexLattice` (flat-top Y / pointy-top X), cube-coord rounding +
  closed-form distance to grid; integrated into `find_cell_recursive`
  and `trace_step_recursive`.
- `geometry::shapes`: `rect_box`, `rect_box_split_bc`, `hex_boundary`,
  `pin_cylinders` — used by all production binaries and exposed via
  Python bindings.

### Depletion
- Bateman + CRAM-16 (Pusa 2016) + CRAM-48.
- Chain JSON loader (3-way `yields` semantics: omitted / `{}` /
  explicit). ENDF default yield inference.
- `BurnupMapping` (table-driven chain↔material walker).
- Predictor-corrector with fresh-corrector (clones materials,
  runs eigenvalue at predicted composition for EOC flux).
- On-the-fly chain-XS spectrum collapse — 9× to 0.77× vs OpenMC.
- Chains shipped: `chains/partial_xe.json` (4 nuclides),
  `chains/pwr_actinides.json` (17, U/Np/Pu + Xe/Sm fission products).

### GPU (CUDA, NVRTC, on RTX A1000 laptop)
- Recursive cell-find / trace-step / multi-step walk — bit-exact
  vs CPU at ≤9.3e-11 max-rel-err, 3–24× speedups.
- Constant-XS transport with collision + scatter + fission banking
  (atomicAdd) — 6.74× speedup, k within MC noise (Δ = 592 pcm,
  expected from float-rounding ordering ties).
- Photon kernels: Compton (fixed-E + per-particle-E variant),
  Rayleigh, pair. Persistent Compton history kernel: **2.22×** wall
  vs 20-thread rayon CPU on 1M histories.
- Random-ray persistent kernel scaffold (`gpu/cuda/random_ray_
  persistent.cu`) — `cargo check --features cuda` clean. Runtime
  parity test against CPU still pending (next-step work).
- HexLattice GPU port (`077db2b`): full device functions
  `gr_hex_*`, dispatched via `GR_FILL_HEX_LATTICE`. Compiles clean;
  large-volume CPU↔GPU parity test on hex 1-ring is the obvious next
  step (CPU `hex_lattice_descent_and_trace_smoke` already passes).

### Python (PyO3)
- `Scene` / `Material` / `Surface` / `PhotonMaterial` builders.
- `run_eigenvalue`, `run_gamma_heating`.
- `XsMode::{Table, Svd, HybridTableWmp, HybridSvdWmp}` per-sim
  toggle, per-MT rank overrides.
- Depletion: `Chain.from_file/from_str`, `CramOrder::{Order16, Order48}`,
  `cram`, `deplete_constant_flux`, `deplete_with_flux_callback`
  (FFI-safe Python flux closure), `Material.set_atom_density` /
  `atom_density_of`.
- `Scene.add_rect_box` / `add_hex_boundary` / `add_pin_cylinders`
  return ready-to-parse region strings.

### Honesty test
- `--mode svd|table|both|hybrid_table_wmp|hybrid_svd_wmp` flags on
  benchmarks. `--mode both` runs back-to-back with comparison print.
- OpenMC reference script: `scripts/honesty_test.py` (WSL + conda).

## What's Open / Research-Tier

- **DXTRAN-style continuous splitting** for >14 mfp photon penetration.
  All `(ratio, growth) ∈ {5,10,20} × {0,1,2,3}` at 300 cm give 0
  transmitted in 500k — `max_split=8` ceiling bounds geometric WW.
- **Full C5G7** (4 fuel × 7 groups × 17×17): data plumbing on top
  of `random_ray::*`, no new solver code.
- **HexLattice GPU runtime parity** vs CPU (large-volume sweep
  equivalent to `hex_lattice_descent_and_trace_smoke`).
- **Linear-source random-ray (1st-order)** — deferred; flat on a
  fine mesh is equivalent for axis-aligned problems.
- **Full PWR depletion bench vs OpenMC** (30–50 GWd/MTU on
  `pwr_actinides.json` + Pu/Np HDF5). Chain-calibration issue
  already addressed in `fd530d0`; the long-burn validation run
  itself is pending.
- **Per-precursor delayed-neutron groups** — only matters for
  time-dependent kinetics, not static k-eff.
- **EADL relaxation cascade on GPU** — long-flagged, still open.
  Not on the critical path for current benchmarks.
- **Source-distribution biasing** (sample initial pos from
  importance CDF) — needed for the textbook 50–1000× Wagner-
  Haghighat FOM on volume/angular sources. For `shield_slab`'s
  monodirectional point beam the importance CDF degenerates.
- **NIST ESTAR brems `S_rad` cross-check** — agrees with OpenMC's
  formula identically but ratio to NIST is element-dependent
  (0.72× H, 4.86× U). Layout vs ICRU-37 model question, doesn't
  affect γ-heating to MC stats.

## Architecture Decisions

### Kept (working well)
- **Enum dispatch for surfaces** (no vtable, jump table).
- **SvdKernel with pre-multiplied basis** (hot path is pure FMA).
- **PCG-64 PRNG** (fast, reproducible, parallel-safe, also on GPU
  as PCG-XSH-RR per-thread).
- **`hdf5-pure`** (no C dependency).
- **`faer` for SVD** (SIMD, correct).
- **`CoordStack` recursive geometry** — depth-1 is bit-identical
  to the pre-refactor flat path; no fast-path special-case needed.
- **`transport::dispatch::EigenvalueRunner`** hides CPU vs CUDA
  backend choice.

### Earlier "to improve" items — status
- ~~"Cell finding is linear scan, BVH built but not used"~~ — wired
  per-universe with the ≥8-cells gate.
- ~~"HDF5 file re-read per reaction"~~ — single-pass per nuclide
  via the per-nuclide loader.
- "Particle transport is history-based" — still true on CPU;
  GPU recursive transport is event-batched (find_cell_batch,
  trace_step_batch, multi_step_walk).

## File Layout

```
rust_prototype/src/
  lib.rs / main.rs                — crate roots
  kernel.rs / decompose.rs / cp_decompose.rs
                                  — SVD reconstruction / decomposition
  hdf5_reader.rs                  — pure-Rust HDF5 + thermal loader
  thermal.rs                      — S(α,β) sampling
  table.rs / wmp.rs               — pointwise / Windowed Multipole providers
  nuclide.rs / loader.rs          — nuclide data
  compare.rs / error.rs
  quadrature.rs / physics_constants.rs
  gpu.rs / gpu_transport.rs / gpu_recursive.rs / gpu_random_ray.rs
                                  — CUDA host-side wrappers

  geometry/                       — recursive universe geometry
    mod.rs surface.rs aabb.rs cell.rs bvh.rs ray.rs
    universe.rs lattice.rs coord.rs scene.rs shapes.rs
    recursive_smoke.rs

  physics/
    mod.rs collision.rs scatter.rs

  transport/
    mod.rs simulate.rs dispatch.rs
    particle.rs rng.rs material.rs nuclides.rs
    xs_provider.rs hybrid_xs.rs
    thermal_library.rs urr_equivalence.rs
    weight_window.rs tally.rs statepoint.rs kinetics.rs
    adjoint_neutron.rs adjoint_photon.rs

  photon/
    mod.rs material.rs data.rs hdf5_reader.rs
    compton.rs coherent.rs photoelectric.rs pair.rs
    bremsstrahlung.rs electron.rs
    transport.rs nee.rs gpu.rs

  random_ray/                     — multigroup TRRM (Tramm 2018, immortal 2021)
    mod.rs mgxs.rs fsr.rs integrator.rs solver.rs
    cadis.rs adjoint_svd.rs

  depletion/
    mod.rs cram.rs chain.rs chain_io.rs
    matrix.rs predictor_corrector.rs mapping.rs flux.rs

  bin/                            — see "Binaries" below
```

Subprojects:
- `bindings/python/` — PyO3 (`open_rust_mc` Python package).
- `cuda_bench/svd_gpu_bench.cu` — standalone GPU recon bench.
- `paper/` — manuscript (TeX + PDF).
- `scripts/` — Python analysis pipeline (phase1–5, honesty test,
  OpenMC export, hex/depletion examples).
- `chains/` — depletion chain JSONs.
- `cuda/` (or `gpu/cuda/`) — kernel sources compiled via NVRTC.

## Binaries

Neutron k-eigenvalue:
- `godiva` — Godiva sphere, 3 nuclides.
- `pwr_pincell` — PWR pin cell, 9 nuclides + S(α,β).
- `pwr_d2o_pincell` — heavy-water variant.
- `pwr_assembly` — 17×17 (use `--shape N` for 3×3 / 5×5 / 7×7).
- `hex_minicore` — N-ring hex array with hex reflective boundary.
- `validate_vs_openmc` — bit-exact validation.
- `bench_mem` / `pareto_bench` — memory/speed sweeps.
- `xs_dump` / `xs_dump_godiva` / `xs_provider_diff` / `cp_analysis`
   / `debug_trace` / `photon_dump` — diagnostics.

Photon / shielding / coupled:
- `pwr_gamma_heating` — PWR γ-heating with full ET + brems.
- `cs137_pulse_height` — pulse-height validation.
- `shield_slab` — fixed-source γ slab benchmark + WW consumer.
- `adjoint_photon_cadis_slab` — CE adjoint photon walker.

Random-ray:
- `rr_pincell` — 2-group UO₂+water pin cell (forward + adjoint).
- `rr_cadis_slab` — slab adjoint → CADIS JSON for `shield_slab`.
- `rr_adjoint_svd` / `rr_adjoint_sweep` — SVD-on-adjoint experiments.

Depletion:
- `deplete_demo` — constant-flux Xe equilibrium.
- `deplete_pwr` — transport-coupled fresh-corrector.

GPU (`--features cuda`):
- `gpu_bench` — SVD recon kernel sweep.
- `gpu_cpu_bench` — CPU/GPU head-to-head.
- `gpu_cpu_trace` / `gpu_recursive_parity` — geometry parity sweeps.
- `gpu_recursive_keff` — recursive transport k-eigenvalue.
- `gpu_const_xs_keff` — constant-XS GPU eigenvalue.
- `gpu_assembly_keff` — full assembly on GPU.
- `gpu_pwr_bench` — PWR pin cell on GPU.
- `gpu_hex_minicore` — hex on GPU.
- `gpu_compton_validate` / `gpu_compton_scaling` — photon GPU validation.
- `gpu_photon_features` — KN+S(x,Z), Doppler, Rayleigh, pair.
- `gpu_wmp_validate` — WMP provider on GPU.

Kinetics:
- `point_kinetics_demo` — point-kinetics ODE driver.

## Build & Run

```powershell
# Build (Windows / PowerShell — primary dev env)
cd rust_prototype; cargo build --release

# All lib tests (287)
cargo test --lib

# Godiva, real ENDF data
cargo run --release --bin godiva -- path\to\endfb-vii.1-hdf5\neutron `
  --rank 5 --batches 80 --inactive 15 --particles 10000

# Honesty test (SVD vs Table head-to-head)
cargo run --release --bin godiva -- path\to\endfb-vii.1-hdf5\neutron `
  --mode both --rank 5 --batches 150 --inactive 20 --particles 20000

# PWR pin cell with S(α,β), multi-seed
cargo run --release --bin pwr_pincell -- path\to\endfb-vii.1-hdf5\neutron `
  --mode both --rank 5 --batches 150 --inactive 20 --particles 50000 --seeds 5

# 17×17 PWR assembly (depth-3 recursive geometry)
cargo run --release --bin pwr_assembly -- path\to\endfb-vii.1-hdf5\neutron

# Hex minicore (1- or 2-ring)
cargo run --release --bin hex_minicore -- path\to\endfb-vii.1-hdf5\neutron --rings 2

# Photon γ-heating PWR
cargo run --release --bin pwr_gamma_heating -- path\to\endfb-vii.1-hdf5 `
  --batches 150 --inactive 20 --particles 50000 --photons 200000

# Shielding + random-ray CADIS pipeline
cargo run --release --bin rr_cadis_slab -- --thickness 100 `
  --output outputs\cadis_water_100cm.json
cargo run --release --bin shield_slab -- --thickness 100 --histories 1000000 `
  --cadis-load outputs\cadis_water_100cm.json

# Depletion (constant flux Xe equilibrium)
cargo run --release --bin deplete_demo

# Depletion (transport-coupled fresh-corrector)
cargo run --release --bin deplete_pwr -- path\to\endfb-vii.1-hdf5\neutron

# GPU (CUDA available on this machine — RTX A1000)
cargo run --release --features cuda --bin gpu_recursive_keff -- `
  path\to\endfb-vii.1-hdf5\neutron --particles 50000
cargo run --release --features cuda --bin gpu_assembly_keff -- `
  path\to\endfb-vii.1-hdf5\neutron

# Full PWR test suite
.\run_pwr_tests.ps1                # all tests
.\run_pwr_tests.ps1 -Download      # download data + run
```

## Nuclear Data

ENDF/B-VII.1 HDF5 from https://openmc.org/data/, extracted to
`data/endfb-vii.1-hdf5/`. Key files:
- `neutron/U234.h5`, `U235.h5`, `U238.h5` (Godiva).
- `neutron/H1.h5`, `O16.h5`, `Zr90-Zr96.h5` (PWR pin cell).
- `neutron/c_H_in_H2O.h5` (S(α,β) for H in water, 9 T, 186 MB).
- `photon/*.h5` (photon library: Compton, form factors, brems DCS,
  per-element I_eV).
- Full library: ~444 nuclide + thermal files, ~5.8 GB.

## Rank-vs-memory-vs-precision sweep — what was measured

### Godiva, 3 nuclides, on-library 294 K (`outputs/sweep_svd_wins.csv`)
80 b × 5 000 p × 3 seeds, `--discrete-rank 1`.

| rank | SVD mem | Table mem | WMP mem | SVD k_eff | Table k_eff | WMP k_eff | SVD ns/p |
|-----:|--------:|----------:|--------:|----------:|------------:|----------:|---------:|
| 1    | 113 MB  | 110 MB    | 107 MB  | 1.00056   | 1.00322     | 1.00372   | 754      |
| 2    | 126 MB  | 110 MB    | 107 MB  | 1.00601   | 1.00322     | 1.00372   | 844      |
| 3    | 138 MB  | 110 MB    | 107 MB  | 1.00563   | 1.00322     | 1.00372   | 845      |
| 5    | 164 MB  | 110 MB    | 107 MB  | 1.00358   | 1.00322     | 1.00372   | 904      |
| 7    | 176 MB  | 110 MB    | 107 MB  | 1.00202   | 1.00322     | 1.00372   | 3 191    |

SVD memory strictly larger than Table on-library at every rank.

### Godiva off-library, stochastic 450 K

| rank | SVD mem | Table mem | WMP mem | SVD k_eff | Table k_eff | WMP k_eff |
|-----:|--------:|----------:|--------:|----------:|------------:|----------:|
| 1    | 113 MB  | 222 MB    | 213 MB  | 0.99699   | 1.00472     | 1.00501   |
| 5    | 164 MB  | 222 MB    | 213 MB  | 1.00372   | 1.00696     | 1.00501   |

Off-library is the regime where SVD wins on memory (1.35× smaller at
rank 5) — Table doubles for pseudo-interpolation, SVD reconstructs
from one library + Ducru weights.

### PWR pin cell on-library (`outputs/sweep_svd_wins_pwr.csv`)

| rank | SVD mem | Table mem | WMP mem | SVD k_inf | Table k_inf | WMP k_inf |
|-----:|--------:|----------:|--------:|----------:|------------:|----------:|
| 1    | 106 MB  | 103 MB    | 100 MB  | 0.871     | 1.327       | 1.329     |
| 3    | 131 MB  | 103 MB    | 100 MB  | 1.321     | 1.327       | 1.329     |
| 5    | 156 MB  | 103 MB    | 100 MB  | 1.328     | 1.327       | 1.329     |
| 7    | 169 MB  | 103 MB    | 100 MB  | 1.328     | 1.327       | 1.329     |

Rank 5 = deployable floor (170 pcm below WMP, 5 pcm above Table).

### PWR pin cell off-library +150 K
SVD wins memory at rank 5: 156 MB vs Table 206 MB = 1.32×.

Plot: `outputs/memory_vs_precision.png`. Paper section: §memprec.

## Key Numbers to Remember (scope-tagged)

| Metric | Scope | Value |
|--------|-------|-------|
| Reactions rank-1 to machine ε | `[micro]` | 47/52 (≠ deployable rank) |
| Reconstruction kernel speedup vs table | `[micro]` | 8–13× (does NOT survive integration) |
| GPU SVD reconstruction kernel speedup | `[micro]` | 2.6–2.8× vs single-thread CPU |
| Godiva SVD k=5 vs Table CPU throughput | `[godiva]` | 1.37×–1.90× (paper §godiva) |
| PWR SVD k=5 vs Table CPU throughput | `[pwr]` | **0.95×** (SVD slightly slower) |
| GPU SVD vs GPU pointwise (Godiva) | `[godiva]` | **0.77×** (GPU SVD 1.3× slower; paper §gpu) |
| GPU recursive transport (const-XS) vs CPU | `[assembly]` | **6.74×** (RTX A1000) |
| GPU multi-step walk vs CPU | `[assembly]` | **24×** at <1e-13 max-rel-err |
| GPU Compton persistent kernel vs 20-thread CPU | `[photon]` | **2.22×** on 1M histories |
| Hybrid SVD+WMP throughput vs CPU SVD | `[pwr]` | **0.49×** (2.06× slower) |
| Hybrid in-engine memory vs Table | `[pwr]` | **5.2× larger** (519 MB vs 100.6 MB) |
| Godiva dk (all rxn SVD k=4, pre-coupling) | `[godiva]` | 3.7 pcm |
| PWR SVD k=5 vs ACE+WMP gap | `[pwr]` | 5 pcm (paper §pwr) |
| ICSBEP HMF-001 (benchmark) | `[godiva]` | 1.0000 ± 100 pcm (σ_exp) |
| Rust Godiva k_eff (SVD k=5) | `[godiva]` | 1.00079 ± 0.00038 |
| **Δ_ICSBEP (pass criterion)** | `[godiva]` | **+79 pcm, inside σ_exp** |
| OpenMC 0.15.3 Godiva k_eff (same HDF5) | `[godiva]` | 0.99901 ± 0.00038 |
| Rust-vs-OpenMC (cross-code) | `[godiva]` | +178 pcm (not a benchmark) |
| PWR Table vs OpenMC 0.15.3 | `[pwr]` | 12 pcm |
| 17×17 assembly k_inf (depth-3) | `[assembly]` | 1.14958 ± 0.00318 |
| Hex 1-ring (7 pins) k_inf | `[hex]` | 1.35829 ± 0.00329 |
| Hex 2-ring (19 pins) k_inf | `[hex]` | 1.36424 ± 0.00399 |
| Depth-3 vs depth-1 same physics | `[assembly]` | 1.07× slower (depth penalty single-digit %) |
| Track-length k σ vs collision σ | `[godiva]` | **3.9× lower** seed-to-seed |
| Survival biasing FOM_collision | `[pwr]` | **4.5×** (412 → 1842) |
| Delayed neutrons Δ_ICSBEP improvement | `[godiva]` | 196 → 19 pcm |
| RR-CADIS FOM 100 cm water 1 MeV γ | `[shield]` | **2.19×** vs analog |
| RR-CADIS FOM 200 cm water 1 MeV γ | `[shield]` | **4.32×** vs analog |
| Lite CADIS proxy (removed) | `[shield]` | 0.69× / 0.20× (worse than analog) |
| PWR γ-heating fuel share (this code / OpenMC) | `[photon]` | 84.1% / ~85% (within 1 pp) |
| Brems emitted at 1 MeV (PWR γ-heat run) | `[photon]` | 0.353% of source energy |
| CRAM-16 vs analytical Xe equilibrium | `[depletion]` | 1e-4 relative |
| CRAM-48 vs CRAM-16 (non-stiff) | `[depletion]` | 1e-13 relative |
| Statepoint warm-restart speedup | `[godiva]` | 33% faster (1542 → 1022 ns/p) |
| **Lib test count** | — | **287 / 287 green** |
