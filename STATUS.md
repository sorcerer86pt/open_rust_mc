# Project status — 2026-05-07

A snapshot of what's on `origin/main` and what's still missing.
Pairs with `resume.md` (the project journal — round-by-round
narrative). This file is forward-looking: "where are we, what's
next".

Deferred specs:
- [`ICSBEP.md`](ICSBEP.md) — phased plan to run the full ICSBEP
  benchmark suite. Engineering-heavy, ~13 weeks; deferred until
  time-dependent kinetics, real continuous-energy adjoint MC, and
  validated full-PWR depletion bench land first.

`origin/main` at `43b3236`. Lib tests **260 / 260 green** on the
default profile; `cargo check --features cuda` clean; Python
bindings (`-p open-rust-mc-py`) clean.

## What's on main

### Geometry
- Recursive `Geometry`: surfaces, cells, universes, rectangular
  lattices, hex lattices, per-cell rotations, BVH-accelerated
  cell finding.
- `geometry::shapes` builders: `rect_box`, `rect_box_split_bc`,
  `hex_boundary`, `hex_side_normals`, `pin_cylinders`. Same
  helpers exposed in Python via `Scene.add_rect_box`,
  `add_hex_boundary`, `add_pin_cylinders`.
- HexLattice transport on **CPU** (descent + trace_step + grid-
  distance dispatch) and on **GPU** (CUDA device functions, 8
  uploaded SoA buffers, 5 kernel signatures). CPU validated end-
  to-end via `hex_minicore`; GPU on-device parity confirmed on
  RTX A1000 (`outputs/gpu_recursive_parity_run.txt`,
  Tests 7–9, 2026-05-08).

### Cross sections
- SVD-compressed reaction kernels (continuous-T via Ducru).
- Pointwise table provider (OpenMC-style, with stochastic
  pseudo-interpolation between library temperatures).
- Hybrid SVD + WMP and Table + WMP.
- URR probability tables (`multiply_smooth=true|false`).
- **URR equivalence theory** (Stoker-Weiss / NJOY rational form):
  Carlvik-Pellaud Dancoff factor for square pin lattices, per-cell
  `(C, l̄)` cache, σ_eff correction wired into `apply_urr` for
  flagged absorber nuclides. `pwr_pincell --urr-equivalence`
  toggles it.
- S(α,β) thermal scattering for H in H₂O (continuous inelastic
  + discrete + coherent / incoherent elastic).
- Delayed-neutron yields ν_d(E) loaded per nuclide.

### Transport
- Eigenvalue power iteration with rayon parallel transport.
- Surface tracking and Woodcock delta tracking, auto-detected
  by material contrast.
- Per-batch results: collisions, fissions, leakage, thermal
  scatters, surface crossings, Shannon entropy, k_eff (collision
  estimator), `k_track` (track-length estimator, surface
  tracking only).
- Variance reduction:
  - Implicit-capture survival biasing + Russian roulette
    (surface tracking standard branch + delta tracking).
  - Cartesian-mesh weight windows (forward application; flux-
    bootstrap generation via `WeightWindow::from_flux`).
  - **`--disable-delayed-neutrons`** ablation flag.
- Tallies: surface currents (J+ / J- by direction), Cartesian
  mesh flux (Amanatides-Woo deposit). Library helpers for the
  common patterns (`for_reflective_surfaces`,
  `for_boundary_surfaces`, `from_aabb`).
- HDF5 statepoint write + read + restart, with chained-restart
  stability validated on 3-hop chains across 3×3 / 5×5 / 7×7
  minicores.
- Delayed-neutron emission: per-fission-neutron prompt vs delayed
  sampling using ν_d / ν_total; soft Watt spectrum (a = 0.4 MeV)
  for delayed energies.
- Backend dispatch: `transport::dispatch::EigenvalueRunner` with
  `CpuRunner` and (cuda-feature-gated) `CudaRunner`.

### Depletion (burnup)
- **CRAM-16 / CRAM-48** matrix exponential evaluator (IPF form,
  Pusa 2016 / OpenMC-canonical poles & residues), dense complex
  Gaussian elimination with partial pivoting.
- `DepletionChain` data structure (decay constants, decay branches,
  per-(parent, MT) one-group reaction XS with default ENDF yield
  inference for `(n,γ) / (n,2n) / (n,3n) / (n,p) / (n,α)`).
- `chain_io` JSON loader with three-way `yields` semantics:
  omitted → default ENDF; `{}` → pure removal; `{daughter: y, ...}`
  → explicit.
- Two shipped chain libraries: `chains/partial_xe.json` (4
  nuclides — Xe poisoning) and `chains/pwr_actinides.json`
  (17 nuclides — actinide buildup + dominant FP poisons).
- CE/LI predictor-corrector step (`deplete_ce_li`) — predictor with
  BOC flux, user-supplied `flux_at` callback for the EOC flux,
  corrector at the average matrix.
- `BurnupMapping` — table-driven walker that pushes chain-evolved
  atom densities back into transport materials. Auto-derived from
  `NUCLIDE_INFO` in `deplete_pwr` so any chain JSON drops in.
- Power-normalised source rate + per-cell mean flux extractor
  (`flux::*`) for transport-coupled burnup.
- `deplete_pwr` driver — full eigenvalue → flux → CRAM →
  composition update loop with **fresh-corrector**: clones
  materials, runs eigenvalue at predicted composition for the EOC
  flux estimate. Every burnup step does 2 transport solves
  (predictor BOC + corrector EOC).
- Validation: Xe-135 equilibrium reproduces analytical formula to
  1e-4 relative; pwr_actinides chain solves 1-day step cleanly with
  qualitatively correct U-238 → Pu-239 buildup.

### Photon transport
- Coupled neutron-photon (driver runs neutron k-eigenvalue, banks
  photon source events, drives photon transport from the bank).
- Compton (free Klein-Nishina + bound rejection + optional
  Doppler), Rayleigh, photoelectric phase 1, Bethe-Heitler pair.
- Full electron transport: Bethe-Bloch dE/dx, Highland multiple
  scattering, Seltzer-Berger bremsstrahlung secondaries.
- GPU photon kernels (Compton / Rayleigh / photoelectric phase 1
  / pair) at bit-parity with CPU; persistent-kernel mode for
  full Compton history loops.
- **Per-photon weight bookkeeping** (`source_weight` +
  `transport_history_csg_with_ww`): tally accumulators all scale
  by current weight; existing analog callers go through the
  weight=1 thin wrapper with no behaviour change.

### Shielding / variance reduction
- `shield_slab` benchmark — fixed-source γ transmission through a
  thick slab (water / concrete / Pb / Fe / W), produces the
  analog FOM = 1 / (σ_rel² · t) reference. 100 cm water at 1 MeV:
  T = 5.26e-3 (matches ANSI/ANS-6.4.3 buildup factor), FOM = 348/s.
- **CADIS-lite calibration**: `--cadis-calibration N` runs
  detector-backward photons and saves the resulting collision-
  density importance map ψ̂\*(z) to JSON (`--cadis-save FILE`).
- **CADIS WW application**: `--cadis-load FILE` ingests the map,
  builds a `WeightWindow` with `w_target ∝ ψ̂\*_max / ψ̂\*`, and
  applies splitting/roulette in the photon hot path. Source-side
  normalisation: `w_ref` calibrated so `w_target(source) ≈ 1.0`,
  keeping the tally in analog units.
- **Status: working framework, FOM gain not yet delivered.**
  Transmission is unbiased (5.21e-3 CADIS vs 5.45e-3 analog,
  agree within 1σ) and 120× more samples reach the detector at
  200 cm. But splits are highly correlated and σ_rel barely
  improves; net FOM is 220/s vs analog 361/s at 100 cm. The
  CADIS-lite proxy isn't a true adjoint flux; real continuous-
  energy adjoint MC + source-distribution biasing is the
  research-tier follow-on.

### GPU
- Recursive geometry kernels (`find_cell_batch`,
  `trace_step_batch`, `multi_step_walk`).
- Constant-XS transport (`const_xs_transport_persistent`).
- Full-physics `transport_recursive` with SVD / Pointwise / WMP /
  URR / S(α,β).
- HexLattice device functions and SoA upload (just landed).
- Persistent-kernel photon Compton.

### Python bindings
- PyO3 builders for `Material`, `Scene`, `Settings`,
  `PhotonMaterial`, `XsMode`.
- Surfaces: `Sphere`, `XCylinder`, `YCylinder`, `ZCylinder`,
  `XPlane`, `YPlane`, `ZPlane`.
- Shape builders: `Scene.add_rect_box`, `add_hex_boundary`,
  `add_pin_cylinders`.
- `run_eigenvalue`, `run_gamma_heating` entry points.
- Convenience material builders (`uranium_oxide_material`,
  `water_material`, `zircaloy4_material`).
- **Depletion API**: `Chain.from_file(path)` / `Chain.from_str`,
  `CramOrder.Order16` / `Order48`, `cram(matrix, n0, order)`,
  `deplete_constant_flux`, `deplete_with_flux_callback` (FFI-
  exception-safe Python `flux_at` closure for predictor-corrector
  with mid-step transport solves).
- `Material.set_atom_density(hdf5_file, density)` /
  `atom_density_of(hdf5_file)` for in-place composition updates
  between burnup steps.
- Examples: `godiva.py`, `pwr_pincell.py`, `pwr_gamma_heating.py`,
  `seed_sweep.py`, `xs_mode_demo.py`, `xs_mode_quick.py`,
  `hex_minicore.py`, `depletion_xe_demo.py`.

## What's missing

### Quick wins (hours to a day each)

- ~~**Hex on GPU runtime parity test on real hardware.**~~
  **Done 2026-05-08, RTX A1000 Laptop GPU.** Geometry-primitive
  parity (Tests 7–9 in `gpu_recursive_parity` for find_cell /
  trace_step / multi_step_walk on a 1-ring hex mini-core) and
  full-physics k_inf parity both confirmed:

  - Geometry primitives: 0/200 000, 0/50 000, 0/20 000 disagreement
    after fixing arbitrary-orientation reflection (`gr_reflect_direction`
    helper added in `geom_recursive.cu`; previously only axis-aligned
    `GR_SURF_PLANE_X/Y/Z` were reflected — hex sides are
    `GR_SURF_PLANE_GENERAL`). Speedups 6.6× / 6.1× / 21.6× vs CPU.
  - Eigenvalue: new `gpu_hex_minicore` binary using the
    dispatch `CudaRunner`. CPU k_inf = 1.36009 ± 0.00137, GPU
    k_inf = 1.35938 ± 0.00341 (4 seeds × 60 batches × 5 000
    particles, rank 5). Δ = 71 pcm < 0.2σ_combined.
  - Artifacts: `outputs/gpu_recursive_parity_run.txt`,
    `outputs/gpu_hex_minicore_4seeds.txt`,
    `outputs/cpu_hex_minicore_4seeds.txt`.
- **Binary refactors to use `EigenvalueRunner`.** `hex_minicore`
  uses `CpuRunner.run()`; godiva / pwr_pincell / pwr_assembly /
  gpu_assembly_keff still call `simulate::run_eigenvalue` directly.
  Optional cleanup; no behaviour change.
- **OpenMC cross-validation on PWR pin cell with URR equivalence
  on.** We ship Carlvik-Pellaud Dancoff for square lattices with
  validated asymptotic limits, but haven't yet measured the
  predicted ~50-200 pcm shift on the existing PWR pin cell vs a
  reference run with `enable_resonance_equivalence` on.
- **`pwr_actinides.json` end-to-end run.** Chain JSON loads and
  solves CRAM cleanly. Running `deplete_pwr` over real burnup
  with this chain needs Pu239/Pu240/Pu241/Pu242/Np237/Np239/etc.
  HDF5 files added to `NUCLIDE_SPECS` + `NUCLIDE_INFO` so the
  `BurnupMapping` includes them. Mechanical 1-day extension once
  the data files are in `data_dir`.

### Medium-effort (1-2 weeks each)

- **Survival biasing in the thermal-scatter path.** Currently the
  analog branch in `transport_particle` for S(α,β) collisions
  doesn't go through `dispatch_real_collision` — it has its own
  path via `process_non_thermal_collision`. Routing it through
  the SB dispatch would extend variance reduction to PWR's
  thermal physics. Net win is small for PWR (H1 has no fission)
  but cleans up the code path.
- ~~**Hex on GPU validated end-to-end on a real eigenvalue.**~~
  **Done 2026-05-08.** `gpu_hex_minicore` binary lives in
  `src/bin/`, reuses the CPU `hex_minicore` geometry, runs the
  GPU recursive transport via the dispatch `CudaRunner`. k_inf
  parity confirmed at 71 pcm < 0.2σ_combined (see Quick wins).
- **Track-length estimator under delta tracking.** Currently the
  delta-tracking path leaves `k_track = 0` — the integrand can't
  be reconstructed when the path crosses material boundaries
  silently. The Sutton-Brown delta-tracking-compatible track-
  length estimator (tally on real events with the local-cell
  ν·σ_f) would close that gap.
- **Surface and mesh tallies under delta tracking.** Same issue
  as the track-length estimator. Currently the helpers are
  surface-tracking-only.

### Substantial (months each — research-grade)

- **Real continuous-energy adjoint Monte Carlo for photon CADIS.**
  The CADIS-lite proxy (forward photons run from the detector,
  collision-density tally as ψ̂\*) ships with the WW-application
  pipeline plumbed through, but it doesn't deliver textbook FOM
  gain (220/s vs analog 361/s on 100 cm water). Real adjoint MC
  needs: (1) transposed Compton kernel — sample E_in given E_out
  via the inverted Klein-Nishina; (2) adjoint photoelectric as a
  source term (the absorption cross-section becomes an emission
  source in the adjoint); (3) energy-dependent WW (4D mesh
  instead of 1D z-only); (4) source-distribution biasing
  (sample initial position from the importance CDF) instead of
  geometric splitting. Wagner-Haghighat 2003 reports 50-1000×
  FOM gain when these are combined. Multi-week effort.
- **CADIS for neutron shielding** — same machinery on the neutron
  side. Photon CADIS is the prototype; once the adjoint MC
  pattern is solid, transposing scatter / fission / capture
  kernels for neutrons follows the same architecture.
- **Doppler-broadened coherent elastic scattering.** Bragg-edge
  treatment beyond the standard S(α,β) tables. Needs per-material
  crystallographic data + phonon spectrum integration.
- **EADL relaxation cascade on GPU.** Photoelectric phase 1 is
  on GPU; the fluorescence + Auger cascade is CPU-only because
  the SoA / thread-divergence design isn't worked out. Roughly
  one week once a clean SoA layout is sketched.
- **Event-based GPU transport.** Current `transport_recursive`
  is history-based (one particle birth-to-death per thread).
  Tramm 2024 reports event-based is ~6× faster on GPU;
  prerequisite is a particle-sort-by-(material, energy) phase
  before XS lookup.
- **Photon depletion / activation transport.** Separate from
  neutron depletion — would track activation products and their
  decay photons over time.
- **Full PWR depletion bench.** With `chains/pwr_actinides.json`
  + extended `NUCLIDE_SPECS` (Pu239/Pu240/Pu241/Pu242/Np ZAIDs),
  run `deplete_pwr` over a 30-50 GWd/MTU burnup history,
  compare U-235 / Pu-239 / Xe-135 / Sm-149 trajectories vs
  OpenMC's depletion solver on the same chain. The framework is
  in place; the long run + comparison is the substantial part.

### Documentation / housekeeping

- **OpenMC cross-validation of the Python path beyond γ-heating.**
  Per `PYTHON.md`, the bindings are validated only at the
  plumbing level for non-γ-heating paths. A k_eff cross-check
  through Python on Godiva / PWR pin cell against a fresh OpenMC
  run would close the loop.
- **Brems DCS vs NIST ESTAR.** Open question from the previous
  resume.md round: per-element ratios are 0.7× / 2.46× / 3.18× /
  4.86× across H / O / Zr / U vs ESTAR. The integrals match
  OpenMC's formula exactly, so the discrepancy is in
  Seltzer-Berger HDF5 layout interpretation. γ-heating numbers
  match OpenMC, so this isn't blocking; flag for any brems-
  dominated benchmark (Møller-Plesset shielding on Pb, etc.).

## How to run

```bash
# Build (default — CPU only)
cd rust_prototype && cargo build --release

# Build with CUDA
cargo build --release --features cuda

# Lib tests (260)
cargo test --lib --release

# Benchmarks
./target/release/godiva ../data/endfb-vii.1-hdf5/neutron \
    --rank 5 --batches 80 --inactive 20 --particles 5000 --seeds 4

./target/release/pwr_pincell ../data/endfb-vii.1-hdf5/neutron \
    --rank 5 --batches 100 --inactive 20 --particles 20000 --seeds 5

./target/release/pwr_assembly ../data/endfb-vii.1-hdf5/neutron \
    --shape 17 --batches 50 --inactive 15 --particles 10000

./target/release/hex_minicore ../data/endfb-vii.1-hdf5/neutron \
    --rings 1 --rank 5 --batches 60 --inactive 15 --particles 5000

# With variance reduction
./target/release/godiva ../data/endfb-vii.1-hdf5/neutron \
    --survival-biasing --weight-window --ww-lower 0.5 --ww-upper 2.0

# PWR pin cell with WW bootstrap + URR equivalence + delayed-ablation
./target/release/pwr_pincell ../data/endfb-vii.1-hdf5/neutron \
    --rank 5 --batches 100 --inactive 20 --particles 20000 --seeds 5 \
    --ww-bootstrap-batches 10 --urr-equivalence

# Statepoint + chained restart
./target/release/pwr_assembly ../data/endfb-vii.1-hdf5/neutron \
    --shape 3 --batches 40 --inactive 10 --particles 2000 \
    --reflective-z --statepoint /tmp/state1.h5
./target/release/pwr_assembly ../data/endfb-vii.1-hdf5/neutron \
    --shape 3 --batches 40 --inactive 5 --particles 2000 \
    --reflective-z --restart-from /tmp/state1.h5

# Depletion (Bateman + CRAM-16 / -48)
./target/release/deplete_demo --steps 16 --total-hours 80 --cram-order 16
./target/release/deplete_pwr ../data/endfb-vii.1-hdf5/neutron \
    --steps 8 --hours-per-step 5 --power-w-per-cm 200 \
    --chain ../chains/pwr_actinides.json --cram-order 48

# Photon shielding benchmark + CADIS-lite
./target/release/shield_slab ../data/endfb-vii.1-hdf5/photon \
    --histories 1_000_000 --thickness-cm 100   # analog FOM baseline
./target/release/shield_slab ../data/endfb-vii.1-hdf5/photon \
    --histories 30_000 --thickness-cm 100 --cadis-z-bins 25 \
    --cadis-calibration 30_000 --cadis-save /tmp/cadis_map.json
./target/release/shield_slab ../data/endfb-vii.1-hdf5/photon \
    --histories 200_000 --thickness-cm 100 \
    --cadis-load /tmp/cadis_map.json --ww-ratio 5

# Python (after maturin develop --release)
python rust_prototype/bindings/python/examples/godiva.py \
    data/endfb-vii.1-hdf5/neutron
python rust_prototype/bindings/python/examples/depletion_xe_demo.py \
    chains/partial_xe.json
```
