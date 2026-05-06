# Project status — 2026-05-06

A snapshot of what's on `origin/main` and what's still missing.
Pairs with `resume.md` (the project journal — round-by-round
narrative). This file is forward-looking: "where are we, what's
next".

`origin/main` at `a17c379`. Lib tests **227 / 227 green** on the
default profile; `cargo check --features cuda` clean; preview
build (`--features preview`) clean.

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
  to-end via `hex_minicore`; GPU compiles clean, on-device
  parity test pending CUDA hardware.

### Cross sections
- SVD-compressed reaction kernels (continuous-T via Ducru).
- Pointwise table provider (OpenMC-style, with stochastic
  pseudo-interpolation between library temperatures).
- Hybrid SVD + WMP and Table + WMP.
- URR probability tables (`multiply_smooth=true|false`).
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
- Examples: `godiva.py`, `pwr_pincell.py`, `pwr_gamma_heating.py`,
  `seed_sweep.py`, `xs_mode_demo.py`, `xs_mode_quick.py`.

## What's missing

### Quick wins (hours to a day each)

- **Python `hex_minicore.py` example.** Wire
  `Scene.add_hex_boundary` + `add_pin_cylinders` end-to-end.
  Demonstrates the new helpers; ~50 lines.
- **Hex on GPU runtime parity test.** Mirror
  `gpu_recursive_parity::hex_lattice_descent_and_trace_smoke`
  with a million-event GPU↔CPU sweep. Schema is in place; needs
  CUDA hardware + a binary that builds a hex universe and calls
  `find_cell_batch` / `trace_step_batch`.
- **Binary refactors to use `EigenvalueRunner`.** godiva,
  pwr_pincell, pwr_assembly, hex_minicore, gpu_assembly_keff
  each shrink ~10–20 lines once they go through `runner.run()`.
  Not urgent.
- **Apply WW bootstrap on a heterogeneous problem.** The
  bootstrap pipeline is correct; Godiva is too uniform to reward
  it. Wire `--ww-bootstrap-batches` into `pwr_pincell` (auto-
  attach the mesh tally over the pin box) and measure FOM gain.
- **Quantify delayed-neutron impact on PWR.** Multi-seed
  pwr_pincell run with delayed neutrons toggled on / off via a
  flag. The current code always has them on; comparing requires a
  conditional path (or running an older commit).

### Medium-effort (1-2 weeks each)

- **Survival biasing in the thermal-scatter path.** Currently the
  analog branch in `transport_particle` for S(α,β) collisions
  doesn't go through `dispatch_real_collision` — it has its own
  path via `process_non_thermal_collision`. Routing it through
  the SB dispatch would extend variance reduction to PWR's
  thermal physics. Net win is small for PWR (H1 has no fission)
  but cleans up the code path.
- **Hex on GPU validated end-to-end on a real eigenvalue.** Once
  hardware is available: build a `gpu_hex_minicore` binary that
  reuses the CPU `hex_minicore` geometry and runs the GPU
  recursive transport. Compare k_inf to the CPU result.
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

- **CADIS / FW-CADIS automatic weight-window generation.**
  Forward bootstrap (now in tree) is the cheap proxy. Real CADIS
  needs a deterministic adjoint transport solver (S_N or adjoint
  MC) to compute the importance map. This is a separate code-
  scale project.
- **Predictor-corrector depletion (CE/LI, CE/CM).** Couples a
  CRAM-based Bateman solver to the transport loop in a feedback
  cycle. Algorithmically known (Pusa 2016 for CRAM-16, Isotalo-
  Aarnio 2011 for the predictor-corrector schemes, OpenMC methods
  paper) but implementationally non-trivial. **Status:** the
  Bateman / CRAM solver does **not** exist in the tree (earlier
  STATUS.md text claiming it did was incorrect). All depletion
  pieces — chain, transmutation matrix, CRAM evaluator,
  predictor-corrector loop, atom-density feedback into `Material`
  — are open work.
- **URR equivalence theory (Stoker-Weiss / NJOY).** Current URR
  probability tables handle stochastic XS sampling correctly for
  infinite medium; the equivalence-theory / Dancoff-factor
  correction for tight lattices is the follow-on. Published but
  subtle.
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

# Lib tests (227)
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

# Statepoint + chained restart
./target/release/pwr_assembly ../data/endfb-vii.1-hdf5/neutron \
    --shape 3 --batches 40 --inactive 10 --particles 2000 \
    --reflective-z --statepoint /tmp/state1.h5
./target/release/pwr_assembly ../data/endfb-vii.1-hdf5/neutron \
    --shape 3 --batches 40 --inactive 5 --particles 2000 \
    --reflective-z --restart-from /tmp/state1.h5

# Python (after maturin develop --release)
python rust_prototype/bindings/python/examples/godiva.py \
    data/endfb-vii.1-hdf5/neutron
```
