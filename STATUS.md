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
- **`NuclideLibrary`** (`transport::nuclides`) — ZAID-keyed
  catalog of structural / actinide / FP nuclides + on-demand
  HDF5 metadata read (AWR + temperature columns) + nearest-
  temperature selection. Replaces the per-binary
  `NUCLIDE_SPECS: &[(&str, f64, f64, usize)]` tables with a
  registry: binaries describe materials by ZAID + target
  temperature, the library resolves the file path / AWR /
  `temp_idx`. Catalog covers H/D/T, B-10/11, C-nat, O-16/17,
  Zr-90/91/92/94/96, Fe-54/56/57/58, U-233 through Cm-247,
  and the major FP poisons (I-135, Xe-135, Cs-135, Pm-149,
  Sm-149, Gd-155, Gd-157).
- **`ThermalLibrary`** (`transport::thermal_library`) — named
  `c_*.h5` resolver covering H-in-H₂O, **D-in-D₂O** (heavy
  water), graphite, ZrH (TRIGA), Be / BeO, polyethylene,
  benzene, α-quartz, bound O / U in UO₂, methane (liquid /
  solid), ortho/para H₂ and D₂, and Al-27 / Fe-56 metals.
- **URR equivalence theory** (Stoker-Weiss / NJOY rational form):
  Carlvik-Pellaud Dancoff factor for square pin lattices, per-cell
  `(C, l̄)` cache, σ_eff correction wired into `apply_urr` for
  flagged absorber nuclides. `pwr_pincell --urr-equivalence`
  toggles it.
- S(α,β) thermal scattering for H in H₂O (continuous inelastic
  + discrete + coherent / incoherent elastic). D-in-D₂O
  validated through `pwr_d2o_pincell` with an end-to-end
  pitch-sweep moderation curve.
- Delayed-neutron yields ν_d(E) loaded per nuclide.

### Adjoint photon Monte Carlo (CADIS, real CE)
- `crate::photon::compton::adjoint_compton_scatter` — inverted
  Klein-Nishina sampler (Wagner-Haghighat 1998 / Lewis-Miller
  §10.3). Given a post-collision energy `E_out` returns the
  pre-collision `E_in` and scattering cosine μ from the transposed
  KN kernel. Rejection ceiling is the analytic supremum
  `2π · r_e²` (proven and empirically scanned in
  `adjoint_compton_envelope_bound`). Conditional density at fixed
  E_out matches the analytic `KN_dcs/dμ(E_in, μ_kin)` curve to
  χ²_red < 2.5 across 25 bins from 200 k samples.
- `crate::transport::adjoint_photon` — slab-geometry adjoint
  walker. Composes the adjoint Compton with self-adjoint Rayleigh
  + photoelectric / pair termination ("absorption-as-source")
  into a track-length tally on a `(z, E)` mesh. Output:
  `ImportanceMap` directly consumable by
  `WeightWindow::from_flux` for the existing CADIS pipeline.
  End-to-end test on a 100-cm water slab at 1 MeV produces a
  diffusion-like ψ̂*(z) peaked at mid-slab with measurable
  up-scatter contribution above the birth-energy bin (571 k vs
  144 k integrated track length). Replacing the random-ray FW
  proxy in `shield_slab` with this walker is the next integration
  step.

### Time-dependent kinetics
- **Point-kinetics** (`transport::kinetics`) with 6-group
  delayed-neutron precursors. Crank-Nicolson 7×7 ODE solver
  (A-stable on the prompt-mode stiffness). Keepin / Hetrick-Roberts
  six-group constants for U-235 thermal, U-238 fast, Pu-239 thermal
  fission shipped as constants; `blend()` combines weighted nuclide
  contributions for mixed cores. Closed-form `prompt_jump_ratio`
  and `inhour_period` (Newton on the inhour equation) for analytic
  cross-check. Demo binary `point_kinetics_demo` runs step / ramp /
  scram reactivity profiles, emits CSV. Validation: equilibrium is
  a fixed point (1e-6 drift over 1 s); prompt-jump matches analytic
  to <5 %; **late-time period at ρ=50 ¢ matches inhour equation
  to −0.37 %** (`outputs/pk_step_50c.csv` + the validation in
  `scripts/`); 10 unit tests green.

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
- ~~**Binary refactors to use `EigenvalueRunner`.**~~
  **Mostly done 2026-05-08.** `hex_minicore`, `gpu_hex_minicore`,
  `godiva`, `pwr_pincell`, `pwr_assembly`, `pwr_gamma_heating`
  now drive the eigenvalue loop through
  `transport::dispatch::{CpuRunner, CudaRunner}.run(&config)`.
  No behaviour change; `cargo test --lib --release` 316/316
  green; smoke runs of godiva (k≈1.0) and pwr_pincell (k≈1.327)
  match the documented values. `gpu_assembly_keff` deliberately
  unchanged — its per-batch live-progress print would be lost
  without adding `verbose` support to `CudaRunner`.
- ~~**OpenMC cross-validation on PWR pin cell with URR equivalence
  on.**~~ **Done 2026-05-08 (fix landed).**
  3 seeds × 60 batches × 10 000 particles, SVD rank 5, 3.1 % UO₂
  pin cell @ 600 K / 900 K, identical geometry across both codes:

  | Config                          | k_inf    | σ (3 seeds) |
  |---------------------------------|----------|-------------|
  | OpenMC 0.15.3                   | 1.32773  | 0.00205     |
  | Rust w/o URR-eq                 | 1.32715  | 0.00153     |
  | Rust w/ URR-eq (rational, old)  | 1.33479  | 0.00226     |
  | **Rust w/ URR-eq (Hwang, fix)** | 1.32892  | 0.00095     |

  Baseline cross-check (no URR-eq): Rust vs OpenMC Δk = -58 pcm
  (0.23σ_combined — pass). The original rational form
  `σ_eff = σ_URR · σ_0/(σ_0+σ_e)` shifted +764 pcm — 3-15× the
  textbook 2-15 % shielding band — because it shielded the full
  URR sample including U-238's smooth ~11.8 b potential elastic
  baseline. Diagnosed via `scripts/urr_eq_dump.py`
  (`outputs/urr_eq_dump.txt`): Carlvik-Pellaud C = 0.68, σ_e
  = 17.4 b, σ_0 = 7.9 b — all correct; the formula's domain of
  applicability was the bug.

  Fix: **Hwang superposition** (`apply_equivalence_correction`,
  `transport/urr_equivalence.rs`):

  ```text
    σ_eff = σ_smooth + (σ_URR − σ_smooth) · σ_0 / (σ_0 + σ_e)
  ```

  The Bondarenko factor `σ_0/(σ_0+σ_e)` shields only the
  resonance-fluctuation `Δσ = σ_URR − σ_smooth` above the
  off-resonance baseline, leaving smooth potential scattering
  and smooth s-wave capture intact (NJOY PURR §13). Threading
  the smooth (pre-`apply_urr`) baseline through to the
  equivalence pass adds one `[MicroXs; MAX_NUCLIDES]` snapshot
  per collision in both surface and delta-tracking hot paths.

  Result: shift = **+177 pcm** (in the textbook 50-200 pcm
  band) and Rust k_inf = 1.32892 lands at 0.53σ_combined of the
  OpenMC reference — consistent with OpenMC at the level of
  3-seed MC noise.

  Artifacts:
  `outputs/pwr_pincell_no_urr_eq.txt`,
  `outputs/pwr_pincell_with_urr_eq.txt` (old rational form),
  `outputs/pwr_pincell_with_urr_eq_hwang.txt` (Hwang fix),
  `outputs/openmc_pwr_urr_ref.json`,
  `outputs/urr_eq_dump.txt`,
  `scripts/urr_eq_dump.py`.
- ~~**`pwr_actinides.json` end-to-end run.**~~ **Done 2026-05-08.**
  `deplete_pwr` now wires all 17 chain ZAIDs into transport
  (0 chain-only); the actinide buildup nuclides (U-236/237/239,
  Np-237/239, Pu-239/240/241/242, Am-241, I-135, Cs-135,
  Pm-149, Sm-149) loaded from HDF5 with initial density 0 and
  fed by CRAM each step. `MAX_NUCLIDES` bumped from 16 → 32
  to fit the 18-nuclide fuel material. Per-step trace adds
  Pu-239/Pu-240/Sm-149 columns when the chain has actinides.
  6 steps × 48 h × 200 W/cm at CRAM-48 (`outputs/deplete_pwr_actinides.txt`):
  Pu-239/U-235 grows linearly 0 → 2.4 %, Pu-240 grows
  quadratically, Sm-149 reaches near-equilibrium ~1.5e-4,
  k_eff 1.331 → 1.206 over 12 d (steep but expected for the
  light pin-cell model). Qualitatively correct trajectories.

### Medium-effort (1-2 weeks each)

- ~~**Survival biasing in the thermal-scatter path.**~~
  **Done 2026-05-08.** The use-thermal-but-non-thermal sub-branch
  in `transport_particle` (a real reaction on a S(α,β) nuclide
  — fission / capture / inelastic, not a thermal scatter) now
  routes through `dispatch_real_collision` instead of the
  dedicated `process_non_thermal_collision` helper. SB is now
  uniform across all collision paths: thermal scatters stay
  analog (no fission/capture to bias), every other reaction —
  including capture on H-1 below 3.75 eV — goes through
  implicit-capture + Russian roulette when SB is enabled.
  `process_non_thermal_collision` deleted. PWR pin cell smoke
  (50 batches × 5 000 particles × 3 seeds): k_inf agrees
  with/without SB at 0.58σ_combined (1.32763 ± 0.00263 vs
  1.32960 ± 0.00218); SB lowers σ slightly. 322/322 lib tests
  green.
- ~~**Hex on GPU validated end-to-end on a real eigenvalue.**~~
  **Done 2026-05-08.** `gpu_hex_minicore` binary lives in
  `src/bin/`, reuses the CPU `hex_minicore` geometry, runs the
  GPU recursive transport via the dispatch `CudaRunner`. k_inf
  parity confirmed at 71 pcm < 0.2σ_combined (see Quick wins).
- ~~**Track-length estimator under delta tracking.**~~
  **Done 2026-05-08.** `transport_particle_delta` now scores
  `w · ν·Σ_f(m, E) / Σ_t(m, E)` at every real (post-acceptance)
  collision — the unbiased Sutton-Brown collision form of the
  track-length estimator. In expectation,
  `Σ_t(m,E) · ν·Σ_f(m,E)/Σ_t(m,E) = ν·Σ_f(m,E)`, the same
  integrand the surface-tracking path accumulates per
  cell-residence segment, so the two `k_track` columns are
  directly comparable. New unit tests
  `delta_tracking_two_material_problem_picks_delta` and
  `delta_tracking_k_track_matches_k_eff` exercise a low-contrast
  two-material geometry that auto-selects delta tracking and
  verify k_track is non-zero and agrees with k_eff within MC
  noise. 320/320 lib tests green.
- ~~**Surface and mesh tallies under delta tracking.**~~
  **Done 2026-05-08.** `transport_particle_delta` now deposits
  mesh flux per Woodcock segment (Amanatides-Woo voxel walker,
  same `MeshFluxTally::deposit` helper as surface tracking) and
  tallies surface currents on the first boundary the segment
  hits — exact for vacuum and reflective BCs (segment ends at
  the surface) and pragmatic for transmission BCs (subsequent
  silently-crossed surfaces in the same Woodcock step are
  skipped, the standard limitation under delta tracking). New
  unit tests `delta_tracking_mesh_flux_populates` and
  `delta_tracking_surface_currents_populate` confirm both
  helpers are wired. 322/322 lib tests green.

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
- ~~**EADL relaxation cascade on GPU.**~~ **Done 2026-05-08.**
  New `relaxation_cascade_batch` CUDA kernel mirrors
  `photoelectric_absorb` byte-for-byte: per-thread fixed-size
  hole stack (16 entries — typical cascade depth is 2-6) and
  per-thread fluorescence buffer (8 entries, configurable via
  `DEFAULT_GPU_MAX_FLUOR_PER_THREAD`). Transition tables
  flattened into `trans_off[n_shells]` /
  `trans_count[n_shells]` / `trans_flat[total_rows × 4]` and
  uploaded once per element alongside the existing phase-1 PE
  data. `GpuPhotoelectricCtx::cascade_batch` /
  `full_cascade_batch` expose the kernel from Rust.

  Validation in `gpu_compton_validate` (CPU vs GPU at N = 1 M
  histories, RTX A1000): all 8 cases pass. Means agree < 0.01 %
  across H / O / Zr / U at 0.1 / 1.0 MeV. Mean fluorescence
  multiplicity matches to 3 decimal places (Zr: 0.67 → 0.68
  CPU/GPU; U at 1 MeV: 1.24 → 1.24); fluorescence-energy
  histogram χ²_red ≤ 1.6 in all heavy-element cases (light
  elements have no above-cutoff fluorescence and the histogram
  is degenerate). 322/322 lib tests green on default and
  `--features cuda`. `outputs/gpu_photon_validate.txt` now
  records cascade rows alongside the existing PE-phase1 ones.
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
