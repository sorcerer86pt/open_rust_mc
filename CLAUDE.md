# open_rust_mc — Claude project memory

## What this is

A pure-Rust continuous-energy Monte Carlo radiation-transport engine.
Started as an SVD cross-section-compression experiment; now also a
coupled neutron-photon code with depletion (CRAM-16 / CRAM-48),
variance reduction (weight windows + random-ray FW-CADIS),
recursive-universe CSG geometry on CPU + CUDA, and a Python (PyO3)
front-end. Reads OpenMC HDF5 nuclear data directly via `hdf5-pure` —
no C dependency.

`origin/main` is at `d80a157` as of this writing. Lib tests
**438 / 438** green on default features, **443 / 443** with
`--features cuda`.

## How to read result numbers

Every quoted result carries a scope tag:

- `[micro]` — isolated kernel / one-nuclide / one-reaction.
  Optimistic; does NOT generalise to whole-engine k_eff.
- `[godiva]` — end-to-end Godiva HEU sphere, 3 nuclides, fast spectrum.
- `[pwr]` — end-to-end PWR pin cell, 9 nuclides, thermal, S(α,β) on.
- `[assembly]` — 17×17 PWR depth-3 recursive geometry.
- `[hex]` — HexLattice mini-core (1- or 2-ring).
- `[shield]` — photon shielding slab.
- `[photon]` — γ-heating / pulse-height / coupled n-γ.
- `[depletion]` — CRAM transmutation, fresh-corrector predictor.
- `[icsbep]` — multi-case ICSBEP regression substrate.
- `[projected]` — analytical / extrapolated. Hypothesis until a
  scoped row replaces it.

A number quoted without a scope tag is a bug. Repeated pattern:
`[micro]` headlines shrink or invert sign under `[pwr]` / `[assembly]`.

## Test acceptance envelope (ICSBEP regressions)

`tests/cuda_runs.rs` and `tests/icsbep_runs.rs` share one envelope:

```
|Δk_pcm| ≤ max(150 pcm, 2 × σ_combined)
σ_combined = sqrt(σ_calc² + σ_exp²)
```

Multi-seed averaging (3 seeds default) drives `σ_calc` from
seed-to-seed stderr — single-seed within-batch stderr underestimates
GPU atomic-ordering nondeterminism. The 150 pcm floor catches small
systematic biases that wide `σ_exp` would otherwise swallow (HEU-SOL-THERM
`σ_exp` = 600 pcm); the 2σ clause stays honest when `σ_exp` is tight
(Godiva `σ_exp` = 100 pcm).

The optional `local_validation` block on a scene JSON points at an
OpenMC k_eff measured on the *same* JSON — used as a secondary target
to grade engine quality apart from any scene-transcription drift from
the registered ICSBEP handbook.

## Hard invariants (don't break these)

- **`RectLattice::local_position` is element-CENTRE-relative**
  (OpenMC convention). Lattice tests place pin surfaces at universe-
  local origin `(0, 0)`, NOT `(pitch/2, pitch/2)`. Same on the GPU.
  Fixing this unblocked LCT-008.

- **`MAX_NUCLIDES_PER_MATERIAL = 128`** at `src/lib.rs` is the single
  source of truth. CPU imports it (`simulate.rs::MAX_NUCLIDES`); GPU
  receives it via NVRTC `-DMAX_NUC_PER_MAT=N` from
  `gpu_recursive.rs::assemble_kernel_source` and
  `gpu_transport.rs::transport_kernel_options`. `transport.cu` has no
  `#define` fallback (`#error`s if the host forgets the flag). Bumping
  is a one-line change followed by a full rebuild + sm_86 register-
  pressure recheck.

- **`SimLimits` (`src/transport/sim_limits.rs`)** is engine policy
  separated from per-run intent (`SimConfig`). Carries
  `max_events_per_history`, `fis_capacity_factor`,
  `sab_temperature_tolerance`, `initial_source_max_attempts_factor`.
  `SimLimits::default()` reproduces the historical magic literals
  bit-for-bit. Long-shielding harnesses can override via
  `SimLimits::from_toml_file(path)`.

- **Initial-source sampler is material-aware, not cell-order-aware.**
  `simulate::try_initial_source_in_materials` walks every cell's
  region tree via `Region::world_aabb(surfaces)`, filters by
  `ResolvedMaterials::fissionable_materials()` (any nuclide with
  `nu_bar_const > 0`), and rejection-samples weighted by per-cell AABB
  volume. Matches Serpent 2's default; replaces the old "first
  Material cell" / "smallest-volume material" heuristics that failed
  on BWR cruciforms, PWR burnable poisons, HFIR plates, CANDU
  spacers, multi-shell HMF.

- **`N_PARAMS = 136`** on `transport.cu` / `gpu_transport.rs`. Slots
  123-135 carry the multi-slot S(α,β) tables (slot 129 = `P_SAB_EMAX`)
  and the Maxwell/Evaporation closed-form χ buffers that fixed
  U-233 / U-234 / Pu-240 fission outgoing-energy spectra. Adding a
  new param slot: update both files atomically.

- **Per-level SVD basis must be uploaded at the global `P_RANK`
  stride.** Each discrete-inelastic level kernel has its own
  `level_rank = min(svd_rank, svd.rank)`. The device kernel has no
  per-level rank slot — it reads `basis[e_idx × P_RANK + j]` for the
  full range. Pad each level's basis to `[n_e × global_rank]` with
  zero columns for `j ∈ [level_rank, global_rank)` and pad coeffs to
  length `global_rank` with zeros. Skipping this silently reads
  adjacent levels' bytes and returns ~10⁰ or ~10⁻⁹⁰ XS values.

- **GPU recursive kernel pinned to sm_86** (Ampere / RTX A1000+) for
  `atomicAdd(double*, double)`. NVRTC arch is hardcoded in
  `gpu_recursive.rs`.

- **CPU transport uses `TransportCtx` worker-local sinks +
  rayon `fold().reduce()`** (not `par_iter().map().collect()`). Plus
  `CollisionOutcome::Fission/Multiplicity` use `SmallVec` (typedefs
  `FissionSites`, `SecondaryList`). Eliminates ~6 MB / batch of
  per-event Vec churn on PWR and pushes ICSBEP-suite wall time down.

- **ENDF/B-VIII.x layout split.** VIII.0 / VIII.1 moved S(α,β) files
  into a sibling `thermal/` directory next to `neutron/`. The
  `data_paths::resolve_thermal_path` helper handles this transparently
  — call sites pass the `neutron/` directory and the resolver finds
  the file in `../thermal/` if the same-dir lookup misses. All TSL
  load sites in binaries and Python bindings route through this.

- **Natural-element ZAIDs (`zaid % 1000 == 0`) auto-split when the
  natural file is missing.** ENDF/B-VIII.x dropped `C0.h5` (and only
  that — `*0.h5` is unique to carbon in our supported libraries).
  `material_resolve::expand_natural_elements` mirrors OpenMC's
  `Material.add_element` behaviour at resolve time: if the natural
  file isn't on disk and the abundance table knows the element, split
  on the fly into isotopic siblings. The on-disk JSON corpus is
  migrated to per-isotope form via `scripts/migrate_natural_elements.py`
  (137 cases / 291 entries rewritten); the engine fallback covers any
  future un-migrated JSON.

## File layout

```
rust_prototype/src/
  lib.rs / main.rs                — crate roots
  data_paths.rs                   — library layout-aware path helpers
                                    (resolve_thermal_path, discover_*)
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

  physics/                        — collision / scatter / kinematics
  transport/                      — simulate, dispatch, rng, materials,
                                    thermal_library, weight_window,
                                    tally, statepoint, kinetics,
                                    adjoint_neutron, adjoint_photon
  photon/                         — Compton, Rayleigh, photoelectric,
                                    pair, brems, electron, transport
  random_ray/                     — multigroup TRRM (Tramm 2018,
                                    immortal Tramm-Siegel 2021),
                                    forward + adjoint + CADIS
  depletion/                      — cram, chain, predictor-corrector,
                                    mapping, flux
  bin/                            — see "Binaries" below

rust_prototype/tests/             — integration tests (438 lib +
                                    integration; cuda_runs gated)
rust_prototype/bindings/python/   — PyO3 wrapper (open_rust_mc package)
cuda/, cuda_bench/                — NVRTC-compiled .cu source
paper/                            — TeX manuscript (SVD paper)
scripts/                          — analysis pipeline + nuclear-data
                                    setup + JSON migration
chains/                           — depletion chain JSONs
bench/icsbep/                     — 375 ICSBEP scene JSONs
data/                             — ENDF HDF5 libraries (gitignored)
outputs/                          — bench CSV/log (gitignored; -f to
                                    commit specific results)
```

## Binaries

Neutron k-eigenvalue:
- `godiva` — Godiva HEU sphere, 3 nuclides.
- `pwr_pincell` — PWR pin cell, 9 nuclides + S(α,β).
- `pwr_d2o_pincell` — heavy-water variant.
- `pwr_assembly` — 17×17 (use `--shape N` for 3×3 / 5×5 / 7×7).
- `hex_minicore` — N-ring hex array with hex reflective boundary.
- `validate_vs_openmc` — bit-exact validation.
- `xs_dump` / `xs_dump_godiva` / `xs_provider_diff` / `debug_trace` /
  `photon_dump` — diagnostics.
- `metal_stats_diag` / `nu_lookup_compare` / `level_xs_compare` /
  `elastic_kinematics_diag` / `chi_compare` / `debug_lct` /
  `icsbep_alloc_bench` — localisation diagnostics.
- `icsbep_bench` — Rust-only ICSBEP harness (no Python in the call
  graph, profile the engine cleanly).
- `preview_scene` — XY cross-section viewer for any scene JSON.
  Interactive window (`--features preview`, default) supports
  cursor-centered scroll-zoom and right-click drag-pan. Headless PNG
  / PPM fallbacks via `--png-out` / `--ppm-out`.

ICSBEP harness (Python, via `bindings/python/examples/`):
- `icsbep_run.py <case> {cpu|gpu}` — single-case run with auto
  data-dir discovery (VIII.1 → VIII.0 → VII.1).
- `icsbep_sweep.py` — full corpus sweep with start / stop / resume,
  per-case CSV durability, multi-seed averaging. Precedence:
  **explicit CLI flag > JSON `recommended_settings` > built-in default**.
- `run_benchmark.ps1` — one-shot wrapper. Picks GPU runner when the
  CUDA extension is loadable. Writes
  `outputs/icsbep_full_<runner>.csv` + matching `.log`.

Photon / shielding / coupled:
- `pwr_gamma_heating` — PWR γ-heating with full ET + brems.
- `cs137_pulse_height` — pulse-height validation.
- `shield_slab` — fixed-source γ slab + WW consumer.
- `adjoint_photon_cadis_slab` — CE adjoint photon walker.

Random-ray:
- `rr_pincell` — 2-group UO₂ + water pin cell (forward + adjoint).
- `rr_cadis_slab` — slab adjoint → CADIS JSON for `shield_slab`.

Depletion:
- `deplete_demo` — constant-flux Xe equilibrium.
- `deplete_pwr` — transport-coupled fresh-corrector.

GPU (`--features cuda`):
- `gpu_bench` — SVD recon kernel sweep.
- `gpu_cpu_bench` — CPU/GPU head-to-head.
- `gpu_recursive_keff` — recursive transport k-eigenvalue.
- `gpu_const_xs_keff` — constant-XS GPU eigenvalue.
- `gpu_assembly_keff` — full assembly on GPU.
- `gpu_pwr_bench` — PWR pin cell on GPU.
- `gpu_hex_minicore` — hex on GPU.
- `gpu_compton_validate` / `gpu_compton_scaling` /
  `gpu_photon_features` / `gpu_wmp_validate` — photon GPU validation.

Kinetics:
- `point_kinetics_demo` — point-kinetics ODE driver.

## Build & run (Windows / PowerShell — primary dev env)

```powershell
cd rust_prototype
cargo build --release                       # CPU only
cargo build --release --features cuda       # + CUDA backend
cargo test --lib                            # 438 / 438 (default)
cargo test --lib --features cuda            # 443 / 443

# Python extension (needs maturin)
cd bindings/python
maturin develop --release                   # CPU only
maturin develop --release --features cuda   # adds Runner.GpuCuda
cd ../../..

# Nuclear data — ENDF/B-VIII.1 by default
.\scripts\setup_nuclear_data.ps1            # ~6.5 GB VIII.1
.\scripts\setup_nuclear_data.ps1 -All       # all four supported libs
.\scripts\setup_nuclear_data.ps1 -Vii1      # legacy VII.1 only

# Single ICSBEP case
python rust_prototype/bindings/python/examples/icsbep_run.py heu-met-fast-001_case-1 gpu

# Full corpus sweep
.\rust_prototype\bindings\python\examples\run_benchmark.ps1

# Override JSON recommended_settings from CLI (precedence: CLI > JSON)
python rust_prototype/bindings/python/examples/icsbep_sweep.py `
    --runner cpu --filter heu-comp-inter-003 `
    --particles 100000 --batches 150 --inactive 40 --seeds 5 `
    --csv outputs/local.csv --stop-file outputs/STOP

# Godiva, real ENDF data
cargo run --release --bin godiva -- data\endfb-viii.1-hdf5\neutron `
  --rank 5 --batches 80 --inactive 15 --particles 10000
```

## Common operations

### Add a new natural-element fallback

`scripts/migrate_natural_elements.py` and
`src/transport/nuclides.rs::natural_isotopic_split` share a table.
Add the new element to both:

```python
# scripts/migrate_natural_elements.py
NATURAL_ABUNDANCES = {
    6: {12: 0.9893, 13: 0.0107},
    # add new Z here
}
```

```rust
// src/transport/nuclides.rs
pub fn natural_isotopic_split(zaid: u32) -> Option<&'static [(u32, f64)]> {
    match zaid {
        6000 => Some(&[(12, 0.9893), (13, 0.0107)]),
        // add new ZAID here
        _ => None,
    }
}
```

Then re-run the script to migrate any new JSONs.

### Bump `MAX_NUCLIDES_PER_MATERIAL`

Edit `src/lib.rs`, full rebuild including the Python wheel
(`maturin develop --release --features cuda`), and confirm GPU
register pressure stayed within sm_86 budget (`nuc_t[N]` is the
hot-path stack-allocated array; ~128 bytes per slot at f64).

### Re-run a failing ICSBEP case

```powershell
python rust_prototype/bindings/python/examples/icsbep_run.py <case-stem> gpu
```

The case stem is the JSON filename without extension. CPU/GPU
chosen via second positional arg.

### Diagnose a CPU vs GPU mismatch

Cascade from cheap to expensive:
1. `level_xs_compare --nuclide U235 --awr 233.025` — per-level SVD XS
   bit-A/B at six probe energies.
2. `nu_lookup_compare` — ν̄(E) CPU-vs-GPU port.
3. `metal_stats_diag <case>` — three-way CPU / GPU / OpenMC integrated
   tallies + per-reaction breakdown.
4. `chi_compare` — fission outgoing-energy spectrum check.

## Open / deferred

- **GPU device-buffer cache for SAB + material payloads.** Per-nuclide
  kernels are cached (`Arc::as_ptr`-keyed). `upload_sab_data_multi`
  and `upload_material_data` rebuild flat arrays + HtoD copy on every
  seed/case. Across a 5-seed × 7-case sweep that's 35 redundant
  ~50 MB SAB uploads. Same LRU + bundle_cache_budget pattern as
  `per_nuclide_cache` should apply.
- **GPU survival biasing / Russian roulette.** CPU has it (4.5× FOM
  on PWR); GPU runs analog. Variance-only — k_eff is unbiased.
- **GPU per-cell `Mat3` rotation.** `GpuRecursiveContext::build` now
  errors loudly if any cell has `rotation = Some(...)` rather than
  silently mis-finding. No ICSBEP scene currently sets `rotation`.
- **GPU discrete S(α,β) inelastic (NJOY iwt=0/1).** CPU has it; GPU
  device sampler is continuous-only. OpenMC's ENDF/B-VII.1 HDF5 ships
  every TSL as `incoherent_inelastic` (continuous), so zero hits in
  the 157-case sweep. `upload_sab_data_multi` errors if the host
  hands it `InelasticDist::Discrete`.
- **DXTRAN-style continuous splitting** for ≥14 mfp photon
  penetration. All `(ratio, growth) ∈ {5,10,20} × {0,1,2,3}` at
  300 cm give 0 transmitted in 500k — `max_split=8` ceiling bounds
  geometric WW.
- **Full C5G7** (4 fuel × 7 groups × 17×17) — data plumbing on top
  of `random_ray::*`, no new solver code.
- **HexLattice GPU runtime parity** vs CPU (large-volume sweep
  equivalent to `hex_lattice_descent_and_trace_smoke`).
- **Per-precursor delayed-neutron groups** — only matters for
  time-dependent kinetics, not static k-eff.
- **EADL relaxation cascade on GPU** — long-flagged.
- **Source-distribution biasing** (sample initial pos from
  importance CDF) — for the Wagner-Haghighat 50-1000× FOM on
  volume / angular sources. For `shield_slab`'s monodirectional
  point beam the importance CDF degenerates.

## Recent session highlights (2026-05-21)

1. **ENDF/B-VIII.1 is now the default library.** `setup_nuclear_data.ps1`
   downloads VIII.1 by default; legacy VII.1 via `-Vii1`. New
   `data_paths` module handles the sibling-`thermal/` layout VIII.x
   uses. 5 new layout tests.
2. **Natural-element handling.** `scripts/migrate_natural_elements.py`
   rewrote 137 JSONs (291 natural-carbon entries split into C-12 +
   C-13 by IUPAC 2021 abundance). Engine-side fallback in
   `material_resolve::expand_natural_elements` covers any future
   un-migrated JSON. 4 new tests.
3. **Sweep precedence inverted.** Explicit CLI flag > JSON
   `recommended_settings` > built-in. Banner shows which knob came
   from where. CPU runs print a note that GPU refill keys are
   ignored (rayon work-stealing already saturates cores).
4. **HEU-COMP-INTER-003 VIII.1 vs VII.1 A/B (CPU, 100k, 7 cases).**
   VIII.1 shifts k upward by +762 to +1035 pcm uniformly. Driver
   localised to CIELO U-235: σ_f at 100 eV grew +5.3%, σ_capture
   dropped −5.5%, so the capture/fission α dropped −10.2% at the
   intermediate-spectrum peak. ν̄(E) essentially unchanged. Cases 2-3
   now FAIL (handbook), cases 5-7 cross zero into PASS.
5. **Library cache infrastructure verified.** Zero "Loading X.h5"
   lines across the 7-case CPU sweep after preload — 19 nuclides
   loaded once and reused (was 125× redundant on the prior baseline
   per the `material_resolve.rs` cache comment).
