# open_rust_mc — Claude project memory

## What this is

A pure-Rust continuous-energy Monte Carlo radiation transport engine.
- Neutron k-eigenvalue and fixed-source, photon transport, coupled
  neutron-photon γ-heating, depletion (CRAM), time-dependent point
  kinetics, continuous-energy adjoint MC.
- CPU backend (rayon, history-based) and CUDA backend (sm_86+,
  event-batched + recursive-CSG), behind a shared `EigenvalueRunner`
  trait.
- Four interchangeable cross-section providers: pointwise `Table`,
  truncated `Svd`, `HybridSvdWmp`, `HybridTableWmp`.
- Reads OpenMC HDF5 nuclear data directly (`hdf5-pure` — no C
  dependency).
- Python (PyO3) bindings as `open_rust_mc` package.

Tests on `origin/main`:
- `cargo test --lib`: **438 / 438 green**.
- `cargo test --lib --features cuda`: **443 / 443 green**.

## How to read results

Numbers in commit messages, docs, and code comments come with a
scope tag. Treating them without the tag is a bug pattern.

- `[micro]` — isolated kernel / one nuclide / one reaction.
  Optimistic and does NOT generalise to k_eff.
- `[godiva]` — end-to-end 3-nuclide HEU sphere, fast spectrum.
- `[pwr]` — end-to-end 9-nuclide PWR pin cell, thermal, S(α,β) on.
- `[assembly]` — depth-3 recursive 17×17 PWR lattice.
- `[hex]` — N-ring hex mini-core.
- `[shield]` — fixed-source photon slab.
- `[photon]` — γ-heating / pulse-height / coupled n-γ.
- `[depletion]` — CRAM transmutation.
- `[icsbep]` — multi-case ICSBEP regression substrate.
- `[projected]` — analytical or extrapolated; hypothesis until a
  scoped row replaces it.

Recurring pattern: `[micro]` headlines shrink or invert under
`[pwr]` / `[assembly]`.

## Build & run

Primary dev env is Windows / PowerShell.

```powershell
cd rust_prototype
cargo build --release                       # CPU only
cargo build --release --features cuda       # + CUDA backend
cargo test --lib                            # 438 / 438
cargo test --lib --features cuda            # 443 / 443

# Python extension
cd bindings/python
maturin develop --release                   # CPU only
maturin develop --release --features cuda   # adds Runner.GpuCuda
cd ../../..

# Nuclear data (ENDF/B-VIII.1 is the default)
.\scripts\setup_nuclear_data.ps1            # VIII.1 only, ~6.5 GB
.\scripts\setup_nuclear_data.ps1 -All       # VIII.1 + VIII.0 + VII.1 + JEFF-3.3
.\scripts\setup_nuclear_data.ps1 -Vii1      # legacy VII.1 only

# Single ICSBEP case (auto-discovers VIII.1 → VIII.0 → VII.1)
python rust_prototype/bindings/python/examples/icsbep_run.py heu-met-fast-001_case-1 gpu

# Full corpus sweep
.\rust_prototype\bindings\python\examples\run_benchmark.ps1

# Scoped sweep with explicit CLI overrides (CLI > JSON > built-in)
python rust_prototype/bindings/python/examples/icsbep_sweep.py `
    --runner cpu --filter heu-comp-inter-003 `
    --particles 100000 --batches 150 --inactive 40 --seeds 5 `
    --csv outputs/scoped.csv --stop-file outputs/STOP
```

## Repository layout

```
rust_prototype/                 — main crate (workspace root)
  Cargo.toml                    — features: default = []; cuda; preview
  src/
    lib.rs                      — crate root, MAX_NUCLIDES_PER_MATERIAL
    data_paths.rs               — ENDF library-layout-aware path helpers
    kernel.rs                   — SVD reconstruction (rank-k FMA)
    decompose.rs cp_decompose.rs — SVD decomposition
    hdf5_reader.rs              — pure-Rust HDF5 + thermal loader
    thermal.rs                  — S(α,β) sampling
    table.rs wmp.rs             — pointwise + Windowed Multipole providers
    nuclide.rs loader.rs        — nuclide loading
    geometry/                   — recursive universes, BVH, hex/rect
    physics/                    — collision / scatter / kinematics
    transport/                  — simulate, dispatch, material_resolve,
                                  xs_provider, hybrid_xs, sim_limits,
                                  nuclides, thermal_library,
                                  urr_equivalence, weight_window,
                                  tally, statepoint, kinetics,
                                  adjoint_neutron, adjoint_photon
    photon/                     — Compton, Rayleigh, photoelectric,
                                  pair, brems, electron, transport, nee
    random_ray/                 — multigroup TRRM (forward + adjoint +
                                  immortal), CADIS, adjoint-SVD
    depletion/                  — cram, chain, predictor-corrector,
                                  mapping, flux
    gpu.rs                      — CUDA host wrappers (root)
    gpu_transport.rs            — `GpuTransportContext`, N_PARAMS=186
    gpu_recursive.rs            — recursive geometry on device
    gpu_random_ray.rs           — random-ray persistent kernel host
    gpu_per_nuclide.rs          — per-nuclide upload + LRU cache
    bin/                        — 40 binaries (see below)
  tests/                        — 9 integration tests
  bindings/python/              — PyO3 wrapper, examples/, run_benchmark.ps1
  gpu/cuda/                     — NVRTC-compiled .cu sources
                                  (transport.cu, transport_recursive.cu,
                                   geom_recursive.cu, ...)

bench/icsbep/                   — 375 scene JSONs
chains/                         — depletion chain JSONs
data/                           — ENDF/B HDF5 libraries (gitignored)
outputs/                        — bench CSV/log (gitignored)
paper/                          — SVD compression paper (TeX + PDF)
scripts/                        — setup_nuclear_data.ps1,
                                  migrate_natural_elements.py,
                                  analysis pipelines

CLAUDE.md STATUS.md README.md   — top-level docs
PYTHON.md BENCHMARKS.md ICSBEP.md — domain docs
```

## Hard invariants (verified 2026-05-21)

These are the ones currently true on `origin/main`. Re-verify before
changing.

- **`MAX_NUCLIDES_PER_MATERIAL = 128`** at `src/lib.rs:19`. Single
  source of truth. CPU imports it as `simulate::MAX_NUCLIDES`;
  GPU receives it via NVRTC `-DMAX_NUC_PER_MAT=N`. Bumping is one
  line + full rebuild + GPU register-pressure recheck.

- **GPU arch adapts to the detected device.** `gpu_recursive::device_nvrtc_arch(ctx)`
  queries CC_MAJOR / CC_MINOR via cudarc device attributes and builds
  the `sm_{major}{minor}` string passed to NVRTC. Both compile sites
  (`gpu_recursive.rs::new()` and `gpu_transport.rs::new()`) route
  through the helper. Minimum is CC 6.0 / sm_60 (Pascal — needed for
  `atomicAdd(double*, double)`); the helper returns an explicit error
  below that. Same binary now targets A1000 (sm_86), A100 (sm_80),
  H100 (sm_90), RTX 5090 (sm_120), etc. without rebuild.

- **`N_PARAMS = 186`** matches between `gpu_transport.rs:18` and
  `gpu/cuda/transport.cu:383`. Adding or removing a slot requires
  touching both atomically. Earlier inline-vec packing sites that
  open-coded `vec![dptr!(...)…]` are now delegated to
  `build_transport_params_vec` (see comment at `gpu_transport.rs:2523`)
  — don't open-code new sites.

- **`SimLimits` (`src/transport/sim_limits.rs`)** is engine policy
  separate from per-run intent (`SimConfig`). `default()` reproduces
  the historical hardcoded values; long-shielding harnesses override
  via `SimLimits::from_toml_file`. No magic literals at construction
  sites.

- **`data_paths::resolve_thermal_path(neutron_dir, name)`** is the
  single layout-aware TSL path resolver. Tries `neutron_dir/name`
  first (VII.1 layout where TSL files mix into neutron/), then
  `neutron_dir/../thermal/name` (VIII.0/VIII.1 layout where they're
  in a sibling dir). All TSL load sites in binaries and Python
  bindings route through this. New TSL load sites must do the same.

- **`data_paths::discover_neutron_dir(workspace_root)`** probes
  libraries in priority order `endfb-viii.1-hdf5` → `endfb-viii.0-hdf5`
  → `endfb-vii.1-hdf5`. Test helpers in `tests/cuda_runs.rs`,
  `tests/cache_roundtrip.rs`, `tests/icsbep_runs.rs`, and 6 diag
  binaries call this — they used to hardcode VII.1.

- **Natural-element ZAID fallback.** When a JSON references
  `zaid: <Z>000` and the natural file isn't on disk (e.g. VIII.x has
  C-12 + C-13 but no C0.h5), `material_resolve::expand_natural_elements`
  splits the entry into isotopic siblings using
  `nuclides::natural_isotopic_split(zaid)` (currently carbon-only).
  Mirrors OpenMC's `Material.add_element` at resolve time. Zero
  overhead on VII.1 (early-out via `lib.has_natural_file(zaid)`).
  Abundance table lives in two places that must stay in sync:
  `scripts/migrate_natural_elements.py::NATURAL_ABUNDANCES` and
  `src/transport/nuclides.rs::natural_isotopic_split`.

- **Sweep precedence (Python harness)**: explicit CLI flag wins
  over JSON `recommended_settings`, which wins over built-in
  default. `icsbep_sweep.py`'s `_pick` helper drives this. Argparse
  defaults are `None` sentinels for "not passed" so the resolver
  can tell. Banner prints `(explicit)` vs `auto (JSON or default)`
  per knob.

## Test acceptance envelope

`tests/cuda_runs.rs` and `tests/icsbep_runs.rs` share one envelope
for ICSBEP regressions:

```
|Δk_pcm| ≤ max(150 pcm, 2 × σ_combined)
σ_combined = sqrt(σ_calc² + σ_exp²)
```

`σ_calc` from multi-seed averaging (3 seeds default) — single-seed
within-batch stderr underestimates GPU atomic-ordering nondeterminism.
The 150 pcm floor catches systematic biases that wide `σ_exp` (e.g.
HEU-SOL-THERM at 600 pcm) would otherwise swallow. The 2σ clause
stays honest when `σ_exp` is tight (Godiva σ_exp = 100 pcm).

Scenes may carry an optional `local_validation` block: an OpenMC
k_eff measured on the same JSON the engine consumes. Used as a
secondary target to grade engine quality apart from any scene-
transcription drift from the registered ICSBEP handbook.

## Binaries (40 in `src/bin/`)

Neutron k-eigenvalue: `godiva`, `pwr_pincell`, `pwr_d2o_pincell`,
`pwr_assembly`, `hex_minicore`, `validate_vs_openmc`.

Diagnostics: `metal_stats_diag` (three-way CPU / GPU / OpenMC
tally compare), `nu_lookup_compare`, `level_xs_compare`,
`elastic_kinematics_diag`, `chi_compare`, `debug_lct`,
`debug_trace`, `xs_dump`, `xs_dump_godiva`, `xs_provider_diff`,
`photon_dump`, `icsbep_alloc_bench`.

ICSBEP harness (Rust): `icsbep_bench` — runs cases without going
through Python, useful for clean engine profiles.

Scene viewer: `preview_scene` — interactive XY cross-section
viewer (`--features preview`, default). Headless PNG / PPM
fallbacks via `--png-out` / `--ppm-out`.

Photon / shielding: `pwr_gamma_heating`, `cs137_pulse_height`,
`shield_slab`, `adjoint_photon_cadis_slab`.

Random-ray: `rr_pincell`, `rr_cadis_slab`, `rr_adjoint_svd`.

Depletion: `deplete_demo`, `deplete_pwr`.

GPU (`--features cuda`): `gpu_bench`, `gpu_recursive_keff`,
`gpu_const_xs_keff`, `gpu_assembly_keff`, `gpu_pwr_bench`,
`gpu_hex_minicore`, `gpu_recursive_parity`, `gpu_photon_features`,
`gpu_wmp_validate`.

Kinetics: `point_kinetics_demo`.

Misc: `nuclide_cache_server` (persistent host nuclide cache).

## Python harness

`rust_prototype/bindings/python/examples/`:

- `icsbep_run.py <case> {cpu|gpu}` — single case, auto data-dir
  discovery (`_find_data_dir`).
- `icsbep_sweep.py` — full corpus or filter-scoped sweep with
  start / stop / resume, per-case CSV durability (`fp.flush()`
  after each row), multi-seed averaging, graceful termination via
  `outputs/STOP` marker file or Ctrl-C.
- `run_benchmark.ps1` — one-shot wrapper. Picks GPU runner when
  `open_rust_mc.Runner.recommended()` returns `gpu_cuda`. Writes
  `outputs/icsbep_full_<runner>.csv` + matching `.log`.

## Common operations

### Add a new natural-element fallback

Two places to update in lockstep:

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

Then run `python scripts/migrate_natural_elements.py` to rewrite
any existing JSONs (idempotent — a second run is a no-op).

### Re-run a single failing ICSBEP case

```powershell
python rust_prototype/bindings/python/examples/icsbep_run.py <case-stem> {cpu|gpu}
```

Case stem is the JSON filename without `.json`. Second positional
arg picks the backend.

### Diagnose a CPU vs GPU mismatch

Cascade cheap → expensive:

1. `level_xs_compare --nuclide U235 --awr 233.025` — per-level SVD XS
   bit-A/B at six probe energies. Fastest sanity check.
2. `nu_lookup_compare` — ν̄(E) CPU vs GPU port. Bit-identical or bug.
3. `metal_stats_diag <case>` — three-way CPU / GPU / OpenMC
   integrated tallies + per-reaction breakdown + ⟨E_in/E_out⟩
   moments.
4. `chi_compare` — fission outgoing-energy spectrum probe.

### Bump `MAX_NUCLIDES_PER_MATERIAL`

Edit `src/lib.rs:19`, full release rebuild including the Python
wheel (`maturin develop --release --features cuda`), confirm GPU
register pressure stayed in budget on sm_86.

### Re-run a sweep on a different machine

The CSV format is hardware-agnostic. `outputs/` is gitignored;
force-add specific files to record results:

```powershell
git add -f outputs/<sweep-name>.csv outputs/<sweep-name>.log
git commit -m "bench: <sweep-name> (<machine>, <date>)"
```

## Recent session highlights (2026-05-21)

Four commits on top of `547dfcb`:

1. `ac282e6` **data: ENDF/B-VIII.1 default + natural-element migration**.
   New `data_paths` module with layout-aware resolver + multi-library
   discovery. `migrate_natural_elements.py` rewrote 137 ICSBEP JSONs
   (291 natural-C entries split into C-12 + C-13 by IUPAC 2021).
   Engine fallback `expand_natural_elements` for un-migrated JSONs.
   `setup_nuclear_data.ps1` defaults to VIII.1. Catalog gained C-12 /
   C-13 entries. 9 new tests.

2. `7994424` **sweep: explicit CLI flag beats JSON
   `recommended_settings`**. Previous precedence was inverted, so
   `--particles 20000` lost to a 500k JSON recommendation. Now CLI >
   JSON > built-in via `None`-sentinel argparse defaults + `_pick`
   helper. Banner shows where each knob's value came from.

3. `d80a157` **bench: heu-comp-inter-003 VIII.1 CPU A/B**. 7 cases
   on this A1000-laptop, 100k particles × 5 seeds × 150 batches /
   40 inactive. VIII.1 shifts k uniformly upward by +820 ± 153 pcm
   vs VII.1 baseline. Localised to **CIELO U-235**: σ_f at 100 eV
   grew +5.3 %, σ_capture dropped −5.5 %, α (capture/fission)
   dropped −10.2 % at the intermediate-spectrum peak. ν̄(E)
   essentially unchanged. Cases 2-3 cross VIII.1 envelope → FAIL;
   cases 5-7 had VII.1 k below 1.000 → +820 pcm pulls them into
   PASS.

4. `3740ae2` **docs: rewrite CLAUDE.md / STATUS.md / README.md;
   drop resume.md**. Top-level docs were drifting; wholesale
   rewrite to reflect current state. Total docs shrink 2469 →
   1043 lines.

## Open / deferred

- **GPU device-buffer cache for SAB + material payloads.**
  `upload_nuclide_data` is cached (per-nuclide LRU keyed on
  `Arc::as_ptr` + rank). `upload_sab_data_multi` and
  `upload_material_data` rebuild flat arrays + HtoD every seed/case.
  A 5-seed × 7-case sweep = 35× redundant ~50 MB SAB uploads.
- **GPU survival biasing / Russian roulette.** CPU has it (4.5×
  FOM on PWR); GPU runs analog. Variance-only, k_eff stays
  unbiased.
- **GPU discrete S(α,β) inelastic (NJOY iwt=0/1).** Continuous
  only on device. OpenMC's HDF5 emits every TSL as
  `incoherent_inelastic` — zero hits in the 375-case corpus.
  `upload_sab_data_multi` panics if the host hands it
  `InelasticDist::Discrete` (silent breakage impossible).
- **GPU per-cell `Mat3` rotation.** `GpuRecursiveContext::build`
  errors loudly if any cell sets `rotation = Some(...)`. No ICSBEP
  scene currently uses it.
- **DXTRAN-style continuous photon splitting** at ≥14 mfp.
  Geometric WW (max_split=8) bounds it.
- **Full C5G7** (4 fuel × 7 groups × 17×17) — data plumbing on
  top of `random_ray::*`; no new solver code.
- **HexLattice GPU runtime parity vs CPU** — full sweep
  equivalent to `hex_lattice_descent_and_trace_smoke`.
- **Full PWR depletion bench vs OpenMC** (30-50 GWd/MTU).
  Chain-calibration fix landed (`fd530d0`); the long-burn
  validation run itself is pending.
- **Backfill the rest of the 137 migrated ICSBEP JSONs** with
  VIII.1 A/B numbers once the GPU device-buffer cache lands and
  the 3080 box is available.
