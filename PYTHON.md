# Python API

A thin PyO3 binding over the Rust engine, designed so Python scripts
can build Monte Carlo neutron-transport scenes (and now coupled
neutron-photon γ-heating runs) without touching Rust. **Rust is the
source of truth** — the builder, region validation, and simulation
core all live on the Rust side; the Python layer is ~200 lines of
class/function exports.

The binding crate is at
[`rust_prototype/bindings/python/`](rust_prototype/bindings/python/);
sample scripts live in
[`rust_prototype/bindings/python/examples/`](rust_prototype/bindings/python/examples/).

What's exposed today (v0.4.0):

- All four cross-section providers via the `XsMode` enum
  (`Table`, `Svd`, `HybridTableWmp`, `HybridSvdWmp`).
- Per-MT SVD rank overrides via `Scene.set_svd_ranks({mt: rank})`.
- S(α,β) thermal scattering attached per nuclide
  (`Material.add_nuclide(..., thermal_file="c_H_in_H2O.h5")`).
- Coupled neutron-photon γ-heating (`run_gamma_heating`) with
  full electron transport (CSDA + Highland MS + Bethe-Bloch
  + Seltzer-Berger bremsstrahlung) on by default.
- Convenience material builders for UO₂, water (with S(α,β)),
  and Zircaloy-4.
- Region grammar: intersection (whitespace), union (`|`),
  and complement (`~`) — flat, no nested parentheses.

## Quick start

Build-from-source only today — the binding is abi3-compatible
(Python ≥ 3.9) but not published to PyPI yet.

```bash
# Prerequisite: Rust toolchain (stable) and Python ≥ 3.9
pip install --user maturin

# Build the wheel
cd rust_prototype/bindings/python
maturin build --release

# Install it
pip install --user --force-reinstall \
    ../../target/wheels/open_rust_mc-0.1.0-cp39-abi3-*.whl
```

Alternatively, `maturin develop --release` inside an activated
virtualenv builds and installs in one step.

## Hello, Godiva

The canonical smoke test — ICSBEP HEU-MET-FAST-001 (a bare 8.7407 cm
93.5 %-enriched uranium sphere) driving the engine from Python in a
few seconds. `k_eff` comes back in ICSBEP's 1.0000 ± 100 pcm band; at
these stats you'll see a few hundred pcm of statistical noise:

```python
from open_rust_mc import Material, Scene, Sphere, Settings, run_eigenvalue

heu = (
    Material("HEU", temperature=294.0, temp_idx=1)
    .add_nuclide("U234.h5", atom_density=0.000483, awr=232.029, nubar=2.49)
    .add_nuclide("U235.h5", atom_density=0.04509,  awr=233.025, nubar=2.43)
    .add_nuclide("U238.h5", atom_density=0.00265,  awr=236.006, nubar=2.49)
)

scene = (
    Scene("data/endfb-vii.1-hdf5/neutron")
    .add_material("heu", heu)
    .add_surface("boundary", Sphere(r=8.7407, bc="vacuum"))
    .add_cell("fuel", region="-boundary", fill="heu", temperature=294.0)
    .add_cell("outside", region="+boundary")  # void outer cell
)

settings = Settings(batches=50, inactive=10, particles=5000, seed=1)
result = run_eigenvalue(scene, settings)

print(f"k_eff = {result.k_eff:.5f} ± {result.k_sigma:.5f}")
print(f"runtime: {result.runtime_seconds:.2f} s over "
      f"{result.active_batches} active batches")
```

Run it:

```bash
python rust_prototype/bindings/python/examples/godiva.py \
    data/endfb-vii.1-hdf5/neutron
```

## Builder API

### `Scene(data_dir)`

The top-level container. Takes a path to the ENDF/B-VII.1 HDF5
directory (`neutron/` subdirectory — same files the Rust binaries
consume). All builder methods return `self` for chaining.

| Method | Purpose |
|---|---|
| `add_material(name, material)` | Register a `Material` under a string name |
| `add_surface(name, surface)` | Register a surface under a name |
| `add_cell(name, region, fill=None, temperature=293.15)` | Register a cell with a region expression and an optional material fill |
| `set_xs_mode(mode)` | Pick the XS provider — `XsMode.Table` (default), `Svd`, `HybridTableWmp`, or `HybridSvdWmp` |
| `set_svd_rank(rank)` | Default SVD truncation rank for `Svd` and `HybridSvdWmp` modes (default 5; ignored in Table mode) |
| `set_svd_ranks({mt: rank, …})` | Per-MT rank overrides — e.g. drop smooth tails (MT=2/18/102) to rank 1 while keeping discrete-level MTs at the default |
| `set_photon_data_dir(path)` | Per-element photon HDF5 directory; required for `run_gamma_heating` |
| `add_photon_material(cell_name, photon_material)` | Attach a `PhotonMaterial` to a previously-registered cell |

### `XsMode`

Selects the cross-section representation. All four providers share
the same `XsProvider` trait on the Rust side, so the same geometry
and physics kernels run unchanged across modes.

| Variant | What it does |
|---|---|
| `XsMode.Table` | Pointwise tables (binary search + log-log interpolation). Lowest load time, fastest single-temperature lookup, most memory. The industry baseline. |
| `XsMode.Svd` | Rank-*k* SVD-compressed kernels. Slower load (SVD decomposition), lower memory at high *k* or multi-temperature. Default rank 5. |
| `XsMode.HybridTableWmp` | Pointwise tables with Windowed Multipole override inside each nuclide's resolved-resonance window. Industry-baseline accuracy at lower memory than pure tables. |
| `XsMode.HybridSvdWmp` | SVD outside the RRR window, exact WMP evaluation inside. The intended production-precision mode. Set per-MT ranks to push smooth tails (MT=2/18/102) to rank 1 — WMP handles all the resonance structure. |

Hybrid modes look for WMP files in `<data_dir>/../wmp/` (standard
`092234.h5`, `092235.h5`, `092238.h5` naming). Nuclides without WMP
coverage transparently fall back to the non-WMP path.

### `Material(name, temperature=293.15, temp_idx=1)`

A nuclide mixture with a bulk temperature (K) and a library-temperature
index (which of the HDF5 column sets to use for on-library lookups).

| Method | Purpose |
|---|---|
| `add_nuclide(hdf5_file, atom_density, awr, nubar=0.0, thermal_file=None)` | Append a nuclide. `hdf5_file` is the basename inside the data dir (e.g. `"U235.h5"`); `atom_density` in atoms/(b·cm); `awr` is atomic-weight ratio; `nubar` is a fallback mean neutron yield used only if the HDF5 doesn't provide energy-dependent ν̄(E). Pass `thermal_file="c_H_in_H2O.h5"` (or similar) to bind an S(α,β) library to this nuclide. |
| `total_atom_density()` | Sum of nuclide atom densities in the material. |

Atom density is the absolute macro unit (`Σ = n·σ` gives cm⁻¹ when
σ is in barns).

**Convenience builders** (in `open_rust_mc`, Python-side):

- `uranium_oxide_material(name, density_g_per_cm3, enrichment, ...)` —
  build UO₂ at a given U-235 enrichment. Does the stoichiometry + atom
  density math from `density × N_A / M_UO₂`. Returns a `Material` with
  U-234, U-235, U-238, O-16 populated.
- `water_material(name="H2O", density_g_per_cm3=0.74, ...)` — H₂O with
  S(α,β) for H in water attached by default (`c_H_in_H2O.h5`).
- `zircaloy4_material(name="Zircaloy-4", density_g_per_cm3=6.55, ...)` —
  four major Zr isotopes at natural abundance.
- `atom_density_from_mass_density(density_g_per_cm3, molar_mass_g_per_mol)` —
  the `ρ · N_A / M · 1e-24` primitive if you want to roll your own.
- `NUCLIDE_DATA` — dict mapping common nuclide names (U235, O16, H1, …)
  to their `{awr, a, nubar, file}` record, pulled from the HDF5 headers.

### `PhotonMaterial(density_g_per_cm3)`

Per-element mixture used for photon transport. Attach to cells via
`Scene.add_photon_material(cell_name, photon_material)`.

| Method | Purpose |
|---|---|
| `add_element(hdf5_file, atom_density)` | Append a per-element photon data file (e.g. `"U.h5"`, `"O.h5"`) in the scene's photon data directory with its atom density in atoms/(b·cm). |

### Surfaces

Surface constructors return a `Surface` handle for `Scene.add_surface`.
All accept a `bc=` keyword — `"transmission"` (default), `"reflective"`,
or `"vacuum"`.

| Constructor | Equation | Keyword arguments |
|---|---|---|
| `Sphere(r, x=0, y=0, z=0)` | (r − c)² = R² | center coords |
| `XCylinder(r, y=0, z=0)` | cylinder along +x axis | center on y,z |
| `YCylinder(r, x=0, z=0)` | cylinder along +y axis | center on x,z |
| `ZCylinder(r, x=0, y=0)` | cylinder along +z axis | center on x,y |
| `XPlane(x0)` | x = x0 | offset |
| `YPlane(y0)` | y = y0 | offset |
| `ZPlane(z0)` | z = z0 | offset |

The engine also supports axis-aligned double cones (`ConeX`, `ConeY`,
`ConeZ`); their Python constructors are a follow-up.

### Region expressions

Cells take a string region expression. The grammar is adequate for
reactor benchmarks without nested parentheses:

| Token | Meaning |
|---|---|
| `"-name"` | inside (negative half-space) of the named surface |
| `"+name"` | outside (positive half-space) of the named surface |
| `"~-name"` / `"~+name"` | complement — `~-name ~-other` reads "outside `name` AND outside `other`" |
| whitespace-separated tokens | AND'd into an intersection |
| `" | "` at the top level | splits OR-groups; the cell is the union of groups |

Examples:

- `"-boundary"` — fuel disc of a Godiva sphere
- `"+fuel_or -clad_ir"` — annular gap in a pin cell
- `"-a | -b"` — inside `a` OR inside `b` (union)
- `"~-a ~-b"` — NOT(inside a) AND NOT(inside b)

Cell AABBs are auto-computed from the axis-aligned half-space tokens:
planes contribute their sided half-space (so `"+xmin -xmax +ymin -ymax
+zmin -zmax"` yields a finite box), cylinders and spheres contribute
their own bounding boxes. A cell with no bounded tokens falls back to
`Aabb::INFINITE`, which is safe because `initial_source` only samples
fissile (material-filled) cells.

### `Settings(batches, inactive, particles, seed=1)`

Eigenvalue power-iteration controls. All fields are writable after
construction (`s.batches = 100`).

### `run_eigenvalue(scene, settings) -> EigenvalueResult`

Materialises the scene, loads the needed HDF5 files once, and runs
the simulation. Holds a rayon thread pool under the hood and
releases Python's GIL for the duration (`py.allow_threads`), so the
Rust side runs fully native with all cores. Returns:

| Field | Type | Meaning |
|---|---|---|
| `k_eff` | `float` | mean of active-batch k_eff |
| `k_sigma` | `float` | standard error of the mean |
| `active_batches` | `int` | number of batches counted toward k |
| `total_histories` | `int` | active_batches × particles |
| `runtime_seconds` | `float` | Rust-side wall time |
| `k_per_batch` | `list[float]` | k_eff value for every batch (active and inactive) |
| `entropy_per_batch` | `list[float]` | Shannon entropy of the fission-site bank, for convergence plots |
| `active_mask` | `list[bool]` | `True` for batches counted toward the active tally |
| `captures_by_cell` | `list[float]` | Non-fission absorption counts per cell, summed over active batches |
| `cell_names` | `list[str]` | Cell names in the same order as `captures_by_cell` |
| `total_collisions` | `int` | Summed across active batches (diagnostic) |
| `total_fissions` | `int` | Summed across active batches |
| `total_leakage` | `int` | Summed across active batches |

Helpers:

- `result.captures_dict()` — `{cell_name: count}` for pandas / plot input.
- `result.stats()` — full diagnostic dict for one-line debug printing
  or JSON serialisation. Includes `mode`, `svd_rank`, `svd_ranks_per_mt`,
  `k_eff`, `k_sigma`, `active_batches`, `total_histories`, `load_time_seconds`,
  `sim_time_seconds`, `runtime_seconds`, `xs_memory_bytes`, `xs_memory_mib`,
  `wmp_covered_nuclides`, `total_collisions`, `total_fissions`,
  `total_leakage`, and `ns_per_history`.

### `run_gamma_heating(scene, neutron_settings, n_photon_histories=200_000, ...)`

Coupled neutron-photon pipeline:

1. Run neutron k-eigenvalue with the XS provider collecting per-MT
   photon source events at every capture, fission, and inelastic
   scattering site (no notional spectra — the outgoing energies are
   sampled from the HDF5 `distribution_0/energy` trees).
2. Run photon transport from the aggregated source bank through the
   same CSG with per-cell `PhotonMaterial`. Electrons born from
   Compton / photoelectric / pair production are track-integrated
   with Highland multiple scattering, Bethe-Bloch-style non-uniform
   `dE/dx`, and single-event Seltzer-Berger bremsstrahlung — all on
   by default in the engine.

Requires `Scene.set_photon_data_dir(path)` and at least one
`Scene.add_photon_material(cell_name, photon_material)` call.

Returns a `GammaHeatingResult` with `k_eff` / `k_sigma`,
`deposition_fraction` + `deposition_ev` per cell, `total_source_energy_ev`,
`escaped_energy_ev`, `orphan_energy_ev`, bremsstrahlung counts, and
neutron / photon runtimes. `result.fractions_dict()` returns
`{cell_name: fraction}`.

## Architecture

### Rust as source of truth

The builder (`Scene`, `Material`, region expression) lives on the
Rust side of the FFI — Python calls typed methods that push into
Rust-owned data structures. Validation (missing surface names,
unknown material references, empty region expressions) happens at
`build()` time in Rust, so Python, a future CLI front-end, and a
future GUI front-end all share the same object model and the same
error messages.

Python is intentionally thin: `__init__.py` only re-exports symbols
from the native `_core` module. No business logic.

### Lazy HDF5 loading

`Scene.add_nuclide(…)` records the file name and physics parameters
but does NOT touch the disk. The HDF5 load happens inside
`run_eigenvalue`, after the whole scene has been validated. You can
build an arbitrarily complex scene in a REPL without paying for I/O
until you press "go".

### GIL handling

The Rust engine runs silent when called from Python
(`SimConfig::verbose = false`) — no per-batch `println!`, which both
avoids stdout-lock contention with Python and lets
`py.allow_threads(|| simulate::run_eigenvalue(…))` release the GIL
cleanly for the whole simulation. Rayon's thread pool lazy-inits the
first time a batch runs, inside the `allow_threads` block where
Python isn't holding any system locks.

The engine's CLI binaries keep `verbose = true` so their stdout
traces are unchanged.

### Windows-specific notes

Building the wheel on Windows requires the MSVC toolchain (free via
Visual Studio Build Tools). maturin handles the rest. Python 3.14
ships a free-threaded build variant (PEP 703); this binding
currently targets the standard GIL-enabled build via `abi3-py39`,
which is enough for the single-simulation-at-a-time flow the builder
is designed around. Parallel parameter sweeps launched from a Python
thread pool will want the free-threaded build eventually — that's a
separate `abi3-py313` build target.

## Known limitations

- **No cone constructors yet.** The engine supports `ConeX`, `ConeY`,
  `ConeZ`; their Python wrappers are a small follow-up.
- **Region grammar is flat.** Unions (`|`), intersections
  (whitespace), and complements (`~`) compose without nested
  parentheses. For any geometry expressible as a finite union of
  half-space-intersections, this is sufficient; arbitrary CSG with
  nested groups would need a recursive-descent parser.
- **Photon materials via cell name.** If two cells share a neutron
  material but should have different photon materials, only the first
  registration is kept. Keep the mapping 1:1 in practice.
- **Fixed-source photon runs are not exposed.** `run_gamma_heating`
  drives the coupled n-γ pipeline; standalone photon-only simulations
  (e.g. Cs-137 spectrum) aren't wired through the FFI yet. Adding them
  is a small extension of the existing photon-transport surface.
- **No per-cell energy-spectrum tallies.** Aggregate tallies (k_eff,
  per-cell captures, γ deposition) are exposed; energy-binned flux /
  current tallies need further FFI surface.
- **No off-library temperature flag.** The Rust binaries take
  `--target-temp` / `--target-temp-offset` to drive the 3-point Ducru
  kernel-Gram QP weights; that knob isn't on `Settings` yet. Workaround:
  set the `temperature` per `Material` / per cell, and pick a `temp_idx`
  bracketing the target.
- **GPU backend is binary-only.** `--features cuda` is a Cargo flag
  on the Rust side; the Python wheel ships CPU-only. Re-exposing
  the GPU XS providers as a `XsMode.GpuSvd` is straightforward but
  not done.

## What the Python layer actually validates

The Python package is not a re-implementation of the engine — it's a
FFI wrapper that constructs `Cell` / `Material` / `Surface` /
`SimConfig` values from Python objects and forwards them to the same
`simulate::run_eigenvalue` and photon-transport paths the Rust
binaries use. A Python call produces the same numbers as the Rust
binary **by construction**, so "Python matches Rust binary" is not a
meaningful physics claim — it's a mechanical consequence of calling
the same function.

The claims that are meaningful are:

- **Plumbing correctness:** the `Scene` builder → engine translation
  (region parsing, AABB derivation from half-space tokens, per-
  material photon-material indexing, S(α,β) wiring, thermal-file
  loading, tally extraction) produces physically correct inputs.
  Failing any of these would show up as bad answers — when the
  photon-material indexing was off-by-one during development, the
  PWR pin cell's clad cell went to 0 % deposition. The 84 / 0 / 10 /
  6 split only emerges once every piece is right.
- **Validation against OpenMC:** the PWR γ-heating example lands at
  84 / 0 / 10 / 6 vs OpenMC's 85 / 0 / 9 / 6 on identical geometry.
  This is the real physics cross-check (see `scripts/openmc_pwr_gamma_heating.py`);
  it validates the engine + Python path together against an
  independent Monte Carlo code.

## Examples

All examples live in
[`rust_prototype/bindings/python/examples/`](rust_prototype/bindings/python/examples/).

| Script | What it shows |
|---|---|
| [`godiva.py`](rust_prototype/bindings/python/examples/godiva.py) | ICSBEP HEU-MET-FAST-001 (bare 8.74 cm HEU sphere). k_eff lands inside ICSBEP's 1.0000 ± 100 pcm band; runtime ~0.2 s at 50 × 5 k. The hello-world for the `Scene` builder. |
| [`xs_mode_demo.py`](rust_prototype/bindings/python/examples/xs_mode_demo.py) | Cycles all four `XsMode` variants on Godiva and prints the `result.stats()` table — load time, sim time, memory, WMP coverage. The shape of the data the engine pushes back. |
| [`xs_mode_quick.py`](rust_prototype/bindings/python/examples/xs_mode_quick.py) | Smaller smoke test of the same matrix, plus a couple of per-MT adaptive-rank rows (drop MT=2/18/102 to rank 1 under the SVD+WMP hybrid). |
| [`pwr_pincell.py`](rust_prototype/bindings/python/examples/pwr_pincell.py) | Neutron-only PWR pin k_inf with reflective BCs, UO₂ + Zr-4 + H₂O-with-S(α,β), and per-cell capture tallies. The starter for any pin-cell-shaped reactor problem. |
| [`pwr_gamma_heating.py`](rust_prototype/bindings/python/examples/pwr_gamma_heating.py) | Coupled neutron-photon γ-heating end-to-end. Deposition split lands at 84 / 0 / 10 / 6, within ~1 pp of OpenMC 0.15.3 on the same geometry. |
| [`seed_sweep.py`](rust_prototype/bindings/python/examples/seed_sweep.py) | Multi-seed Godiva run with a Student-t 95 % confidence interval on k_eff. The pattern any benchmark-quality run wants. |
