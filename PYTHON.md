# Python API

A thin PyO3 binding over the Rust engine, designed so Python scripts
can build Monte Carlo neutron-transport scenes and run eigenvalue
simulations without touching Rust. **Rust is the source of truth** —
the builder, region validation, and simulation core all live on the
Rust side; the Python layer is ~200 lines of class/function exports.

The binding crate is at
[`rust_prototype/bindings/python/`](rust_prototype/bindings/python/);
sample scripts live in
[`rust_prototype/bindings/python/examples/`](rust_prototype/bindings/python/examples/).

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
93.5 %-enriched uranium sphere) reproducing the Rust binary's
answer of `k_eff ≈ 1.000` to within ~300 pcm statistical noise in a
few seconds:

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

### `Material(name, temperature=293.15, temp_idx=1)`

A nuclide mixture with a bulk temperature (K) and a library-temperature
index (which of the HDF5 column sets to use for on-library lookups).

| Method | Purpose |
|---|---|
| `add_nuclide(hdf5_file, atom_density, awr, nubar=0.0)` | Append a nuclide. `hdf5_file` is the basename inside the data dir (e.g. `"U235.h5"`); `atom_density` in atoms/(b·cm); `awr` is atomic-weight ratio; `nubar` is a fallback mean neutron yield used only if the HDF5 doesn't provide energy-dependent ν̄(E). |

Atom density is the absolute macro unit (`Σ = n·σ` gives cm⁻¹ when
σ is in barns). Compute it on the Python side from macro density and
stoichiometry, or pass a pre-computed value.

### Surfaces

All eight variants from the engine are exposed as constructor
functions. Each returns a `Surface` handle that you pass to
`Scene.add_surface`. Every surface accepts a `bc=` keyword for the
boundary condition — one of `"transmission"` (default), `"reflective"`,
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
`ConeZ`); Python constructors for them land in a follow-up change.

### Region expressions

A cell's `region=` argument is a boolean-AND expression over registered
surface names, using OpenMC's sign convention:

| Token | Meaning |
|---|---|
| `"-name"` | inside (negative half-space) of the named surface |
| `"+name"` | outside (positive half-space) of the named surface |
| `"-a -b +c"` | intersection of the listed half-spaces |

Godiva uses just two cells: the fuel sphere is `"-boundary"`, the
outer vacuum region is `"+boundary"`. A PWR pin cell would use
`"-fuel_or"` for the fuel, `"+fuel_or -clad_ir"` for the He gap, and
so on. Unions and complements are a planned follow-up; for ICSBEP
benchmarks and reactor pin cells, intersections are sufficient.

Cell AABBs are computed automatically from the `-name` tokens in the
region: each inside-surface's own bounding box contributes to the
cell's bounding box via intersection. This is what keeps the
rejection-sampled initial fission source finite — a cell with no
inside tokens (pure-`+name` outer cells) gets `Aabb::INFINITE`, which
is safe because only cells with material fills are sampled.

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

## Known limitations (first cut)

- **No region unions or complements.** `-a -b` (intersection) works;
  `-a | -b` (union) and `~-a` (complement) don't. Region grammar is
  a whitespace-separated list of signed surface-name tokens.
- **Atom densities only** — no enrichment-to-density conversion helper.
  For a 3.1 % enriched UO₂ you pass the three pre-computed densities.
  A `Material.add_nuclide_by_enrichment()` helper is a natural follow-up.
- **No S(α,β) thermal scattering binding.** The engine supports it
  (see `pwr_pincell.rs`); wiring up the Python path is follow-up work.
- **No photon transport, no γ-heating.** The scaffold is in place
  (all physics kernels live in the engine crate); exposing
  `Scene.add_photon_material` and `run_gamma_heating` is the next
  chunk of FFI surface.
- **No tallies.** Today you get `k_eff` and timing; per-cell heating,
  fission-rate maps, and energy-spectrum tallies are planned but
  currently require dropping to Rust.

## Examples

- [`examples/godiva.py`](rust_prototype/bindings/python/examples/godiva.py) — ICSBEP HEU-MET-FAST-001 reproduction.

More examples will land as the FFI surface grows.
