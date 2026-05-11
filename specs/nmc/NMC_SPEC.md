# open_rust_mc — NMC Bundle & Visual IDE Specification

**Version:** 0.1-draft  
**Engine reference:** `open_rust_mc` commit `43b3236`, STATUS.md 2026-05-07  
**Status:** Draft for implementation

---

## 1. Overview

An `.nmc` file is a ZIP archive (deflate compressed, `.nmcz` alias accepted) that bundles
a complete, self-contained Monte Carlo simulation scene — geometry, materials, simulation
settings, verification render, and optionally results — into a single portable artifact.

Design goals:

- **One file = one reproducible simulation.** Drop anywhere, run anywhere.
- **No translation step.** The geometry the visual IDE holds in memory, the geometry the
  verify render uses, and the geometry the transport engine runs are all derived from the
  same `scene.json` — zero intermediate format, zero divergence risk.
- **Human-inspectable.** Standard ZIP + JSON + PNG. No proprietary binary format.
  `unzip -l scene.nmc` on any system shows all contents.
- **Tamper-evident.** `manifest.json` carries SHA-256 hashes of `scene.json` and
  `verify.png`. The engine refuses to run if hashes do not match.
- **Results are optional.** An unrun bundle is valid. Results populate `results/` after
  simulation.

---

## 2. Bundle Layout

```
scene_name.nmc          (ZIP, deflate compressed)
├── manifest.json       Required. Metadata, hashes, engine pin.
├── scene.json          Required. Geometry + materials + sim settings.
├── verify.png          Required after verification. PhongPlot render.
├── verify_xy.png       Optional. XY cross-section slice at z=0.
├── verify_xz.png       Optional. XZ cross-section slice at y=0.
└── results/            Optional. Populated by engine after run.
    ├── summary.json    k_eff, k_sigma, timing, particle count.
    ├── keff_history.json   Per-batch k_coll and k_track.
    ├── entropy_history.json  Per-batch Shannon entropy.
    ├── captures_by_cell.json  Per-cell absorption counts.
    ├── mesh_flux.npy   Optional. Cartesian mesh flux array (NumPy format).
    ├── surface_currents.json  Optional. J+ and J- per tagged surface.
    ├── gamma_heating.json  Optional. Per-cell photon energy deposition.
    └── statepoint.h5   Optional. HDF5 statepoint for restart.
```

### 2.1 File naming

The archive filename carries the scene name. Internally all paths are relative to the
archive root with no leading `/`. Path separators are `/` on all platforms.

Accepted extensions: `.nmc` (recommended), `.nmcz` (explicit alias).

---

## 3. `manifest.json`

```json
{
  "nmc_format_version": "1.0",
  "created_utc": "2026-05-10T14:32:00Z",
  "engine_version": "0.4.1",
  "engine_commit": "43b3236",
  "scene_hash":  "sha256:a3f9c2...",
  "verify_hash": "sha256:b72c1d...",
  "verified": true,
  "author": "",
  "notes": "",
  "tags": [],
  "benchmark": {
    "suite":           "ICSBEP",
    "case_id":         "HEU-MET-FAST-001",
    "case_name":       "Godiva",
    "k_eff_reference": 1.0000,
    "k_eff_sigma":     0.0010,
    "source":          "ICSBEP Handbook 2022, HMF-001",
    "category":        "HEU-MET-FAST"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `nmc_format_version` | string | yes | Semver of this spec. Currently `"1.0"`. |
| `created_utc` | string | yes | ISO 8601 UTC timestamp of bundle creation. |
| `engine_version` | string | yes | `open_rust_mc` crate version that produced this bundle. |
| `engine_commit` | string | yes | Short git SHA of the engine at verification time. |
| `scene_hash` | string | yes | `"sha256:" + hex(SHA-256(scene.json bytes))`. |
| `verify_hash` | string | yes if `verified=true` | `"sha256:" + hex(SHA-256(verify.png bytes))`. |
| `verified` | bool | yes | `true` only after `find_cell_recursive` render has run and hashes recorded. A bundle with `verified=false` can be edited and saved but the engine will warn before running. |
| `author` | string | no | Free text. |
| `notes` | string | no | Free text description. |
| `tags` | string[] | no | e.g. `["pwr", "fresh-fuel", "hzp"]`. |
| `benchmark` | object | no | Optional reference-case metadata. Present when the bundle represents a published benchmark (ICSBEP, KRITZ, MISTRAL, …). See §3.1. |

**Hash enforcement.** On load, the engine computes `SHA-256(scene.json)` and compares to
`scene_hash`. If mismatched, load fails with a clear error. Same for `verify_hash` when
`verified=true`. This guarantees the verify render corresponds exactly to the geometry
that will run.

### 3.1 `benchmark` block

Present when the bundle represents a published benchmark case with a known reference
k_eff and experimental uncertainty. The `icsbep_bench` runner reads this block to compute
the regression metric `Δ_pcm = (k_calc − k_ref) · 10⁵` and the pass criterion
`|k_calc − k_ref| < n_σ · √(σ_calc² + σ_exp²)`.

| Field | Type | Required | Description |
|---|---|---|---|
| `suite` | string | yes | Benchmark suite identifier — e.g. `"ICSBEP"`, `"KRITZ"`, `"MISTRAL"`. |
| `case_id` | string | yes | Canonical case identifier — e.g. `"HEU-MET-FAST-001"`. |
| `case_name` | string | no | Human-readable name — e.g. `"Godiva"`, `"Jezebel"`. |
| `k_eff_reference` | number | yes | Reference k_eff from the published evaluation. |
| `k_eff_sigma` | number | yes | 1-σ experimental uncertainty on `k_eff_reference` (absolute, not pcm). |
| `source` | string | yes | Citation — e.g. `"ICSBEP Handbook 2022, HMF-001, Table 3"`. |
| `category` | string | no | Sub-class — e.g. `"HEU-MET-FAST"`, `"LEU-COMP-THERM"`. |
| `notes` | string | no | Free-text caveats (e.g. "spec uses ENDF/B-VII.0; we run VII.1"). |

A bundle without a `benchmark` block is a valid `.nmc` — `benchmark` is only for
validation / regression workflows. Production simulations omit it.

---

## 4. `scene.json`

Full JSON Schema: `open_rust_mc_geometry.schema.json` (companion file).

Top-level structure:

```json
{
  "surfaces":        [...],
  "cells":           [...],
  "universes":       [...],
  "rect_lattices":   [...],
  "hex_lattices":    [...],
  "materials":       [...],
  "root_universe_id": 0,
  "sim_settings":    { ... }
}
```

`surfaces`, `cells`, `universes`, `materials` are required. `rect_lattices` and
`hex_lattices` are optional (omit if none). `sim_settings` is optional; CLI flags
override any values present.

### 4.1 Geometry types

Derived exactly from `geometry/surface.rs` and `geometry/cell.rs`.

**Surfaces** — all enum variants supported:

| Type | Key parameters |
|---|---|
| `Plane` | `normal: [f64;3]`, `offset: f64` |
| `PlaneX` | `x0: f64` |
| `PlaneY` | `y0: f64` |
| `PlaneZ` | `z0: f64` |
| `Sphere` | `center: [f64;3]`, `radius: f64` |
| `CylinderZ` | `center_x`, `center_y`, `radius` |
| `CylinderX` | `center_y`, `center_z`, `radius` |
| `CylinderY` | `center_x`, `center_z`, `radius` |
| `ConeZ` | `x0`, `y0`, `z0`, `r_sq` (= tan²(half-angle)) |
| `ConeX` | `x0`, `y0`, `z0`, `r_sq` |
| `ConeY` | `x0`, `y0`, `z0`, `r_sq` |

All surfaces carry `bc`: `"Transmission"` | `"Reflective"` | `"Vacuum"`.

**Region CSG tree** — recursive, matches `geometry/cell.rs Region` enum:

```
{ "op": "HalfSpace",    "surface_idx": 0, "positive": false }
{ "op": "Intersection", "left": <Region>, "right": <Region> }
{ "op": "Union",        "left": <Region>, "right": <Region> }
{ "op": "Complement",   "inner": <Region> }
```

**Cell fill** — matches `CellFill` enum:

```
{ "type": "Material",   "material_idx": 0 }
{ "type": "Universe",   "universe_id":  1 }
{ "type": "Lattice",    "lattice_idx":  0 }
{ "type": "HexLattice", "hex_lattice_idx": 0 }
{ "type": "Void" }
```

**RectLattice** — `origin [x,y,z]`, `pitch [x,y,z]`, `shape [nx,ny,nz]`,
`universes [UniverseId]` (row-major, length = nx×ny×nz),
`material_overrides` (optional, one `{cell_idx: material_idx}` map per element).

**HexLattice** — `center`, `pitch_xy`, `pitch_z`, `n_rings`, `n_axial`,
`orientation: "Y"|"X"`, `universes` (axial-major, doubled (2N+1)² grid),
`material_overrides` (optional).

### 4.2 `sim_settings`

All fields optional. CLI flags take precedence.

```json
{
  "batches":              100,
  "inactive_batches":     20,
  "particles_per_batch":  10000,
  "seed":                 12345,
  "xs_mode":              "Table",
  "svd_rank":             5,
  "survival_biasing":     { "w_min": 0.25, "w_survive": 0.5 },
  "weight_window":        null,
  "urr_equivalence":      false,
  "delayed_neutrons":     true,
  "run_photon_transport": false,
  "mesh_flux": {
    "origin":  [-10, -10, -10],
    "upper":   [ 10,  10,  10],
    "shape":   [50, 50, 50]
  },
  "surface_current_tags": [0, 3],
  "statepoint_every_n_batches": null
}
```

---

## 5. `verify.png`

A PNG image rendered by calling `find_cell_recursive` (the actual transport geometry
query) using Phong illumination, as described in Ridley et al. SNA+MC 2024.

**Not an approximation.** The primitive meshes displayed in the IDE edit mode are
approximate (configurable subdivision). The verify render is exact — the same function
the engine calls during particle tracking determines every pixel's colour.

**Render parameters** stored in `manifest.json` under `verify_params`:

```json
"verify_params": {
  "camera_pos":    [0, 0, 30],
  "look_at":       [0, 0,  0],
  "up":            [0, 1,  0],
  "fov_deg":       70,
  "width_px":      1920,
  "height_px":     1080,
  "light_positions": [[20, 20, 20], [-20, 10, 15]],
  "opaque_materials": "all"
}
```

Lighting model: Phong ambient + diffuse. Surface normals from `Surface::normal_at`.
Shadow rays via `find_cell_recursive` from hit point to each light source.

---

## 6. `results/summary.json`

Populated by the engine after a completed run. Never written by the IDE.

```json
{
  "completed_utc":        "2026-05-10T15:14:22Z",
  "engine_version":       "0.4.1",
  "engine_commit":        "43b3236",
  "scene_hash":           "sha256:a3f9c2...",
  "n_batches_total":      100,
  "n_batches_active":     80,
  "n_particles_per_batch": 10000,
  "k_eff":                1.00342,
  "k_eff_sigma":          0.00018,
  "k_track":              1.00311,
  "k_track_sigma":        0.00016,
  "entropy_converged_at_batch": 12,
  "wall_time_s":          142.3,
  "xs_load_time_s":       3.1,
  "transport_time_s":     139.2,
  "particles_per_second": 5614,
  "xs_mode":              "Svd",
  "svd_rank":             5,
  "gpu_used":             false
}
```

---

## 7. Visual IDE — NMC Studio

### 7.1 Architecture

```
NMC Studio
├── Editor (egui panels + wgpu viewport)
│   ├── Palette panel          Surface types, material presets, object templates
│   ├── Geometry viewport      wgpu render, switchable Edit/Verify/Slice modes
│   ├── Scene tree panel       Hierarchical view: universes → cells → surfaces
│   └── Inspector panel        Selected object properties, numeric fields
├── Engine bridge              Calls open_rust_mc via Rust API (same binary)
│   ├── verify()               Triggers find_cell_recursive render → verify.png
│   ├── run()                  Runs full simulation, populates results/
│   └── run_batch_parametric() Sweeps a parameter, produces Vec<.nmc>
└── Bundle I/O                 ZIP read/write, hash computation, manifest
```

**Framework:** `egui` + `egui-wgpu` + `egui-winit` + `wgpu` + `winit`.

The engine and IDE live in the same binary. Verify and run are function calls, not
process spawns. No IPC, no serialisation round-trip for the verify render.

### 7.2 Geometry viewport — three modes

**Edit mode** (default, real-time)

Renders pre-computed surface primitive meshes via wgpu rasterisation.
Instanced rendering: one vertex buffer per surface type, one instance buffer updated
per frame from the current scene state.

Primitive mesh library (pre-computed once at startup):

| Surface type | Mesh |
|---|---|
| PlaneX/Y/Z/Plane | Large quad (1000 cm × 1000 cm), oriented to normal |
| Sphere | UV sphere, 32 longitude × 16 latitude bands |
| CylinderZ/X/Y | Ring of 64 vertices, extruded ±500 cm along axis, no end caps |
| ConeZ/X/Y | Double cone, 64 vertices per ring, apex shared |

Per-instance vertex shader transform:
- Sphere: `pos * radius + center`
- CylinderZ: `[pos.x * radius + cx, pos.y * radius + cy, pos.z]`
- ConeZ: `[pos.x * sqrt(r_sq) * abs(pos.z) + x0, ..., pos.z + z0]`

Surfaces render as semi-transparent filled meshes with an opaque wireframe overlay.
Selected surface: highlighted in accent colour. Boundary condition encoded by colour:
`Transmission` = grey, `Reflective` = blue, `Vacuum` = red.

Cells are not directly rendered in edit mode. The scene tree panel shows the CSG
expression; the inspector shows which surfaces participate.

Controls:
- Orbit: left-drag
- Pan: middle-drag or shift+left-drag
- Zoom: scroll wheel
- Select: left-click on surface mesh (ray-pick against instance bounding spheres)
- Axis indicator: XYZ gizmo bottom-left
- Grid: optional XY grid plane, spacing configurable

**Slice mode** (2D cross-section)

Renders a 2D pixel buffer by calling `find_cell_recursive` on a flat grid of points
at a fixed axis/position. Exact — same function as the transport engine.

Controls:
- Axis selector: XY / XZ / YZ buttons
- Slice position: slider + numeric input
- Colour map: per-material colours, configurable
- Resolution: 512 / 1024 / 2048 px (trades speed for detail)

This is the existing `preview/render.rs` logic, exposed as a viewport tab. Runs on
CPU in a background thread; UI stays responsive. Result uploaded as a wgpu texture
and displayed via `egui::Image`.

**Verify mode** (3D PhongPlot)

Triggered by pressing **Verify** button. Runs the full Phong ray-trace using
`find_cell_recursive` on a background thread. Progress bar shows completion.
On finish: updates `verify.png`, updates `manifest.json` hashes, marks bundle
`verified: true`.

Camera controls same as Edit mode. Re-render button re-runs if geometry changed.

A red banner is shown whenever geometry has been modified since the last verify render,
indicating the bundle is no longer verified.

### 7.3 Palette panel

Three tabs:

**Surfaces** — click to instantiate at scene origin, then drag to position.

```
Primitives:
  □ PlaneX    □ PlaneY    □ PlaneZ
  ○ Sphere    ⌀ CylinderZ ⌀ CylinderX ⌀ CylinderY
  △ ConeZ     △ ConeX     △ ConeY     ⬡ Plane (general)

Assembly helpers:
  ▦ Rect box (6 planes)
  ⬡ Hex boundary
  ◎ Pin cylinders (fuel + gap + clad)
```

**Materials** — pre-loaded nuclear material library.

Standard compositions auto-populate atom densities from embedded tables:

| Preset | Nuclides | Notes |
|---|---|---|
| UO₂ (enrichment %, T K) | U-234/235/238, O-16 | Density from standard correlation |
| He gap (T K) | He-4 | At fill pressure |
| Zircaloy-4 (T K) | Zr-90/91/92/94/96, Sn-116, Fe-56, Cr-52 | ASTM B353 |
| H₂O (T K, P MPa) | H-1 + S(α,β), O-16 | Density from IAPWS-IF97 |
| D₂O (T K) | D-2 + S(α,β), O-16 | |
| B₄C (enrichment %) | B-10/11, C-nat | Absorber rod material |
| Gd₂O₃-UO₂ (wt% Gd, enrich %) | U-234/235/238, Gd-154/155/156/157/158, O-16 | Burnable absorber |
| SS-304 | Fe-54/56/57/58, Cr-50/52/53/54, Ni-58/60/61/62/64 | Structural |
| Graphite (T K) | C-nat + S(α,β) graphite | |
| Air | N-14, O-16 | At 293 K, 1 atm |
| HEU metal (enrichment %) | U-235, U-238 | Criticality safety |
| Pu metal | Pu-239/240/241/242 | Criticality safety |

Slider controls for temperature, enrichment, pressure where applicable. Atom densities
update in real time. HDF5 file paths resolved via `NuclideLibrary` (ZAID registry).

**Templates** — pre-built common objects:

```
PWR fuel pin    (UO₂ + He gap + Zr-4, standard dimensions)
BWR fuel pin    (UO₂ + He gap + Zr-2)
Control rod     (B₄C + SS clad)
Guide tube      (water-filled + Zr-4)
TRIGA fuel rod  (U-ZrH + clad)
Godiva sphere   (HEU metal, critical radius)
```

Instantiating a template adds all required surfaces, a pin universe, and pre-wired
cells to the scene. Dimensions and materials are editable in the inspector.

### 7.4 Scene tree panel

Hierarchical view of the scene:

```
▼ Scene
  ▼ Universe 0 (root)
    ▼ Cell 0  [Material: UO₂_900K]
        Surface 0 (Sphere r=8.74, Vacuum)
    ▼ Cell 1  [Void]
        ~Surface 0
  ▼ Materials
      UO₂_900K
      H₂O_600K
```

Click any node to select and show in Inspector panel. Right-click for context menu:
duplicate, delete, rename, add child.

### 7.5 Inspector panel

Shows properties of the selected object. All fields are editable numeric inputs with
real-time update of the Edit viewport.

**Surface selected:**
```
Type:         CylinderZ  ▾
Center X:     [  0.0000 ] cm
Center Y:     [  0.0000 ] cm
Radius:       [  0.4096 ] cm
Boundary:     Transmission ▾
```

**Cell selected:**
```
ID:           0
Temperature:  [ 900.0 ] K
Fill:         Material ▾   UO₂_900K ▾
Region:       inside(0)
Rotation:     None ▾
```

**Material selected:**
```
Name:         UO₂_3.1pct_900K
Temperature:  [ 900.0 ] K
─────────────────────────────────
H1.h5         [           ] atoms/b-cm    [remove]
U235.h5       [ 7.0714e-4 ] atoms/b-cm   [remove]
U238.h5       [ 2.2050e-2 ] atoms/b-cm   [remove]
O16.h5        [ 4.5514e-2 ] atoms/b-cm   [remove]
                                          [+ add nuclide]
Thermal:      c_H_in_H2O.h5 ▾
```

### 7.6 Simulation settings panel

Collapsible panel (default collapsed) below the scene tree.

```
── Simulation settings ────────────────────────
Batches:          [ 100 ]   Inactive: [ 20 ]
Particles/batch:  [ 10000 ]  Seed: [ 12345 ]

XS mode:          Table ▾    SVD rank: [ 5 ]
Survival biasing: ☑   w_min: [0.25]  w_survive: [0.5]
URR equivalence:  ☐
Delayed neutrons: ☑
Photon transport: ☐

Mesh flux tally:  ☐
  Origin: [-10,-10,-10]  Upper: [10,10,10]  Shape: [50,50,50]
───────────────────────────────────────────────
```

### 7.7 Toolbar actions

```
[New]  [Open .nmc]  [Save]  [Save As]
─────────────────────────────────────────────────────
[Verify ▶]   Runs PhongPlot, locks bundle as verified
[Run ▶]      Runs full simulation (warns if not verified)
[Parametric] Opens parametric sweep dialog
─────────────────────────────────────────────────────
[Export JSON]   Saves scene.json standalone
[Export Script] Generates equivalent Python API script
```

### 7.8 Verify flow (detailed)

1. User presses **Verify**.
2. IDE serialises current scene state to `scene.json` bytes.
3. Computes `SHA-256(scene.json)` → `scene_hash`.
4. Calls engine's `render_phong(scene, camera_params)` on background thread.
   Uses `find_cell_recursive` per pixel. Same code path as particle transport.
5. On completion: saves `verify.png`, computes `verify_hash`.
6. Updates `manifest.json`: `verified=true`, both hashes, `engine_commit`.
7. Viewport switches to Verify mode showing the render.
8. Red "unverified" banner clears.
9. Bundle is now safe to publish/archive/run.

If any geometry edit occurs after step 8, the banner returns and `verified` is set
`false` in memory (not written to disk until next save).

### 7.9 Parametric sweep dialog

```
Parameter:    materials[0].nuclides[0].atom_density  ▾  (or any JSON path)
From:         0.045
To:           0.060
Steps:        8
─────────────────────────────────────────────────────────
Output dir:   /home/user/sweeps/enrichment_sweep/
Bundle prefix: uo2_enrich_
─────────────────────────────────────────────────────────
[ Verify all ]   [ Run all ]   [ Verify + Run all ]
```

Produces N `.nmc` files, each verified independently. Results populate `results/` in
each bundle as simulations complete. Progress shown per bundle.

---

## 8. CLI — `open_rust_mc`

### 8.1 Commands

```
open_rust_mc run    <file.nmc>   [options]
open_rust_mc verify <file.nmc>   [options]
open_rust_mc info   <file.nmc>
open_rust_mc unpack <file.nmc>   [--output-dir DIR]
open_rust_mc pack   <dir>        [--output FILE]
open_rust_mc sweep  <file.nmc>   --param PATH --from F --to T --steps N
```

### 8.2 `run` options

All `sim_settings` fields in `scene.json` are overridable:

```
--batches N              Total batch count
--inactive N             Inactive batches
--particles N            Particles per batch
--seed N                 RNG seed
--xs-mode MODE           Table|Svd|HybridSvdWmp|HybridTableWmp
--svd-rank K             SVD rank (Svd/Hybrid modes)
--survival-biasing       Enable implicit capture
--ww-lower F             Weight window lower bound
--ww-upper F             Weight window upper bound
--urr-equivalence        Enable Stoker-Weiss URR spatial self-shielding
--no-delayed             Disable delayed neutron emission
--photons                Enable coupled neutron-photon transport
--statepoint FILE.h5     Write HDF5 statepoint every active batch
--restart-from FILE.h5   Load source bank from statepoint
--gpu                    Use CUDA transport (requires cuda feature)
--threads N              Rayon thread count (default: all cores)
--output FILE.nmc        Write results into a new bundle (default: in-place)
--no-verify-check        Skip hash verification (not recommended)
```

### 8.3 `info` output

```
$ open_rust_mc info godiva.nmc

Bundle:     godiva.nmc
Format:     NMC 1.0
Created:    2026-05-10 14:32 UTC
Author:     sorcerer86pt
Notes:      Bare HEU sphere, ICSBEP HEU-MET-FAST-001

Engine:     open_rust_mc 0.4.1 (43b3236)
Verified:   yes  (SHA-256 match confirmed)

Geometry:
  Surfaces:   1  (Sphere)
  Cells:      2
  Universes:  1
  Materials:  1  (HEU_metallic)
  Lattices:   none

Sim settings:
  Batches: 100 (20 inactive)  Particles: 10000/batch
  XS mode: Svd rank=5  Survival biasing: on

Results:    present
  k_eff:    1.00024 ± 0.00031
  k_track:  1.00019 ± 0.00028
  Runtime:  14.2 s  (5614 particles/s)
```

---

## 9. Python API

The Python bindings (`open_rust_mc` wheel) expose bundle I/O alongside the existing
`Scene` / `run_eigenvalue` API:

```python
import open_rust_mc as mc

# Load bundle
bundle = mc.Bundle.open("godiva.nmc")

# Inspect
print(bundle.manifest)        # dict
print(bundle.is_verified)     # bool
scene = bundle.scene          # mc.Scene object, fully editable

# Edit and re-verify
scene.materials[0].set_atom_density("U235.h5", 0.046)
bundle.verify(camera_pos=[0,0,30], look_at=[0,0,0])  # blocks until done

# Run
result = bundle.run(batches=200, inactive=40, particles=20000)
print(result.k_eff, result.k_sigma)

# Save
bundle.save("godiva_modified.nmc")

# Parametric sweep (returns list of bundles with results)
bundles = mc.sweep(
    template="godiva.nmc",
    param="materials[0].nuclides[0].atom_density",
    values=[0.043, 0.045, 0.047, 0.049],
    batches=100, inactive=20, particles=5000,
    verify_each=True,
)
for b in bundles:
    print(b.manifest["notes"], b.results["k_eff"])
```

---

## 10. Versioning and compatibility

`nmc_format_version` in `manifest.json` follows semver. Minor version increments
(1.1, 1.2) add optional fields; engines reading 1.x must ignore unknown fields.
Major version increments signal breaking changes; engines must reject bundles with a
higher major version than they support.

The `engine_commit` field is informational only and does not affect compatibility.
It enables auditing: if a bug is found in a specific commit range, all `.nmc` files
produced in that range can be identified and flagged for re-verification.

---

## 11. What is explicitly out of scope (v1.0)

- CAD/DAGMC geometry import (CSG only in v1.0)
- Hex lattice Python API builder (Rust binaries only in v1.0)
- Lattice/universe builder in the visual IDE (flat CSG scenes only in v1.0;
  lattice support is the primary v1.1 target)
- MPI distributed runs (single-node only)
- Results visualisation beyond summary JSON (mesh flux viewer, power maps)
- MCNP input deck import/export

---

## 12. File format summary table

| File | Format | Required | Written by |
|---|---|---|---|
| `manifest.json` | JSON | yes | IDE, CLI |
| `scene.json` | JSON (geometry schema) | yes | IDE, Python, CLI |
| `verify.png` | PNG | after verify | Engine (find_cell_recursive) |
| `verify_xy.png` | PNG | no | Engine (slice render) |
| `verify_xz.png` | PNG | no | Engine (slice render) |
| `results/summary.json` | JSON | after run | Engine |
| `results/keff_history.json` | JSON | after run | Engine |
| `results/entropy_history.json` | JSON | after run | Engine |
| `results/captures_by_cell.json` | JSON | after run | Engine |
| `results/mesh_flux.npy` | NumPy .npy | if mesh tally | Engine |
| `results/surface_currents.json` | JSON | if surface tags | Engine |
| `results/gamma_heating.json` | JSON | if photon transport | Engine |
| `results/statepoint.h5` | HDF5 | if `--statepoint` | Engine |
