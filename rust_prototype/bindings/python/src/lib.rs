// Surface constructor functions use CamelCase to match the Python class
// names the users see (`Sphere`, `ZCylinder`, ...). Rust's snake_case
// lint would rename them; suppress project-wide in this binding crate.
#![allow(non_snake_case)]
// pyo3's `?` propagation of `PyErr` triggers a `useless_conversion`
// false positive on rust 1.94 / clippy 1.94 inside `PyResult<T>` return
// types. The pattern is idiomatic across pyo3 examples, so suppress
// crate-wide here rather than per-call-site.
#![allow(clippy::useless_conversion)]

//! Python bindings for the `open-rust-mc` engine.
//!
//! Exposes a Rust-side builder (`Scene`) so the Python and (eventual)
//! CLI/GUI front-ends share a single validated object model. The
//! engine itself lives in the sibling `open-rust-mc` crate; this file
//! is pure glue.
//!
//! # Design
//!
//! The builder stores symbolic specifications (nuclide name, surface
//! name, region expression) and materialises them into the engine's
//! concrete types at `run_eigenvalue` time. This deferred binding
//! means `add_nuclide("U235.h5", …)` does NOT block on HDF5 I/O — the
//! user can build the whole scene interactively in a REPL and only
//! pay the data-load cost when they press "go".
//!
//! # Region expressions
//!
//! Cells take a string region expression of the form:
//!   - `"-<name>"`   → inside the named surface (negative half-space)
//!   - `"+<name>"`   → outside the named surface (positive half-space)
//!   - `"-a -b +c"`  → intersection of the listed half-spaces
//!
//! This is the OpenMC convention, minus unions. Full boolean regions
//! (unions, complement) are a natural follow-up; for Godiva and PWR
//! pin cells, intersections cover the problem.

use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId, Region};
use open_rust_mc::geometry::lattice::HexOrientation;
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{ray, Aabb, Vec3};
#[cfg(feature = "cuda")]
use open_rust_mc::geometry::Geometry;
use open_rust_mc::photon::bremsstrahlung::MaterialBremss;
use open_rust_mc::photon::electron::{radiation_length_cm, track_integrate_electron_csg_with_ms};
use open_rust_mc::photon::material::PhotonMaterial as RustPhotonMaterial;
use open_rust_mc::photon::transport::transport_history_csg;
use open_rust_mc::photon::PhotonElement;
use std::sync::Arc;

use open_rust_mc::hdf5_reader::NuclideFileReader;
use open_rust_mc::transport::hybrid_xs::{HybridSvdWmpXsProvider, HybridTableWmpXsProvider};
use open_rust_mc::transport::material::Material as RustMaterial;
use open_rust_mc::transport::rng::Rng;
#[cfg(feature = "cuda")]
use open_rust_mc::transport::sim_limits::SimLimits;
use open_rust_mc::transport::simulate::{self, SimConfig};
use open_rust_mc::transport::xs_provider::{self, RankPolicy, SvdXsProvider, TableXsProvider};
use open_rust_mc::wmp::WindowedMultipole;

// ── Surfaces ──────────────────────────────────────────────────────────────

/// Parse a boundary-condition string. Accepts OpenMC-style names.
fn parse_bc(s: &str) -> PyResult<BoundaryCondition> {
    match s {
        "transmission" => Ok(BoundaryCondition::Transmission),
        "reflective" | "reflection" => Ok(BoundaryCondition::Reflective),
        "vacuum" => Ok(BoundaryCondition::Vacuum),
        _ => Err(PyValueError::new_err(format!(
            "unknown boundary condition {s:?}; expected 'transmission', 'reflective', or 'vacuum'"
        ))),
    }
}

/// Opaque wrapper over a Rust `Surface`. Python constructs these via
/// `Sphere(...)`, `ZCylinder(...)`, etc. — the actual variant is
/// resolved inside the wrapper at `Scene.add_surface` time.
#[pyclass(name = "Surface", module = "open_rust_mc._core")]
#[derive(Clone)]
struct PySurface {
    inner: Surface,
}

#[pymethods]
impl PySurface {
    fn __repr__(&self) -> String {
        format!("<Surface: {:?}>", self.inner)
    }
}

#[pyfunction]
#[pyo3(signature = (r, x=0.0, y=0.0, z=0.0, bc="transmission"))]
fn Sphere(r: f64, x: f64, y: f64, z: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::Sphere {
            center: Vec3::new(x, y, z),
            radius: r,
            bc: parse_bc(bc)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (r, x=0.0, y=0.0, bc="transmission"))]
fn ZCylinder(r: f64, x: f64, y: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::CylinderZ {
            center_x: x,
            center_y: y,
            radius: r,
            bc: parse_bc(bc)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (r, y=0.0, z=0.0, bc="transmission"))]
fn XCylinder(r: f64, y: f64, z: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::CylinderX {
            center_y: y,
            center_z: z,
            radius: r,
            bc: parse_bc(bc)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (r, x=0.0, z=0.0, bc="transmission"))]
fn YCylinder(r: f64, x: f64, z: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::CylinderY {
            center_x: x,
            center_z: z,
            radius: r,
            bc: parse_bc(bc)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (x0, bc="transmission"))]
fn XPlane(x0: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::PlaneX {
            x0,
            bc: parse_bc(bc)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (y0, bc="transmission"))]
fn YPlane(y0: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::PlaneY {
            y0,
            bc: parse_bc(bc)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (z0, bc="transmission"))]
fn ZPlane(z0: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::PlaneZ {
            z0,
            bc: parse_bc(bc)?,
        },
    })
}

// ── Material ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct NuclideSpec {
    hdf5_file: String,
    atom_density: f64,
    awr: f64,
    nubar: f64,
    /// Optional S(α,β) thermal-scattering HDF5 file name (e.g.
    /// "c_H_in_H2O.h5"), resolved relative to the scene's data dir.
    thermal_file: Option<String>,
}

#[pyclass(name = "Material", module = "open_rust_mc._core")]
#[derive(Clone)]
struct PyMaterial {
    name: String,
    temperature: f64,
    temp_idx: usize,
    nuclides: Vec<NuclideSpec>,
}

#[pymethods]
impl PyMaterial {
    #[new]
    #[pyo3(signature = (name, temperature=293.15, temp_idx=1))]
    fn new(name: String, temperature: f64, temp_idx: usize) -> Self {
        Self {
            name,
            temperature,
            temp_idx,
            nuclides: Vec::new(),
        }
    }

    /// Append a nuclide via its HDF5 file name, atom density
    /// [atoms/(barn·cm)], atomic weight ratio, and mean nu-bar fallback.
    ///
    /// `thermal_file` attaches an S(α,β) thermal-scattering library
    /// to this nuclide (e.g. `"c_H_in_H2O.h5"` for hydrogen in water).
    /// Pass `None` to use free-gas kinematics only. The file must
    /// live in the scene's data directory.
    ///
    /// Atom density is the absolute macro unit used throughout the
    /// engine (Σ = n · σ in cm⁻¹ when σ is in barn). Compute it from
    /// macro density + stoichiometry on the Python side, or pass an
    /// already-computed value.
    #[pyo3(signature = (hdf5_file, atom_density, awr, nubar=0.0, thermal_file=None))]
    fn add_nuclide(
        mut slf: PyRefMut<'_, Self>,
        hdf5_file: String,
        atom_density: f64,
        awr: f64,
        nubar: f64,
        thermal_file: Option<String>,
    ) -> PyRefMut<'_, Self> {
        slf.nuclides.push(NuclideSpec {
            hdf5_file,
            atom_density,
            awr,
            nubar,
            thermal_file,
        });
        slf
    }

    /// Total atom density of the material, summed over its nuclides.
    fn total_atom_density(&self) -> f64 {
        self.nuclides.iter().map(|n| n.atom_density).sum()
    }

    /// In-place update: set the atom density of the nuclide entry
    /// whose `hdf5_file` matches. Returns `True` if a matching entry
    /// was updated, `False` if no nuclide with that file is in the
    /// material (no error — caller can branch on the bool).
    ///
    /// Use this between burnup steps to push CRAM-evolved number
    /// densities back into the live material before the next
    /// `run_eigenvalue` call.
    fn set_atom_density<'a>(
        mut slf: PyRefMut<'a, Self>,
        hdf5_file: &str,
        atom_density: f64,
    ) -> (PyRefMut<'a, Self>, bool) {
        let mut updated = false;
        for nuc in slf.nuclides.iter_mut() {
            if nuc.hdf5_file == hdf5_file {
                nuc.atom_density = atom_density;
                updated = true;
                break;
            }
        }
        (slf, updated)
    }

    /// Read the atom density of the nuclide with this `hdf5_file`,
    /// or `None` if not in the material.
    fn atom_density_of(&self, hdf5_file: &str) -> Option<f64> {
        self.nuclides
            .iter()
            .find(|n| n.hdf5_file == hdf5_file)
            .map(|n| n.atom_density)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Material {:?} at {:.1} K, {} nuclides>",
            self.name,
            self.temperature,
            self.nuclides.len()
        )
    }
}

// ── PhotonMaterial ────────────────────────────────────────────────────────

/// Per-element entry in a photon material (HDF5 file name + atom density).
#[derive(Clone)]
struct PhotonElementSpec {
    hdf5_file: String,
    atom_density: f64,
}

/// Photon-transport material: a homogeneous element mixture with a
/// mass density. Atom densities are in atoms/(b·cm), same convention
/// as the neutron `Material`. Mass density (g/cm³) is used for the
/// Katz-Penfold CSDA electron-range scaling in the track-integrated
/// electron deposit.
#[pyclass(name = "PhotonMaterial", module = "open_rust_mc._core")]
#[derive(Clone)]
struct PyPhotonMaterial {
    density_g_per_cm3: f64,
    elements: Vec<PhotonElementSpec>,
}

#[pymethods]
impl PyPhotonMaterial {
    #[new]
    #[pyo3(signature = (density_g_per_cm3))]
    fn new(density_g_per_cm3: f64) -> Self {
        Self {
            density_g_per_cm3,
            elements: Vec::new(),
        }
    }

    /// Append an element. `hdf5_file` is the per-element photon data
    /// file name (e.g. `"U.h5"`, `"O.h5"`) located in the scene's
    /// photon data directory. `atom_density` in atoms/(b·cm).
    #[pyo3(signature = (hdf5_file, atom_density))]
    fn add_element(
        mut slf: PyRefMut<'_, Self>,
        hdf5_file: String,
        atom_density: f64,
    ) -> PyRefMut<'_, Self> {
        slf.elements.push(PhotonElementSpec {
            hdf5_file,
            atom_density,
        });
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "<PhotonMaterial density={:.3} g/cm3, {} elements>",
            self.density_g_per_cm3,
            self.elements.len()
        )
    }
}

// ── Settings ──────────────────────────────────────────────────────────────

#[pyclass(name = "Settings", module = "open_rust_mc._core")]
#[derive(Clone)]
struct PySettings {
    #[pyo3(get, set)]
    batches: u32,
    #[pyo3(get, set)]
    inactive: u32,
    #[pyo3(get, set)]
    particles: u32,
    #[pyo3(get, set)]
    seed: u64,
}

#[pymethods]
impl PySettings {
    #[new]
    #[pyo3(signature = (batches=50, inactive=10, particles=5000, seed=1))]
    fn new(batches: u32, inactive: u32, particles: u32, seed: u64) -> Self {
        Self {
            batches,
            inactive,
            particles,
            seed,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<Settings batches={} inactive={} particles={} seed={}>",
            self.batches, self.inactive, self.particles, self.seed
        )
    }
}

// ── Cross-section provider mode (the builder toggle) ─────────────────────

/// Selects which cross-section representation the simulation uses.
///
/// - `Table`: OpenMC-style pointwise tables. Industry baseline. Lowest
///   load time, fastest single-temperature lookup, most memory.
/// - `Svd`: rank-`k` SVD-compressed kernels. Lower memory at high k or
///   multi-temperature, slower load (SVD decomposition). Default rank 5.
/// - `HybridSvdWmp`: SVD outside the resolved-resonance window, exact
///   Windowed-Multipole evaluation inside. Requires WMP HDF5 files in
///   `<data_dir>/../wmp/`. The intended production-precision mode.
/// - `HybridTableWmp`: pointwise tables with WMP override in the
///   resonance window. Industry-baseline accuracy at lower memory.
#[pyclass(eq, eq_int, name = "XsMode", module = "open_rust_mc._core")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PyXsMode {
    Table,
    Svd,
    HybridSvdWmp,
    HybridTableWmp,
}

#[pymethods]
impl PyXsMode {
    fn __repr__(&self) -> String {
        match self {
            PyXsMode::Table => "XsMode.Table",
            PyXsMode::Svd => "XsMode.Svd",
            PyXsMode::HybridSvdWmp => "XsMode.HybridSvdWmp",
            PyXsMode::HybridTableWmp => "XsMode.HybridTableWmp",
        }
        .to_string()
    }

    fn name(&self) -> &'static str {
        match self {
            PyXsMode::Table => "table",
            PyXsMode::Svd => "svd",
            PyXsMode::HybridSvdWmp => "hybrid_svd_wmp",
            PyXsMode::HybridTableWmp => "hybrid_table_wmp",
        }
    }
}

// ── Runner (backend device selector) ─────────────────────────────────────

/// Compute backend the simulation runs on. The CPU path is always
/// available; `GpuCuda` requires the bindings to be built with
/// `--features cuda` (which forwards to `open-rust-mc/cuda`).
///
/// Use `Runner.recommended()` to pick the build's optimal default,
/// or `Scene.set_runner(Runner.Cpu | Runner.GpuCuda)` to force one.
/// Future variants (e.g. ROCm, Metal, distributed cluster runners)
/// will be added as new enum members without breaking the existing
/// names.
#[pyclass(eq, eq_int, name = "Runner", module = "open_rust_mc._core")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PyRunner {
    Cpu,
    GpuCuda,
}

#[pymethods]
impl PyRunner {
    fn __repr__(&self) -> String {
        match self {
            PyRunner::Cpu => "Runner.Cpu",
            PyRunner::GpuCuda => "Runner.GpuCuda",
        }
        .to_string()
    }

    fn name(&self) -> &'static str {
        match self {
            PyRunner::Cpu => "cpu",
            PyRunner::GpuCuda => "gpu_cuda",
        }
    }

    /// Default runner for the current build. Returns `GpuCuda` when the
    /// bindings were built with `--features cuda`, else `Cpu`. Mirrors
    /// `open_rust_mc::transport::dispatch::Backend::recommended`.
    #[staticmethod]
    fn recommended() -> Self {
        #[cfg(feature = "cuda")]
        {
            PyRunner::GpuCuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            PyRunner::Cpu
        }
    }
}

/// Standard 9-nuclide WMP filename convention (ZZAAA.h5) for U-235,
/// U-238, U-234. Other nuclides return `None`. Used by the hybrid
/// modes to discover WMP coverage.
fn wmp_path_for(file: &str) -> Option<&'static str> {
    match file {
        "U234.h5" => Some("092234.h5"),
        "U235.h5" => Some("092235.h5"),
        "U238.h5" => Some("092238.h5"),
        _ => None,
    }
}

// ── Scene (the builder) ───────────────────────────────────────────────────

/// A spec for a cell, resolved at `run_eigenvalue` time.
#[derive(Clone)]
struct CellSpec {
    name: String,
    region_expr: String,
    fill: Option<String>, // material name, or None for void
    temperature: f64,
}

#[pyclass(name = "Scene", module = "open_rust_mc._core")]
struct PyScene {
    data_dir: PathBuf,
    /// Optional photon data directory (per-element HDF5 files). Only
    /// needed when calling `run_gamma_heating`.
    photon_data_dir: Option<PathBuf>,
    materials: HashMap<String, PyMaterial>,
    material_order: Vec<String>,
    surfaces: HashMap<String, PySurface>,
    surface_order: Vec<String>,
    cells: Vec<CellSpec>,
    /// Per-cell photon material, indexed by cell name. `None` means
    /// the cell is void for photons.
    photon_materials: HashMap<String, PyPhotonMaterial>,
    /// Cross-section provider mode. Default Table.
    xs_mode: PyXsMode,
    /// SVD truncation rank used as the default for any reaction MT not
    /// listed in `svd_ranks_per_mt`. Default 5.
    svd_rank: usize,
    /// Per-MT rank overrides. Empty by default — every reaction uses
    /// `svd_rank`. Set via `set_svd_ranks({mt: rank, ...})`.
    svd_ranks_per_mt: HashMap<u32, usize>,
    /// Compute backend. Default `Cpu`. Set via `set_runner`.
    runner: PyRunner,
}

#[pymethods]
impl PyScene {
    #[new]
    #[pyo3(signature = (data_dir))]
    fn new(data_dir: PathBuf) -> PyResult<Self> {
        if !data_dir.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "data_dir does not exist: {}",
                data_dir.display()
            )));
        }
        Ok(Self {
            data_dir,
            photon_data_dir: None,
            materials: HashMap::new(),
            material_order: Vec::new(),
            surfaces: HashMap::new(),
            surface_order: Vec::new(),
            cells: Vec::new(),
            photon_materials: HashMap::new(),
            xs_mode: PyXsMode::Table,
            svd_rank: 5,
            svd_ranks_per_mt: HashMap::new(),
            runner: PyRunner::Cpu,
        })
    }

    /// Select the compute backend. Builder-style, returns `self`.
    ///
    /// `Runner.Cpu` (the default) runs the rayon-parallel CPU
    /// transport. `Runner.GpuCuda` dispatches through `CudaRunner` and
    /// requires the bindings to be built with `--features cuda`. All
    /// four `XsMode` values dispatch on either backend; on GPU the
    /// pure-`Table` modes are uploaded via a rank-1 SVD layout that is
    /// numerically equivalent to the CPU `TableXsProvider`.
    fn set_runner<'a>(
        mut slf: PyRefMut<'a, Self>,
        runner: PyRunner,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.runner = runner;
        Ok(slf)
    }

    /// Read back the currently selected runner.
    #[getter]
    fn runner(&self) -> PyRunner {
        self.runner
    }

    /// Set the cross-section provider mode. Builder-style, returns `self`.
    fn set_xs_mode<'a>(
        mut slf: PyRefMut<'a, Self>,
        mode: PyXsMode,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.xs_mode = mode;
        Ok(slf)
    }

    /// Set the SVD truncation rank used by the Svd and HybridSvdWmp
    /// modes. Builder-style, returns `self`. Default 5. No effect on
    /// Table-mode runs.
    fn set_svd_rank<'a>(mut slf: PyRefMut<'a, Self>, rank: usize) -> PyResult<PyRefMut<'a, Self>> {
        if rank == 0 {
            return Err(PyValueError::new_err("svd_rank must be ≥ 1"));
        }
        slf.svd_rank = rank;
        Ok(slf)
    }

    /// Read back the currently selected XsMode (for asserts in tests).
    #[getter]
    fn xs_mode(&self) -> PyXsMode {
        self.xs_mode
    }

    /// Read back the currently selected SVD rank.
    #[getter]
    fn svd_rank(&self) -> usize {
        self.svd_rank
    }

    /// Set per-reaction SVD rank overrides. The dict keys are MT
    /// numbers (2 elastic, 4 inelastic, 16 n2n, 17 n3n, 18 fission,
    /// 102 capture; discrete-level MTs 51..91 are GPU-stride-locked
    /// to `svd_rank` and ignored here). Reactions not listed fall
    /// back to `svd_rank`. Builder-style; returns `self`.
    ///
    /// Recommended for the production-precision Hybrid SVD+WMP path:
    ///
    /// ```python
    /// scene = (scene
    ///     .set_xs_mode(XsMode.HybridSvdWmp)
    ///     .set_svd_rank(5)                     # default
    ///     .set_svd_ranks({2: 1, 18: 1, 102: 1})  # smooth tails — WMP handles resonance
    /// )
    /// ```
    fn set_svd_ranks<'a>(
        mut slf: PyRefMut<'a, Self>,
        ranks: HashMap<u32, usize>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        for (&mt, &rank) in ranks.iter() {
            if rank == 0 {
                return Err(PyValueError::new_err(format!(
                    "rank for MT={mt} must be ≥ 1, got 0"
                )));
            }
        }
        slf.svd_ranks_per_mt = ranks;
        Ok(slf)
    }

    /// Read back the per-MT rank overrides as a dict.
    #[getter]
    fn svd_ranks_per_mt(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);
        for (&mt, &rank) in &self.svd_ranks_per_mt {
            d.set_item(mt, rank)?;
        }
        Ok(d.into())
    }

    /// Set the photon (per-element) HDF5 data directory, required for
    /// `run_gamma_heating`. Builder-style, returns `self`.
    fn set_photon_data_dir<'a>(
        mut slf: PyRefMut<'a, Self>,
        path: PathBuf,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if !path.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "photon_data_dir does not exist: {}",
                path.display()
            )));
        }
        slf.photon_data_dir = Some(path);
        Ok(slf)
    }

    /// Attach a `PhotonMaterial` to a previously-registered cell by
    /// name. Cells without a photon material are treated as void for
    /// photon transport.
    fn add_photon_material<'a>(
        mut slf: PyRefMut<'a, Self>,
        cell_name: String,
        material: PyPhotonMaterial,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.photon_materials.insert(cell_name, material);
        Ok(slf)
    }

    fn add_material<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        material: PyMaterial,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if slf.materials.contains_key(&name) {
            return Err(PyValueError::new_err(format!(
                "material {name:?} already registered"
            )));
        }
        slf.material_order.push(name.clone());
        slf.materials.insert(name, material);
        Ok(slf)
    }

    fn add_surface<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        surface: PySurface,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if slf.surfaces.contains_key(&name) {
            return Err(PyValueError::new_err(format!(
                "surface {name:?} already registered"
            )));
        }
        slf.surface_order.push(name.clone());
        slf.surfaces.insert(name, surface);
        Ok(slf)
    }

    /// Register a cell by name. `region` is a boolean-AND expression
    /// over surfaces, using `-name` for the negative (inside) half-space
    /// and `+name` for the positive (outside). `fill` is a material
    /// name, or `None` for a void cell.
    #[pyo3(signature = (name, region, fill=None, temperature=293.15))]
    fn add_cell<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        region: String,
        fill: Option<String>,
        temperature: f64,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.cells.push(CellSpec {
            name,
            region_expr: region,
            fill,
            temperature,
        });
        Ok(slf)
    }

    /// Register the 6 axis-aligned planes of an axis-aligned box
    /// centred at the origin with half-extents `half = [hx, hy, hz]`.
    /// All 6 planes carry `bc` (typically "reflective" or "vacuum").
    /// Surfaces are named `{prefix}_xmin`, `{prefix}_xmax`, etc.
    /// Returns the region expression for "inside the box" suitable
    /// for `add_cell(..., region=...)`.
    #[pyo3(signature = (prefix, half, bc="reflective"))]
    fn add_rect_box<'a>(
        mut slf: PyRefMut<'a, Self>,
        prefix: &str,
        half: [f64; 3],
        bc: &str,
    ) -> PyResult<(PyRefMut<'a, Self>, String)> {
        let bc_parsed = parse_bc(bc)?;
        let names = [
            format!("{prefix}_xmin"),
            format!("{prefix}_xmax"),
            format!("{prefix}_ymin"),
            format!("{prefix}_ymax"),
            format!("{prefix}_zmin"),
            format!("{prefix}_zmax"),
        ];
        let surfaces = [
            Surface::PlaneX {
                x0: -half[0],
                bc: bc_parsed,
            },
            Surface::PlaneX {
                x0: half[0],
                bc: bc_parsed,
            },
            Surface::PlaneY {
                y0: -half[1],
                bc: bc_parsed,
            },
            Surface::PlaneY {
                y0: half[1],
                bc: bc_parsed,
            },
            Surface::PlaneZ {
                z0: -half[2],
                bc: bc_parsed,
            },
            Surface::PlaneZ {
                z0: half[2],
                bc: bc_parsed,
            },
        ];
        for (name, surface) in names.iter().zip(surfaces.iter()) {
            if slf.surfaces.contains_key(name) {
                return Err(PyValueError::new_err(format!(
                    "surface {name:?} already registered"
                )));
            }
            slf.surface_order.push(name.clone());
            slf.surfaces.insert(
                name.clone(),
                PySurface {
                    inner: surface.clone(),
                },
            );
        }
        // Inside region: above xmin, below xmax, etc.
        let region = format!(
            "+{}_xmin & -{}_xmax & +{}_ymin & -{}_ymax & +{}_zmin & -{}_zmax",
            prefix, prefix, prefix, prefix, prefix, prefix
        );
        Ok((slf, region))
    }

    /// Register the 6 reflective hex-side planes plus 2 z planes for
    /// a hex-shaped 3D region. The hex outer-boundary inradius is
    /// `(rings + 0.5) * pitch` — sized to enclose an `n_rings`
    /// tessellation of `pitch`-spaced hex elements.
    ///
    /// `orientation`: "flat" (flat-top, side midpoints at 30° steps
    /// from +x) or "pointy" (vertex-up, midpoints at 0° steps).
    /// Surfaces are named `{prefix}_side0`..`{prefix}_side5`,
    /// `{prefix}_zmin`, `{prefix}_zmax`. Returns the
    /// "inside the hex" region expression.
    #[pyo3(signature = (prefix, rings, pitch, orientation="flat", xy_bc="reflective", z_half=10.0, z_bc="reflective"))]
    fn add_hex_boundary<'a>(
        mut slf: PyRefMut<'a, Self>,
        prefix: &str,
        rings: usize,
        pitch: f64,
        orientation: &str,
        xy_bc: &str,
        z_half: f64,
        z_bc: &str,
    ) -> PyResult<(PyRefMut<'a, Self>, String)> {
        let xy_bc = parse_bc(xy_bc)?;
        let z_bc = parse_bc(z_bc)?;
        let orient = match orientation {
            "flat" | "Y" | "y" => HexOrientation::Y,
            "pointy" | "X" | "x" => HexOrientation::X,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown hex orientation {orientation:?}; expected 'flat' or 'pointy'"
                )));
            }
        };
        let inradius = (rings as f64 + 0.5) * pitch;
        let normals = open_rust_mc::geometry::shapes::hex_side_normals(orient);
        let mut names: Vec<String> = (0..6).map(|i| format!("{prefix}_side{i}")).collect();
        names.push(format!("{prefix}_zmin"));
        names.push(format!("{prefix}_zmax"));
        let mut surfaces: Vec<Surface> = normals
            .iter()
            .map(|&n| Surface::Plane {
                normal: n,
                offset: inradius,
                bc: xy_bc,
            })
            .collect();
        surfaces.push(Surface::PlaneZ {
            z0: -z_half,
            bc: z_bc,
        });
        surfaces.push(Surface::PlaneZ {
            z0: z_half,
            bc: z_bc,
        });
        for (name, surface) in names.iter().zip(surfaces.iter()) {
            if slf.surfaces.contains_key(name) {
                return Err(PyValueError::new_err(format!(
                    "surface {name:?} already registered"
                )));
            }
            slf.surface_order.push(name.clone());
            slf.surfaces.insert(
                name.clone(),
                PySurface {
                    inner: surface.clone(),
                },
            );
        }
        // Inside region: below all 6 hex sides (-side*) plus z bounds.
        let mut parts: Vec<String> = (0..6).map(|i| format!("-{prefix}_side{i}")).collect();
        parts.push(format!("+{prefix}_zmin"));
        parts.push(format!("-{prefix}_zmax"));
        Ok((slf, parts.join(" & ")))
    }

    /// Register N concentric Z cylinders centred at `(center_x,
    /// center_y)` with the given radii (must be sorted ascending).
    /// All cylinders carry `bc="transmission"`. Surfaces are named
    /// `{prefix}_r{i}` (i = 0..N). Returns the list of registered
    /// names so the caller can build pin region expressions like
    /// `-{prefix}_r0` (inside fuel) / `-{prefix}_r1 & +{prefix}_r0`
    /// (annular gap) etc.
    #[pyo3(signature = (prefix, radii, center_x=0.0, center_y=0.0))]
    fn add_pin_cylinders<'a>(
        mut slf: PyRefMut<'a, Self>,
        prefix: &str,
        radii: Vec<f64>,
        center_x: f64,
        center_y: f64,
    ) -> PyResult<(PyRefMut<'a, Self>, Vec<String>)> {
        let mut names = Vec::with_capacity(radii.len());
        for (i, &r) in radii.iter().enumerate() {
            let name = format!("{prefix}_r{i}");
            if slf.surfaces.contains_key(&name) {
                return Err(PyValueError::new_err(format!(
                    "surface {name:?} already registered"
                )));
            }
            slf.surface_order.push(name.clone());
            slf.surfaces.insert(
                name.clone(),
                PySurface {
                    inner: Surface::CylinderZ {
                        center_x,
                        center_y,
                        radius: r,
                        bc: BoundaryCondition::Transmission,
                    },
                },
            );
            names.push(name);
        }
        Ok((slf, names))
    }

    fn __repr__(&self) -> String {
        format!(
            "<Scene data_dir={} materials={} surfaces={} cells={}>",
            self.data_dir.display(),
            self.materials.len(),
            self.surfaces.len(),
            self.cells.len()
        )
    }
}

// ── Run eigenvalue ────────────────────────────────────────────────────────

#[pyclass(name = "EigenvalueResult", module = "open_rust_mc._core")]
struct PyEigenvalueResult {
    #[pyo3(get)]
    k_eff: f64,
    #[pyo3(get)]
    k_sigma: f64,
    #[pyo3(get)]
    active_batches: usize,
    #[pyo3(get)]
    total_histories: u64,
    #[pyo3(get)]
    runtime_seconds: f64,
    /// Per-batch k_eff values, one per batch (both inactive and
    /// active). Use to plot convergence or compute custom statistics.
    #[pyo3(get)]
    k_per_batch: Vec<f64>,
    /// Per-batch Shannon entropy of the fission-source bank.
    #[pyo3(get)]
    entropy_per_batch: Vec<f64>,
    /// Mask parallel to `k_per_batch` — `true` for active batches.
    #[pyo3(get)]
    active_mask: Vec<bool>,
    /// Per-cell capture counts summed across all active batches.
    /// Indexed by the order cells were registered via
    /// `Scene.add_cell`. Use as a capture-rate tally proxy.
    #[pyo3(get)]
    captures_by_cell: Vec<f64>,
    /// Names of cells, in the same order as `captures_by_cell`.
    #[pyo3(get)]
    cell_names: Vec<String>,
    /// Total collisions across all active batches (diagnostic).
    #[pyo3(get)]
    total_collisions: u64,
    /// Total fissions across all active batches.
    #[pyo3(get)]
    total_fissions: u64,
    /// Total leakages (particles escaping through vacuum BC).
    #[pyo3(get)]
    total_leakage: u64,
    /// Cross-section mode actually used by the run, as a string
    /// ("table", "svd", "hybrid_svd_wmp", "hybrid_table_wmp").
    #[pyo3(get)]
    mode_used: String,
    /// SVD rank actually used (0 for Table mode).
    #[pyo3(get)]
    svd_rank: usize,
    /// Wall time spent loading the cross-section data (HDF5 reads + SVD
    /// decomposition + WMP loads), in seconds. Excludes simulation.
    #[pyo3(get)]
    load_time_seconds: f64,
    /// Wall time spent in `simulate::run_eigenvalue` (transport loop
    /// only — no IO, no provider construction). In seconds.
    #[pyo3(get)]
    sim_time_seconds: f64,
    /// In-solver memory footprint of the cross-section provider (bytes).
    /// Hybrid modes report current scaffolding; tables report packed XS
    /// arrays; SVD reports basis + coefficient + grid bytes.
    #[pyo3(get)]
    xs_memory_bytes: u64,
    /// Number of WMP-covered nuclides (0 in Table/Svd modes).
    #[pyo3(get)]
    wmp_covered_nuclides: usize,
    /// Per-MT rank overrides effectively used by the run, captured from
    /// the scene at run start. Empty `{}` means uniform `svd_rank`.
    /// Same `{mt: rank}` shape as `Scene.set_svd_ranks`.
    #[pyo3(get)]
    svd_ranks_per_mt: HashMap<u32, usize>,
}

#[pymethods]
impl PyEigenvalueResult {
    fn __repr__(&self) -> String {
        format!(
            "<EigenvalueResult k_eff={:.5} ± {:.5} from {} active batches>",
            self.k_eff, self.k_sigma, self.active_batches
        )
    }

    /// Captures by cell as a `{name: count}` dictionary.
    fn captures_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);
        for (n, c) in self.cell_names.iter().zip(self.captures_by_cell.iter()) {
            d.set_item(n, *c)?;
        }
        Ok(d.into())
    }

    /// Return a dictionary with the run's full diagnostic statistics:
    /// k_eff, sigma, batch counts, timing, memory, mode metadata. Useful
    /// in notebooks for one-line debug printing or for serialising a run
    /// to JSON.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("mode", &self.mode_used)?;
        d.set_item("svd_rank", self.svd_rank)?;
        d.set_item("k_eff", self.k_eff)?;
        d.set_item("k_sigma", self.k_sigma)?;
        d.set_item("active_batches", self.active_batches)?;
        d.set_item("total_histories", self.total_histories)?;
        d.set_item("load_time_seconds", self.load_time_seconds)?;
        d.set_item("sim_time_seconds", self.sim_time_seconds)?;
        d.set_item("runtime_seconds", self.runtime_seconds)?;
        d.set_item("xs_memory_bytes", self.xs_memory_bytes)?;
        d.set_item(
            "xs_memory_mib",
            self.xs_memory_bytes as f64 / (1024.0 * 1024.0),
        )?;
        d.set_item("wmp_covered_nuclides", self.wmp_covered_nuclides)?;
        let ranks = pyo3::types::PyDict::new_bound(py);
        for (&mt, &r) in &self.svd_ranks_per_mt {
            ranks.set_item(mt, r)?;
        }
        d.set_item("svd_ranks_per_mt", ranks)?;
        d.set_item("total_collisions", self.total_collisions)?;
        d.set_item("total_fissions", self.total_fissions)?;
        d.set_item("total_leakage", self.total_leakage)?;
        let ns_per_history = if self.total_histories > 0 {
            self.sim_time_seconds * 1e9 / self.total_histories as f64
        } else {
            0.0
        };
        d.set_item("ns_per_history", ns_per_history)?;
        Ok(d.into())
    }
}

// ── Gamma-heating pipeline ────────────────────────────────────────────────

#[pyclass(name = "GammaHeatingResult", module = "open_rust_mc._core")]
struct PyGammaHeatingResult {
    #[pyo3(get)]
    k_eff: f64,
    #[pyo3(get)]
    k_sigma: f64,
    /// Per-cell γ-deposition fractions (sum over cells = 1 − escaped −
    /// orphan). Order matches `cell_names`.
    #[pyo3(get)]
    deposition_fraction: Vec<f64>,
    /// Per-cell absolute γ deposition in eV (sum over all photon
    /// histories).
    #[pyo3(get)]
    deposition_ev: Vec<f64>,
    #[pyo3(get)]
    cell_names: Vec<String>,
    /// Total photon source energy summed over all photon histories (eV).
    #[pyo3(get)]
    total_source_energy_ev: f64,
    /// Energy that escaped through vacuum BCs (eV).
    #[pyo3(get)]
    escaped_energy_ev: f64,
    /// Energy whose deposit position didn't resolve to any cell (eV).
    #[pyo3(get)]
    orphan_energy_ev: f64,
    /// Number of bremsstrahlung photons emitted during photon transport.
    #[pyo3(get)]
    brems_photons_emitted: u64,
    /// Total energy carried by emitted brems photons (eV).
    #[pyo3(get)]
    brems_energy_ev: f64,
    #[pyo3(get)]
    neutron_runtime_seconds: f64,
    #[pyo3(get)]
    photon_runtime_seconds: f64,
    #[pyo3(get)]
    photon_events: u64,
}

#[pymethods]
impl PyGammaHeatingResult {
    fn __repr__(&self) -> String {
        let fracs: Vec<String> = self
            .cell_names
            .iter()
            .zip(self.deposition_fraction.iter())
            .map(|(n, f)| format!("{n}={:.2}%", 100.0 * f))
            .collect();
        format!(
            "<GammaHeatingResult k={:.5}, [{}]>",
            self.k_eff,
            fracs.join(" ")
        )
    }

    /// Deposition fractions as a `{cell_name: fraction}` dictionary.
    fn fractions_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);
        for (n, f) in self.cell_names.iter().zip(self.deposition_fraction.iter()) {
            d.set_item(n, *f)?;
        }
        Ok(d.into())
    }
}

/// Run the coupled neutron-photon pipeline:
/// 1. neutron k-eigenvalue, collecting photon source events at each
///    capture / fission / inelastic collision;
/// 2. photon transport with track-integrated CSDA electrons, Highland
///    multiple scattering, Seltzer-Berger bremsstrahlung, Bethe-Bloch-
///    style non-uniform dE/dx — all enabled by default in the engine.
///
/// Returns per-cell γ-deposition fractions, escaped energy, and
/// diagnostics.
#[pyfunction]
#[pyo3(signature = (scene, neutron_settings, n_photon_histories=200_000, photon_energy_cutoff_ev=1000.0, photon_seed_base=0xB0F1_0000))]
fn run_gamma_heating(
    py: Python<'_>,
    scene: &PyScene,
    neutron_settings: &PySettings,
    n_photon_histories: usize,
    photon_energy_cutoff_ev: f64,
    photon_seed_base: u64,
) -> PyResult<PyGammaHeatingResult> {
    // ── Preflight: photon data dir must be set ─────────────────────
    let photon_dir = scene.photon_data_dir.as_ref().ok_or_else(|| {
        PyValueError::new_err(
            "run_gamma_heating requires Scene.set_photon_data_dir(path) to point at \
             the per-element photon HDF5 directory (e.g. 'photon/' in the ENDF release)",
        )
    })?;

    // ── Phase 0: same geometry + neutron-side build as run_eigenvalue ──
    // Collect nuclides and XS tables.
    #[derive(PartialEq, Clone)]
    struct NuclideKey {
        hdf5_file: String,
        temp_idx: usize,
    }
    let mut nuclide_keys: Vec<NuclideKey> = Vec::new();
    let mut nuclide_specs: Vec<(String, f64, f64, usize)> = Vec::new();
    for mat_name in &scene.material_order {
        let mat = &scene.materials[mat_name];
        for n in &mat.nuclides {
            let key = NuclideKey {
                hdf5_file: n.hdf5_file.clone(),
                temp_idx: mat.temp_idx,
            };
            if !nuclide_keys.contains(&key) {
                nuclide_keys.push(key.clone());
                nuclide_specs.push((n.hdf5_file.clone(), n.awr, n.nubar, mat.temp_idx));
            }
        }
    }
    let mut tables: Vec<std::sync::Arc<xs_provider::NuclideTableData>> =
        Vec::with_capacity(nuclide_specs.len());
    for (file, awr, nubar, temp_idx) in &nuclide_specs {
        let path = scene.data_dir.join(file);
        if !path.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "missing nuclide file: {}",
                path.display()
            )));
        }
        tables.push(std::sync::Arc::new(xs_provider::load_nuclide_table(
            &path, *temp_idx, *awr, *nubar,
        )));
    }
    // Thermal scattering (same logic as run_eigenvalue).
    let mut thermal_files: Vec<Option<String>> = vec![None; nuclide_specs.len()];
    for mat_name in &scene.material_order {
        let mat = &scene.materials[mat_name];
        for n in &mat.nuclides {
            if let Some(tf) = &n.thermal_file {
                let key = NuclideKey {
                    hdf5_file: n.hdf5_file.clone(),
                    temp_idx: mat.temp_idx,
                };
                let idx = nuclide_keys.iter().position(|k| k == &key).unwrap();
                thermal_files[idx] = Some(tf.clone());
            }
        }
    }
    let mut thermal: Vec<Option<std::sync::Arc<open_rust_mc::thermal::ThermalScatteringData>>> =
        vec![None; nuclide_specs.len()];
    for (i, tf) in thermal_files.iter().enumerate() {
        if let Some(tf) = tf {
            let path = scene.data_dir.join(tf);
            if let Ok(tsl) = open_rust_mc::hdf5_reader::load_thermal_scattering(&path) {
                thermal[i] = Some(std::sync::Arc::new(tsl));
            }
        }
    }
    let xs = TableXsProvider {
        nuclides: tables,
        thermal,
    };

    // Surfaces, materials, cells.
    let mut surf_idx: HashMap<String, usize> = HashMap::new();
    let mut surfaces_rt: Vec<Surface> = Vec::with_capacity(scene.surface_order.len());
    for (i, name) in scene.surface_order.iter().enumerate() {
        surf_idx.insert(name.clone(), i);
        surfaces_rt.push(scene.surfaces[name].inner.clone());
    }
    let mut material_idx: HashMap<String, usize> = HashMap::new();
    let mut materials_rt: Vec<RustMaterial> = Vec::with_capacity(scene.material_order.len());
    for (i, name) in scene.material_order.iter().enumerate() {
        let mat = &scene.materials[name];
        let mut m = RustMaterial::new(&mat.name, mat.temperature);
        for n in &mat.nuclides {
            let key = NuclideKey {
                hdf5_file: n.hdf5_file.clone(),
                temp_idx: mat.temp_idx,
            };
            let nuc_idx = nuclide_keys.iter().position(|k| k == &key).unwrap();
            m.add_nuclide(n.atom_density, nuc_idx);
        }
        material_idx.insert(name.clone(), i);
        materials_rt.push(m);
    }
    let mut cells_rt: Vec<Cell> = Vec::with_capacity(scene.cells.len());
    for (i, c) in scene.cells.iter().enumerate() {
        let region = build_region(&c.region_expr, &surf_idx)?;
        let aabb = aabb_from_region(&c.region_expr, &surfaces_rt, &surf_idx);
        let fill = match &c.fill {
            None => CellFill::Void,
            Some(mat_name) => {
                let idx = material_idx.get(mat_name).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "cell {:?} references unknown material {:?}",
                        c.name, mat_name
                    ))
                })?;
                CellFill::Material(*idx as u32)
            }
        };
        cells_rt.push(
            Cell::new(CellId(i as u32), region, fill)
                .with_aabb(aabb)
                .with_temperature(c.temperature),
        );
    }

    // ── Phase 1: neutron transport ─────────────────────────────────
    let config = SimConfig {
        batches: neutron_settings.batches,
        inactive: neutron_settings.inactive,
        particles_per_batch: neutron_settings.particles,
        seed: neutron_settings.seed,
        auto_inactive: None,
        verbose: false,
        parallel: true,
        tallies: Default::default(),
        statepoint_path: None,
        survival_biasing: None,
        initial_source_bank: None,
        weight_window: None,
        disable_delayed_neutrons: false,
        urr_equivalence: None,
    };
    let t_neu = std::time::Instant::now();
    let (batch_results, k_running) = py.allow_threads(|| {
        simulate::run_eigenvalue(&config, &surfaces_rt, &cells_rt, &materials_rt, &xs)
    });
    let neutron_runtime = t_neu.elapsed().as_secs_f64();

    // Gather photon source bank from active batches.
    let mut photon_bank: Vec<simulate::PhotonSourceEvent> = Vec::new();
    for br in &batch_results {
        if br.active {
            photon_bank.extend(br.photon_events.iter().copied());
        }
    }
    if photon_bank.is_empty() {
        return Err(PyValueError::new_err(
            "no photon source events produced — make sure the XS provider carries \
             photon product data (neutron HDF5 files must have `/reaction_MT/product_photon` groups)",
        ));
    }
    let active: Vec<f64> = batch_results
        .iter()
        .filter(|b| b.active)
        .map(|b| b.k_eff)
        .collect();
    let n_active = active.len().max(1);
    let k_mean: f64 = active.iter().sum::<f64>() / n_active as f64;
    let k_var = if n_active > 1 {
        active.iter().map(|k| (k - k_mean).powi(2)).sum::<f64>() / (n_active - 1) as f64
    } else {
        0.0
    };
    let k_sigma = (k_var / n_active as f64).sqrt();
    let _ = k_running;

    // ── Phase 2: build photon-side materials and run photon transport ──
    // Per-cell PhotonMaterial + per-cell X₀ + per-cell brems sampler.
    // Build `materials_p` indexed by MATERIAL id (same as `materials_rt`),
    // which is what the engine expects — `CellFill::Material(m)` maps
    // to `materials_p[m]`. Photon materials from the Python side are
    // attached by cell name, so we resolve each `cell_name ->
    // material_name -> material_idx`. Two cells sharing a material
    // must agree on the photon material.
    let mut materials_p: Vec<Option<RustPhotonMaterial>> =
        (0..scene.material_order.len()).map(|_| None).collect();
    for (cell_name, py_pm) in &scene.photon_materials {
        let cell_spec = scene
            .cells
            .iter()
            .find(|c| &c.name == cell_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "photon material attached to unknown cell {:?}",
                    cell_name
                ))
            })?;
        let mat_name = cell_spec.fill.as_ref().ok_or_else(|| {
            PyValueError::new_err(format!(
                "cannot attach photon material to void cell {:?}",
                cell_name
            ))
        })?;
        let mat_idx = *material_idx.get(mat_name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "cell {:?} references unknown material {:?}",
                cell_name, mat_name
            ))
        })?;
        let mut entries = Vec::with_capacity(py_pm.elements.len());
        for e in &py_pm.elements {
            let path = photon_dir.join(&e.hdf5_file);
            if !path.exists() {
                return Err(PyFileNotFoundError::new_err(format!(
                    "missing photon element file: {}",
                    path.display()
                )));
            }
            let elem = PhotonElement::from_hdf5(&path).map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to load photon element {}: {err}",
                    path.display()
                ))
            })?;
            entries.push((e.atom_density, elem));
        }
        if materials_p[mat_idx].is_some() {
            // Silently keep the first one — consistent with "two cells
            // sharing a material agree on the photon material". We
            // could check for equality but Rust's PhotonMaterial isn't
            // PartialEq.
            continue;
        }
        materials_p[mat_idx] =
            Some(RustPhotonMaterial::new(entries).with_density(py_pm.density_g_per_cm3));
    }
    // Brems samplers are also per-material.
    let brems_p: Vec<Option<MaterialBremss>> = materials_p
        .iter()
        .map(|m| m.as_ref().map(MaterialBremss::from_photon_material))
        .collect();
    // X₀ is per-cell — indirect through the cell's material id. Cells
    // with void fill get ∞ (no MS).
    let x0_per_cell: Vec<f64> = cells_rt
        .iter()
        .map(|c| match c.fill {
            CellFill::Material(m) => materials_p
                .get(m as usize)
                .and_then(|o| o.as_ref())
                .map(radiation_length_cm)
                .unwrap_or(f64::INFINITY),
            _ => f64::INFINITY,
        })
        .collect();

    // Photon transport loop (mirrors pwr_gamma_heating.rs).
    let t_ph = std::time::Instant::now();
    let mut deposited_per_cell = vec![0.0_f64; cells_rt.len()];
    let mut escaped_energy = 0.0_f64;
    let mut orphan_deposit = 0.0_f64;
    let mut total_source_energy = 0.0_f64;
    let mut brems_photons_emitted = 0_u64;
    let mut brems_energy_emitted = 0.0_f64;

    py.allow_threads(|| {
        for i in 0..n_photon_histories {
            let mut rng = Rng::new(photon_seed_base.wrapping_add(i as u64), 1);
            let ev_idx = (rng.uniform() * photon_bank.len() as f64) as usize;
            let ev = photon_bank[ev_idx.min(photon_bank.len() - 1)];
            let pos = Vec3::new(ev.pos[0], ev.pos[1], ev.pos[2]);
            let (dx, dy, dz) = rng.isotropic_direction();
            let e_src = ev.energy;
            total_source_energy += e_src;

            let mut photon_bank_local: Vec<(Vec3, Vec3, f64, usize)> =
                vec![(pos, Vec3::new(dx, dy, dz), e_src, 0)];

            while let Some((p0, d0, e0, _c0)) = photon_bank_local.pop() {
                let r = transport_history_csg(
                    p0,
                    d0,
                    e0,
                    &surfaces_rt,
                    &cells_rt,
                    &materials_p,
                    photon_energy_cutoff_ev,
                    &mut rng,
                );

                escaped_energy += r.energy_escaped;
                for (p, e) in &r.deposits {
                    if let Some(idx) = ray::find_cell(*p, &surfaces_rt, &cells_rt) {
                        deposited_per_cell[idx] += e;
                    } else {
                        orphan_deposit += e;
                    }
                }
                for ele in &r.electrons {
                    let mut csda_energy = ele.e_kin_ev;
                    // Look up the cell's material id first, then fetch
                    // brems/photon materials for that material — not
                    // the cell index directly. materials_p is indexed
                    // by material id (same as the engine's convention).
                    let mat_id = match cells_rt.get(ele.cell_idx).map(|c| &c.fill) {
                        Some(CellFill::Material(m)) => *m as usize,
                        _ => usize::MAX,
                    };
                    if let (Some(Some(bs)), Some(Some(_mat))) =
                        (brems_p.get(mat_id), materials_p.get(mat_id))
                    {
                        let p_brems = bs.radiative_yield_approx(ele.e_kin_ev);
                        if p_brems > 0.0 && rng.uniform() < p_brems {
                            if let Some(e_gamma) = bs.sample_photon_energy(ele.e_kin_ev, &mut rng) {
                                let (gx, gy, gz) = rng.isotropic_direction();
                                photon_bank_local.push((
                                    ele.pos,
                                    Vec3::new(gx, gy, gz),
                                    e_gamma,
                                    ele.cell_idx,
                                ));
                                brems_photons_emitted += 1;
                                brems_energy_emitted += e_gamma;
                                csda_energy -= e_gamma;
                                if csda_energy < 0.0 {
                                    csda_energy = 0.0;
                                }
                            }
                        }
                    }
                    track_integrate_electron_csg_with_ms(
                        ele.pos,
                        ele.dir,
                        csda_energy,
                        ele.cell_idx,
                        &surfaces_rt,
                        &cells_rt,
                        &materials_p,
                        &x0_per_cell,
                        0.005,
                        &mut rng,
                        &mut deposited_per_cell,
                    );
                }
            }
        }
    });
    let photon_runtime = t_ph.elapsed().as_secs_f64();

    let cell_names: Vec<String> = scene.cells.iter().map(|c| c.name.clone()).collect();
    let deposition_fraction: Vec<f64> = deposited_per_cell
        .iter()
        .map(|d| {
            if total_source_energy > 0.0 {
                d / total_source_energy
            } else {
                0.0
            }
        })
        .collect();

    Ok(PyGammaHeatingResult {
        k_eff: k_mean,
        k_sigma,
        deposition_fraction,
        deposition_ev: deposited_per_cell,
        cell_names,
        total_source_energy_ev: total_source_energy,
        escaped_energy_ev: escaped_energy,
        orphan_energy_ev: orphan_deposit,
        brems_photons_emitted,
        brems_energy_ev: brems_energy_emitted,
        neutron_runtime_seconds: neutron_runtime,
        photon_runtime_seconds: photon_runtime,
        photon_events: photon_bank.len() as u64,
    })
}

/// AABB of a surface's half-space, given the sign.
///
/// - `+name` (positive half-space): for axis-aligned planes, the
///   open half above the plane; for closed quadrics (sphere, cylinder,
///   cone), the complement of the surface's own bounding box which
///   we can't represent tightly — fall back to infinite.
/// - `-name` (negative half-space): for closed quadrics, the surface's
///   bounding box; for planes, the open half below the plane.
fn half_space_aabb(surface: &Surface, positive: bool) -> Aabb {
    let inf = f64::INFINITY;
    let neg_inf = f64::NEG_INFINITY;
    match surface {
        Surface::PlaneX { x0, .. } => {
            if positive {
                Aabb::new(Vec3::new(*x0, neg_inf, neg_inf), Vec3::new(inf, inf, inf))
            } else {
                Aabb::new(
                    Vec3::new(neg_inf, neg_inf, neg_inf),
                    Vec3::new(*x0, inf, inf),
                )
            }
        }
        Surface::PlaneY { y0, .. } => {
            if positive {
                Aabb::new(Vec3::new(neg_inf, *y0, neg_inf), Vec3::new(inf, inf, inf))
            } else {
                Aabb::new(
                    Vec3::new(neg_inf, neg_inf, neg_inf),
                    Vec3::new(inf, *y0, inf),
                )
            }
        }
        Surface::PlaneZ { z0, .. } => {
            if positive {
                Aabb::new(Vec3::new(neg_inf, neg_inf, *z0), Vec3::new(inf, inf, inf))
            } else {
                Aabb::new(
                    Vec3::new(neg_inf, neg_inf, neg_inf),
                    Vec3::new(inf, inf, *z0),
                )
            }
        }
        // Oblique plane, or non-axis-aligned: we don't have a tight
        // half-space AABB; use the surface's own AABB for `-` (which
        // for closed quadrics is their bounding sphere/cylinder box)
        // and infinity for `+` (open outside).
        _ => {
            if positive {
                Aabb::INFINITE
            } else {
                surface.aabb()
            }
        }
    }
}

/// AABB for a cell derived from its region expression.
///
/// Each OR-group contributes its own intersected half-space AABBs;
/// the whole cell's AABB encloses every group's box. Both `+name`
/// and `-name` tokens contribute (via `half_space_aabb`), so
/// axis-aligned plane-bounded cells like a PWR pin cell get finite
/// AABBs on every axis. "~name" and unknown names are skipped.
fn aabb_from_region(expr: &str, surfaces: &[Surface], surf_idx: &HashMap<String, usize>) -> Aabb {
    let mut group_boxes: Vec<Aabb> = Vec::new();
    for group in expr.split('|') {
        let mut acc: Option<Aabb> = None;
        for token in group.split_whitespace() {
            if token.starts_with('~') {
                continue; // complement — hard to derive AABB
            }
            let (positive, name) = match token.chars().next() {
                Some('-') => (false, &token[1..]),
                Some('+') => (true, &token[1..]),
                _ => continue,
            };
            let Some(&i) = surf_idx.get(name) else {
                continue;
            };
            let a = half_space_aabb(&surfaces[i], positive);
            acc = Some(match acc {
                None => a,
                Some(b) => Aabb::new(
                    Vec3::new(
                        b.min.x.max(a.min.x),
                        b.min.y.max(a.min.y),
                        b.min.z.max(a.min.z),
                    ),
                    Vec3::new(
                        b.max.x.min(a.max.x),
                        b.max.y.min(a.max.y),
                        b.max.z.min(a.max.z),
                    ),
                ),
            });
        }
        if let Some(b) = acc {
            group_boxes.push(b);
        }
    }
    if group_boxes.is_empty() {
        return Aabb::INFINITE;
    }
    group_boxes
        .into_iter()
        .reduce(|a, b| {
            Aabb::new(
                Vec3::new(
                    a.min.x.min(b.min.x),
                    a.min.y.min(b.min.y),
                    a.min.z.min(b.min.z),
                ),
                Vec3::new(
                    a.max.x.max(b.max.x),
                    a.max.y.max(b.max.y),
                    a.max.z.max(b.max.z),
                ),
            )
        })
        .unwrap()
}

/// Build a `Region` from a string expression.
///
/// Grammar (no nested parentheses, adequate for reactor benchmarks):
/// - `-name`    — inside the named surface (negative half-space)
/// - `+name`    — outside the named surface (positive half-space)
/// - `~-name`   — complement of inside, i.e. NOT(inside) — equivalent
///   to `+name` for a simple half-space, but the `~` form composes
///   through unions/intersections so `~(-a -b)` means "not in both"
/// - whitespace-separated tokens within a group are AND'd
/// - `|` at the top level splits OR-groups; each OR-group is an AND
///   of its tokens, and the overall region is the union of groups
///
/// Examples:
/// - `"-fuel_or"`                       fuel disc
/// - `"+fuel_or -clad_ir"`              annular gap
/// - `"-inner | -other"`                union of two regions
/// - `"~-a ~-b"`                        outside a AND outside b
fn build_region(expr: &str, surf_idx: &HashMap<String, usize>) -> PyResult<Region> {
    let or_groups: Vec<&str> = expr
        .split('|')
        .map(|g| g.trim())
        .filter(|g| !g.is_empty())
        .collect();
    if or_groups.is_empty() {
        return Err(PyValueError::new_err("region expression is empty"));
    }

    let mut group_regions: Vec<Region> = Vec::with_capacity(or_groups.len());
    for group in or_groups {
        let mut and_terms: Vec<Region> = Vec::new();
        for token in group.split_whitespace() {
            // Treat `&` as an explicit intersection separator (whitespace-
            // equivalent). The `add_rect_box` / `add_hex_boundary` helpers
            // return `" & "`-joined region strings, and downstream user
            // code typically concatenates them with their own clauses
            // using `&` as the visible separator.
            if token == "&" {
                continue;
            }
            let (complement, rest) = match token.strip_prefix('~') {
                Some(stripped) => (true, stripped),
                None => (false, token),
            };
            let (is_inside, name) = match rest.chars().next() {
                Some('-') => (true, &rest[1..]),
                Some('+') => (false, &rest[1..]),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "region token {token:?} must be '-name', '+name', '~-name', or '~+name'"
                    )));
                }
            };
            let idx = *surf_idx.get(name).ok_or_else(|| {
                PyValueError::new_err(format!("unknown surface {name:?} in region {expr:?}"))
            })?;
            let half = if is_inside {
                cell::inside(idx)
            } else {
                cell::outside(idx)
            };
            and_terms.push(if complement {
                Region::Complement(Box::new(half))
            } else {
                half
            });
        }
        if and_terms.is_empty() {
            return Err(PyValueError::new_err(format!(
                "empty or-group in region {expr:?}"
            )));
        }
        group_regions.push(if and_terms.len() == 1 {
            and_terms.into_iter().next().unwrap()
        } else {
            cell::intersect_all(and_terms)
        });
    }

    Ok(group_regions
        .into_iter()
        .reduce(|a, b| Region::Union(Box::new(a), Box::new(b)))
        .expect("or_groups is non-empty"))
}

/// GPU dispatch path. Uploads NuclideKernels / material / SAB / WMP
/// once per run, builds a recursive geometry context, and drives
/// `CudaRunner::run` through the same `EigenvalueRunner` trait the
/// CLI binaries use.
///
/// All four `XsMode` values arrive here with `kernels` already loaded
/// at the appropriate rank:
///   - `Svd` / `HybridSvdWmp`: rank = `scene.svd_rank` (with per-MT
///     overrides applied to MTs 2/4/16/17/18/102; discrete-level MTs
///     stay at the global rank because of the GPU's per-level
///     basis-stride invariant).
///   - `Table` / `HybridTableWmp`: rank = 1 (the rank-1 SVD layout
///     reconstructs the loaded pointwise table exactly).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_gpu_eigenvalue(
    py: Python<'_>,
    scene: &PyScene,
    config: &SimConfig,
    kernels: Vec<Arc<xs_provider::NuclideKernels>>,
    rank: usize,
    materials_rt: &[RustMaterial],
    surfaces: &[Surface],
    cells_rt: &[Cell],
    nuclide_specs: &[(String, f64, f64, usize)],
    thermal: &[Option<Arc<open_rust_mc::thermal::ThermalScatteringData>>],
    wmps: &[Option<(Arc<WindowedMultipole>, f64)>],
) -> PyResult<(
    Vec<open_rust_mc::transport::simulate::BatchResult>,
    f64,
    usize,
)> {
    use open_rust_mc::gpu_recursive::GpuRecursiveContext;
    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::transport::dispatch::{CudaRunner, EigenvalueRunner};

    let gpu_err = |label: &str, e: Box<dyn std::error::Error>| {
        PyValueError::new_err(format!("GPU {label}: {e}"))
    };

    // ── Build the recursive geometry (single-universe, flat cells) ──
    let geometry = Geometry::from_slices(surfaces, cells_rt).map_err(|e| {
        PyValueError::new_err(format!(
            "failed to assemble GPU geometry from {} surfaces / {} cells: {e:?}",
            surfaces.len(),
            cells_rt.len()
        ))
    })?;

    let n = config.particles_per_batch as usize;
    let xs_memory_bytes: usize = kernels.iter().map(|n| n.svd_memory_bytes()).sum();
    let limits = SimLimits::default();

    // ── Upload XS + materials + (optional) S(α,β) + (optional) WMP ──
    let gpu = GpuTransportContext::shared().map_err(|e| gpu_err("init", e))?;
    let nuc_data = gpu
        .upload_nuclide_data(&kernels, rank)
        .map_err(|e| gpu_err("upload nuclides", e))?;

    let awrs: Vec<f64> = nuclide_specs.iter().map(|(_, awr, _, _)| *awr).collect();
    let nu_bars: Vec<f64> = nuclide_specs.iter().map(|(_, _, nu, _)| *nu).collect();
    let mat_data = gpu
        .upload_material_data(materials_rt, &awrs, &nu_bars)
        .map_err(|e| gpu_err("upload materials", e))?;

    // Collect every nuclide that has a thermal-scattering library
    // attached and upload all of them in one multi-slot pack. The GPU
    // kernel routes each collision to the matching TSL via the
    // per-nuclide `slot_per_nuc` lookup table, so problems with
    // simultaneous H-in-H₂O + D-in-D₂O + C-in-graphite all sample
    // correctly on the GPU.
    let n_nuc = nuclide_specs.len();
    let sab_slots: Vec<(
        &open_rust_mc::thermal::ThermalScatteringData,
        usize,
        usize,
    )> = thermal
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            t.as_ref().map(|tsl| {
                let (_, _, _, temp_idx) = &nuclide_specs[i];
                (tsl.as_ref(), *temp_idx, i)
            })
        })
        .collect();
    // sab_nuc_idx is retained for compatibility with the CudaRunner
    // field but is no longer authoritative — the slot table is. Set it
    // to the first SAB-bearing nuclide so legacy kernel paths that
    // still consult it stay correct for single-TSL scenes.
    let sab_nuc_idx: i32 = sab_slots
        .first()
        .map(|(_, _, idx)| *idx as i32)
        .unwrap_or(-1);
    let sab_data = gpu
        .upload_sab_data_multi(&sab_slots, n_nuc)
        .map_err(|e| gpu_err("upload S(α,β)", e))?;

    let wmp_has_any = wmps.iter().any(|w| w.is_some());
    let wmp_data = if wmp_has_any {
        gpu.upload_wmp_data(wmps)
            .map_err(|e| gpu_err("upload WMP", e))?
    } else {
        gpu.upload_wmp_data_empty(nuclide_specs.len())
            .map_err(|e| gpu_err("upload empty WMP", e))?
    };

    let recursive = GpuRecursiveContext::build(&geometry, n).map_err(|e| {
        PyValueError::new_err(format!("GPU recursive context build failed: {e}"))
    })?;

    // Per-material kT in eV (Boltzmann constant in eV/K).
    const K_B_EV_PER_K: f64 = 8.617_333_262e-5;
    let mat_k_t: Vec<f64> = scene
        .material_order
        .iter()
        .map(|name| scene.materials[name].temperature * K_B_EV_PER_K)
        .collect();

    // Closure that seeds batch 1 using the same rejection sampler the
    // CPU path uses. CudaRunner's per-batch loop normalises the
    // fission bank between iterations internally.
    let geom_ref = &geometry;
    let cells_ref = cells_rt;
    let init_src: Box<dyn Fn(usize, u64) -> Vec<(f64, f64, f64, f64)>> =
        Box::new(move |n, seed| {
            simulate::initial_source(n, geom_ref, cells_ref, seed)
                .into_iter()
                .map(|s| (s.pos.x, s.pos.y, s.pos.z, s.energy))
                .collect()
        });

    let runner = CudaRunner {
        recursive: &recursive,
        transport: &gpu,
        nuc_data: &nuc_data,
        mat_data: &mat_data,
        sab_data: &sab_data,
        wmp_data: &wmp_data,
        mat_k_t: &mat_k_t,
        sab_nuc_idx,
        max_events_per_history: limits.max_events_per_history as i32,
        fis_capacity: limits.fis_capacity(n),
        initial_source: init_src,
        buffers: std::cell::RefCell::new(None),
    };

    let _ = py;
    let outcome = runner.run(config);
    Ok((outcome.batches, outcome.k_eff, xs_memory_bytes))
}

#[pyfunction]
fn run_eigenvalue(
    py: Python<'_>,
    scene: &PyScene,
    settings: &PySettings,
) -> PyResult<PyEigenvalueResult> {
    // 1. Collect all unique (hdf5_file, temp_idx, awr, nubar) specs from
    //    all materials, in first-appearance order — that's the nuclide
    //    table indexing the engine expects.
    #[derive(PartialEq, Clone)]
    struct NuclideKey {
        hdf5_file: String,
        temp_idx: usize,
    }
    let mut nuclide_keys: Vec<NuclideKey> = Vec::new();
    let mut nuclide_specs: Vec<(String, f64, f64, usize)> = Vec::new(); // (file, awr, nubar, temp_idx)
    for mat_name in &scene.material_order {
        let mat = &scene.materials[mat_name];
        for n in &mat.nuclides {
            let key = NuclideKey {
                hdf5_file: n.hdf5_file.clone(),
                temp_idx: mat.temp_idx,
            };
            if !nuclide_keys.contains(&key) {
                nuclide_keys.push(key.clone());
                nuclide_specs.push((n.hdf5_file.clone(), n.awr, n.nubar, mat.temp_idx));
            }
        }
    }

    // 2. Load XS data — backend × mode specific. Time the load step.
    //
    // CPU dispatch reads `tables` for Table*-mode and `svd_kernels` for
    // Svd*-mode. GPU dispatch always reads `svd_kernels` (the device
    // upload accepts NuclideKernels with mixed Svd / Table per-MT
    // variants); for the pure-Table modes the kernels are loaded at
    // rank 1, which the upload adapts into a rank-1 SVD layout that
    // reconstructs the original pointwise table exactly at the loaded
    // temperature.
    let t_load_start = std::time::Instant::now();
    let mut tables: Vec<std::sync::Arc<xs_provider::NuclideTableData>> = Vec::new();
    let mut svd_kernels: Vec<std::sync::Arc<xs_provider::NuclideKernels>> = Vec::new();
    let mode_uses_svd = matches!(scene.xs_mode, PyXsMode::Svd | PyXsMode::HybridSvdWmp);
    let mode_uses_table = matches!(scene.xs_mode, PyXsMode::Table | PyXsMode::HybridTableWmp);
    let on_gpu = matches!(scene.runner, PyRunner::GpuCuda);
    let need_table = mode_uses_table && !on_gpu;
    let need_svd = mode_uses_svd || on_gpu;
    let kernel_rank: usize = if mode_uses_svd { scene.svd_rank } else { 1 };
    for (file, awr, nubar, temp_idx) in &nuclide_specs {
        let path = scene.data_dir.join(file);
        if !path.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "missing nuclide file: {}",
                path.display()
            )));
        }
        if need_table {
            tables.push(std::sync::Arc::new(xs_provider::load_nuclide_table(
                &path, *temp_idx, *awr, *nubar,
            )));
        }
        if need_svd {
            // Build the per-MT rank policy once per call (cheap, but keep
            // outside the hot path anyway).
            let mut policy = RankPolicy::new(kernel_rank);
            for (&mt, &rank) in &scene.svd_ranks_per_mt {
                policy = policy.with_mt(mt, rank);
            }
            svd_kernels.push(std::sync::Arc::new(xs_provider::load_nuclide_with_policy(
                &path, &policy, *temp_idx, *awr, *nubar,
            )));
        }
    }

    // 2b. Build per-nuclide thermal scattering table. Indexed by the
    //     same nuclide index as `tables`. A nuclide is "thermal" if
    //     ANY material using it specified a `thermal_file` — we pick
    //     the first such file (all uses must agree). This mirrors
    //     pwr_pincell.rs's `load_thermal` path.
    let mut thermal_files: Vec<Option<String>> = vec![None; nuclide_specs.len()];
    for mat_name in &scene.material_order {
        let mat = &scene.materials[mat_name];
        for n in &mat.nuclides {
            if let Some(tf) = &n.thermal_file {
                let key = NuclideKey {
                    hdf5_file: n.hdf5_file.clone(),
                    temp_idx: mat.temp_idx,
                };
                let idx = nuclide_keys.iter().position(|k| k == &key).unwrap();
                match &thermal_files[idx] {
                    None => thermal_files[idx] = Some(tf.clone()),
                    Some(existing) if existing != tf => {
                        return Err(PyValueError::new_err(format!(
                            "conflicting thermal files for {}: {} vs {}",
                            n.hdf5_file, existing, tf
                        )));
                    }
                    _ => {}
                }
            }
        }
    }
    let mut thermal: Vec<Option<std::sync::Arc<open_rust_mc::thermal::ThermalScatteringData>>> =
        vec![None; nuclide_specs.len()];
    for (i, tf) in thermal_files.iter().enumerate() {
        if let Some(tf) = tf {
            let path = scene.data_dir.join(tf);
            if !path.exists() {
                return Err(PyFileNotFoundError::new_err(format!(
                    "missing thermal scattering file: {}",
                    path.display()
                )));
            }
            match open_rust_mc::hdf5_reader::load_thermal_scattering(&path) {
                Ok(tsl) => thermal[i] = Some(std::sync::Arc::new(tsl)),
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "failed to load thermal scattering {}: {e}",
                        path.display()
                    )));
                }
            }
        }
    }
    // 3. Build surface vec.
    let mut surf_idx: HashMap<String, usize> = HashMap::new();
    let mut surfaces: Vec<Surface> = Vec::with_capacity(scene.surface_order.len());
    for (i, name) in scene.surface_order.iter().enumerate() {
        surf_idx.insert(name.clone(), i);
        surfaces.push(scene.surfaces[name].inner.clone());
    }

    // 4. Build materials (engine-side).
    let mut material_idx: HashMap<String, usize> = HashMap::new();
    let mut materials_rt: Vec<RustMaterial> = Vec::with_capacity(scene.material_order.len());
    for (i, name) in scene.material_order.iter().enumerate() {
        let mat = &scene.materials[name];
        let mut m = RustMaterial::new(&mat.name, mat.temperature);
        for n in &mat.nuclides {
            let key = NuclideKey {
                hdf5_file: n.hdf5_file.clone(),
                temp_idx: mat.temp_idx,
            };
            let nuc_idx = nuclide_keys.iter().position(|k| k == &key).unwrap();
            m.add_nuclide(n.atom_density, nuc_idx);
        }
        material_idx.insert(name.clone(), i);
        materials_rt.push(m);
    }

    // 5. Build cells.
    let mut cells_rt: Vec<Cell> = Vec::with_capacity(scene.cells.len());
    for (i, c) in scene.cells.iter().enumerate() {
        let region = build_region(&c.region_expr, &surf_idx)?;
        let aabb = aabb_from_region(&c.region_expr, &surfaces, &surf_idx);
        let fill = match &c.fill {
            None => CellFill::Void,
            Some(mat_name) => {
                let idx = material_idx.get(mat_name).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "cell {:?} references unknown material {:?}",
                        c.name, mat_name
                    ))
                })?;
                CellFill::Material(*idx as u32)
            }
        };
        cells_rt.push(
            Cell::new(CellId(i as u32), region, fill)
                .with_aabb(aabb)
                .with_temperature(c.temperature),
        );
    }

    // 6. Build the chosen XS provider, then run eigenvalue.
    //
    // For Hybrid* modes, also load WMP files from `<data_dir>/../wmp/`
    // using the standard ZZAAA.h5 naming convention. Nuclides without
    // a known WMP filename (anything other than U-234/235/238) get
    // `None` and fall through to the inner provider.
    let mut wmps: Vec<Option<(Arc<WindowedMultipole>, f64)>> =
        Vec::with_capacity(nuclide_specs.len());
    let mut covered = 0usize;
    let needs_wmp = matches!(
        scene.xs_mode,
        PyXsMode::HybridSvdWmp | PyXsMode::HybridTableWmp
    );
    if needs_wmp {
        let wmp_dir = scene.data_dir.join("..").join("wmp");
        for (file, _, _, _) in &nuclide_specs {
            let entry = match wmp_path_for(file) {
                None => None,
                Some(wmp_file) => {
                    let path = wmp_dir.join(wmp_file);
                    if !path.exists() {
                        None
                    } else {
                        match WindowedMultipole::from_hdf5(&path) {
                            Ok(wmp) => {
                                covered += 1;
                                let t_kelvin = 294.0; // matches default temp_idx=1 -> ~294 K
                                Some((Arc::new(wmp), t_kelvin))
                            }
                            Err(_) => None,
                        }
                    }
                }
            };
            wmps.push(entry);
        }
    }

    let load_time = t_load_start.elapsed().as_secs_f64();

    let config = SimConfig {
        batches: settings.batches,
        inactive: settings.inactive,
        particles_per_batch: settings.particles,
        seed: settings.seed,
        auto_inactive: None,
        verbose: false,
        parallel: true,
        tallies: Default::default(),
        statepoint_path: None,
        survival_biasing: None,
        initial_source_bank: None,
        weight_window: None,
        disable_delayed_neutrons: false,
        urr_equivalence: None,
    };
    let t_sim_start = std::time::Instant::now();
    let (batch_results, _k_running, xs_memory_bytes) = match scene.runner {
        PyRunner::Cpu => match scene.xs_mode {
            PyXsMode::Table => {
                let xs_mem: usize = tables.iter().map(|t| t.table_memory_bytes()).sum();
                let xs = TableXsProvider {
                    nuclides: tables,
                    thermal,
                };
                let (br, k) = py.allow_threads(|| {
                    simulate::run_eigenvalue(&config, &surfaces, &cells_rt, &materials_rt, &xs)
                });
                (br, k, xs_mem)
            }
            PyXsMode::Svd => {
                let xs_mem: usize = svd_kernels.iter().map(|n| n.svd_memory_bytes()).sum();
                let xs = SvdXsProvider {
                    nuclides: svd_kernels,
                    thermal,
                };
                let (br, k) = py.allow_threads(|| {
                    simulate::run_eigenvalue(&config, &surfaces, &cells_rt, &materials_rt, &xs)
                });
                (br, k, xs_mem)
            }
            PyXsMode::HybridSvdWmp => {
                let inner = SvdXsProvider {
                    nuclides: svd_kernels,
                    thermal,
                };
                let xs = HybridSvdWmpXsProvider::new(inner, wmps);
                // (smooth-only rebuild disabled here pending diagnosis of
                // a Godiva-specific zero-fission regression in the Python
                // path; pwr_pincell binary still does the rebuild and
                // reports the realised memory drop. See xs_mode_demo.)
                let xs_mem = xs.memory_report().current_total();
                let (br, k) = py.allow_threads(|| {
                    simulate::run_eigenvalue(&config, &surfaces, &cells_rt, &materials_rt, &xs)
                });
                (br, k, xs_mem)
            }
            PyXsMode::HybridTableWmp => {
                let inner = TableXsProvider {
                    nuclides: tables,
                    thermal,
                };
                let xs = HybridTableWmpXsProvider::new(inner, wmps);
                let xs_mem = xs.memory_report().current_total();
                let (br, k) = py.allow_threads(|| {
                    simulate::run_eigenvalue(&config, &surfaces, &cells_rt, &materials_rt, &xs)
                });
                (br, k, xs_mem)
            }
        },
        PyRunner::GpuCuda => {
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (kernel_rank, &svd_kernels, &nuclide_specs, &wmps);
                return Err(PyValueError::new_err(
                    "Runner.GpuCuda was requested but these bindings were built without \
                     the cuda feature. Rebuild the Python extension with \
                     `maturin develop --features cuda` (CUDA toolkit + an sm_86+ GPU \
                     required), or call `Scene.set_runner(Runner.Cpu)` to stay on the \
                     CPU backend.",
                ));
            }
            #[cfg(feature = "cuda")]
            {
                run_gpu_eigenvalue(
                    py,
                    scene,
                    &config,
                    svd_kernels,
                    kernel_rank,
                    &materials_rt,
                    &surfaces,
                    &cells_rt,
                    &nuclide_specs,
                    &thermal,
                    &wmps,
                )?
            }
        }
    };
    let sim_time = t_sim_start.elapsed().as_secs_f64();
    let runtime = load_time + sim_time;

    // 7. Compute k mean/std and roll up per-batch tallies.
    let active: Vec<f64> = batch_results
        .iter()
        .filter(|b| b.active)
        .map(|b| b.k_eff)
        .collect();
    let n = active.len();
    if n == 0 {
        return Err(PyValueError::new_err(
            "no active batches produced — check settings.batches > settings.inactive",
        ));
    }
    let mean: f64 = active.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        active.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let std_dev = (variance / n as f64).sqrt();

    let k_per_batch: Vec<f64> = batch_results.iter().map(|b| b.k_eff).collect();
    let entropy_per_batch: Vec<f64> = batch_results.iter().map(|b| b.shannon_entropy).collect();
    let active_mask: Vec<bool> = batch_results.iter().map(|b| b.active).collect();

    // Per-cell capture tally: sum over active batches.
    let mut captures_by_cell = vec![0.0_f64; cells_rt.len()];
    let mut total_collisions: u64 = 0;
    let mut total_fissions: u64 = 0;
    let mut total_leakage: u64 = 0;
    for br in batch_results.iter().filter(|b| b.active) {
        for (i, c) in br.captures_by_cell.iter().enumerate() {
            if i < captures_by_cell.len() {
                captures_by_cell[i] += *c;
            }
        }
        total_collisions += br.collisions as u64;
        total_fissions += br.fissions as u64;
        total_leakage += br.leakage as u64;
    }
    let cell_names: Vec<String> = scene.cells.iter().map(|c| c.name.clone()).collect();

    let mode_name = match scene.xs_mode {
        PyXsMode::Table => "table",
        PyXsMode::Svd => "svd",
        PyXsMode::HybridSvdWmp => "hybrid_svd_wmp",
        PyXsMode::HybridTableWmp => "hybrid_table_wmp",
    };
    let svd_rank_used = match scene.xs_mode {
        PyXsMode::Table | PyXsMode::HybridTableWmp => 0,
        PyXsMode::Svd | PyXsMode::HybridSvdWmp => scene.svd_rank,
    };

    Ok(PyEigenvalueResult {
        k_eff: mean,
        k_sigma: std_dev,
        active_batches: n,
        total_histories: n as u64 * settings.particles as u64,
        runtime_seconds: runtime,
        k_per_batch,
        entropy_per_batch,
        active_mask,
        captures_by_cell,
        cell_names,
        total_collisions,
        total_fissions,
        total_leakage,
        mode_used: mode_name.to_string(),
        svd_rank: svd_rank_used,
        load_time_seconds: load_time,
        sim_time_seconds: sim_time,
        xs_memory_bytes: xs_memory_bytes as u64,
        wmp_covered_nuclides: covered,
        svd_ranks_per_mt: scene.svd_ranks_per_mt.clone(),
    })
}

// ── Depletion bindings ────────────────────────────────────────────────────

/// CRAM approximation order — Python-visible `CramOrder` enum.
/// `Order16` is the default for PWR-typical Δt; `Order48` for stiff
/// activation chains and geologic Δt.
#[pyclass(eq, eq_int, name = "CramOrder", module = "open_rust_mc._core")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PyCramOrder {
    Order16,
    Order48,
}

#[pymethods]
impl PyCramOrder {
    fn __repr__(&self) -> String {
        match self {
            PyCramOrder::Order16 => "CramOrder.Order16".into(),
            PyCramOrder::Order48 => "CramOrder.Order48".into(),
        }
    }
}

impl From<PyCramOrder> for open_rust_mc::depletion::cram::CramOrder {
    fn from(o: PyCramOrder) -> Self {
        match o {
            PyCramOrder::Order16 => open_rust_mc::depletion::cram::CramOrder::Cram16,
            PyCramOrder::Order48 => open_rust_mc::depletion::cram::CramOrder::Cram48,
        }
    }
}

/// A loaded depletion chain. Build from a JSON file
/// (`Chain.from_file`) or from a JSON string (`Chain.from_str`).
/// Holds the runtime `DepletionChain` plus the original `ChainSpec`
/// so name / description survive round-trip.
#[pyclass(name = "Chain", module = "open_rust_mc._core")]
struct PyChain {
    spec: open_rust_mc::depletion::chain_io::ChainSpec,
    chain: open_rust_mc::depletion::DepletionChain,
}

#[pymethods]
impl PyChain {
    /// Load a chain from a JSON file on disk. See
    /// `chains/partial_xe.json` for the schema.
    #[staticmethod]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        let spec = open_rust_mc::depletion::chain_io::ChainSpec::from_file(&path)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        let chain = spec.build();
        Ok(Self { spec, chain })
    }

    /// Load a chain from a JSON string (no I/O). Useful for unit
    /// tests and notebooks that paste-in a small chain literal.
    #[staticmethod]
    fn from_str(text: &str) -> PyResult<Self> {
        let spec = open_rust_mc::depletion::chain_io::ChainSpec::from_str(text)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        let chain = spec.build();
        Ok(Self { spec, chain })
    }

    #[getter]
    fn name(&self) -> String {
        self.spec.name.clone()
    }

    #[getter]
    fn description(&self) -> String {
        self.spec.description.clone()
    }

    #[getter]
    fn n_nuclides(&self) -> usize {
        self.chain.len()
    }

    #[getter]
    fn n_reactions(&self) -> usize {
        self.chain.reactions.len()
    }

    /// List of `(zaid, name)` tuples in chain order. Useful for
    /// driving an `n0` vector indexed by chain position.
    fn nuclide_list(&self) -> Vec<(u32, String)> {
        self.chain
            .nuclides
            .iter()
            .map(|n| (n.zaid, n.name.clone()))
            .collect()
    }

    /// Index of `zaid` in this chain, or `None` if not present.
    fn index_of_zaid(&self, zaid: u32) -> Option<usize> {
        self.chain.index_of_zaid(zaid)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Chain '{}': {} nuclides, {} reactions>",
            self.spec.name,
            self.chain.len(),
            self.chain.reactions.len()
        )
    }
}

/// One CE/LI predictor-corrector step at constant flux. Returns the
/// updated composition vector. Composition is indexed by the chain's
/// nuclide order (see `Chain.nuclide_list`).
///
/// `flux` is the one-group flux in `n / (cm² · s)`. `dt_seconds` is
/// the time step. `order` selects CRAM-16 (default) or CRAM-48.
///
/// Constant-flux variant — the corrector matrix matches the
/// predictor matrix, so the result is the predictor estimate. For
/// flux feedback (transport-solve mid-step), call CRAM-16 / CRAM-48
/// directly with the desired matrix or extend this to take a Python
/// callback.
#[pyfunction]
#[pyo3(signature = (chain, n0, flux, dt_seconds, order=PyCramOrder::Order16))]
fn deplete_constant_flux(
    chain: &PyChain,
    n0: Vec<f64>,
    flux: f64,
    dt_seconds: f64,
    order: PyCramOrder,
) -> PyResult<Vec<f64>> {
    if n0.len() != chain.chain.len() {
        return Err(PyValueError::new_err(format!(
            "n0 length {} does not match chain length {}",
            n0.len(),
            chain.chain.len()
        )));
    }
    let order = open_rust_mc::depletion::cram::CramOrder::from(order);
    let step =
        open_rust_mc::depletion::deplete_ce_li(&chain.chain, &n0, flux, dt_seconds, order, |_| {
            flux
        });
    Ok(step.corrected)
}

/// One CE/LI predictor-corrector step where the corrector flux
/// comes from a Python callable. `flux_at` is invoked once per
/// step with the predicted composition and must return the EOC
/// one-group flux. Use this to plug a real transport solve into
/// the depletion loop:
///
/// ```python
/// def flux_at(predicted_composition):
///     # Push predicted composition into materials, run eigenvalue,
///     # extract mean fuel-cell flux, return it.
///     for zaid, density in zip(chain.nuclide_list(), predicted_composition):
///         ...
///     result = run_eigenvalue(scene, settings)
///     return mean_flux  # n / (cm² · s)
///
/// new_composition = deplete_with_flux_callback(
///     chain, n0, flux_boc, dt_seconds, flux_at, CramOrder.Order16,
/// )
/// ```
///
/// The callback runs once per step. Re-entry into Rust is fine —
/// `flux_at` can call `run_eigenvalue` (or anything else) inside
/// the closure; the GIL is held during the callback only.
#[pyfunction]
#[pyo3(signature = (chain, n0, flux_boc, dt_seconds, flux_at, order=PyCramOrder::Order16))]
fn deplete_with_flux_callback(
    py: Python<'_>,
    chain: &PyChain,
    n0: Vec<f64>,
    flux_boc: f64,
    dt_seconds: f64,
    flux_at: PyObject,
    order: PyCramOrder,
) -> PyResult<Vec<f64>> {
    if n0.len() != chain.chain.len() {
        return Err(PyValueError::new_err(format!(
            "n0 length {} does not match chain length {}",
            n0.len(),
            chain.chain.len()
        )));
    }
    let order = open_rust_mc::depletion::cram::CramOrder::from(order);

    // Capture any Python exception thrown by the callback so we can
    // re-raise it through the PyResult after the Rust call returns.
    let mut callback_error: Option<PyErr> = None;
    let result = open_rust_mc::depletion::deplete_ce_li(
        &chain.chain,
        &n0,
        flux_boc,
        dt_seconds,
        order,
        |predicted: &[f64]| {
            if callback_error.is_some() {
                return flux_boc;
            }
            let predicted_vec: Vec<f64> = predicted.to_vec();
            match flux_at.call1(py, (predicted_vec,)) {
                Ok(retval) => retval.extract::<f64>(py).unwrap_or_else(|e| {
                    callback_error = Some(e);
                    flux_boc
                }),
                Err(e) => {
                    callback_error = Some(e);
                    flux_boc
                }
            }
        },
    );
    if let Some(e) = callback_error {
        return Err(e);
    }
    Ok(result.corrected)
}

/// Direct CRAM matrix-exponential evaluator: returns
/// `exp(matrix) · n0` where `matrix` is the row-major flattened
/// `n × n` real matrix (already pre-multiplied by `Δt` in the
/// caller — see `chain.matrix.build_transmutation_matrix`).
///
/// This is the low-level primitive; for typical depletion use
/// `deplete_constant_flux` which builds the matrix from the chain
/// + flux for you.
#[pyfunction]
#[pyo3(signature = (matrix, n0, order=PyCramOrder::Order16))]
fn cram(matrix: Vec<f64>, n0: Vec<f64>, order: PyCramOrder) -> PyResult<Vec<f64>> {
    let n = n0.len();
    if matrix.len() != n * n {
        return Err(PyValueError::new_err(format!(
            "matrix length {} does not match n0.len() squared = {}",
            matrix.len(),
            n * n
        )));
    }
    let order = open_rust_mc::depletion::cram::CramOrder::from(order);
    Ok(open_rust_mc::depletion::cram::cram(order, &matrix, &n0))
}

// ── Nuclear-data introspection ────────────────────────────────────────────
//
// Lightweight read-only handle on an ENDF HDF5 nuclide file, for
// diagnostic work (e.g. the U-233 −2876 pcm Jezebel-23 bias deep-dive).
// Exposes ν̄(E) and fission χ(E_in, E_out) tables verbatim from the
// file so they can be diffed against OpenMC's Python API
// (`openmc.data.IncidentNeutron.from_hdf5(...)`).

/// Open a nuclide HDF5 file for read-only introspection.
///
/// >>> nuc = NuclideFile.open("data/endfb-vii.1-hdf5/neutron/U233.h5")
/// >>> nuc.nuclide_name
/// 'U233'
/// >>> e, v = nuc.nu_bar()
/// >>> len(e)
/// 13
#[pyclass(name = "NuclideFile", module = "open_rust_mc._core")]
pub struct PyNuclideFile {
    reader: NuclideFileReader,
}

#[pymethods]
impl PyNuclideFile {
    /// Open a nuclide HDF5 file. Raises FileNotFoundError on missing
    /// file, ValueError on any HDF5 / layout issue.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let p = PathBuf::from(path);
        if !p.exists() {
            return Err(PyFileNotFoundError::new_err(format!("no such file: {path}")));
        }
        NuclideFileReader::open(&p)
            .map(|reader| Self { reader })
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Nuclide name as stored in the HDF5 root group (e.g. "U233").
    #[getter]
    fn nuclide_name(&self) -> String {
        self.reader.nuclide_name.clone()
    }

    /// Temperatures (K) present in the file, sorted ascending.
    #[getter]
    fn temperatures(&self) -> Vec<f64> {
        self.reader.temperatures.clone()
    }

    /// Atomic weight ratio (AWR). Returns 0.0 if the attribute is
    /// missing.
    fn awr(&self) -> f64 {
        self.reader.awr().unwrap_or(0.0)
    }

    /// Prompt + delayed ν̄(E) table.
    ///
    /// Returns ``(energies_eV, nu_bar)`` as two equal-length lists.
    /// Empty lists if the nuclide is non-fissile.
    fn nu_bar(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let nb = self
            .reader
            .nu_bar()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok((nb.energies, nb.values))
    }

    /// Linear-interpolated ν̄ at a single energy. Same code path as
    /// the transport engine, so values match the engine bit-for-bit.
    fn nu_bar_at(&self, energy_eV: f64) -> PyResult<f64> {
        let nb = self
            .reader
            .nu_bar()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(nb.lookup(energy_eV))
    }

    /// Delayed-only ν̄(E) table. ``None`` if the file ships no
    /// delayed-product entries (always the case for non-fissile
    /// nuclides; some fissile files also omit it).
    fn delayed_nu_bar(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        self.reader
            .delayed_nu_bar()
            .map(|nb| (nb.energies, nb.values))
    }

    /// Fission χ(E_in, E_out) incident-energy grid (eV). Returns
    /// ``None`` if the file does not ship MT=18 outgoing-energy data
    /// in a form the engine recognises (uncorrelated tabular law).
    fn fission_incident_energies(&self) -> Option<Vec<f64>> {
        self.reader.fission_energy_dist().map(|ed| ed.energies)
    }

    /// Per-incident-energy outgoing distribution at index ``i``.
    /// Returns ``(e_out_eV, pdf, cdf)`` — three equal-length lists.
    /// ``pdf`` may be empty when the file only stores ``(e_out, cdf)``
    /// pairs without an explicit PDF channel (the engine then falls
    /// back to histogram-CDF inversion at sample time).
    ///
    /// Raises ValueError if the file has no fission spectrum, or if
    /// ``i`` is out of range.
    fn fission_outgoing_at(&self, i: usize) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let ed = self
            .reader
            .fission_energy_dist()
            .ok_or_else(|| PyValueError::new_err("no fission energy distribution in this file"))?;
        let d = ed
            .distributions
            .get(i)
            .ok_or_else(|| PyValueError::new_err(format!(
                "index {i} out of range (have {} incident-energy nodes)",
                ed.energies.len()
            )))?;
        Ok((d.e_out.clone(), d.pdf.clone(), d.cdf.clone()))
    }

    /// Sample ``n`` outgoing fission-neutron energies (eV) at the
    /// given incident energy, using the engine's exact sampling
    /// kernel. Returns an empty list if the file has no fission
    /// energy distribution.
    #[pyo3(signature = (energy_eV, n, seed = 0xD1A6_F133))]
    fn sample_fission_outgoing(&self, energy_eV: f64, n: usize, seed: u64) -> Vec<f64> {
        let ed = match self.reader.fission_energy_dist() {
            Some(ed) => ed,
            None => return Vec::new(),
        };
        let mut rng = Rng::new(seed, 0);
        (0..n).map(|_| ed.sample(energy_eV, &mut rng)).collect()
    }
}

// ── ICSBEP loader + runner ────────────────────────────────────────────────

/// Best-effort decode of a `catch_unwind` payload into a human-readable
/// message. Rust's panic payload is `Box<dyn Any + Send>` — typically
/// either a `&'static str` or a `String`, depending on whether the
/// `panic!()` site used a literal or a formatted message.
fn panic_to_string(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Result of an ICSBEP regression run. Carries the engine's `k_calc`
/// alongside the case's acceptance reference (handbook or — when the
/// JSON ships one — the `local_validation` block recording OpenMC's
/// measurement on the same scene) and the pass envelope.
#[pyclass(name = "IcsbepResult", module = "open_rust_mc._core")]
struct PyIcsbepResult {
    #[pyo3(get)]
    case: String,
    #[pyo3(get)]
    k_eff: f64,
    #[pyo3(get)]
    k_sigma: f64,
    #[pyo3(get)]
    k_ref: f64,
    #[pyo3(get)]
    sigma_exp: f64,
    #[pyo3(get)]
    handbook_k: f64,
    #[pyo3(get)]
    handbook_sigma: f64,
    #[pyo3(get)]
    ref_source: String,
    #[pyo3(get)]
    delta_pcm: f64,
    #[pyo3(get)]
    sigma_combined: f64,
    #[pyo3(get)]
    sigma_ratio: f64,
    #[pyo3(get)]
    bound_pcm: f64,
    // `pass` is a Python keyword — expose as `passed`.
    #[pyo3(get, name = "passed")]
    pass_ok: bool,
    #[pyo3(get)]
    runner: PyRunner,
    #[pyo3(get)]
    runtime_seconds: f64,
    #[pyo3(get)]
    load_time_seconds: f64,
    #[pyo3(get)]
    sim_time_seconds: f64,
    #[pyo3(get)]
    total_collisions: u64,
    #[pyo3(get)]
    total_fissions: u64,
    #[pyo3(get)]
    total_leakage: u64,
    #[pyo3(get)]
    active_batches: usize,
    #[pyo3(get)]
    total_histories: u64,
}

#[pymethods]
impl PyIcsbepResult {
    fn __repr__(&self) -> String {
        format!(
            "<IcsbepResult {} {} k={:.5}±{:.5} k_ref={:.5}±{:.5} Δ={:+.0}pcm {:.2}σ bound=±{:.0}pcm [{}]>",
            self.case,
            self.runner.__repr__(),
            self.k_eff,
            self.k_sigma,
            self.k_ref,
            self.sigma_exp,
            self.delta_pcm,
            self.sigma_ratio,
            self.bound_pcm,
            if self.pass_ok { "PASS" } else { "FAIL" },
        )
    }
}

/// Run an ICSBEP regression case end-to-end from its JSON specification.
///
/// `case_json` is a path to one of the `bench/icsbep/*.json` files
/// (e.g. `bench/icsbep/heu-met-fast-001_case-1.json`). `data_dir`
/// points at the OpenMC HDF5 distribution's neutron sub-directory.
///
/// The function:
///   1. Parses the JSON, extracts the handbook `k_eff_reference`
///      and (if present) `local_validation.openmc_k_eff` to derive
///      the acceptance reference.
///   2. Loads the scene's recursive geometry via
///      `scene_io::load_scene_from_json` and resolves materials via
///      `material_resolve::resolve_materials` against the HDF5
///      library at `data_dir`.
///   3. Dispatches the eigenvalue loop through the selected runner
///      (`Runner.Cpu` — `simulate::run_eigenvalue_with_geometry`;
///      `Runner.GpuCuda` — `dispatch::CudaRunner`, requires the
///      bindings built with `--features cuda`).
///   4. Applies the `|Δ| ≤ max(150 pcm, 2·σ_combined)` envelope —
///      the same criterion `tests/cuda_runs.rs` and
///      `tests/icsbep_runs.rs` use.
///
/// Returns an `IcsbepResult` with `k_eff`, `k_sigma`, `k_ref`,
/// `sigma_exp`, `delta_pcm`, `bound_pcm`, `pass`, and timing info.
#[pyfunction]
#[pyo3(signature = (case_json, data_dir, settings, runner=PyRunner::Cpu, rank=15))]
fn run_icsbep_case(
    py: Python<'_>,
    case_json: PathBuf,
    data_dir: PathBuf,
    settings: &PySettings,
    runner: PyRunner,
    rank: usize,
) -> PyResult<PyIcsbepResult> {
    use open_rust_mc::geometry::scene_io;
    use open_rust_mc::transport::material_resolve;
    use open_rust_mc::transport::nuclides::NuclideLibrary;

    open_rust_mc::hardware_profile::log_startup_banner();

    if !case_json.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "ICSBEP case JSON not found: {}",
            case_json.display()
        )));
    }
    if !data_dir.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "data_dir not found: {}",
            data_dir.display()
        )));
    }

    let case_label: String = case_json
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();

    let text = std::fs::read_to_string(&case_json)
        .map_err(|e| PyValueError::new_err(format!("read {}: {e}", case_json.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| PyValueError::new_err(format!("parse {}: {e}", case_json.display())))?;
    let benchmark = &value["benchmark"];
    let scene = &value["scene"];
    // Some `bench/icsbep/*.json` files are CLI-runner manifests (no
    // `scene` block; `runner.binary` names a CLI binary like
    // `godiva` / `pwr_pincell`) rather than scene specifications.
    // Surface them with a clear error instead of letting the
    // SceneDto deserializer choke on the missing object.
    if scene.is_null() {
        let runner_binary = value
            .get("runner")
            .and_then(|r| r.get("binary"))
            .and_then(|b| b.as_str())
            .unwrap_or("<unknown>");
        return Err(PyValueError::new_err(format!(
            "case JSON has no `scene` block — this is a CLI-runner manifest \
             pointing at the `{runner_binary}` binary. Only scene-based \
             cases are runnable through `run_icsbep_case`. Use the named \
             CLI binary directly or pick a *_case-*.json file."
        )));
    }

    let handbook_k = benchmark["k_eff_reference"]
        .as_f64()
        .ok_or_else(|| PyValueError::new_err("benchmark.k_eff_reference missing"))?;
    let handbook_sigma = benchmark["k_eff_sigma"]
        .as_f64()
        .ok_or_else(|| PyValueError::new_err("benchmark.k_eff_sigma missing"))?;

    let (k_ref, sigma_exp, ref_source): (f64, f64, String) = match benchmark.get("local_validation") {
        Some(lv) if lv.get("openmc_k_eff").and_then(|v: &serde_json::Value| v.as_f64()).is_some() => {
            let k = lv["openmc_k_eff"].as_f64().unwrap();
            let s_omc = lv["openmc_k_sigma_seeds"]
                .as_f64()
                .unwrap_or(0.001);
            (
                k,
                s_omc.max(handbook_sigma),
                "local_validation (OpenMC on this scene)".to_string(),
            )
        }
        _ => (
            handbook_k,
            handbook_sigma,
            "k_eff_reference (ICSBEP handbook)".to_string(),
        ),
    };

    let t_load_start = std::time::Instant::now();
    let loaded = scene_io::load_scene_from_json(&scene.to_string())
        .map_err(|e| PyValueError::new_err(format!("scene_io: {e:?}")))?;
    let lib = NuclideLibrary::from_data_dir(&data_dir);
    let resolved = material_resolve::resolve_materials(&loaded.materials, &lib, rank)
        .map_err(|e| PyValueError::new_err(format!("material_resolve: {e}")))?;
    // Engine hard limit — single source of truth lives at
    // `open_rust_mc::MAX_NUCLIDES_PER_MATERIAL`. Both the CPU's
    // fixed-size MicroXs arrays in `simulate.rs` and the GPU's
    // `nuc_t[MAX_NUC_PER_MAT]` register array read from the same
    // value (the GPU sees it through an NVRTC `-D` flag). Materials
    // that exceed the cap hit an index-out-of-bounds panic at the
    // first collision; bail early here so sweeps can categorise the
    // failure cleanly.
    let max_nuclides = open_rust_mc::MAX_NUCLIDES_PER_MATERIAL;
    for (mi, mat) in resolved.materials.iter().enumerate() {
        if mat.nuclides.len() > max_nuclides {
            return Err(PyValueError::new_err(format!(
                "material[{mi}] {:?} has {} nuclides, but the engine \
                 supports at most {max_nuclides} per material \
                 (MAX_NUCLIDES_PER_MATERIAL — bump in lib.rs and rebuild). \
                 Split the material or raise the limit.",
                mat.name,
                mat.nuclides.len(),
            )));
        }
    }
    let load_time = t_load_start.elapsed().as_secs_f64();

    // Pre-seed the source bank via the fallible, fissionability-aware
    // sampler. Mirrors Serpent 2's default: per-cell region-tree AABB
    // weighted by volume, accept any draw that lands in a cell whose
    // material has `nu_bar_const > 0`. Replaces the historical
    // "first Material cell" / "smallest-volume material" heuristics
    // that broke on multi-shell HMF, BWR control blades, PWR burnable
    // poisons, HFIR plate cladding, CANDU spacers.
    let fissionable = resolved.fissionable_materials();
    let initial_bank = simulate::try_initial_source_in_materials(
        settings.particles as usize,
        &loaded.geometry,
        loaded.geometry.cells.as_slice(),
        Some(&fissionable),
        settings.seed,
    )
    .map_err(|e| PyValueError::new_err(format!("initial_source: {}", e.message)))?;

    let config = SimConfig {
        batches: settings.batches,
        inactive: settings.inactive,
        particles_per_batch: settings.particles,
        seed: settings.seed,
        auto_inactive: None,
        verbose: false,
        parallel: true,
        tallies: Default::default(),
        statepoint_path: None,
        survival_biasing: None,
        initial_source_bank: Some(initial_bank),
        weight_window: None,
        disable_delayed_neutrons: false,
        urr_equivalence: None,
    };

    // Some ICSBEP cases (degenerate world AABB, missing fissile region,
    // upload overflow) trigger a Rust `panic!` deep inside the engine
    // — typically from `simulate::initial_source`'s rejection sampler
    // or from CUDA's MAX_NUC_PER_MAT check. Without `catch_unwind` the
    // panic propagates out and aborts the host Python process, which
    // breaks any sweep that hits one bad case. Wrap the sim call so a
    // panic surfaces as a normal `PyValueError`, letting the sweep
    // continue with that case marked ERROR.
    use std::panic::{self, AssertUnwindSafe};

    let t_sim_start = std::time::Instant::now();
    let runner_label = match runner {
        PyRunner::Cpu => "CPU",
        PyRunner::GpuCuda => "GPU",
    };
    let sim_result = panic::catch_unwind(AssertUnwindSafe(|| -> PyResult<(
        Vec<open_rust_mc::transport::simulate::BatchResult>,
        f64,
    )> {
        match runner {
            PyRunner::Cpu => {
                let provider = &resolved.provider;
                let materials = &resolved.materials;
                let geometry = &loaded.geometry;
                let out = py.allow_threads(|| {
                    simulate::run_eigenvalue_with_geometry(
                        &config, geometry, materials, provider,
                    )
                });
                Ok(out)
            }
            PyRunner::GpuCuda => {
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (&resolved, &loaded, &config, settings);
                    return Err(PyValueError::new_err(
                        "Runner.GpuCuda was requested but these bindings were built without \
                         the cuda feature. Rebuild with `maturin develop --features cuda` \
                         (CUDA toolkit + an sm_86+ GPU required) or select Runner.Cpu.",
                    ));
                }
                #[cfg(feature = "cuda")]
                {
                    run_gpu_icsbep(&config, &loaded.geometry, &resolved)
                }
            }
        }
    }));
    let (batch_results, _k_running) = match sim_result {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => return Err(e),
        Err(payload) => {
            return Err(PyValueError::new_err(format!(
                "engine panic during {runner_label} dispatch: {}",
                panic_to_string(&payload),
            )));
        }
    };
    let sim_time = t_sim_start.elapsed().as_secs_f64();
    let runtime = load_time + sim_time;

    // Active-batch mean / stderr.
    let active: Vec<f64> = batch_results
        .iter()
        .filter(|b| b.active)
        .map(|b| b.k_eff)
        .collect();
    let n = active.len();
    if n == 0 {
        return Err(PyValueError::new_err(
            "no active batches produced — check settings.batches > settings.inactive",
        ));
    }
    let mean = active.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        active.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let k_sigma = (variance / n as f64).sqrt();

    let sigma_combined: f64 = (k_sigma * k_sigma + sigma_exp * sigma_exp).sqrt();
    let delta: f64 = mean - k_ref;
    let delta_pcm: f64 = delta * 1.0e5;
    let bound_pcm: f64 = (150.0_f64).max(2.0 * sigma_combined * 1.0e5);
    let sigma_ratio: f64 = if sigma_combined > 0.0 {
        delta.abs() / sigma_combined
    } else {
        0.0
    };
    let pass_ok: bool = delta_pcm.abs() <= bound_pcm;

    let mut total_collisions: u64 = 0;
    let mut total_fissions: u64 = 0;
    let mut total_leakage: u64 = 0;
    for br in batch_results.iter().filter(|b| b.active) {
        total_collisions += br.collisions as u64;
        total_fissions += br.fissions as u64;
        total_leakage += br.leakage as u64;
    }

    Ok(PyIcsbepResult {
        case: case_label,
        k_eff: mean,
        k_sigma,
        k_ref,
        sigma_exp,
        handbook_k,
        handbook_sigma,
        ref_source,
        delta_pcm,
        sigma_combined,
        sigma_ratio,
        bound_pcm,
        pass_ok,
        runner,
        runtime_seconds: runtime,
        load_time_seconds: load_time,
        sim_time_seconds: sim_time,
        total_collisions,
        total_fissions,
        total_leakage,
        active_batches: n,
        total_histories: n as u64 * settings.particles as u64,
    })
}

/// GPU dispatch for an ICSBEP case. Uploads the resolved
/// [`SvdXsProvider`] (kernels + thermal) plus an empty WMP placeholder
/// and drives [`CudaRunner`].
#[cfg(feature = "cuda")]
fn run_gpu_icsbep(
    config: &SimConfig,
    geometry: &Geometry,
    resolved: &open_rust_mc::transport::material_resolve::ResolvedMaterials,
) -> PyResult<(
    Vec<open_rust_mc::transport::simulate::BatchResult>,
    f64,
)> {
    use open_rust_mc::gpu_recursive::GpuRecursiveContext;
    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::transport::dispatch::{CudaRunner, EigenvalueRunner};

    let gpu_err = |label: &str, e: Box<dyn std::error::Error>| {
        PyValueError::new_err(format!("GPU {label}: {e}"))
    };

    let provider = &resolved.provider;
    let materials_rt = &resolved.materials;
    let n_nuc = provider.nuclides.len();
    let n = config.particles_per_batch as usize;
    let limits = SimLimits::default();

    let gpu = GpuTransportContext::shared().map_err(|e| gpu_err("init", e))?;
    let nuc_data = gpu
        .upload_nuclide_data(&provider.nuclides, /* rank = global */ 15)
        .map_err(|e| gpu_err("upload nuclides", e))?;

    let awrs: Vec<f64> = provider.nuclides.iter().map(|n| n.awr).collect();
    let nu_bars: Vec<f64> = provider
        .nuclides
        .iter()
        .map(|n| n.nu_bar_const)
        .collect();
    let mat_data = gpu
        .upload_material_data(materials_rt, &awrs, &nu_bars)
        .map_err(|e| gpu_err("upload materials", e))?;

    // Multi-slot SAB upload — every thermal nuclide goes into its own
    // slot. The first material temperature is used to pick each TSL's
    // temperature index (mirrors `tests/cuda_runs.rs` semantics).
    let pick_temp = if !materials_rt.is_empty() {
        materials_rt[0].temperature
    } else {
        294.0
    };
    let sab_slots: Vec<(
        &open_rust_mc::thermal::ThermalScatteringData,
        usize,
        usize,
    )> = provider
        .thermal
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            t.as_ref().map(|tsl| {
                let t_idx = tsl.select_temperature(pick_temp, limits.sab_temperature_tolerance);
                (tsl.as_ref(), t_idx, i)
            })
        })
        .collect();
    let sab_nuc_idx: i32 = sab_slots
        .first()
        .map(|(_, _, idx)| *idx as i32)
        .unwrap_or(-1);
    let sab_data = gpu
        .upload_sab_data_multi(&sab_slots, n_nuc)
        .map_err(|e| gpu_err("upload S(α,β)", e))?;

    let wmp_data = gpu
        .upload_wmp_data_empty(n_nuc)
        .map_err(|e| gpu_err("upload empty WMP", e))?;

    let recursive = GpuRecursiveContext::build(geometry, n).map_err(|e| {
        PyValueError::new_err(format!("GPU recursive context build failed: {e}"))
    })?;

    const K_B_EV_PER_K: f64 = 8.617_333_262e-5;
    let mat_k_t: Vec<f64> = materials_rt
        .iter()
        .map(|m| m.temperature * K_B_EV_PER_K)
        .collect();

    let cells_ref = geometry.cells.as_slice();
    let init_src: Box<dyn Fn(usize, u64) -> Vec<(f64, f64, f64, f64)>> =
        Box::new(move |n, seed| {
            // Use the fallible sampler; if geometry can't be sampled
            // return an empty bank rather than aborting the process.
            // The CudaRunner then sees zero histories and produces a
            // zero k — the wrapping `run_icsbep_case` already
            // pre-validated the bank above, so this branch only runs
            // for the per-batch resamples (which reuse fission sites
            // from the previous batch and never re-enter the rejection
            // sampler) — leaving the fallback in place is purely
            // defense-in-depth.
            simulate::try_initial_source(n, geometry, cells_ref, seed)
                .unwrap_or_default()
                .into_iter()
                .map(|s| (s.pos.x, s.pos.y, s.pos.z, s.energy))
                .collect()
        });

    let runner = CudaRunner {
        recursive: &recursive,
        transport: &gpu,
        nuc_data: &nuc_data,
        mat_data: &mat_data,
        sab_data: &sab_data,
        wmp_data: &wmp_data,
        mat_k_t: &mat_k_t,
        sab_nuc_idx,
        max_events_per_history: limits.max_events_per_history as i32,
        fis_capacity: limits.fis_capacity(n),
        initial_source: init_src,
        buffers: std::cell::RefCell::new(None),
    };

    let outcome = runner.run(config);
    Ok((outcome.batches, outcome.k_eff))
}

/// Pre-warm the L1 nuclide cache with frequency hints from a sweep
/// manifest pre-scan.
///
/// `weights` is a list of `(zaid, temperature_k, count)` triples —
/// typically built by walking every case JSON in the sweep,
/// counting how often each `(zaid, temperature)` pair appears
/// across the whole corpus. `rank` is the SVD rank the sweep runs
/// with (the rank participates in the cache key, so the weights
/// only apply to runs at that rank).
///
/// Each ZAID is resolved to an HDF5 path via `NuclideLibrary` (same
/// catalog `run_icsbep_case` uses) and the temperature is mapped to
/// the nearest library column via `pick_temperature`. First call
/// hashes every referenced HDF5 file (~30-60 s for ~50 actinide
/// files on SSD); subsequent `run_icsbep_case` lookups promote
/// pre-marked nuclides under the LFU-with-recency policy.
///
/// Returns the number of weights successfully resolved. ZAIDs not
/// in the catalog or files that can't be hashed are silently
/// skipped — the cache still works, just without the warm-start
/// hint for that nuclide.
#[pyfunction]
fn preload_nuclide_cache_weights(
    data_dir: PathBuf,
    weights: Vec<(u32, f64, u64)>,
    rank: usize,
) -> PyResult<usize> {
    use open_rust_mc::transport::nuclide_cache;
    use open_rust_mc::transport::nuclide_cache::NuclideKey;
    use open_rust_mc::transport::nuclides::NuclideLibrary;
    use open_rust_mc::transport::xs_provider::RankPolicy;

    let lib = NuclideLibrary::from_data_dir(&data_dir);
    let policy = RankPolicy::new(rank);
    let mut map: HashMap<NuclideKey, u64> = HashMap::new();
    for (zaid, temp_k, w) in &weights {
        let resolved = match lib.resolve(*zaid, *temp_k) {
            Ok(r) => r,
            Err(_) => continue,
        };
        match NuclideKey::from_inputs(&resolved.path, &policy, resolved.temp_idx) {
            Ok(key) => {
                map.insert(key, *w);
            }
            Err(_) => continue,
        }
    }
    let n = map.len();
    nuclide_cache::set_preload_weights(&map);
    Ok(n)
}

// ── Module init ───────────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Rayon's thread pool lazy-initialises the first time `par_iter`
    // runs. That's fine here because the simulate call wraps itself
    // in `py.allow_threads` — the GIL is released before rayon's
    // first call, so Python is not holding the Windows loader lock
    // while rayon spawns worker threads. Eagerly building the pool
    // during pymodule init would deadlock on Windows because that
    // IS inside the loader lock (module init runs from LoadLibrary).
    m.add_class::<PySurface>()?;
    m.add_class::<PyMaterial>()?;
    m.add_class::<PyPhotonMaterial>()?;
    m.add_class::<PySettings>()?;
    m.add_class::<PyScene>()?;
    m.add_class::<PyXsMode>()?;
    m.add_class::<PyRunner>()?;
    m.add_class::<PyEigenvalueResult>()?;
    m.add_class::<PyGammaHeatingResult>()?;
    m.add_class::<PyIcsbepResult>()?;
    m.add_class::<PyChain>()?;
    m.add_class::<PyCramOrder>()?;
    m.add_class::<PyNuclideFile>()?;

    m.add_function(wrap_pyfunction!(Sphere, m)?)?;
    m.add_function(wrap_pyfunction!(ZCylinder, m)?)?;
    m.add_function(wrap_pyfunction!(XCylinder, m)?)?;
    m.add_function(wrap_pyfunction!(YCylinder, m)?)?;
    m.add_function(wrap_pyfunction!(XPlane, m)?)?;
    m.add_function(wrap_pyfunction!(YPlane, m)?)?;
    m.add_function(wrap_pyfunction!(ZPlane, m)?)?;

    m.add_function(wrap_pyfunction!(run_eigenvalue, m)?)?;
    m.add_function(wrap_pyfunction!(run_icsbep_case, m)?)?;
    m.add_function(wrap_pyfunction!(run_gamma_heating, m)?)?;
    m.add_function(wrap_pyfunction!(cram, m)?)?;
    m.add_function(wrap_pyfunction!(deplete_constant_flux, m)?)?;
    m.add_function(wrap_pyfunction!(deplete_with_flux_callback, m)?)?;
    m.add_function(wrap_pyfunction!(preload_nuclide_cache_weights, m)?)?;

    Ok(())
}
