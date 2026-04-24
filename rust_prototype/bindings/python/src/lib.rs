// Surface constructor functions use CamelCase to match the Python class
// names the users see (`Sphere`, `ZCylinder`, ...). Rust's snake_case
// lint would rename them; suppress project-wide in this binding crate.
#![allow(non_snake_case)]

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
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3};
use open_rust_mc::transport::material::Material as RustMaterial;
use open_rust_mc::transport::simulate::{self, SimConfig};
use open_rust_mc::transport::xs_provider::{self, TableXsProvider};

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
        inner: Surface::PlaneX { x0, bc: parse_bc(bc)? },
    })
}

#[pyfunction]
#[pyo3(signature = (y0, bc="transmission"))]
fn YPlane(y0: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::PlaneY { y0, bc: parse_bc(bc)? },
    })
}

#[pyfunction]
#[pyo3(signature = (z0, bc="transmission"))]
fn ZPlane(z0: f64, bc: &str) -> PyResult<PySurface> {
    Ok(PySurface {
        inner: Surface::PlaneZ { z0, bc: parse_bc(bc)? },
    })
}

// ── Material ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct NuclideSpec {
    hdf5_file: String,
    atom_density: f64,
    awr: f64,
    nubar: f64,
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
    /// Atom density is the absolute macro unit used throughout the
    /// engine (Σ = n · σ in cm⁻¹ when σ is in barn). Compute it from
    /// macro density + stoichiometry on the Python side, or pass an
    /// already-computed value.
    #[pyo3(signature = (hdf5_file, atom_density, awr, nubar=0.0))]
    fn add_nuclide(
        mut slf: PyRefMut<'_, Self>,
        hdf5_file: String,
        atom_density: f64,
        awr: f64,
        nubar: f64,
    ) -> PyRefMut<'_, Self> {
        slf.nuclides.push(NuclideSpec {
            hdf5_file,
            atom_density,
            awr,
            nubar,
        });
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "<Material {:?} at {:.1} K, {} nuclides>",
            self.name, self.temperature, self.nuclides.len()
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
    materials: HashMap<String, PyMaterial>,
    material_order: Vec<String>,
    surfaces: HashMap<String, PySurface>,
    surface_order: Vec<String>,
    cells: Vec<CellSpec>,
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
            materials: HashMap::new(),
            material_order: Vec::new(),
            surfaces: HashMap::new(),
            surface_order: Vec::new(),
            cells: Vec::new(),
        })
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
}

#[pymethods]
impl PyEigenvalueResult {
    fn __repr__(&self) -> String {
        format!(
            "<EigenvalueResult k_eff={:.5} ± {:.5} from {} active batches>",
            self.k_eff, self.k_sigma, self.active_batches
        )
    }
}

/// AABB for a cell derived from its region expression.
///
/// For each `-<name>` token (inside half-space), the cell is bounded
/// by that surface's own AABB. We take the intersection of all such
/// "inside" AABBs — giving a tight bounding box for rejection
/// sampling in `initial_source`. `+<name>` tokens (outside half-space)
/// are ignored: the cell extends to infinity on those sides unless
/// constrained by another surface's inside token.
///
/// For a void cell with no `-name` tokens (e.g. the "outside" cell of
/// Godiva), the result is `Aabb::INFINITE`. That's fine because
/// initial_source only samples fissile cells (those with
/// `CellFill::Material`), not void cells.
fn aabb_from_region(
    expr: &str,
    surfaces: &[Surface],
    surf_idx: &HashMap<String, usize>,
) -> Aabb {
    let mut acc: Option<Aabb> = None;
    for token in expr.split_whitespace() {
        let (sign, name) = match token.chars().next() {
            Some('-') => ('-', &token[1..]),
            _ => continue, // only inside tokens constrain the AABB
        };
        if sign != '-' {
            continue;
        }
        let Some(&i) = surf_idx.get(name) else { continue };
        let a = surfaces[i].aabb();
        acc = Some(match acc {
            None => a,
            Some(b) => Aabb::new(
                Vec3::new(b.min.x.max(a.min.x), b.min.y.max(a.min.y), b.min.z.max(a.min.z)),
                Vec3::new(b.max.x.min(a.max.x), b.max.y.min(a.max.y), b.max.z.min(a.max.z)),
            ),
        });
    }
    acc.unwrap_or(Aabb::INFINITE)
}

/// Build the Region for `expr` like `-a +b -c` with name→index map.
fn build_region(expr: &str, surf_idx: &HashMap<String, usize>) -> PyResult<Region> {
    let mut parts = Vec::new();
    for token in expr.split_whitespace() {
        let (sign, name) = match token.chars().next() {
            Some('-') => ('-', &token[1..]),
            Some('+') => ('+', &token[1..]),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "region token {token:?} must start with '+' or '-'"
                )));
            }
        };
        let idx = *surf_idx.get(name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown surface {name:?} in region {expr:?}"))
        })?;
        parts.push(if sign == '-' {
            cell::inside(idx)
        } else {
            cell::outside(idx)
        });
    }
    if parts.is_empty() {
        return Err(PyValueError::new_err("region expression is empty"));
    }
    Ok(if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        cell::intersect_all(parts)
    })
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

    // 2. Load XS tables.
    let mut tables = Vec::with_capacity(nuclide_specs.len());
    for (file, awr, nubar, temp_idx) in &nuclide_specs {
        let path = scene.data_dir.join(file);
        if !path.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "missing nuclide file: {}",
                path.display()
            )));
        }
        tables.push(xs_provider::load_nuclide_table(&path, *temp_idx, *awr, *nubar));
    }
    let thermal = vec![None; nuclide_specs.len()];
    let xs = TableXsProvider {
        nuclides: tables,
        thermal,
    };

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

    // 6. Run eigenvalue — release the GIL for the long-running Rust loop.
    let config = SimConfig {
        batches: settings.batches,
        inactive: settings.inactive,
        particles_per_batch: settings.particles,
        seed: settings.seed,
        auto_inactive: None,
        // Rust engine stays silent — Python owns reporting. This also
        // means no stdout contention between rayon workers and
        // Python's own stdout when the GIL is released.
        verbose: false,
        // Parallel transport via rayon. `py.allow_threads` below
        // releases the GIL so Python blocks while Rust's rayon pool
        // runs natively. No stdout contention because the engine is
        // silent.
        parallel: true,
    };
    let t0 = std::time::Instant::now();
    let (batch_results, _k_running) = py.allow_threads(|| {
        simulate::run_eigenvalue(&config, &surfaces, &cells_rt, &materials_rt, &xs)
    });
    let runtime = t0.elapsed().as_secs_f64();

    // 7. Compute k mean/std over active batches.
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
    let std = (variance / n as f64).sqrt();

    Ok(PyEigenvalueResult {
        k_eff: mean,
        k_sigma: std,
        active_batches: n,
        total_histories: n as u64 * settings.particles as u64,
        runtime_seconds: runtime,
    })
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
    m.add_class::<PySettings>()?;
    m.add_class::<PyScene>()?;
    m.add_class::<PyEigenvalueResult>()?;

    m.add_function(wrap_pyfunction!(Sphere, m)?)?;
    m.add_function(wrap_pyfunction!(ZCylinder, m)?)?;
    m.add_function(wrap_pyfunction!(XCylinder, m)?)?;
    m.add_function(wrap_pyfunction!(YCylinder, m)?)?;
    m.add_function(wrap_pyfunction!(XPlane, m)?)?;
    m.add_function(wrap_pyfunction!(YPlane, m)?)?;
    m.add_function(wrap_pyfunction!(ZPlane, m)?)?;

    m.add_function(wrap_pyfunction!(run_eigenvalue, m)?)?;

    Ok(())
}
