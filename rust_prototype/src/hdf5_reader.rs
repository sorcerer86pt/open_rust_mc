//! Pure-Rust HDF5 reader for OpenMC nuclear data files.
//!
//! Uses `hdf5-pure` (zero C dependencies) to read OpenMC nuclide HDF5 files.
//!
//! OpenMC HDF5 layout per nuclide (e.g. U235.h5):
//!   /energy/{temp}                          — energy grid per temperature
//!   /reactions/reaction_{MT:03}/{temp}/xs    — cross-section values

use std::path::Path;

use crate::error::{Result, SvdError};

/// Cross-section data for a single nuclide extracted from an OpenMC HDF5 file.
pub struct NuclideData {
    /// Temperatures in Kelvin, sorted ascending.
    pub temperatures: Vec<f64>,
    /// Temperature labels as stored in HDF5 (e.g. "294K").
    pub temp_labels: Vec<String>,
    /// Unionized energy grid in eV, sorted ascending.
    pub energies: Vec<f64>,
    /// Cross-section columns: one `Vec<f64>` per temperature, each length = N_E,
    /// interpolated onto the unionized energy grid.
    pub xs_per_temp: Vec<Vec<f64>>,
    /// Reaction MT number.
    pub mt: u32,
}

fn parse_temp_kelvin(label: &str) -> Option<f64> {
    label.strip_suffix('K')?.parse::<f64>().ok()
}

impl NuclideData {
    /// Read a single reaction from an OpenMC HDF5 nuclide file.
    ///
    /// Pure Rust — no C library required.
    pub fn from_hdf5(path: &Path, mt: u32) -> Result<Self> {
        let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("{e}"),
        })?;

        // Discover the nuclide name (top-level group, e.g. "U235")
        let root = file.root();
        let nuclide_name = root.groups().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot list root groups: {e}"),
        })?
        .into_iter()
        .next()
        .ok_or_else(|| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: "no nuclide group found at root".into(),
        })?;

        println!("  Nuclide: {nuclide_name}");

        let nuclide_group = root.group(&nuclide_name).map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot open /{nuclide_name}: {e}"),
        })?;

        // Discover temperatures from /{nuclide}/energy group
        let energy_group = nuclide_group.group("energy").map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot open /{nuclide_name}/energy: {e}"),
        })?;

        let mut temp_labels = energy_group.datasets().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot list /energy datasets: {e}"),
        })?;

        // Filter out 0K (unbroadened) and sort numerically
        temp_labels.retain(|l| parse_temp_kelvin(l).is_some_and(|t| t > 0.0));
        temp_labels.sort_by(|a, b| {
            let va = parse_temp_kelvin(a).unwrap_or(0.0);
            let vb = parse_temp_kelvin(b).unwrap_or(0.0);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let temperatures: Vec<f64> = temp_labels.iter().filter_map(|l| parse_temp_kelvin(l)).collect();

        println!("  Temperatures: {temp_labels:?}");

        // Read per-temperature energy grids
        let mut energy_grids: Vec<Vec<f64>> = Vec::with_capacity(temp_labels.len());
        for label in &temp_labels {
            let ds = energy_group.dataset(label).map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("cannot open /energy/{label}: {e}"),
            })?;
            let grid = ds.read_f64().map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("cannot read /energy/{label}: {e}"),
            })?;
            energy_grids.push(grid);
        }

        // Unionize
        let mut union: Vec<f64> = energy_grids.iter().flatten().copied().collect();
        union.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        union.dedup();
        println!("  Unionized energy grid: {} points", union.len());

        // Read cross-sections per temperature
        let rxn_name = format!("reaction_{mt:03}");
        let reactions_group = nuclide_group.group("reactions").map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot open /{nuclide_name}/reactions: {e}"),
        })?;
        let rxn_group = reactions_group.group(&rxn_name).map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot open /{nuclide_name}/reactions/{rxn_name}: {e}"),
        })?;

        let mut xs_per_temp = Vec::with_capacity(temp_labels.len());
        for (t_idx, label) in temp_labels.iter().enumerate() {
            let temp_group = rxn_group.group(label).map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("cannot open /{nuclide_name}/reactions/{rxn_name}/{label}: {e}"),
            })?;
            let xs_ds = temp_group.dataset("xs").map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("cannot open /{nuclide_name}/reactions/{rxn_name}/{label}/xs: {e}"),
            })?;
            let xs_raw = xs_ds.read_f64().map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("cannot read /{nuclide_name}/reactions/{rxn_name}/{label}/xs: {e}"),
            })?;

            let e_grid = &energy_grids[t_idx];
            let n_grid = e_grid.len();
            let n_xs = xs_raw.len();

            // Handle threshold reactions: xs array may be shorter than energy grid.
            // The xs values correspond to the LAST n_xs points of the energy grid.
            // Below threshold, cross-section is zero.
            let xs_full = if n_xs < n_grid {
                let mut full = vec![0.0_f64; n_grid];
                let offset = n_grid - n_xs;
                full[offset..].copy_from_slice(&xs_raw);
                full
            } else {
                xs_raw
            };

            let xs_interp = interpolate_to_grid(e_grid, &xs_full, &union);
            xs_per_temp.push(xs_interp);
        }

        Ok(NuclideData {
            temperatures,
            temp_labels,
            energies: union,
            xs_per_temp,
            mt,
        })
    }

    pub fn n_energy(&self) -> usize { self.energies.len() }
    pub fn n_temp(&self) -> usize { self.temperatures.len() }

    /// Build N_E × N_T matrix in log₁₀ scale, row-major.
    pub fn to_log_matrix(&self) -> Vec<f64> {
        let (n_e, n_t) = (self.n_energy(), self.n_temp());
        let mut mat = vec![0.0_f64; n_e * n_t];
        for t in 0..n_t {
            for i in 0..n_e {
                mat[i * n_t + t] = self.xs_per_temp[t][i].max(1e-30).log10();
            }
        }
        mat
    }

    /// Build N_E × N_T matrix in linear scale, row-major.
    pub fn to_linear_matrix(&self) -> Vec<f64> {
        let (n_e, n_t) = (self.n_energy(), self.n_temp());
        let mut mat = vec![0.0_f64; n_e * n_t];
        for t in 0..n_t {
            for i in 0..n_e {
                mat[i * n_t + t] = self.xs_per_temp[t][i].max(1e-30);
            }
        }
        mat
    }
}

/// Linear interpolation of (x_src, y_src) onto x_dst.
/// Both x_src and x_dst must be sorted ascending.
fn interpolate_to_grid(x_src: &[f64], y_src: &[f64], x_dst: &[f64]) -> Vec<f64> {
    let n = x_src.len();
    let mut out = Vec::with_capacity(x_dst.len());
    let mut j = 0_usize;

    for &x in x_dst {
        while j + 1 < n && x_src[j + 1] < x {
            j += 1;
        }
        if x <= x_src[0] {
            out.push(y_src[0]);
        } else if x >= x_src[n - 1] {
            out.push(y_src[n - 1]);
        } else if (x - x_src[j]).abs() < 1e-15 {
            out.push(y_src[j]);
        } else if j + 1 < n {
            let frac = (x - x_src[j]) / (x_src[j + 1] - x_src[j]);
            out.push(y_src[j] + frac * (y_src[j + 1] - y_src[j]));
        } else {
            out.push(y_src[j]);
        }
    }
    out
}
