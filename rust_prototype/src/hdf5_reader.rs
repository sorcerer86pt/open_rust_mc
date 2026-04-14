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

/// Tabulated nu-bar (average neutrons per fission) as a function of energy.
pub struct NuBarTable {
    /// Energy grid in eV, sorted ascending.
    pub energies: Vec<f64>,
    /// Nu-bar values at each energy point.
    pub values: Vec<f64>,
}

impl NuBarTable {
    /// Interpolate nu-bar at a given energy (linear interpolation).
    pub fn lookup(&self, energy: f64) -> f64 {
        let n = self.energies.len();
        if n == 0 { return 2.43; } // fallback
        if n == 1 { return self.values[0]; }
        if energy <= self.energies[0] { return self.values[0]; }
        if energy >= self.energies[n - 1] { return self.values[n - 1]; }

        // Binary search
        let idx = match self.energies.binary_search_by(|e| {
            e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => return self.values[i],
            Err(i) => i,
        };

        let i = idx - 1;
        let frac = (energy - self.energies[i]) / (self.energies[i + 1] - self.energies[i]);
        self.values[i] + frac * (self.values[i + 1] - self.values[i])
    }
}

/// Discrete inelastic level info (for MT=51-91).
#[derive(Debug, Clone)]
pub struct DiscreteLevelInfo {
    /// ENDF MT number (51-91).
    pub mt: u32,
    /// Q-value in eV (negative = excitation energy).
    pub q_value: f64,
    /// Threshold energy in eV: E_threshold = |Q| * (A+1)/A.
    pub threshold: f64,
}

/// Read total nu-bar (neutron yield) from fission reaction products.
///
/// Total nu-bar = sum of all neutron product yields (prompt + delayed).
/// OpenMC stores each neutron product under reaction_018/product_N with
/// `@particle = "neutron"`. The yield dataset has shape [2, N] (tabulated)
/// or shape [1] (constant).
pub fn read_nu_bar(path: &Path) -> Result<NuBarTable> {
    let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    let root = file.root();
    let nuclide_name = root.groups().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot list root groups: {e}"),
    })?
    .into_iter()
    .next()
    .ok_or_else(|| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: "no nuclide group found".into(),
    })?;

    let nuc = root.group(&nuclide_name).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    let rxn_group = nuc.group("reactions").and_then(|r| r.group("reaction_018"));
    let rxn_group = match rxn_group {
        Ok(g) => g,
        Err(_) => return Ok(NuBarTable { energies: vec![], values: vec![] }),
    };

    // Collect all neutron product yield tables (prompt + delayed)
    let subgroups = rxn_group.groups().unwrap_or_default();
    let mut prompt_table: Option<NuBarTable> = None;
    let mut delayed_constants: Vec<f64> = Vec::new();

    for product_name in &subgroups {
        if !product_name.starts_with("product_") { continue; }

        let product = match rxn_group.group(product_name) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let attrs = product.attrs().unwrap_or_default();

        let is_neutron = matches!(
            attrs.get("particle"),
            Some(hdf5_pure::AttrValue::String(s)) if s == "neutron"
        );
        if !is_neutron { continue; }

        let is_prompt = matches!(
            attrs.get("emission_mode"),
            Some(hdf5_pure::AttrValue::String(s)) if s == "prompt"
        );

        let yield_ds = match product.dataset("yield") {
            Ok(ds) => ds,
            Err(_) => continue,
        };

        let shape = yield_ds.shape().unwrap_or_default();
        let raw = match yield_ds.read_f64() {
            Ok(r) => r,
            Err(_) => continue,
        };

        if is_prompt {
            // Prompt neutron yield — the main tabulated table
            if shape.len() == 2 && shape[0] == 2 {
                let n = shape[1] as usize;
                if raw.len() >= 2 * n {
                    prompt_table = Some(NuBarTable {
                        energies: raw[..n].to_vec(),
                        values: raw[n..2 * n].to_vec(),
                    });
                }
            } else if shape.len() == 1 && shape[0] == 1 {
                prompt_table = Some(NuBarTable {
                    energies: vec![1e-5, 20.0e6],
                    values: vec![raw[0], raw[0]],
                });
            }
        } else {
            // Delayed neutron group — usually constant yield
            if shape.len() == 2 && shape[0] == 2 {
                let n = shape[1] as usize;
                if raw.len() >= 2 * n {
                    // Use average of tabulated values
                    let sum: f64 = raw[n..2 * n].iter().sum();
                    delayed_constants.push(sum / n as f64);
                }
            } else if shape.len() == 1 && shape[0] == 1 {
                delayed_constants.push(raw[0]);
            }
        }
    }

    match prompt_table {
        Some(mut table) => {
            // Add delayed neutron yield to the prompt table
            let delayed_total: f64 = delayed_constants.iter().sum();
            if delayed_total > 0.0 {
                for v in &mut table.values {
                    *v += delayed_total;
                }
            }
            Ok(table)
        }
        None => Ok(NuBarTable { energies: vec![], values: vec![] }),
    }
}

/// Read the atomic weight ratio from the HDF5 nuclide file.
pub fn read_awr(path: &Path) -> Result<f64> {
    let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    let root = file.root();
    let nuclide_name = root.groups().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot list root groups: {e}"),
    })?
    .into_iter()
    .next()
    .ok_or_else(|| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: "no nuclide group found".into(),
    })?;

    let nuc = root.group(&nuclide_name).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    let attrs = nuc.attrs().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read attributes: {e}"),
    })?;

    if let Some(hdf5_pure::AttrValue::F64(awr)) = attrs.get("atomic_weight_ratio") {
        Ok(*awr)
    } else {
        Err(SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: "missing atomic_weight_ratio attribute".into(),
        })
    }
}

/// Discover discrete inelastic levels (MT=51-91) and read their Q-values.
///
/// Returns a sorted list of levels with Q-values and thresholds.
pub fn read_discrete_levels(path: &Path, awr: f64) -> Result<Vec<DiscreteLevelInfo>> {
    let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    let root = file.root();
    let nuclide_name = root.groups().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot list root groups: {e}"),
    })?
    .into_iter()
    .next()
    .ok_or_else(|| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: "no nuclide group found".into(),
    })?;

    let nuc = root.group(&nuclide_name).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    let reactions = match nuc.group("reactions") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };

    let rxn_names = reactions.groups().unwrap_or_default();

    let mut levels = Vec::new();
    for name in &rxn_names {
        // Parse MT number from "reaction_051" etc.
        let mt: u32 = match name.strip_prefix("reaction_").and_then(|s| s.parse().ok()) {
            Some(mt) if (51..=91).contains(&mt) => mt,
            _ => continue,
        };

        let rxn = match reactions.group(name) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let attrs = rxn.attrs().unwrap_or_default();
        let q_value = if let Some(hdf5_pure::AttrValue::F64(q)) = attrs.get("Q_value") {
            *q
        } else {
            continue; // no Q-value — skip
        };

        // Threshold: E_threshold = |Q| * (A+1)/A for endothermic reactions
        let threshold = if q_value < 0.0 {
            (-q_value) * (awr + 1.0) / awr
        } else {
            0.0
        };

        levels.push(DiscreteLevelInfo { mt, q_value, threshold });
    }

    // Sort by MT number (which corresponds to ascending excitation energy)
    levels.sort_by_key(|l| l.mt);
    Ok(levels)
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
