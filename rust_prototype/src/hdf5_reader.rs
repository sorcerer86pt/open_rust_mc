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

/// Tabular angular distribution for scattering — mu(cosine) vs energy.
pub struct AngularDistribution {
    /// Energy grid (eV) at which distributions are tabulated.
    pub energies: Vec<f64>,
    /// Per-energy distribution: (mu, cdf) pairs for inverse CDF sampling.
    pub distributions: Vec<TabularMuDist>,
    /// Whether the distribution is in the center-of-mass frame.
    pub center_of_mass: bool,
}

/// Tabular mu distribution at a single energy — for inverse CDF sampling.
pub struct TabularMuDist {
    /// Cosine values, sorted ascending.
    pub mu: Vec<f64>,
    /// Cumulative distribution function values, sorted ascending [0, 1].
    pub cdf: Vec<f64>,
}

impl AngularDistribution {
    /// Sample the scattering cosine mu at a given energy.
    pub fn sample_mu(&self, energy: f64, rng: &mut crate::transport::rng::Rng) -> f64 {
        if self.energies.is_empty() {
            return 2.0 * rng.uniform() - 1.0; // isotropic fallback
        }

        // Find energy bracket
        let n = self.energies.len();
        let idx = if energy <= self.energies[0] {
            0
        } else if energy >= self.energies[n - 1] {
            n - 1
        } else {
            match self.energies.binary_search_by(|e| {
                e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less)
            }) {
                Ok(i) => i,
                Err(i) => if i > 0 { i - 1 } else { 0 },
            }
        };

        // Sample from the distribution at this energy index
        self.distributions[idx].sample(rng)
    }
}

impl TabularMuDist {
    /// Sample mu using inverse CDF method.
    fn sample(&self, rng: &mut crate::transport::rng::Rng) -> f64 {
        let xi = rng.uniform();
        let n = self.cdf.len();
        if n < 2 {
            return 2.0 * rng.uniform() - 1.0;
        }

        // Binary search on CDF
        let idx = match self.cdf.binary_search_by(|c| {
            c.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(i) => if i > 0 { i - 1 } else { 0 },
        };

        let idx = idx.min(n - 2);
        let cdf_lo = self.cdf[idx];
        let cdf_hi = self.cdf[idx + 1];
        let mu_lo = self.mu[idx];
        let mu_hi = self.mu[idx + 1];

        if (cdf_hi - cdf_lo).abs() < 1e-15 {
            return mu_lo;
        }

        // Linear interpolation within the CDF bracket
        let frac = (xi - cdf_lo) / (cdf_hi - cdf_lo);
        let mu = mu_lo + frac * (mu_hi - mu_lo);
        mu.clamp(-1.0, 1.0)
    }
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

/// Tabular outgoing energy distribution — for fission spectrum sampling.
pub struct EnergyDistribution {
    /// Incident energy grid (eV).
    pub energies: Vec<f64>,
    /// Per-energy outgoing energy distribution (E_out, cdf) for inverse CDF sampling.
    pub distributions: Vec<TabularEnergyDist>,
}

/// Tabular outgoing energy distribution at a single incident energy.
pub struct TabularEnergyDist {
    /// Outgoing energies (eV), sorted ascending.
    pub e_out: Vec<f64>,
    /// CDF values, sorted ascending [0, 1].
    pub cdf: Vec<f64>,
}

impl EnergyDistribution {
    /// Sample an outgoing energy at a given incident energy.
    pub fn sample(&self, incident_energy: f64, rng: &mut crate::transport::rng::Rng) -> f64 {
        if self.energies.is_empty() {
            return incident_energy; // fallback
        }

        let n = self.energies.len();
        let idx = if incident_energy <= self.energies[0] {
            0
        } else if incident_energy >= self.energies[n - 1] {
            n - 1
        } else {
            match self.energies.binary_search_by(|e| {
                e.partial_cmp(&incident_energy).unwrap_or(std::cmp::Ordering::Less)
            }) {
                Ok(i) => i,
                Err(i) => if i > 0 { i - 1 } else { 0 },
            }
        };

        self.distributions[idx].sample(rng)
    }
}

impl TabularEnergyDist {
    fn sample(&self, rng: &mut crate::transport::rng::Rng) -> f64 {
        let xi = rng.uniform();
        let n = self.cdf.len();
        if n < 2 {
            return self.e_out.first().copied().unwrap_or(1.0e6);
        }

        let idx = match self.cdf.binary_search_by(|c| {
            c.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(i) => if i > 0 { i - 1 } else { 0 },
        };

        let idx = idx.min(n - 2);
        let cdf_lo = self.cdf[idx];
        let cdf_hi = self.cdf[idx + 1];
        let e_lo = self.e_out[idx];
        let e_hi = self.e_out[idx + 1];

        if (cdf_hi - cdf_lo).abs() < 1e-15 {
            return e_lo;
        }

        let frac = (xi - cdf_lo) / (cdf_hi - cdf_lo);
        (e_lo + frac * (e_hi - e_lo)).max(1e-5)
    }
}

/// Read the fission energy distribution for the prompt neutron product.
pub fn read_fission_energy_dist(path: &Path) -> Result<Option<EnergyDistribution>> {
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

    let rxn = match nuc.group("reactions").and_then(|r| r.group("reaction_018")) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };

    // Find the prompt neutron product
    let subgroups = rxn.groups().unwrap_or_default();
    for product_name in &subgroups {
        if !product_name.starts_with("product_") { continue; }

        let product = match rxn.group(product_name) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let attrs = product.attrs().unwrap_or_default();
        let is_neutron = matches!(attrs.get("particle"), Some(hdf5_pure::AttrValue::String(s)) if s == "neutron");
        let is_prompt = matches!(attrs.get("emission_mode"), Some(hdf5_pure::AttrValue::String(s)) if s == "prompt");
        if !is_neutron || !is_prompt { continue; }

        // Navigate to distribution_0/energy/
        let dist = match product.group("distribution_0") {
            Ok(g) => g,
            Err(_) => continue,
        };
        let edist = match dist.group("energy") {
            Ok(g) => g,
            Err(_) => continue,
        };

        // Read energy grid (incident energies)
        let energy_ds = match edist.dataset("energy") {
            Ok(ds) => ds,
            Err(_) => continue,
        };
        let energies = energy_ds.read_f64().unwrap_or_default();
        if energies.is_empty() { continue; }

        // Read distribution dataset [3, N_total]
        let dist_ds = match edist.dataset("distribution") {
            Ok(ds) => ds,
            Err(_) => continue,
        };
        let dist_shape = dist_ds.shape().unwrap_or_default();
        let dist_raw = dist_ds.read_f64().unwrap_or_default();

        if dist_shape.len() != 2 || dist_shape[0] != 3 { continue; }
        let n_total = dist_shape[1] as usize;

        // Read offsets attribute
        let dist_attrs = dist_ds.attrs().unwrap_or_default();
        let offsets: Vec<usize> = if let Some(hdf5_pure::AttrValue::I64Array(arr)) = dist_attrs.get("offsets") {
            arr.iter().map(|&v| v as usize).collect()
        } else {
            let per_e = n_total / energies.len();
            (0..energies.len()).map(|i| i * per_e).collect()
        };

        // Parse per-energy distributions
        let e_out_values = &dist_raw[..n_total];
        let cdf_values = &dist_raw[2 * n_total..3 * n_total];

        let n_energies = energies.len();
        let mut distributions = Vec::with_capacity(n_energies);
        for i in 0..n_energies {
            let start = offsets.get(i).copied().unwrap_or(0);
            let end = offsets.get(i + 1).copied().unwrap_or(n_total).min(n_total);

            if start >= end || start >= n_total {
                distributions.push(TabularEnergyDist {
                    e_out: vec![1e5, 2e6],
                    cdf: vec![0.0, 1.0],
                });
                continue;
            }

            distributions.push(TabularEnergyDist {
                e_out: e_out_values[start..end].to_vec(),
                cdf: cdf_values[start..end].to_vec(),
            });
        }

        return Ok(Some(EnergyDistribution { energies, distributions }));
    }

    Ok(None)
}

/// Unresolved Resonance Range (URR) probability tables.
///
/// In the URR, cross-sections fluctuate statistically around the average.
/// The probability table captures these fluctuations as a set of "bands"
/// (discrete realizations). Each band has consistent cross-section factors
/// for all reaction channels.
pub struct UrrProbabilityTables {
    /// Energy grid for the URR (eV), sorted ascending.
    pub energies: Vec<f64>,
    /// Number of probability bands (typically 20).
    pub n_bands: usize,
    /// Cumulative probabilities: [n_energy][n_bands].
    pub cum_prob: Vec<Vec<f64>>,
    /// Cross-section factors for total: [n_energy][n_bands].
    pub total_factor: Vec<Vec<f64>>,
    /// Cross-section factors for elastic: [n_energy][n_bands].
    pub elastic_factor: Vec<Vec<f64>>,
    /// Cross-section factors for fission: [n_energy][n_bands].
    pub fission_factor: Vec<Vec<f64>>,
    /// Cross-section factors for capture: [n_energy][n_bands].
    pub capture_factor: Vec<Vec<f64>>,
    /// Whether to multiply by smooth (average) cross-sections.
    pub multiply_smooth: bool,
}

/// Sampled URR cross-section multipliers for one collision.
#[derive(Debug, Clone, Copy)]
pub struct UrrFactors {
    pub total: f64,
    pub elastic: f64,
    pub fission: f64,
    pub capture: f64,
}

impl UrrProbabilityTables {
    /// Check if the given energy is within the URR range.
    #[inline]
    pub fn in_range(&self, energy: f64) -> bool {
        if self.energies.is_empty() { return false; }
        energy >= self.energies[0] && energy <= *self.energies.last().unwrap_or(&0.0)
    }

    /// Sample URR factors at a given energy.
    ///
    /// Returns cross-section multipliers if `multiply_smooth=true`,
    /// or absolute cross-sections if `multiply_smooth=false`.
    pub fn sample(&self, energy: f64, xi: f64) -> UrrFactors {
        let n_e = self.energies.len();
        if n_e == 0 {
            return UrrFactors { total: 1.0, elastic: 1.0, fission: 1.0, capture: 1.0 };
        }

        // Find energy bracket
        let idx = match self.energies.binary_search_by(|e| {
            e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(i) => if i > 0 { i - 1 } else { 0 },
        };
        let idx = idx.min(n_e - 1);

        // Find the probability band using cumulative probability
        let cum = &self.cum_prob[idx];
        let mut band = 0;
        for (j, &cp) in cum.iter().enumerate() {
            if xi <= cp {
                band = j;
                break;
            }
            band = j;
        }

        UrrFactors {
            total: self.total_factor[idx][band],
            elastic: self.elastic_factor[idx][band],
            fission: self.fission_factor[idx][band],
            capture: self.capture_factor[idx][band],
        }
    }
}

/// Read URR probability tables from HDF5.
///
/// Data is at: `{nuclide}/urr/{temp}/`
/// - `energy`: [N_E] energy grid
/// - `table`: [N_E, 6, N_bands] probability tables
///   Channel 0: cumulative probability
///   Channel 1: total XS factor
///   Channel 2: elastic XS factor
///   Channel 3: fission XS factor
///   Channel 4: capture XS factor
///   Channel 5: heating (unused)
pub fn read_urr_tables(path: &Path, temp_label: &str) -> Result<Option<UrrProbabilityTables>> {
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

    let urr = match nuc.group("urr") {
        Ok(g) => g,
        Err(_) => return Ok(None), // No URR data
    };

    let temp_group = match urr.group(temp_label) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };

    // Read attributes
    let attrs = temp_group.attrs().unwrap_or_default();
    let multiply_smooth = matches!(
        attrs.get("multiply_smooth"),
        Some(hdf5_pure::AttrValue::I64(1))
    );

    // Read energy grid
    let energy_ds = match temp_group.dataset("energy") {
        Ok(ds) => ds,
        Err(_) => return Ok(None),
    };
    let energies = energy_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read urr energy: {e}"),
    })?;
    let n_e = energies.len();

    // Read table: shape [N_E, 6, N_bands]
    let table_ds = match temp_group.dataset("table") {
        Ok(ds) => ds,
        Err(_) => return Ok(None),
    };
    let table_shape = table_ds.shape().unwrap_or_default();
    let table_raw = table_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read urr table: {e}"),
    })?;

    if table_shape.len() != 3 || table_shape[0] as usize != n_e || table_shape[1] != 6 {
        return Ok(None);
    }
    let n_bands = table_shape[2] as usize;
    let n_channels = 6_usize;

    // Parse the 3D table: element at [e, c, b] = raw[e * 6 * n_bands + c * n_bands + b]
    let mut cum_prob = Vec::with_capacity(n_e);
    let mut total_factor = Vec::with_capacity(n_e);
    let mut elastic_factor = Vec::with_capacity(n_e);
    let mut fission_factor = Vec::with_capacity(n_e);
    let mut capture_factor = Vec::with_capacity(n_e);

    for e in 0..n_e {
        let base = e * n_channels * n_bands;
        let prob: Vec<f64> = (0..n_bands).map(|b| table_raw[base + 0 * n_bands + b]).collect();
        let total: Vec<f64> = (0..n_bands).map(|b| table_raw[base + 1 * n_bands + b]).collect();
        let elastic: Vec<f64> = (0..n_bands).map(|b| table_raw[base + 2 * n_bands + b]).collect();
        let fission: Vec<f64> = (0..n_bands).map(|b| table_raw[base + 3 * n_bands + b]).collect();
        let capture: Vec<f64> = (0..n_bands).map(|b| table_raw[base + 4 * n_bands + b]).collect();

        cum_prob.push(prob);
        total_factor.push(total);
        elastic_factor.push(elastic);
        fission_factor.push(fission);
        capture_factor.push(capture);
    }

    Ok(Some(UrrProbabilityTables {
        energies,
        n_bands,
        cum_prob,
        total_factor,
        elastic_factor,
        fission_factor,
        capture_factor,
        multiply_smooth,
    }))
}

/// Read the angular distribution for a specific reaction from HDF5.
///
/// Angular data is at: `{nuclide}/reactions/reaction_{MT}/product_0/distribution_0/angle/`
/// - `energy`: [N_E] energy grid
/// - `mu`: [3, N_total] packed (mu, pdf, cdf) data
///
/// The `mu` dataset has an `offsets` attribute indicating where each energy's
/// distribution starts in the packed array.
pub fn read_angular_distribution(path: &Path, mt: u32) -> Result<Option<AngularDistribution>> {
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

    // Check if this reaction is in center-of-mass frame
    let rxn_name = format!("reaction_{mt:03}");
    let reactions = match nuc.group("reactions") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let rxn = match reactions.group(&rxn_name) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let rxn_attrs = rxn.attrs().unwrap_or_default();
    let center_of_mass = matches!(
        rxn_attrs.get("center_of_mass"),
        Some(hdf5_pure::AttrValue::I64(1))
    );

    // Navigate to product_0/distribution_0/angle/
    let product = match rxn.group("product_0") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let dist = match product.group("distribution_0") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let angle = match dist.group("angle") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };

    // Read energy grid
    let energy_ds = match angle.dataset("energy") {
        Ok(ds) => ds,
        Err(_) => return Ok(None),
    };
    let energies = energy_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read angle/energy: {e}"),
    })?;
    let n_energies = energies.len();
    if n_energies == 0 {
        return Ok(None);
    }

    // Read mu dataset
    let mu_ds = match angle.dataset("mu") {
        Ok(ds) => ds,
        Err(_) => return Ok(None),
    };
    let mu_shape = mu_ds.shape().unwrap_or_default();
    let mu_raw = mu_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read angle/mu: {e}"),
    })?;

    if mu_shape.len() != 2 || mu_shape[0] != 3 {
        return Ok(None);
    }
    let n_total = mu_shape[1] as usize;

    // Try to read offsets attribute from mu dataset
    let mu_attrs = mu_ds.attrs().unwrap_or_default();
    let offsets: Vec<usize> = if let Some(hdf5_pure::AttrValue::I64Array(arr)) = mu_attrs.get("offsets") {
        arr.iter().map(|&v| v as usize).collect()
    } else if let Some(hdf5_pure::AttrValue::I64(v)) = mu_attrs.get("offsets") {
        vec![*v as usize]
    } else {
        // No offsets attribute — try uniform distribution
        // Each energy gets n_total / n_energies points
        let per_e = n_total / n_energies;
        (0..n_energies).map(|i| i * per_e).collect()
    };

    // Parse per-energy distributions
    // mu_raw is stored as [3, N_total] in row-major: first N_total values are row 0 (mu),
    // next N_total are row 1 (pdf), next N_total are row 2 (cdf)
    let mu_values = &mu_raw[..n_total];
    let _pdf_values = &mu_raw[n_total..2 * n_total];
    let cdf_values = &mu_raw[2 * n_total..3 * n_total];

    let mut distributions = Vec::with_capacity(n_energies);
    for i in 0..n_energies {
        let start = offsets.get(i).copied().unwrap_or(0);
        let end = offsets.get(i + 1).copied().unwrap_or(n_total);
        let end = end.min(n_total);

        if start >= end || start >= n_total {
            // Empty distribution — isotropic
            distributions.push(TabularMuDist {
                mu: vec![-1.0, 1.0],
                cdf: vec![0.0, 1.0],
            });
            continue;
        }

        let mu_slice = mu_values[start..end].to_vec();
        let cdf_slice = cdf_values[start..end].to_vec();
        distributions.push(TabularMuDist { mu: mu_slice, cdf: cdf_slice });
    }

    Ok(Some(AngularDistribution {
        energies,
        distributions,
        center_of_mass,
    }))
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
