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
        let nuclide_name = root
            .groups()
            .map_err(|e| SvdError::Hdf5 {
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

        let temperatures: Vec<f64> = temp_labels
            .iter()
            .filter_map(|l| parse_temp_kelvin(l))
            .collect();

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
        let reactions_group = nuclide_group
            .group("reactions")
            .map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("cannot open /{nuclide_name}/reactions: {e}"),
            })?;
        let rxn_group = reactions_group
            .group(&rxn_name)
            .map_err(|e| SvdError::Hdf5 {
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

    pub fn n_energy(&self) -> usize {
        self.energies.len()
    }
    pub fn n_temp(&self) -> usize {
        self.temperatures.len()
    }

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

/// Cached HDF5 file reader for a single nuclide.
///
/// Opens the file once, pre-reads the energy grids and unionized grid,
/// then provides fast access for reading individual reactions and metadata.
/// Eliminates redundant file opens and energy grid unionization.
pub struct NuclideFileReader {
    /// The opened HDF5 file (kept alive for the Group references).
    file: hdf5_pure::File,
    /// Nuclide name (e.g., "U235").
    pub nuclide_name: String,
    /// Temperature labels, sorted ascending (e.g., ["250K", "294K", ...]).
    pub temp_labels: Vec<String>,
    /// Temperatures in Kelvin.
    pub temperatures: Vec<f64>,
    /// Per-temperature energy grids.
    pub energy_grids: Vec<Vec<f64>>,
    /// Unionized energy grid.
    pub union_grid: Vec<f64>,
}

impl NuclideFileReader {
    /// Open a nuclide HDF5 file and preload the energy grids.
    pub fn open(path: &Path) -> Result<Self> {
        let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("{e}"),
        })?;

        let root = file.root();
        let nuclide_name = root
            .groups()
            .map_err(|e| SvdError::Hdf5 {
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

        let energy_group = nuc.group("energy").map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot open energy group: {e}"),
        })?;

        let mut temp_labels = energy_group.datasets().unwrap_or_default();
        temp_labels.retain(|l| parse_temp_kelvin(l).is_some_and(|t| t > 0.0));
        temp_labels.sort_by(|a, b| {
            let va = parse_temp_kelvin(a).unwrap_or(0.0);
            let vb = parse_temp_kelvin(b).unwrap_or(0.0);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let temperatures: Vec<f64> = temp_labels
            .iter()
            .filter_map(|l| parse_temp_kelvin(l))
            .collect();

        let mut energy_grids = Vec::with_capacity(temp_labels.len());
        for label in &temp_labels {
            let ds = energy_group.dataset(label).map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("cannot read energy/{label}: {e}"),
            })?;
            energy_grids.push(ds.read_f64().map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("cannot read energy/{label} data: {e}"),
            })?);
        }

        let mut union_grid: Vec<f64> = energy_grids.iter().flatten().copied().collect();
        union_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        union_grid.dedup();

        Ok(Self {
            file,
            nuclide_name,
            temp_labels,
            temperatures,
            energy_grids,
            union_grid,
        })
    }

    /// Read a single reaction's cross-section data using the cached grids.
    pub fn read_reaction(&self, mt: u32) -> Result<NuclideData> {
        let root = self.file.root();
        let nuc = root.group(&self.nuclide_name).map_err(|e| SvdError::Hdf5 {
            path: self.nuclide_name.clone(),
            detail: format!("{e}"),
        })?;

        let rxn_name = format!("reaction_{mt:03}");
        let reactions = nuc.group("reactions").map_err(|e| SvdError::Hdf5 {
            path: self.nuclide_name.clone(),
            detail: format!("cannot open reactions: {e}"),
        })?;
        let rxn = reactions.group(&rxn_name).map_err(|e| SvdError::Hdf5 {
            path: self.nuclide_name.clone(),
            detail: format!("cannot open {rxn_name}: {e}"),
        })?;

        let mut xs_per_temp = Vec::with_capacity(self.temp_labels.len());
        for (t_idx, label) in self.temp_labels.iter().enumerate() {
            let temp_group = rxn.group(label).map_err(|e| SvdError::Hdf5 {
                path: self.nuclide_name.clone(),
                detail: format!("cannot open {rxn_name}/{label}: {e}"),
            })?;
            let xs_ds = temp_group.dataset("xs").map_err(|e| SvdError::Hdf5 {
                path: self.nuclide_name.clone(),
                detail: format!("cannot read {rxn_name}/{label}/xs: {e}"),
            })?;
            let xs_raw = xs_ds.read_f64().map_err(|e| SvdError::Hdf5 {
                path: self.nuclide_name.clone(),
                detail: format!("cannot read {rxn_name}/{label}/xs data: {e}"),
            })?;

            let e_grid = &self.energy_grids[t_idx];
            let n_grid = e_grid.len();
            let n_xs = xs_raw.len();

            let xs_full = if n_xs < n_grid {
                let mut full = vec![0.0_f64; n_grid];
                let offset = n_grid - n_xs;
                full[offset..].copy_from_slice(&xs_raw);
                full
            } else {
                xs_raw
            };

            let xs_interp = interpolate_to_grid(e_grid, &xs_full, &self.union_grid);
            xs_per_temp.push(xs_interp);
        }

        Ok(NuclideData {
            temperatures: self.temperatures.clone(),
            temp_labels: self.temp_labels.clone(),
            energies: self.union_grid.clone(),
            xs_per_temp,
            mt,
        })
    }

    /// Compute pointwise XS for all 7 channels (el, inel, n2n, n3n, fis, cap, total)
    /// at one temperature on the unionized grid. Returns [n_energy * 7] flat array.
    /// Channel order: 0=elastic, 1=inelastic, 2=n2n, 3=n3n, 4=fission, 5=capture, 6=total.
    pub fn compute_pointwise_xs(&self, temp_idx: usize) -> Option<Vec<f64>> {
        let n = self.union_grid.len();
        if n == 0 {
            return None;
        }

        let read = |mt: u32| -> Vec<f64> {
            self.read_reaction(mt)
                .ok()
                .and_then(|d| d.xs_per_temp.get(temp_idx).cloned())
                .unwrap_or_else(|| vec![0.0; n])
        };

        let elastic = read(2);
        let mut inelastic = read(4);
        if inelastic.iter().all(|&v| v == 0.0) {
            for mt in 51..=91 {
                if let Ok(data) = self.read_reaction(mt)
                    && let Some(xs) = data.xs_per_temp.get(temp_idx)
                {
                    for i in 0..n.min(xs.len()) {
                        inelastic[i] += xs[i].max(0.0);
                    }
                }
            }
        }
        let n2n = read(16);
        let n3n = read(17);
        let fission = read(18);
        let capture = read(102);
        let total = self.compute_total_xs(temp_idx).unwrap_or_else(|| {
            (0..n)
                .map(|i| elastic[i] + inelastic[i] + n2n[i] + n3n[i] + fission[i] + capture[i])
                .collect()
        });

        let mut out = vec![0.0_f64; n * 7];
        for i in 0..n {
            out[i * 7] = elastic[i].max(0.0);
            out[i * 7 + 1] = inelastic[i].max(0.0);
            out[i * 7 + 2] = n2n[i].max(0.0);
            out[i * 7 + 3] = n3n[i].max(0.0);
            out[i * 7 + 4] = fission[i].max(0.0);
            out[i * 7 + 5] = capture[i].max(0.0);
            out[i * 7 + 6] = total[i].max(0.0);
        }
        println!(
            "    Pointwise XS: {} energy pts × 7 channels = {:.1} KB",
            n,
            out.len() as f64 * 8.0 / 1024.0
        );
        Some(out)
    }

    /// List all physics reaction MT numbers in the HDF5 file (MT < 200).
    pub fn list_reaction_mts(&self) -> Vec<u32> {
        let root = self.file.root();
        let nuc = match root.group(&self.nuclide_name) {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let reactions = match nuc.group("reactions") {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let groups = match reactions.groups() {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        groups
            .iter()
            .filter_map(|name| {
                let mt: u32 = name.strip_prefix("reaction_")?.parse().ok()?;
                if mt < 200 { Some(mt) } else { None }
            })
            .collect()
    }

    /// Compute total XS from HDF5 at one temperature on the unionized grid.
    ///
    /// Uses MT=2 (elastic) + MT=3 (nonelastic) when available (exact match to OpenMC).
    /// Otherwise sums individual leaf reactions, excluding sum-MTs (1, 3, 4).
    pub fn compute_total_xs(&self, temp_idx: usize) -> Option<Vec<f64>> {
        let mts = self.list_reaction_mts();
        if mts.is_empty() {
            return None;
        }

        let n = self.union_grid.len();
        let has_mt3 = mts.contains(&3);

        if has_mt3 {
            let el = self.read_reaction(2).ok()?;
            let nel = self.read_reaction(3).ok()?;
            let xs_el = el.xs_per_temp.get(temp_idx)?;
            let xs_nel = nel.xs_per_temp.get(temp_idx)?;
            let total: Vec<f64> = (0..n)
                .map(|i| {
                    xs_el.get(i).copied().unwrap_or(0.0).max(0.0)
                        + xs_nel.get(i).copied().unwrap_or(0.0).max(0.0)
                })
                .collect();
            println!("    Total XS: MT=2 + MT=3 (exact nonelastic from HDF5)");
            return Some(total);
        }

        // No MT=3: sum leaf reactions (exclude MT=1,3,4 which are sums of others)
        let mut total = vec![0.0_f64; n];
        let mut count = 0_u32;
        for mt in &mts {
            if matches!(*mt, 1 | 3 | 4) {
                continue;
            }
            if let Ok(data) = self.read_reaction(*mt)
                && let Some(xs) = data.xs_per_temp.get(temp_idx)
            {
                for i in 0..n.min(xs.len()) {
                    total[i] += xs[i].max(0.0);
                }
                count += 1;
            }
        }

        if count > 0 {
            println!("    Total XS: summed {count} leaf reactions (no MT=3 available)");
            Some(total)
        } else {
            None
        }
    }

    /// Read the AWR attribute.
    pub fn awr(&self) -> Result<f64> {
        let root = self.file.root();
        let nuc = root.group(&self.nuclide_name).map_err(|e| SvdError::Hdf5 {
            path: self.nuclide_name.clone(),
            detail: format!("{e}"),
        })?;
        let attrs = nuc.attrs().map_err(|e| SvdError::Hdf5 {
            path: self.nuclide_name.clone(),
            detail: format!("cannot read attrs: {e}"),
        })?;
        if let Some(hdf5_pure::AttrValue::F64(awr)) = attrs.get("atomic_weight_ratio") {
            Ok(*awr)
        } else {
            Err(SvdError::Hdf5 {
                path: self.nuclide_name.clone(),
                detail: "missing atomic_weight_ratio".into(),
            })
        }
    }

    /// Read nu-bar (total neutron yield) from fission products.
    pub fn nu_bar(&self) -> Result<NuBarTable> {
        let root = self.file.root();
        let nuc = root.group(&self.nuclide_name).map_err(|e| SvdError::Hdf5 {
            path: self.nuclide_name.clone(),
            detail: format!("{e}"),
        })?;
        let rxn = match nuc.group("reactions").and_then(|r| r.group("reaction_018")) {
            Ok(g) => g,
            Err(_) => {
                return Ok(NuBarTable {
                    energies: vec![],
                    values: vec![],
                });
            }
        };
        read_nu_bar_from_group(&rxn)
    }

    /// Read discrete inelastic levels.
    pub fn discrete_levels(&self, awr: f64) -> Vec<DiscreteLevelInfo> {
        let root = self.file.root();
        let nuc = match root.group(&self.nuclide_name) {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let reactions = match nuc.group("reactions") {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let rxn_names = reactions.groups().unwrap_or_default();

        let mut levels = Vec::new();
        for name in &rxn_names {
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
                continue;
            };
            let threshold = if q_value < 0.0 {
                (-q_value) * (awr + 1.0) / awr
            } else {
                0.0
            };
            levels.push(DiscreteLevelInfo {
                mt,
                q_value,
                threshold,
            });
        }
        levels.sort_by_key(|l| l.mt);
        levels
    }

    /// Read angular distribution for a reaction.
    pub fn angular_distribution(&self, mt: u32) -> Option<AngularDistribution> {
        read_angular_dist_from_file(&self.file, &self.nuclide_name, mt)
    }

    /// Read fission energy distribution.
    pub fn fission_energy_dist(&self) -> Option<EnergyDistribution> {
        read_fission_edist_from_file(&self.file, &self.nuclide_name)
    }

    /// Read the outgoing-energy distribution for any reaction MT that
    /// stores `product_0/distribution_0/energy/{energy, distribution}`
    /// in the ContinuousTabular (ENDF Law 4) layout. Used for MT=91
    /// (continuum inelastic), MT=16 (n,2n), MT=17 (n,3n). The returned
    /// distribution is in whatever frame the reaction declares via its
    /// `center_of_mass` attribute; the caller must handle the frame.
    pub fn reaction_energy_dist(&self, mt: u32) -> Option<EnergyDistribution> {
        read_reaction_edist_from_file(&self.file, &self.nuclide_name, mt)
    }

    /// Read URR probability tables.
    pub fn urr_tables(&self, temp_label: &str) -> Option<UrrProbabilityTables> {
        read_urr_from_file(&self.file, &self.nuclide_name, temp_label)
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
        if n == 0 {
            return 2.43;
        } // fallback
        if n == 1 {
            return self.values[0];
        }
        if energy <= self.energies[0] {
            return self.values[0];
        }
        if energy >= self.energies[n - 1] {
            return self.values[n - 1];
        }

        // Binary search
        let idx = match self
            .energies
            .binary_search_by(|e| e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
        {
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
///
/// ENDF/B-VII.1 stores angular distributions as (mu, pdf, cdf) triples with
/// an `interpolation` attribute indicating how the PDF varies between
/// tabulated points (1 = histogram / constant PDF per bin → linear CDF,
/// 2 = linear-linear / PDF linearly interpolated → **quadratic** CDF).
/// For uranium elastic scattering in the forward-peaked fast range, every
/// energy uses `interpolation=2`. Treating those as histogram (linear CDF
/// inversion) systematically under-samples the forward peak and shifts
/// leakage-dominated benchmarks (e.g. Godiva) by hundreds of pcm.
pub struct TabularMuDist {
    /// Cosine values, sorted ascending.
    pub mu: Vec<f64>,
    /// PDF values at each mu breakpoint.
    pub pdf: Vec<f64>,
    /// Cumulative distribution function values, sorted ascending [0, 1].
    pub cdf: Vec<f64>,
    /// If true, PDF is constant within each bin (histogram interpolation,
    /// OpenMC `interpolation=1`) and we invert a linear CDF. If false,
    /// PDF is linearly interpolated (OpenMC `interpolation=2`) and we solve
    /// a quadratic for the inverse CDF.
    pub histogram: bool,
}

impl AngularDistribution {
    /// Sample the scattering cosine mu at a given energy.
    ///
    /// Uses correlated CDF interpolation: the same random number is used
    /// We follow OpenMC's convention (src/distribution_angle.cpp):
    /// stochastic bin selection plus a single CDF inversion inside the
    /// chosen bin, rather than a correlated double inversion with
    /// linear interpolation. With interpolation factor
    /// `r = (E - E_lo) / (E_hi - E_lo)`, we pick the high-bin
    /// distribution with probability `r` and sample from it directly.
    /// Uses two random draws total (one for bin, one for mu) — the
    /// same number OpenMC uses.
    pub fn sample_mu(&self, energy: f64, rng: &mut crate::transport::rng::Rng) -> f64 {
        if self.energies.is_empty() {
            return 2.0 * rng.uniform() - 1.0; // isotropic fallback
        }

        let n = self.energies.len();
        if energy <= self.energies[0] {
            return self.distributions[0].sample(rng);
        }
        if energy >= self.energies[n - 1] {
            return self.distributions[n - 1].sample(rng);
        }

        let idx = match self
            .energies
            .binary_search_by(|e| e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => return self.distributions[i].sample(rng),
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        };

        if idx + 1 >= n {
            return self.distributions[idx].sample(rng);
        }

        let e_lo = self.energies[idx];
        let e_hi = self.energies[idx + 1];
        let r = (energy - e_lo) / (e_hi - e_lo);

        // OpenMC-style stochastic bin selection:
        //   if r > ξ_bin: take high bin; else take low bin.
        //   Then sample mu from the chosen bin with a fresh ξ_mu.
        let pick_hi = rng.uniform() < r;
        let dist = if pick_hi {
            &self.distributions[idx + 1]
        } else {
            &self.distributions[idx]
        };
        dist.sample(rng).clamp(-1.0, 1.0)
    }
}

impl TabularMuDist {
    /// Sample mu using inverse CDF, drawing a fresh random number.
    fn sample(&self, rng: &mut crate::transport::rng::Rng) -> f64 {
        self.sample_with_xi(rng.uniform())
    }

    /// Sample mu using inverse CDF with a pre-drawn random number.
    ///
    /// For histogram interpolation the PDF is constant within each bin so
    /// the CDF is linear and a single linear inversion gives mu directly.
    /// For linear-linear interpolation the PDF is linearly interpolated
    /// between (mu_lo, pdf_lo) and (mu_hi, pdf_hi); integrating, the CDF
    /// inside the bin is quadratic in (mu - mu_lo) and we invert it with
    /// the standard quadratic formula, taking the physical root in
    /// [0, mu_hi - mu_lo]. This matches OpenMC's Tabular::sample in
    /// src/distribution.cpp.
    fn sample_with_xi(&self, xi: f64) -> f64 {
        let n = self.cdf.len();
        if n < 2 {
            return 2.0 * xi - 1.0;
        }

        let idx = match self
            .cdf
            .binary_search_by(|c| c.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i,
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        };

        let idx = idx.min(n - 2);
        let cdf_lo = self.cdf[idx];
        let cdf_hi = self.cdf[idx + 1];
        let mu_lo = self.mu[idx];
        let mu_hi = self.mu[idx + 1];
        let dmu = mu_hi - mu_lo;

        if (cdf_hi - cdf_lo).abs() < 1e-15 || dmu.abs() < 1e-15 {
            return mu_lo.clamp(-1.0, 1.0);
        }

        if self.histogram {
            // Constant PDF in bin → linear CDF → single step.
            let frac = (xi - cdf_lo) / (cdf_hi - cdf_lo);
            return (mu_lo + frac * dmu).clamp(-1.0, 1.0);
        }

        // Linear-linear: PDF(mu) = pdf_lo + (pdf_hi - pdf_lo)/dmu * (mu - mu_lo).
        // CDF(mu) = cdf_lo + pdf_lo * (mu - mu_lo)
        //         + 0.5 * (pdf_hi - pdf_lo)/dmu * (mu - mu_lo)^2
        // Setting CDF(mu) = xi and solving for x = mu - mu_lo:
        //   a x^2 + b x + c = 0, with a = (pdf_hi - pdf_lo)/(2 dmu),
        //   b = pdf_lo, c = cdf_lo - xi.
        let pdf_lo = if idx < self.pdf.len() {
            self.pdf[idx]
        } else {
            0.0
        };
        let pdf_hi = if idx + 1 < self.pdf.len() {
            self.pdf[idx + 1]
        } else {
            pdf_lo
        };
        let a = (pdf_hi - pdf_lo) / (2.0 * dmu);
        let b = pdf_lo;
        let c = cdf_lo - xi;

        let x = if a.abs() < 1e-14 {
            // Degenerate: PDF is constant in this bin. Fall back to linear.
            if b.abs() < 1e-30 {
                // No PDF info → uniform in bin.
                (xi - cdf_lo) / (cdf_hi - cdf_lo) * dmu
            } else {
                -c / b
            }
        } else {
            // Discriminant clamped to zero to avoid spurious NaNs from
            // sub-ULP negative values when xi is essentially cdf_lo.
            let disc = (b * b - 4.0 * a * c).max(0.0);
            let sqrt_disc = disc.sqrt();
            // The physical root is the one in [0, dmu]. With a > 0 and
            // c <= 0 the "+" root is always the right one; with a < 0
            // (PDF decreasing) it's also the "+" root. Using "+" works
            // for both signs because we picked b = pdf_lo ≥ 0.
            (-b + sqrt_disc) / (2.0 * a)
        };

        let x = x.clamp(0.0, dmu);
        (mu_lo + x).clamp(-1.0, 1.0)
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
    let nuclide_name = root
        .groups()
        .map_err(|e| SvdError::Hdf5 {
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
        Err(_) => {
            return Ok(NuBarTable {
                energies: vec![],
                values: vec![],
            });
        }
    };

    // Collect all neutron product yield tables (prompt + delayed)
    let subgroups = rxn_group.groups().unwrap_or_default();
    let mut prompt_table: Option<NuBarTable> = None;
    let mut delayed_constants: Vec<f64> = Vec::new();

    for product_name in &subgroups {
        if !product_name.starts_with("product_") {
            continue;
        }

        let product = match rxn_group.group(product_name) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let attrs = product.attrs().unwrap_or_default();

        let is_neutron = matches!(
            attrs.get("particle"),
            Some(hdf5_pure::AttrValue::String(s)) if s == "neutron"
        );
        if !is_neutron {
            continue;
        }

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
        None => Ok(NuBarTable {
            energies: vec![],
            values: vec![],
        }),
    }
}

/// Read the atomic weight ratio from the HDF5 nuclide file.
pub fn read_awr(path: &Path) -> Result<f64> {
    let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    let root = file.root();
    let nuclide_name = root
        .groups()
        .map_err(|e| SvdError::Hdf5 {
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
    let nuclide_name = root
        .groups()
        .map_err(|e| SvdError::Hdf5 {
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

        levels.push(DiscreteLevelInfo {
            mt,
            q_value,
            threshold,
        });
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
    /// PDF values at each `e_out`, used for the quadratic lin-lin CDF
    /// inversion. Populated from HDF5 channel 1 when available; if the
    /// reader left it empty, `sample_with_xi` falls back to the linear
    /// (histogram-PDF) approximation.
    pub pdf: Vec<f64>,
    /// CDF values, sorted ascending [0, 1].
    pub cdf: Vec<f64>,
}

impl EnergyDistribution {
    /// Sample an outgoing energy at a given incident energy.
    ///
    /// Matches OpenMC's ContinuousTabular::sample (ENDF Law 4 / Law 61)
    /// convention (src/distribution_energy.cpp): stochastic bin
    /// selection between the two bracketing incident-energy bins, plus
    /// a scaled kinematic adjustment that remaps the sampled outgoing
    /// energy from the chosen bin's [E_min, E_max] to the
    /// linearly-interpolated bounds [E_1, E_K] between both bins.
    /// Uses two random draws total (bin selection + CDF inversion).
    pub fn sample(&self, incident_energy: f64, rng: &mut crate::transport::rng::Rng) -> f64 {
        if self.energies.is_empty() {
            return incident_energy;
        }

        let n = self.energies.len();
        if incident_energy <= self.energies[0] {
            return self.distributions[0].sample(rng).max(1e-5);
        }
        if incident_energy >= self.energies[n - 1] {
            return self.distributions[n - 1].sample(rng).max(1e-5);
        }

        let idx = match self.energies.binary_search_by(|e| {
            e.partial_cmp(&incident_energy)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => return self.distributions[i].sample(rng).max(1e-5),
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        };

        if idx + 1 >= n {
            return self.distributions[idx].sample(rng).max(1e-5);
        }

        let e_lo = self.energies[idx];
        let e_hi = self.energies[idx + 1];
        let r = (incident_energy - e_lo) / (e_hi - e_lo);

        // Stochastic bin selection: take high bin with probability r.
        let pick_hi = rng.uniform() < r;
        let l = if pick_hi { idx + 1 } else { idx };
        let dist_l = &self.distributions[l];

        // Sample E_out from the chosen bin.
        let e_out = dist_l.sample(rng);

        // Scaled kinematic adjustment: map E_out from the chosen bin's
        // [e_out_min_l, e_out_max_l] to the linearly interpolated
        // [E_1, E_K] where
        //   E_1 = (1-r) e_out_min[idx] + r e_out_min[idx+1]
        //   E_K = (1-r) e_out_max[idx] + r e_out_max[idx+1]
        let (el1_lo, el1_hi) = dist_l.bounds();
        let (ea_lo, ea_hi) = self.distributions[idx].bounds();
        let (eb_lo, eb_hi) = self.distributions[idx + 1].bounds();
        let e1 = (1.0 - r) * ea_lo + r * eb_lo;
        let ek = (1.0 - r) * ea_hi + r * eb_hi;
        let span_l = el1_hi - el1_lo;
        let adjusted = if span_l.abs() < 1e-30 {
            e_out
        } else {
            e1 + (e_out - el1_lo) * (ek - e1) / span_l
        };
        adjusted.max(1e-5)
    }
}

impl TabularEnergyDist {
    fn sample(&self, rng: &mut crate::transport::rng::Rng) -> f64 {
        self.sample_with_xi(rng.uniform())
    }

    /// Return (E_out_min, E_out_max) for this incident-energy bin, used
    /// by EnergyDistribution::sample for the OpenMC scaled kinematic
    /// adjustment.
    fn bounds(&self) -> (f64, f64) {
        match self.e_out.last() {
            None => (0.0, 0.0),
            Some(&last) => (self.e_out[0], last),
        }
    }

    /// Quadratic lin-lin CDF inversion (OpenMC Tabular::sample,
    /// src/distribution.cpp). With PDF `p_k` at `E_k`, slope
    /// `m = (p_{k+1} - p_k)/(E_{k+1} - E_k)`, and `Δc = ξ − c_k`,
    ///     E = E_k + (√(p_k² + 2 m Δc) − p_k) / m   (m ≠ 0)
    ///     E = E_k + Δc / p_k                        (histogram, m = 0)
    /// Falls back to linear `(cdf, E_out)` when PDF is unavailable —
    /// which matches the old implementation bit-for-bit.
    fn sample_with_xi(&self, xi: f64) -> f64 {
        let n = self.cdf.len();
        if n < 2 {
            return self.e_out.first().copied().unwrap_or(1.0e6);
        }

        let idx = match self
            .cdf
            .binary_search_by(|c| c.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i,
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        };

        let idx = idx.min(n - 2);
        let cdf_lo = self.cdf[idx];
        let cdf_hi = self.cdf[idx + 1];
        let e_lo = self.e_out[idx];
        let e_hi = self.e_out[idx + 1];
        let de = e_hi - e_lo;

        if (cdf_hi - cdf_lo).abs() < 1e-15 {
            return e_lo.max(1e-5);
        }

        // Quadratic lin-lin path requires PDF aligned 1:1 with e_out.
        if self.pdf.len() == n && de > 0.0 {
            let p_lo = self.pdf[idx];
            let p_hi = self.pdf[idx + 1];
            let m = (p_hi - p_lo) / de;
            let dc = xi - cdf_lo;
            let e = if m.abs() < 1e-30 {
                if p_lo.abs() < 1e-30 {
                    e_lo
                } else {
                    e_lo + dc / p_lo
                }
            } else {
                let disc = p_lo * p_lo + 2.0 * m * dc;
                if disc < 0.0 {
                    e_lo
                } else {
                    e_lo + (disc.sqrt() - p_lo) / m
                }
            };
            return e.max(1e-5);
        }

        // Fallback: histogram-PDF (linear CDF) inversion.
        let frac = (xi - cdf_lo) / (cdf_hi - cdf_lo);
        (e_lo + frac * de).max(1e-5)
    }
}

/// Read the fission energy distribution for the prompt neutron product.
pub fn read_fission_energy_dist(path: &Path) -> Result<Option<EnergyDistribution>> {
    let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    let root = file.root();
    let nuclide_name = root
        .groups()
        .map_err(|e| SvdError::Hdf5 {
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
        if !product_name.starts_with("product_") {
            continue;
        }

        let product = match rxn.group(product_name) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let attrs = product.attrs().unwrap_or_default();
        let is_neutron = matches!(attrs.get("particle"), Some(hdf5_pure::AttrValue::String(s)) if s == "neutron");
        let is_prompt = matches!(attrs.get("emission_mode"), Some(hdf5_pure::AttrValue::String(s)) if s == "prompt");
        if !is_neutron || !is_prompt {
            continue;
        }

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
        if energies.is_empty() {
            continue;
        }

        // Read distribution dataset [3, N_total]
        let dist_ds = match edist.dataset("distribution") {
            Ok(ds) => ds,
            Err(_) => continue,
        };
        let dist_shape = dist_ds.shape().unwrap_or_default();
        let dist_raw = dist_ds.read_f64().unwrap_or_default();

        if dist_shape.len() != 2 || dist_shape[0] != 3 {
            continue;
        }
        let n_total = dist_shape[1] as usize;

        // Read offsets attribute
        let dist_attrs = dist_ds.attrs().unwrap_or_default();
        let offsets: Vec<usize> =
            if let Some(hdf5_pure::AttrValue::I64Array(arr)) = dist_attrs.get("offsets") {
                arr.iter().map(|&v| v as usize).collect()
            } else {
                let per_e = n_total / energies.len();
                (0..energies.len()).map(|i| i * per_e).collect()
            };

        // Parse per-energy distributions. Layout is [3, n_total] =
        // (e_out, pdf, cdf). PDF is needed for the quadratic lin-lin
        // CDF inversion (OpenMC Tabular::sample).
        let e_out_values = &dist_raw[..n_total];
        let pdf_values = &dist_raw[n_total..2 * n_total];
        let cdf_values = &dist_raw[2 * n_total..3 * n_total];

        let n_energies = energies.len();
        let mut distributions = Vec::with_capacity(n_energies);
        for i in 0..n_energies {
            let start = offsets.get(i).copied().unwrap_or(0);
            let end = offsets.get(i + 1).copied().unwrap_or(n_total).min(n_total);

            if start >= end || start >= n_total {
                distributions.push(TabularEnergyDist {
                    e_out: vec![1e5, 2e6],
                    pdf: vec![1.0, 1.0],
                    cdf: vec![0.0, 1.0],
                });
                continue;
            }

            distributions.push(TabularEnergyDist {
                e_out: e_out_values[start..end].to_vec(),
                pdf: pdf_values[start..end].to_vec(),
                cdf: cdf_values[start..end].to_vec(),
            });
        }

        return Ok(Some(EnergyDistribution {
            energies,
            distributions,
        }));
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
    /// ENDF interpolation scheme between adjacent URR energies.
    /// 2 = lin-lin (default), 5 = log-log.
    pub interpolation: u8,
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
        if self.energies.is_empty() {
            return false;
        }
        energy >= self.energies[0] && energy <= *self.energies.last().unwrap_or(&0.0)
    }

    /// Sample URR factors at a given energy.
    ///
    /// Follows OpenMC `calculate_urr_xs` (src/nuclide.cpp): sample one ξ,
    /// look up the band independently at both bracketing energies via the
    /// per-energy cumulative CDF, then interpolate the XS values between
    /// the two energies (lin-lin for interpolation=2, log-log for =5).
    /// At the lower/upper grid edge we fall back to single-energy lookup.
    ///
    /// Returns cross-section multipliers if `multiply_smooth=true`,
    /// or absolute cross-sections if `multiply_smooth=false`.
    pub fn sample(&self, energy: f64, xi: f64) -> UrrFactors {
        let n_e = self.energies.len();
        if n_e == 0 {
            return UrrFactors {
                total: 1.0,
                elastic: 1.0,
                fission: 1.0,
                capture: 1.0,
            };
        }

        // Bracket [i_lo, i_lo+1] with E_i <= energy < E_{i+1}; clamp at edges.
        let i_lo = match self
            .energies
            .binary_search_by(|e| e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i,
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        };

        // Single-energy sampling helper (used at edges or when n_e == 1).
        let pick = |idx: usize| -> (f64, f64, f64, f64) {
            let cum = &self.cum_prob[idx];
            // upper_bound on the CDF: the first band whose cumulative prob
            // is strictly greater than ξ. Matches OpenMC's upper_bound_index.
            let mut band = cum.len() - 1;
            for (j, &cp) in cum.iter().enumerate() {
                if xi < cp {
                    band = j;
                    break;
                }
            }
            (
                self.total_factor[idx][band],
                self.elastic_factor[idx][band],
                self.fission_factor[idx][band],
                self.capture_factor[idx][band],
            )
        };

        if n_e == 1 || i_lo + 1 >= n_e || energy <= self.energies[0] {
            let (total, elastic, fission, capture) = pick(i_lo.min(n_e - 1));
            return UrrFactors {
                total,
                elastic,
                fission,
                capture,
            };
        }

        // Interpolate between E_i and E_{i+1}.
        let e_lo = self.energies[i_lo];
        let e_hi = self.energies[i_lo + 1];
        let f = match self.interpolation {
            5 => (energy / e_lo).ln() / (e_hi / e_lo).ln(),
            _ => (energy - e_lo) / (e_hi - e_lo),
        };
        let (t_lo, el_lo, fi_lo, c_lo) = pick(i_lo);
        let (t_hi, el_hi, fi_hi, c_hi) = pick(i_lo + 1);
        UrrFactors {
            total: (1.0 - f) * t_lo + f * t_hi,
            elastic: (1.0 - f) * el_lo + f * el_hi,
            fission: (1.0 - f) * fi_lo + f * fi_hi,
            capture: (1.0 - f) * c_lo + f * c_hi,
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
    let nuclide_name = root
        .groups()
        .map_err(|e| SvdError::Hdf5 {
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
    // ENDF interpolation scheme for the URR energy grid.
    // 2 = lin-lin (OpenMC default), 5 = log-log.
    let interpolation = match attrs.get("interpolation") {
        Some(hdf5_pure::AttrValue::I64(v)) => *v as u8,
        _ => 2,
    };

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
        let prob: Vec<f64> = (0..n_bands).map(|b| table_raw[base + b]).collect();
        let total: Vec<f64> = (0..n_bands)
            .map(|b| table_raw[base + n_bands + b])
            .collect();
        let elastic: Vec<f64> = (0..n_bands)
            .map(|b| table_raw[base + 2 * n_bands + b])
            .collect();
        let fission: Vec<f64> = (0..n_bands)
            .map(|b| table_raw[base + 3 * n_bands + b])
            .collect();
        let capture: Vec<f64> = (0..n_bands)
            .map(|b| table_raw[base + 4 * n_bands + b])
            .collect();

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
        interpolation,
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
    let nuclide_name = root
        .groups()
        .map_err(|e| SvdError::Hdf5 {
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
    let offsets: Vec<usize> =
        if let Some(hdf5_pure::AttrValue::I64Array(arr)) = mu_attrs.get("offsets") {
            arr.iter().map(|&v| v as usize).collect()
        } else if let Some(hdf5_pure::AttrValue::I64(v)) = mu_attrs.get("offsets") {
            vec![*v as usize]
        } else {
            // No offsets attribute — try uniform distribution
            let per_e = n_total / n_energies;
            (0..n_energies).map(|i| i * per_e).collect()
        };

    // Per-energy interpolation type: OpenMC writes an `interpolation`
    // attribute on the mu dataset. Value `1` means histogram (constant PDF
    // in each mu bin, linear CDF); `2` means linear-linear (PDF linearly
    // interpolated in each bin, quadratic CDF). Missing/unknown → default
    // to linear-linear, which is the convention for ENDF/B-VII.1 uranium
    // elastic scattering at all tabulated energies.
    let interp_flags: Vec<u8> =
        if let Some(hdf5_pure::AttrValue::I64Array(arr)) = mu_attrs.get("interpolation") {
            arr.iter().map(|&v| v as u8).collect()
        } else if let Some(hdf5_pure::AttrValue::I64(v)) = mu_attrs.get("interpolation") {
            vec![*v as u8]
        } else {
            vec![2u8; n_energies]
        };

    // Parse per-energy distributions
    // mu_raw is stored as [3, N_total] in row-major: first N_total values are row 0 (mu),
    // next N_total are row 1 (pdf), next N_total are row 2 (cdf)
    let mu_values = &mu_raw[..n_total];
    let pdf_values = &mu_raw[n_total..2 * n_total];
    let cdf_values = &mu_raw[2 * n_total..3 * n_total];

    let mut distributions = Vec::with_capacity(n_energies);
    for i in 0..n_energies {
        let start = offsets.get(i).copied().unwrap_or(0);
        let end = offsets.get(i + 1).copied().unwrap_or(n_total);
        let end = end.min(n_total);
        let histogram = interp_flags.get(i).copied().unwrap_or(2) == 1;

        if start >= end || start >= n_total {
            // Empty distribution — isotropic
            distributions.push(TabularMuDist {
                mu: vec![-1.0, 1.0],
                pdf: vec![0.5, 0.5],
                cdf: vec![0.0, 1.0],
                histogram: true,
            });
            continue;
        }

        let mu_slice = mu_values[start..end].to_vec();
        let pdf_slice = pdf_values[start..end].to_vec();
        let cdf_slice = cdf_values[start..end].to_vec();
        distributions.push(TabularMuDist {
            mu: mu_slice,
            pdf: pdf_slice,
            cdf: cdf_slice,
            histogram,
        });
    }

    Ok(Some(AngularDistribution {
        energies,
        distributions,
        center_of_mass,
    }))
}

// ── Internal helpers for NuclideFileReader ──────────────────────────────────

/// Read nu-bar from an already-opened reaction_018 group.
fn read_nu_bar_from_group(rxn_group: &hdf5_pure::Group<'_>) -> Result<NuBarTable> {
    // Collect prompt + each delayed product as an energy-dependent yield
    // table (or a constant). Previous implementation averaged tabulated
    // delayed yields into a single constant, which introduced a
    // systematic bias (U-238: -0.009, U-235: -0.002 in ν̄) because the
    // delayed yield varies with incident energy. Total ν̄(E) is the
    // sum of prompt(E) + Σ delayed_i(E); when values differ on the
    // energy grid we interpolate linearly.
    let subgroups = rxn_group.groups().unwrap_or_default();
    let mut prompt_table: Option<NuBarTable> = None;
    // Each delayed product contributes an (energies, values) table or a
    // single scalar ("constant over all E").
    let mut delayed_tables: Vec<NuBarTable> = Vec::new();
    let mut delayed_constants: Vec<f64> = Vec::new();

    for product_name in &subgroups {
        if !product_name.starts_with("product_") {
            continue;
        }
        let product = match rxn_group.group(product_name) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let attrs = product.attrs().unwrap_or_default();
        let is_neutron = matches!(attrs.get("particle"), Some(hdf5_pure::AttrValue::String(s)) if s == "neutron");
        if !is_neutron {
            continue;
        }
        let is_prompt = matches!(attrs.get("emission_mode"), Some(hdf5_pure::AttrValue::String(s)) if s == "prompt");
        let yield_ds = match product.dataset("yield") {
            Ok(ds) => ds,
            Err(_) => continue,
        };
        let shape = yield_ds.shape().unwrap_or_default();
        let raw = match yield_ds.read_f64() {
            Ok(r) => r,
            Err(_) => continue,
        };

        if shape.len() == 2 && shape[0] == 2 {
            let n = shape[1] as usize;
            if raw.len() >= 2 * n {
                let t = NuBarTable {
                    energies: raw[..n].to_vec(),
                    values: raw[n..2 * n].to_vec(),
                };
                if is_prompt {
                    prompt_table = Some(t);
                } else {
                    delayed_tables.push(t);
                }
            }
        } else if shape.len() == 1 && shape[0] == 1 {
            if is_prompt {
                prompt_table = Some(NuBarTable {
                    energies: vec![1e-5, 20.0e6],
                    values: vec![raw[0], raw[0]],
                });
            } else {
                delayed_constants.push(raw[0]);
            }
        }
    }

    let mut prompt = match prompt_table {
        Some(t) => t,
        None => {
            return Ok(NuBarTable {
                energies: vec![],
                values: vec![],
            });
        }
    };

    let d_const: f64 = delayed_constants.iter().sum();
    for v in &mut prompt.values {
        *v += d_const;
    }

    for d in &delayed_tables {
        for (i, &e) in prompt.energies.iter().enumerate() {
            prompt.values[i] += d.lookup(e);
        }
    }

    Ok(prompt)
}

/// Read angular distribution from an already-open file.
fn read_angular_dist_from_file(
    file: &hdf5_pure::File,
    nuclide_name: &str,
    mt: u32,
) -> Option<AngularDistribution> {
    let root = file.root();
    let nuc = root.group(nuclide_name).ok()?;
    let rxn_name = format!("reaction_{mt:03}");
    let rxn = nuc.group("reactions").ok()?.group(&rxn_name).ok()?;
    let rxn_attrs = rxn.attrs().unwrap_or_default();
    let center_of_mass = matches!(
        rxn_attrs.get("center_of_mass"),
        Some(hdf5_pure::AttrValue::I64(1))
    );
    let angle = rxn
        .group("product_0")
        .ok()?
        .group("distribution_0")
        .ok()?
        .group("angle")
        .ok()?;
    let energies = angle.dataset("energy").ok()?.read_f64().ok()?;
    if energies.is_empty() {
        return None;
    }
    let n_energies = energies.len();
    let mu_ds = angle.dataset("mu").ok()?;
    let mu_shape = mu_ds.shape().ok()?;
    let mu_raw = mu_ds.read_f64().ok()?;
    if mu_shape.len() != 2 || mu_shape[0] != 3 {
        return None;
    }
    let n_total = mu_shape[1] as usize;
    let mu_attrs = mu_ds.attrs().unwrap_or_default();
    let offsets: Vec<usize> =
        if let Some(hdf5_pure::AttrValue::I64Array(arr)) = mu_attrs.get("offsets") {
            arr.iter().map(|&v| v as usize).collect()
        } else {
            let per_e = n_total / n_energies;
            (0..n_energies).map(|i| i * per_e).collect()
        };
    let interp_flags: Vec<u8> =
        if let Some(hdf5_pure::AttrValue::I64Array(arr)) = mu_attrs.get("interpolation") {
            arr.iter().map(|&v| v as u8).collect()
        } else {
            vec![2u8; n_energies]
        };
    let mu_values = &mu_raw[..n_total];
    let pdf_values = &mu_raw[n_total..2 * n_total];
    let cdf_values = &mu_raw[2 * n_total..3 * n_total];
    let mut distributions = Vec::with_capacity(n_energies);
    for i in 0..n_energies {
        let start = offsets.get(i).copied().unwrap_or(0);
        let end = offsets.get(i + 1).copied().unwrap_or(n_total).min(n_total);
        let histogram = interp_flags.get(i).copied().unwrap_or(2) == 1;
        if start >= end || start >= n_total {
            distributions.push(TabularMuDist {
                mu: vec![-1.0, 1.0],
                pdf: vec![0.5, 0.5],
                cdf: vec![0.0, 1.0],
                histogram: true,
            });
        } else {
            distributions.push(TabularMuDist {
                mu: mu_values[start..end].to_vec(),
                pdf: pdf_values[start..end].to_vec(),
                cdf: cdf_values[start..end].to_vec(),
                histogram,
            });
        }
    }
    Some(AngularDistribution {
        energies,
        distributions,
        center_of_mass,
    })
}

/// Read fission energy distribution from an already-open file.
/// Generic loader for a reaction's outgoing-energy distribution.
///
/// Returns the tabulated (E_out, PDF, CDF) distribution at each incident
/// energy for `reaction_{mt:03}/product_0/distribution_0/energy/`.
/// Works for any reaction that stores a ContinuousTabular (ENDF Law 4)
/// distribution — continuum inelastic (MT=91), (n,2n) (MT=16), (n,3n)
/// (MT=17). Returns `None` for reactions where the product group is
/// absent, stores a different distribution law, or has no tabulated
/// outgoing-energy data.
fn read_reaction_edist_from_file(
    file: &hdf5_pure::File,
    nuclide_name: &str,
    mt: u32,
) -> Option<EnergyDistribution> {
    let root = file.root();
    let nuc = root.group(nuclide_name).ok()?;
    let rxn_name = format!("reaction_{mt:03}");
    let rxn = nuc.group("reactions").ok()?.group(&rxn_name).ok()?;
    // For non-fission reactions `product_0` is the emitted neutron and
    // has no prompt/delayed distinction; no filtering needed.
    let product = rxn.group("product_0").ok()?;
    let dist0 = product.group("distribution_0").ok()?;
    // Two layouts in OpenMC HDF5:
    //   a) nested (fission, reaction_018/product_0/distribution_0/energy
    //      is a group with `energy` + `distribution` datasets).
    //   b) flat (non-fission, MT=91/16/17: distribution_0 directly
    //      contains `energy` and `distribution` datasets).
    // `hdf5_pure::Group::group("X")` returns Ok even when X is a
    // dataset, so probe via `datasets()` to pick the right branch.
    let d0_datasets: Vec<String> = dist0.datasets().unwrap_or_default();
    let is_flat = d0_datasets.iter().any(|n| n == "energy")
        && d0_datasets.iter().any(|n| n == "distribution");
    let (energies, dist_ds) = if is_flat {
        let energies = dist0.dataset("energy").ok()?.read_f64().ok()?;
        let dist_ds = dist0.dataset("distribution").ok()?;
        (energies, dist_ds)
    } else {
        let edist_grp = dist0.group("energy").ok()?;
        let energies = edist_grp.dataset("energy").ok()?.read_f64().ok()?;
        let dist_ds = edist_grp.dataset("distribution").ok()?;
        (energies, dist_ds)
    };
    if energies.is_empty() {
        return None;
    }
    let dist_shape = dist_ds.shape().ok()?;
    let dist_raw = dist_ds.read_f64().ok()?;
    // OpenMC layout: rows are [E_out, PDF, CDF, ...] — we use the first
    // three. Additional rows (e.g. angular moments for Law 61) are
    // ignored; we isotropise in CM.
    if dist_shape.len() != 2 || dist_shape[0] < 3 {
        return None;
    }
    let n_total = dist_shape[1] as usize;
    let n_rows = dist_shape[0] as usize;
    let dist_attrs = dist_ds.attrs().unwrap_or_default();
    let offsets: Vec<usize> =
        if let Some(hdf5_pure::AttrValue::I64Array(arr)) = dist_attrs.get("offsets") {
            arr.iter().map(|&v| v as usize).collect()
        } else {
            let per_e = n_total / energies.len();
            (0..energies.len()).map(|i| i * per_e).collect()
        };
    let e_out_values = &dist_raw[..n_total];
    let pdf_values = &dist_raw[n_total..2 * n_total];
    let cdf_values = &dist_raw[2 * n_total..3 * n_total];
    let _ = n_rows; // silenced: only rows 0..3 are used
    let n_energies = energies.len();
    let mut distributions = Vec::with_capacity(n_energies);
    for i in 0..n_energies {
        let start = offsets.get(i).copied().unwrap_or(0);
        let end = offsets.get(i + 1).copied().unwrap_or(n_total).min(n_total);
        if start >= end || start >= n_total {
            distributions.push(TabularEnergyDist {
                e_out: vec![1e5, 2e6],
                pdf: vec![1.0, 1.0],
                cdf: vec![0.0, 1.0],
            });
        } else {
            distributions.push(TabularEnergyDist {
                e_out: e_out_values[start..end].to_vec(),
                pdf: pdf_values[start..end].to_vec(),
                cdf: cdf_values[start..end].to_vec(),
            });
        }
    }
    Some(EnergyDistribution {
        energies,
        distributions,
    })
}

fn read_fission_edist_from_file(
    file: &hdf5_pure::File,
    nuclide_name: &str,
) -> Option<EnergyDistribution> {
    let root = file.root();
    let nuc = root.group(nuclide_name).ok()?;
    let rxn = nuc.group("reactions").ok()?.group("reaction_018").ok()?;
    let subgroups = rxn.groups().unwrap_or_default();
    for product_name in &subgroups {
        if !product_name.starts_with("product_") {
            continue;
        }
        let product = match rxn.group(product_name) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let attrs = product.attrs().unwrap_or_default();
        let is_neutron = matches!(attrs.get("particle"), Some(hdf5_pure::AttrValue::String(s)) if s == "neutron");
        let is_prompt = matches!(attrs.get("emission_mode"), Some(hdf5_pure::AttrValue::String(s)) if s == "prompt");
        if !is_neutron || !is_prompt {
            continue;
        }
        let edist = product.group("distribution_0").ok()?.group("energy").ok()?;
        let energies = edist.dataset("energy").ok()?.read_f64().ok()?;
        if energies.is_empty() {
            continue;
        }
        let dist_ds = edist.dataset("distribution").ok()?;
        let dist_shape = dist_ds.shape().ok()?;
        let dist_raw = dist_ds.read_f64().ok()?;
        if dist_shape.len() != 2 || dist_shape[0] != 3 {
            continue;
        }
        let n_total = dist_shape[1] as usize;
        let dist_attrs = dist_ds.attrs().unwrap_or_default();
        let offsets: Vec<usize> =
            if let Some(hdf5_pure::AttrValue::I64Array(arr)) = dist_attrs.get("offsets") {
                arr.iter().map(|&v| v as usize).collect()
            } else {
                let per_e = n_total / energies.len();
                (0..energies.len()).map(|i| i * per_e).collect()
            };
        let e_out_values = &dist_raw[..n_total];
        let pdf_values = &dist_raw[n_total..2 * n_total];
        let cdf_values = &dist_raw[2 * n_total..3 * n_total];
        let n_energies = energies.len();
        let mut distributions = Vec::with_capacity(n_energies);
        for i in 0..n_energies {
            let start = offsets.get(i).copied().unwrap_or(0);
            let end = offsets.get(i + 1).copied().unwrap_or(n_total).min(n_total);
            if start >= end || start >= n_total {
                distributions.push(TabularEnergyDist {
                    e_out: vec![1e5, 2e6],
                    pdf: vec![1.0, 1.0],
                    cdf: vec![0.0, 1.0],
                });
            } else {
                distributions.push(TabularEnergyDist {
                    e_out: e_out_values[start..end].to_vec(),
                    pdf: pdf_values[start..end].to_vec(),
                    cdf: cdf_values[start..end].to_vec(),
                });
            }
        }
        return Some(EnergyDistribution {
            energies,
            distributions,
        });
    }
    None
}

/// Read URR probability tables from an already-open file.
fn read_urr_from_file(
    file: &hdf5_pure::File,
    nuclide_name: &str,
    temp_label: &str,
) -> Option<UrrProbabilityTables> {
    let root = file.root();
    let nuc = root.group(nuclide_name).ok()?;
    let urr = nuc.group("urr").ok()?;
    let temp_group = urr.group(temp_label).ok()?;
    let attrs = temp_group.attrs().unwrap_or_default();
    let multiply_smooth = matches!(
        attrs.get("multiply_smooth"),
        Some(hdf5_pure::AttrValue::I64(1))
    );
    let interpolation = match attrs.get("interpolation") {
        Some(hdf5_pure::AttrValue::I64(v)) => *v as u8,
        _ => 2,
    };
    let energies = temp_group.dataset("energy").ok()?.read_f64().ok()?;
    let n_e = energies.len();
    let table_ds = temp_group.dataset("table").ok()?;
    let table_shape = table_ds.shape().ok()?;
    let table_raw = table_ds.read_f64().ok()?;
    if table_shape.len() != 3 || table_shape[0] as usize != n_e || table_shape[1] != 6 {
        return None;
    }
    let n_bands = table_shape[2] as usize;
    let n_ch = 6_usize;
    let mut cum_prob = Vec::with_capacity(n_e);
    let mut total_factor = Vec::with_capacity(n_e);
    let mut elastic_factor = Vec::with_capacity(n_e);
    let mut fission_factor = Vec::with_capacity(n_e);
    let mut capture_factor = Vec::with_capacity(n_e);
    for e in 0..n_e {
        let base = e * n_ch * n_bands;
        cum_prob.push((0..n_bands).map(|b| table_raw[base + b]).collect());
        total_factor.push(
            (0..n_bands)
                .map(|b| table_raw[base + n_bands + b])
                .collect(),
        );
        elastic_factor.push(
            (0..n_bands)
                .map(|b| table_raw[base + 2 * n_bands + b])
                .collect(),
        );
        fission_factor.push(
            (0..n_bands)
                .map(|b| table_raw[base + 3 * n_bands + b])
                .collect(),
        );
        capture_factor.push(
            (0..n_bands)
                .map(|b| table_raw[base + 4 * n_bands + b])
                .collect(),
        );
    }
    Some(UrrProbabilityTables {
        energies,
        n_bands,
        cum_prob,
        total_factor,
        elastic_factor,
        fission_factor,
        capture_factor,
        multiply_smooth,
        interpolation,
    })
}

// ── Thermal Scattering HDF5 Reader ─────────────────────────────────────

use crate::thermal::{
    ContinuousInelastic, DiscreteInelastic, ElasticThermal, InelasticDist, InelasticThermal,
    ThermalScatteringData,
};

/// Load thermal scattering data from an OpenMC HDF5 file (e.g., `c_H_in_H2O.h5`).
///
/// HDF5 layout (Section 3.3 of OpenMC docs):
/// ```text
/// /<thermal_name>/
///   @atomic_weight_ratio, @energy_max, @nuclides
///   kTs/<TTT>K  (double, kT in eV)
///   inelastic/<TTT>K/inelastic/
///     xs (Tabulated1D: [2, n_e])
///     distribution/ (@type, energy, energy_out[5][N], mu[3][M])
///   elastic/<TTT>K/elastic/  (optional)
///     xs (CoherentElastic / IncoherentElastic / Tabulated1D)
///     distribution/ (...)
/// ```
pub fn load_thermal_scattering(path: &Path) -> Result<ThermalScatteringData> {
    let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;
    let root = file.root();

    // Find the thermal material group (first group under root)
    let name = root
        .groups()
        .map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot list root: {e}"),
        })?
        .into_iter()
        .next()
        .ok_or_else(|| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: "no group at root".into(),
        })?;

    let g = root.group(&name).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot open /{name}: {e}"),
    })?;

    // Read top-level attributes
    let attrs = g.attrs().unwrap_or_default();
    let awr = match attrs.get("atomic_weight_ratio") {
        Some(hdf5_pure::AttrValue::F64(v)) => *v,
        _ => 1.0,
    };
    let energy_max = match attrs.get("energy_max") {
        Some(hdf5_pure::AttrValue::F64(v)) => *v,
        _ => 4.0,
    };
    let nuclides: Vec<String> = match attrs.get("nuclides") {
        Some(hdf5_pure::AttrValue::StringArray(arr)) => arr.clone(),
        _ => vec![],
    };

    println!(
        "  Thermal: {name}  nuclides={nuclides:?}  energy_max={energy_max:.2} eV  AWR={awr:.3}"
    );

    // Read temperatures from kTs group
    let kts_group = g.group("kTs").map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot open kTs: {e}"),
    })?;
    let kt_labels = kts_group.datasets().unwrap_or_default();

    let mut temp_data: Vec<(String, f64)> = Vec::new();
    for label in &kt_labels {
        let ds = kts_group.dataset(label).map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot read kTs/{label}: {e}"),
        })?;
        let val = ds.read_f64().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("kTs/{label} read error: {e}"),
        })?;
        if let Some(&kt) = val.first() {
            temp_data.push((label.clone(), kt));
        }
    }
    // Sort by kT value
    temp_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let temp_labels: Vec<String> = temp_data.iter().map(|(l, _)| l.clone()).collect();
    let kts: Vec<f64> = temp_data.iter().map(|(_, kt)| *kt).collect();

    println!("  Temperatures: {temp_labels:?}");

    // Read inelastic data per temperature
    let mut inelastic = Vec::with_capacity(temp_labels.len());
    for label in &temp_labels {
        let inel = read_thermal_inelastic(&file, &name, label, path)?;
        inelastic.push(inel);
    }

    // Read elastic data per temperature (optional)
    let elastic = read_thermal_elastic_all(&file, &name, &temp_labels, path);

    Ok(ThermalScatteringData {
        name,
        nuclides,
        energy_max,
        awr,
        kts,
        temp_labels,
        inelastic,
        elastic,
    })
}

/// Read inelastic thermal scattering for one temperature.
fn read_thermal_inelastic(
    file: &hdf5_pure::File,
    thermal_name: &str,
    temp_label: &str,
    path: &Path,
) -> Result<InelasticThermal> {
    let root = file.root();
    let base = root.group(thermal_name).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("{e}"),
    })?;

    // Navigate: /<thermal>/<TTT>K/inelastic/
    let temp_group = base.group(temp_label).map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot open {temp_label}: {e}"),
    })?;
    let inel_group = temp_group.group("inelastic").map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot open {temp_label}/inelastic: {e}"),
    })?;

    // Read cross section (Tabulated1D: [2, n_e])
    let xs_ds = inel_group.dataset("xs").map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read {temp_label}/inelastic/xs: {e}"),
    })?;
    let xs_shape = xs_ds.shape().unwrap_or_default();
    let xs_raw = xs_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("xs read error: {e}"),
    })?;

    let n_e = if xs_shape.len() == 2 {
        xs_shape[1] as usize
    } else {
        xs_raw.len() / 2
    };
    let energy: Vec<f64> = xs_raw[..n_e].to_vec();
    let xs: Vec<f64> = xs_raw[n_e..2 * n_e].to_vec();

    // Read distribution
    let dist_group = inel_group
        .group("distribution")
        .map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot open distribution: {e}"),
        })?;

    let dist_attrs = dist_group.attrs().unwrap_or_default();
    let dist_type = match dist_attrs.get("type") {
        Some(hdf5_pure::AttrValue::String(s)) => s.clone(),
        _ => "incoherent_inelastic".to_string(),
    };

    let dist = if dist_type == "incoherent_inelastic_discrete" {
        read_discrete_inelastic_dist(&dist_group, path)?
    } else {
        // "incoherent_inelastic" or "correlated" — continuous tabular
        read_continuous_inelastic_dist(&dist_group, path)?
    };

    println!("    {temp_label} inelastic: {n_e} energies, type={dist_type}");

    Ok(InelasticThermal { energy, xs, dist })
}

/// Read continuous inelastic distribution (type="incoherent_inelastic" / "correlated").
fn read_continuous_inelastic_dist(group: &hdf5_pure::Group, path: &Path) -> Result<InelasticDist> {
    // energy: double[n_inc]
    let energy_ds = group.dataset("energy").map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read distribution/energy: {e}"),
    })?;
    let inc_energy = energy_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("energy read error: {e}"),
    })?;
    let n_inc = inc_energy.len();

    // energy_out: double[5][N_total]
    let eout_ds = group.dataset("energy_out").map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read energy_out: {e}"),
    })?;
    let eout_shape = eout_ds.shape().unwrap_or_default();
    let eout_raw = eout_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("energy_out read error: {e}"),
    })?;

    let n_total = if eout_shape.len() == 2 {
        eout_shape[1] as usize
    } else {
        eout_raw.len() / 5
    };

    // Unpack 5 rows
    let e_out = eout_raw[..n_total].to_vec();
    let pdf_e = eout_raw[n_total..2 * n_total].to_vec();
    let cdf_e = eout_raw[2 * n_total..3 * n_total].to_vec();
    // Row 3: interpolation codes for angular distributions
    let mu_interp_f64 = &eout_raw[3 * n_total..4 * n_total];
    let mu_interp: Vec<u32> = mu_interp_f64.iter().map(|&v| v as u32).collect();
    // Row 4: offsets into mu array
    let mu_offsets_f64 = &eout_raw[4 * n_total..5 * n_total];
    let mu_offsets: Vec<usize> = mu_offsets_f64.iter().map(|&v| v as usize).collect();

    // Read offsets attribute for energy_out distributions
    let eout_attrs = eout_ds.attrs().unwrap_or_default();
    let offsets: Vec<usize> =
        if let Some(hdf5_pure::AttrValue::I64Array(arr)) = eout_attrs.get("offsets") {
            arr.iter().map(|&v| v as usize).collect()
        } else if let Some(hdf5_pure::AttrValue::F64Array(arr)) = eout_attrs.get("offsets") {
            arr.iter().map(|&v| v as usize).collect()
        } else {
            // Fallback: evenly split
            let per_e = n_total / n_inc.max(1);
            (0..n_inc).map(|i| i * per_e).collect()
        };

    let interp: Vec<u32> =
        if let Some(hdf5_pure::AttrValue::I64Array(arr)) = eout_attrs.get("interpolation") {
            arr.iter().map(|&v| v as u32).collect()
        } else {
            vec![2; n_inc] // default: linear-linear
        };

    // mu: double[3][M_total]
    let mu_ds = group.dataset("mu").map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read mu: {e}"),
    })?;
    let mu_shape = mu_ds.shape().unwrap_or_default();
    let mu_raw = mu_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("mu read error: {e}"),
    })?;

    let n_mu_total = if mu_shape.len() == 2 {
        mu_shape[1] as usize
    } else {
        mu_raw.len() / 3
    };

    let mu = mu_raw[..n_mu_total].to_vec();
    let pdf_mu = mu_raw[n_mu_total..2 * n_mu_total].to_vec();
    let cdf_mu = mu_raw[2 * n_mu_total..3 * n_mu_total].to_vec();

    println!("    Continuous: {n_inc} inc energies, {n_total} E_out pts, {n_mu_total} mu pts");

    Ok(InelasticDist::Continuous(ContinuousInelastic {
        n_inc,
        offsets,
        interp,
        e_out,
        pdf_e,
        cdf_e,
        mu_interp,
        mu_offsets,
        mu,
        pdf_mu,
        cdf_mu,
    }))
}

/// Read discrete inelastic distribution (type="incoherent_inelastic_discrete").
fn read_discrete_inelastic_dist(group: &hdf5_pure::Group, path: &Path) -> Result<InelasticDist> {
    // energy_out: double[n_inc][n_out]
    let eout_ds = group.dataset("energy_out").map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read energy_out: {e}"),
    })?;
    let eout_shape = eout_ds.shape().unwrap_or_default();
    let energy_out = eout_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("energy_out read error: {e}"),
    })?;
    let n_out = if eout_shape.len() >= 2 {
        eout_shape[1] as usize
    } else {
        1
    };

    // mu_out: double[n_inc][n_out][n_mu]
    let mu_ds = group.dataset("mu_out").map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("cannot read mu_out: {e}"),
    })?;
    let mu_shape = mu_ds.shape().unwrap_or_default();
    let mu_out = mu_ds.read_f64().map_err(|e| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail: format!("mu_out read error: {e}"),
    })?;
    let n_mu = if mu_shape.len() >= 3 {
        mu_shape[2] as usize
    } else {
        1
    };

    // skewed: int8 (0 = equi-probable, 1 = skewed)
    let dist_attrs = group.attrs().unwrap_or_default();
    let skewed = match dist_attrs.get("skewed") {
        Some(hdf5_pure::AttrValue::I64(v)) => *v != 0,
        _ => {
            // Check for the dataset version
            if let Ok(ds) = group.dataset("skewed") {
                ds.read_f64()
                    .ok()
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0.0)
                    != 0.0
            } else {
                false
            }
        }
    };

    Ok(InelasticDist::Discrete(DiscreteInelastic {
        energy_out,
        n_out,
        mu_out,
        n_mu,
        skewed,
    }))
}

/// Read elastic thermal scattering for all temperatures (returns None if absent).
fn read_thermal_elastic_all(
    file: &hdf5_pure::File,
    thermal_name: &str,
    temp_labels: &[String],
    path: &Path,
) -> Option<Vec<ElasticThermal>> {
    let root = file.root();
    let base = root.group(thermal_name).ok()?;

    // Check if elastic exists at the first temperature
    let first_temp = base.group(temp_labels.first()?).ok()?;
    first_temp.group("elastic").ok()?; // If this fails, no elastic data

    let mut elastic = Vec::with_capacity(temp_labels.len());
    for label in temp_labels {
        let el = read_thermal_elastic_one(file, thermal_name, label, path);
        elastic.push(el);
    }
    Some(elastic)
}

/// Read elastic thermal scattering for one temperature.
fn read_thermal_elastic_one(
    file: &hdf5_pure::File,
    thermal_name: &str,
    temp_label: &str,
    _path: &Path,
) -> ElasticThermal {
    let root = file.root();
    let base = match root.group(thermal_name) {
        Ok(g) => g,
        Err(_) => return default_elastic(),
    };
    let temp = match base.group(temp_label) {
        Ok(g) => g,
        Err(_) => return default_elastic(),
    };
    let el = match temp.group("elastic") {
        Ok(g) => g,
        Err(_) => return default_elastic(),
    };

    // Read xs dataset and check type attribute
    let xs_ds = match el.dataset("xs") {
        Ok(d) => d,
        Err(_) => return default_elastic(),
    };
    let xs_attrs = xs_ds.attrs().unwrap_or_default();
    let xs_type = match xs_attrs.get("type") {
        Some(hdf5_pure::AttrValue::String(s)) => s.clone(),
        _ => String::new(),
    };

    match xs_type.as_str() {
        "CoherentElastic" => {
            // [2, n_bragg]: row 0 = bragg edges, row 1 = cumulative factors
            let raw = xs_ds.read_f64().unwrap_or_default();
            let shape = xs_ds.shape().unwrap_or_default();
            let n = if shape.len() == 2 {
                shape[1] as usize
            } else {
                raw.len() / 2
            };
            let bragg_edges = raw[..n].to_vec();
            let factors = raw[n..2 * n].to_vec();
            println!("    {temp_label} elastic: coherent, {n} Bragg edges");
            ElasticThermal::Coherent {
                bragg_edges,
                factors,
            }
        }
        "IncoherentElastic" => {
            // [2]: bound_xs, debye_waller
            let raw = xs_ds.read_f64().unwrap_or_default();
            let bound_xs = raw.first().copied().unwrap_or(0.0);
            let debye_waller = raw.get(1).copied().unwrap_or(0.0);
            println!(
                "    {temp_label} elastic: incoherent, σ_b={bound_xs:.1} b, W'={debye_waller:.4} eV⁻¹"
            );
            ElasticThermal::Incoherent {
                bound_xs,
                debye_waller,
            }
        }
        _ => {
            // Unknown or Tabulated1D — try to read as tabulated cross section
            println!("    {temp_label} elastic: type={xs_type} (unsupported, skipping)");
            default_elastic()
        }
    }
}

fn default_elastic() -> ElasticThermal {
    ElasticThermal::Incoherent {
        bound_xs: 0.0,
        debye_waller: 0.0,
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

#[cfg(test)]
mod sampling_tests {
    //! Statistical tests for the OpenMC stochastic-bin samplers.
    //!
    //! These verify behavior invariant under the sampler rewrite
    //! (correlated single-ξ → stochastic-bin two-ξ). They use a small
    //! deterministic seed and enough draws that the statistics are
    //! stable to several sigma.
    use super::*;
    use crate::transport::rng::Rng;

    fn build_two_bin_angular(
        energies: &[f64],
        low_bin_mu_cdf: &[(f64, f64)],
        high_bin_mu_cdf: &[(f64, f64)],
    ) -> AngularDistribution {
        let split = |pts: &[(f64, f64)]| {
            // Tests construct (mu, cdf) pairs; use histogram interpolation
            // so the CDF inversion matches the pre-existing linear-CDF
            // expectations that these tests assert against.
            let mu: Vec<f64> = pts.iter().map(|p| p.0).collect();
            let cdf: Vec<f64> = pts.iter().map(|p| p.1).collect();
            // Derive PDF consistent with histogram interpretation.
            let mut pdf = vec![0.0; mu.len()];
            for i in 0..mu.len().saturating_sub(1) {
                let dmu = (mu[i + 1] - mu[i]).max(1e-30);
                let dcdf = cdf[i + 1] - cdf[i];
                pdf[i] = dcdf / dmu;
            }
            if let Some(last) = pdf.last_mut().copied() {
                *pdf.last_mut().unwrap() = last;
            }
            TabularMuDist {
                mu,
                pdf,
                cdf,
                histogram: true,
            }
        };
        AngularDistribution {
            energies: energies.to_vec(),
            distributions: vec![split(low_bin_mu_cdf), split(high_bin_mu_cdf)],
            center_of_mass: true,
        }
    }

    #[test]
    fn angular_sample_mu_stays_in_bounds() {
        // Two bins: low bin isotropic, high bin forward-peaked.
        let dist = build_two_bin_angular(
            &[1e4, 1e6],
            &[(-1.0, 0.0), (1.0, 1.0)],             // uniform [-1, 1]
            &[(-1.0, 0.0), (0.0, 0.1), (1.0, 1.0)], // more weight near +1
        );
        let mut rng = Rng::new(42, 0);
        for _ in 0..10_000 {
            let e = 1e4 + rng.uniform() * (1e6 - 1e4);
            let mu = dist.sample_mu(e, &mut rng);
            assert!((-1.0..=1.0).contains(&mu), "mu out of range: {mu}");
        }
    }

    #[test]
    fn angular_sample_mu_isotropic_within_bin_has_zero_mean() {
        // Both bins isotropic → <mu> ≈ 0.
        let dist = build_two_bin_angular(
            &[1.0, 2.0],
            &[(-1.0, 0.0), (1.0, 1.0)],
            &[(-1.0, 0.0), (1.0, 1.0)],
        );
        let mut rng = Rng::new(7, 0);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| dist.sample_mu(1.5, &mut rng)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-2, "<mu> = {mean}, expected ≈ 0");
    }

    #[test]
    fn angular_sample_mu_forward_peaked_is_positive() {
        // Both bins strongly forward-peaked — most mass near mu = +1.
        // CDF = mu^10-ish via (mu, cdf) = [(-1, 0), (0.8, 0.1), (1.0, 1.0)].
        let dist = build_two_bin_angular(
            &[1.0, 2.0],
            &[(-1.0, 0.0), (0.8, 0.1), (1.0, 1.0)],
            &[(-1.0, 0.0), (0.8, 0.1), (1.0, 1.0)],
        );
        let mut rng = Rng::new(11, 0);
        let n = 50_000;
        let mean: f64 = (0..n).map(|_| dist.sample_mu(1.5, &mut rng)).sum::<f64>() / n as f64;
        assert!(mean > 0.5, "<mu> = {mean}, expected strongly positive");
    }

    #[test]
    fn angular_sample_mu_stochastic_bin_respects_r() {
        // Low bin always returns near -1, high bin always returns near +1.
        // At E = E_lo + 0.25*(E_hi - E_lo), pick_hi fires 25% of the time
        // → <mu> ≈ 0.25*(+1) + 0.75*(-1) = -0.5.
        let dist = build_two_bin_angular(
            &[0.0, 1.0],
            &[(-1.0, 0.0), (-0.999, 1.0)], // nearly δ at -1
            &[(0.999, 0.0), (1.0, 1.0)],   // nearly δ at +1
        );
        let mut rng = Rng::new(3, 0);
        let e = 0.25; // r = 0.25
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| dist.sample_mu(e, &mut rng)).sum::<f64>() / n as f64;
        assert!(
            (mean - (-0.5)).abs() < 2e-2,
            "stochastic-bin mean {mean} does not match r=0.25 prediction -0.5"
        );
    }

    #[test]
    fn angular_sample_mu_edge_below_grid_uses_first_bin() {
        // E below first energy → always sample from first bin.
        let dist = build_two_bin_angular(
            &[10.0, 20.0],
            &[(-1.0, 0.0), (-0.99, 1.0)],
            &[(0.99, 0.0), (1.0, 1.0)],
        );
        let mut rng = Rng::new(99, 0);
        for _ in 0..1_000 {
            let mu = dist.sample_mu(5.0, &mut rng);
            assert!(mu < -0.9, "below grid should sample first bin: got {mu}");
        }
    }

    #[test]
    fn angular_sample_mu_edge_above_grid_uses_last_bin() {
        let dist = build_two_bin_angular(
            &[10.0, 20.0],
            &[(-1.0, 0.0), (-0.99, 1.0)],
            &[(0.99, 0.0), (1.0, 1.0)],
        );
        let mut rng = Rng::new(100, 0);
        for _ in 0..1_000 {
            let mu = dist.sample_mu(50.0, &mut rng);
            assert!(mu > 0.9, "above grid should sample last bin: got {mu}");
        }
    }

    fn build_two_bin_energy(
        incident: &[f64],
        low_bin_eout_cdf: &[(f64, f64)],
        high_bin_eout_cdf: &[(f64, f64)],
    ) -> EnergyDistribution {
        let split = |pts: &[(f64, f64)]| {
            // Derive a histogram-PDF so the quadratic inverter reduces to
            // linear behaviour, preserving legacy test expectations.
            let e_out: Vec<f64> = pts.iter().map(|p| p.0).collect();
            let cdf: Vec<f64> = pts.iter().map(|p| p.1).collect();
            let mut pdf = vec![0.0_f64; e_out.len()];
            for k in 0..e_out.len().saturating_sub(1) {
                let de = e_out[k + 1] - e_out[k];
                if de > 0.0 {
                    pdf[k] = (cdf[k + 1] - cdf[k]) / de;
                }
            }
            let n = e_out.len();
            if n >= 2 {
                let tail = pdf[n - 2];
                pdf[n - 1] = tail;
            }
            TabularEnergyDist { e_out, pdf, cdf }
        };
        EnergyDistribution {
            energies: incident.to_vec(),
            distributions: vec![split(low_bin_eout_cdf), split(high_bin_eout_cdf)],
        }
    }

    #[test]
    fn energy_sample_stays_positive_and_bounded() {
        let dist = build_two_bin_energy(
            &[1e4, 1e6],
            &[(0.1e6, 0.0), (5.0e6, 1.0)],  // [0.1, 5] MeV
            &[(0.1e6, 0.0), (10.0e6, 1.0)], // [0.1, 10] MeV
        );
        let mut rng = Rng::new(123, 0);
        for _ in 0..10_000 {
            let e_inc = 1e4 + rng.uniform() * (1e6 - 1e4);
            let e_out = dist.sample(e_inc, &mut rng);
            assert!(e_out >= 1e-5, "sample below floor: {e_out}");
            // Scaled kinematic remap can stretch beyond a single bin's nominal
            // range, but must stay inside the union-of-bins envelope + tolerance.
            assert!(e_out <= 1.1e7, "sample implausibly large: {e_out}");
        }
    }

    #[test]
    fn energy_sample_scaled_kinematic_interpolates_bounds() {
        // Low bin E_out in [1, 2] MeV, high bin in [3, 6] MeV.
        // At r = 0.5, scaled kinematic remap should place samples roughly
        // in [2, 4] MeV regardless of which bin was drawn.
        let dist = build_two_bin_energy(
            &[0.0, 1.0],
            &[(1.0e6, 0.0), (2.0e6, 1.0)],
            &[(3.0e6, 0.0), (6.0e6, 1.0)],
        );
        let mut rng = Rng::new(77, 0);
        let n = 5_000;
        let mut min_e = f64::INFINITY;
        let mut max_e = 0.0_f64;
        for _ in 0..n {
            let e = dist.sample(0.5, &mut rng);
            if e < min_e {
                min_e = e;
            }
            if e > max_e {
                max_e = e;
            }
        }
        // Allow generous margin; the essential claim is that the scaled
        // remap keeps samples in the interpolated [E_1, E_K] envelope.
        assert!(
            (1.9e6..=2.1e6).contains(&min_e),
            "min E_out = {min_e} should be near 2.0e6"
        );
        assert!(
            (3.8e6..=4.2e6).contains(&max_e),
            "max E_out = {max_e} should be near 4.0e6"
        );
    }

    #[test]
    fn energy_sample_below_grid_uses_first_bin() {
        let dist = build_two_bin_energy(
            &[10.0, 20.0],
            &[(1.0e6, 0.0), (2.0e6, 1.0)],
            &[(5.0e6, 0.0), (6.0e6, 1.0)],
        );
        let mut rng = Rng::new(44, 0);
        for _ in 0..500 {
            let e = dist.sample(1.0, &mut rng);
            assert!(
                (1.0e6..=2.0e6).contains(&e),
                "below-grid sample {e} not in first bin [1e6, 2e6]"
            );
        }
    }

    // ── Quadratic lin-lin CDF inversion (OpenMC Tabular::sample) ──
    //
    // With a linear-rising PDF from 0 to p_max over a single bin, the
    // closed-form inverse gives E = E_lo + sqrt(2 * m * Δc) / m when
    // cdf[0] = 0. Feeding ξ = 0.5 should place the sample at the point
    // where the cumulative area equals 0.5 — for a triangular PDF over
    // [0, 1] MeV with total area 1, that is E ≈ 0.707 MeV.

    #[test]
    fn energy_tabular_quadratic_triangular_pdf_inverse() {
        let e_out = vec![0.0_f64, 1.0e6];
        let pdf = vec![0.0_f64, 2.0e-6]; // triangular, ∫ = 1
        let cdf = vec![0.0_f64, 1.0];
        let t = TabularEnergyDist { e_out, pdf, cdf };
        let sample = t.sample_with_xi(0.5);
        // Analytical inverse: E = sqrt(0.5) * 1e6 ≈ 707_107 eV.
        let expected = (0.5_f64).sqrt() * 1.0e6;
        assert!(
            (sample - expected).abs() / expected < 1e-5,
            "quadratic sample {sample} vs expected {expected}"
        );
    }

    #[test]
    fn energy_tabular_quadratic_falls_back_to_linear_when_pdf_missing() {
        let e_out = vec![0.0_f64, 1.0e6];
        let cdf = vec![0.0_f64, 1.0];
        let t = TabularEnergyDist {
            e_out,
            pdf: vec![],
            cdf,
        };
        let sample = t.sample_with_xi(0.5);
        // Linear fallback: E = 0.5 * 1e6 = 5e5 eV.
        assert!((sample - 5.0e5).abs() < 1e-3, "fallback {sample}");
    }

    // ── URR probability table interpolation (OpenMC calculate_urr_xs) ──

    fn build_two_point_urr(interpolation: u8) -> UrrProbabilityTables {
        // Single-band tables at two energies: factor 1.0 at E_lo, factor 2.0
        // at E_hi. All bands share the same factor so the band pick is
        // irrelevant; this isolates the energy interpolation.
        UrrProbabilityTables {
            energies: vec![100.0, 1000.0],
            n_bands: 1,
            cum_prob: vec![vec![1.0], vec![1.0]],
            total_factor: vec![vec![1.0], vec![2.0]],
            elastic_factor: vec![vec![1.0], vec![2.0]],
            fission_factor: vec![vec![1.0], vec![2.0]],
            capture_factor: vec![vec![1.0], vec![2.0]],
            multiply_smooth: true,
            interpolation,
        }
    }

    #[test]
    fn urr_lin_lin_interpolation_midpoint() {
        let urr = build_two_point_urr(2);
        let f = urr.sample(550.0, 0.5); // (550-100)/(1000-100) = 0.5
        assert!(
            (f.elastic - 1.5).abs() < 1e-12,
            "lin-lin elastic {}",
            f.elastic
        );
        assert!((f.total - 1.5).abs() < 1e-12);
    }

    #[test]
    fn urr_log_log_interpolation_sqrt_point() {
        // Log-log with factors 1.0 and 2.0 at 100 and 1000 eV; at E = 100*sqrt(10)
        // the log-fraction is 0.5, so lin interpolation of the factors is 1.5.
        let urr = build_two_point_urr(5);
        let e = 100.0 * 10_f64.sqrt();
        let f = urr.sample(e, 0.5);
        assert!(
            (f.elastic - 1.5).abs() < 1e-10,
            "log-log elastic {}",
            f.elastic
        );
    }

    #[test]
    fn urr_band_selected_independently_per_energy() {
        // Two bands, cutoff shifts with energy so that ξ = 0.5 picks band 0
        // at E_lo but band 1 at E_hi. Factors encode which band was chosen.
        let urr = UrrProbabilityTables {
            energies: vec![100.0, 200.0],
            n_bands: 2,
            cum_prob: vec![
                vec![0.9, 1.0], // at E_lo: ξ=0.5 < 0.9 → band 0
                vec![0.1, 1.0], // at E_hi: ξ=0.5 > 0.1 → band 1
            ],
            total_factor: vec![vec![10.0, 99.0], vec![88.0, 20.0]],
            elastic_factor: vec![vec![10.0, 99.0], vec![88.0, 20.0]],
            fission_factor: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            capture_factor: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            multiply_smooth: false,
            interpolation: 2,
        };
        // At E=150, f=0.5. Expected elastic = 0.5*10 + 0.5*20 = 15.
        let f = urr.sample(150.0, 0.5);
        assert!(
            (f.elastic - 15.0).abs() < 1e-12,
            "indep band elastic {}",
            f.elastic
        );
    }

    #[test]
    fn urr_at_grid_edge_falls_back_to_single_energy() {
        let urr = build_two_point_urr(2);
        let f_lo = urr.sample(100.0, 0.5);
        let f_hi = urr.sample(1000.0, 0.5);
        assert!((f_lo.elastic - 1.0).abs() < 1e-12);
        assert!((f_hi.elastic - 2.0).abs() < 1e-12);
    }
}
