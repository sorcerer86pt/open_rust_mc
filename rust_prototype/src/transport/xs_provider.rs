//! SVD-based cross-section provider — connects the SVD reconstruction
//! kernel to the transport loop.
//!
//! For each nuclide, stores SVD kernels for key reactions (elastic,
//! fission, capture). At lookup time, reconstructs sigma(E) via a dot
//! product instead of binary-searching a table.

use std::sync::Arc;

use crate::decompose;
use crate::hdf5_reader::{self, AngularDistribution, DiscreteLevelInfo, EnergyDistribution, NuBarTable, NuclideData, NuclideFileReader, UrrProbabilityTables};
use crate::kernel::SvdKernel;
use crate::physics::collision::MicroXs;
use crate::table::PointwiseTable;
use crate::thermal::ThermalScatteringData;
use crate::transport::simulate::XsProvider;

/// Per-nuclide SVD-compressed cross-section data.
pub struct NuclideKernels {
    /// SVD kernel for elastic scattering (MT=2).
    pub elastic: Option<ReactionKernel>,
    /// Pointwise total XS (sum of ALL reactions from HDF5). Used for accurate collision distance.
    pub total_table: Option<PointwiseTable>,
    /// Raw total XS values on the shared energy grid (for GPU upload as pointwise table).
    pub total_xs_raw: Option<Vec<f64>>,
    /// Missing channel XS = total_hdf5 - (el + inel + n2n + n3n + fis + cap) from pointwise data.
    pub missing_xs: Option<Vec<f64>>,
    /// SVD kernel for inelastic scattering (MT=4, total inelastic).
    pub inelastic: Option<ReactionKernel>,
    /// SVD kernel for (n,2n) reaction (MT=16).
    pub n2n: Option<ReactionKernel>,
    /// SVD kernel for (n,3n) reaction (MT=17).
    pub n3n: Option<ReactionKernel>,
    /// SVD kernel for fission (MT=18). None for non-fissile nuclides.
    pub fission: Option<ReactionKernel>,
    /// SVD kernel for capture (MT=102).
    pub capture: Option<ReactionKernel>,
    /// Atomic weight ratio.
    pub awr: f64,
    /// Constant fallback nu-bar.
    pub nu_bar_const: f64,
    /// Energy-dependent nu-bar table (if available).
    pub nu_bar_table: Option<NuBarTable>,
    /// Discrete inelastic level data (MT=51-91) with SVD kernels.
    pub discrete_levels: Vec<DiscreteLevel>,
    /// Whether continuum inelastic (MT=91) is present.
    pub has_continuum_inelastic: bool,
    /// Angular distribution for elastic scattering (MT=2).
    pub elastic_angle: Option<AngularDistribution>,
    /// Fission energy distribution for prompt neutrons.
    pub fission_energy_dist: Option<EnergyDistribution>,
    /// URR probability tables.
    pub urr_tables: Option<UrrProbabilityTables>,
}

/// A discrete inelastic level with its cross-section kernel and Q-value.
pub struct DiscreteLevel {
    pub info: DiscreteLevelInfo,
    pub kernel: Option<ReactionKernel>,
}

/// SVD kernel + energy grid for a single reaction.
pub struct ReactionKernel {
    pub kernel: SvdKernel,
    /// Pre-computed temperature coefficients (one set per temperature).
    pub coeffs: Vec<f64>,
}

impl ReactionKernel {
    /// Reconstruct cross-section at a given energy (linear scale, barns).
    /// Performs its own binary search — use `reconstruct_at_index` when
    /// the index is already known from a shared grid search.
    #[inline]
    pub fn lookup(&self, energy: f64) -> f64 {
        let idx = self.kernel.energy_index(energy);
        self.reconstruct_at_index(idx)
    }

    /// Reconstruct cross-section at a pre-computed energy grid index.
    /// Skips the binary search — caller must provide a valid index from
    /// the same shared energy grid.
    #[inline]
    pub fn reconstruct_at_index(&self, idx: usize) -> f64 {
        let log_val = self.kernel.reconstruct_single_log(idx, &self.coeffs);
        f64::exp2(log_val * std::f64::consts::LOG2_10)
    }

    /// Reconstruct with log-log interpolation between grid points (OpenMC scheme).
    ///
    /// `idx` is the lower bracket index, `log_frac` is the interpolation fraction
    /// in log-energy space: `(ln(E) - ln(E_lo)) / (ln(E_hi) - ln(E_lo))`.
    #[inline]
    pub fn reconstruct_interp(&self, idx: usize, log_frac: f64) -> f64 {
        let log_lo = self.kernel.reconstruct_single_log(idx, &self.coeffs);
        if idx + 1 >= self.kernel.n_energy() || log_frac <= 0.0 {
            return f64::exp2(log_lo * std::f64::consts::LOG2_10);
        }
        let log_hi = self.kernel.reconstruct_single_log(idx + 1, &self.coeffs);
        let log_interp = log_lo + log_frac * (log_hi - log_lo);
        f64::exp2(log_interp * std::f64::consts::LOG2_10)
    }
}

impl NuclideKernels {
    /// Get energy-dependent nu-bar at the given energy.
    pub fn nu_bar_at(&self, energy: f64) -> f64 {
        self.nu_bar_table
            .as_ref()
            .map_or(self.nu_bar_const, |t| t.lookup(energy))
    }

    /// Lookup cross-sections for each discrete level at the given energy.
    pub fn discrete_level_xs(&self, energy: f64) -> Vec<f64> {
        self.discrete_levels
            .iter()
            .map(|lvl| {
                if energy < lvl.info.threshold {
                    0.0
                } else {
                    lvl.kernel.as_ref().map_or(0.0, |k| k.lookup(energy))
                }
            })
            .collect()
    }

    /// Get the discrete level info slices.
    pub fn discrete_level_info(&self) -> Vec<DiscreteLevelInfo> {
        self.discrete_levels.iter().map(|l| l.info.clone()).collect()
    }

    /// Total memory of SVD kernels (bytes), excluding metadata.
    pub fn svd_memory_bytes(&self) -> usize {
        let mut total = 0;
        if let Some(k) = &self.elastic { total += k.kernel.memory_bytes(); }
        if let Some(t) = &self.total_table { total += t.memory_bytes(); }
        if let Some(k) = &self.inelastic { total += k.kernel.memory_bytes(); }
        if let Some(k) = &self.n2n { total += k.kernel.memory_bytes(); }
        if let Some(k) = &self.n3n { total += k.kernel.memory_bytes(); }
        if let Some(k) = &self.fission { total += k.kernel.memory_bytes(); }
        if let Some(k) = &self.capture { total += k.kernel.memory_bytes(); }
        for lvl in &self.discrete_levels {
            if let Some(k) = &lvl.kernel { total += k.kernel.memory_bytes(); }
        }
        total
    }

    /// Apply URR probability table factors to a MicroXs if the energy is in the URR.
    ///
    /// When `multiply_smooth=true`, the URR factors multiply the smooth XS values.
    /// A random number `xi` selects the probability band for consistent sampling.
    pub fn apply_urr(&self, xs: &mut MicroXs, energy: f64, xi: f64) {
        let urr = match &self.urr_tables {
            Some(u) if u.in_range(energy) => u,
            _ => return,
        };

        let factors = urr.sample(energy, xi);

        if urr.multiply_smooth {
            // Multiply smooth XS by URR factors
            xs.elastic *= factors.elastic;
            xs.fission *= factors.fission;
            xs.capture *= factors.capture;
        } else {
            // Replace smooth XS with absolute URR values
            xs.elastic = factors.elastic;
            xs.fission = factors.fission;
            xs.capture = factors.capture;
        }
        // Recompute total from partials for consistency
        xs.total = xs.elastic + xs.inelastic + xs.n2n + xs.n3n + xs.fission + xs.capture;
    }
}

/// Cross-section provider backed by SVD-compressed kernels.
pub struct SvdXsProvider {
    pub nuclides: Vec<NuclideKernels>,
    /// Thermal scattering data per nuclide (None if no S(α,β) for this nuclide).
    pub thermal: Vec<Option<Arc<ThermalScatteringData>>>,
}

impl XsProvider for SvdXsProvider {
    fn lookup(&self, nuclide_idx: usize, energy: f64) -> MicroXs {
        let nuc = &self.nuclides[nuclide_idx];

        // Single binary search on the shared energy grid — reused by all 6 reactions.
        let any_kernel = nuc.elastic.as_ref()
            .or(nuc.fission.as_ref())
            .or(nuc.capture.as_ref())
            .or(nuc.inelastic.as_ref())
            .or(nuc.n2n.as_ref())
            .or(nuc.n3n.as_ref());

        let (idx, log_frac) = match any_kernel {
            Some(k) => {
                let idx = k.kernel.energy_index(energy);
                let grid = k.kernel.energies();
                let frac = if idx + 1 < grid.len() && grid[idx] > 0.0 && grid[idx + 1] > grid[idx] {
                    let log_e = energy.ln();
                    let log_lo = grid[idx].ln();
                    let log_hi = grid[idx + 1].ln();
                    ((log_e - log_lo) / (log_hi - log_lo)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (idx, frac)
            }
            None => (0, 0.0),
        };

        // Log-log interpolation between grid points (matching OpenMC/pointwise table)
        let elastic = nuc.elastic.as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let inelastic = match &nuc.inelastic {
            Some(k) => k.reconstruct_interp(idx, log_frac),
            None if !nuc.discrete_levels.is_empty() => {
                nuc.discrete_levels.iter()
                    .filter(|lvl| energy >= lvl.info.threshold)
                    .filter_map(|lvl| lvl.kernel.as_ref())
                    .map(|k| k.reconstruct_interp(idx, log_frac).max(0.0))
                    .sum::<f64>()
            }
            None => 0.0,
        };
        let n2n = nuc.n2n.as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let n3n = nuc.n3n.as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let fission = nuc.fission.as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let mut capture = nuc.capture.as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));

        let total = match &nuc.total_table {
            Some(t) => {
                let tot = t.lookup(energy);
                capture = (tot - elastic - inelastic - n2n - n3n - fission).max(0.0);
                tot
            }
            None => elastic + inelastic + n2n + n3n + fission + capture,
        };
        let nu_bar = nuc.nu_bar_at(energy);

        MicroXs {
            total,
            elastic,
            inelastic,
            n2n,
            n3n,
            fission,
            capture,
            nu_bar,
            awr: nuc.awr,
        }
    }

    fn discrete_level_info(&self, nuclide_idx: usize) -> Vec<DiscreteLevelInfo> {
        self.nuclides[nuclide_idx].discrete_level_info()
    }

    fn discrete_level_xs(&self, nuclide_idx: usize, energy: f64) -> Vec<f64> {
        self.nuclides[nuclide_idx].discrete_level_xs(energy)
    }

    fn has_continuum_inelastic(&self, nuclide_idx: usize) -> bool {
        self.nuclides[nuclide_idx].has_continuum_inelastic
    }

    fn elastic_angular_dist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::AngularDistribution> {
        self.nuclides[nuclide_idx].elastic_angle.as_ref()
    }

    fn fission_energy_dist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx].fission_energy_dist.as_ref()
    }

    fn apply_urr(&self, nuclide_idx: usize, xs: &mut MicroXs, energy: f64, xi: f64) {
        self.nuclides[nuclide_idx].apply_urr(xs, energy, xi);
    }

    fn thermal_scattering(&self, nuclide_idx: usize) -> Option<&ThermalScatteringData> {
        self.thermal.get(nuclide_idx)?.as_deref()
    }
}

/// Build an SVD kernel for one reaction of one nuclide from HDF5 data.
///
/// Returns None if the reaction doesn't exist or has no data.
pub fn build_reaction_kernel(
    h5_path: &std::path::Path,
    mt: u32,
    svd_rank: usize,
    temp_idx: usize,
) -> Option<ReactionKernel> {
    let data = NuclideData::from_hdf5(h5_path, mt).ok()?;

    if data.n_energy() == 0 || data.n_temp() == 0 {
        return None;
    }

    let log_matrix = data.to_log_matrix();
    let svd = decompose::svd(&log_matrix, data.n_energy(), data.n_temp());

    let rank = svd_rank.min(svd.rank);
    let n_e = svd.n_e;
    let n_t = svd.n_t;

    // Pre-multiply basis
    let mut basis = vec![0.0_f64; n_e * rank];
    for j in 0..rank {
        let s_j = svd.s[j];
        for i in 0..n_e {
            basis[i * rank + j] = svd.u[i * svd.rank + j] * s_j;
        }
    }

    let mut vt_coeffs = vec![0.0_f64; rank * n_t];
    for j in 0..rank {
        for t in 0..n_t {
            vt_coeffs[j * n_t + t] = svd.vt[j * n_t + t];
        }
    }

    let energies: Arc<[f64]> = data.energies.into();
    let kernel = SvdKernel::new(basis, vt_coeffs, energies, rank, n_e, n_t);

    let t_idx = temp_idx.min(n_t - 1);
    let coeffs = kernel.temp_coeffs(t_idx);

    Some(ReactionKernel { kernel, coeffs })
}

/// Build an SVD kernel from a `NuclideFileReader` using a shared energy grid.
///
/// Uses Ducru kernel reconstruction (2017) for temperature interpolation:
/// the coefficients are a physically optimal weighted sum of all V^T columns,
/// not just a single temperature index. This enables accurate cross-section
/// reconstruction at arbitrary temperatures.
fn build_kernel_from_reader(
    reader: &NuclideFileReader,
    mt: u32,
    svd_rank: usize,
    temp_idx: usize,
    shared_grid: &Arc<[f64]>,
) -> Option<ReactionKernel> {
    let data = reader.read_reaction(mt).ok()?;

    if data.n_energy() == 0 || data.n_temp() == 0 {
        return None;
    }

    let log_matrix = data.to_log_matrix();
    let svd = decompose::svd(&log_matrix, data.n_energy(), data.n_temp());

    let rank = svd_rank.min(svd.rank);
    let n_e = svd.n_e;
    let n_t = svd.n_t;

    let mut basis = vec![0.0_f64; n_e * rank];
    for j in 0..rank {
        let s_j = svd.s[j];
        for i in 0..n_e {
            basis[i * rank + j] = svd.u[i * svd.rank + j] * s_j;
        }
    }

    let mut vt_coeffs = vec![0.0_f64; rank * n_t];
    for j in 0..rank {
        for t in 0..n_t {
            vt_coeffs[j * n_t + t] = svd.vt[j * n_t + t];
        }
    }

    let kernel = SvdKernel::new(basis, vt_coeffs, Arc::clone(shared_grid), rank, n_e, n_t);

    // Use Ducru kernel reconstruction if multiple temperatures available,
    // otherwise fall back to direct index lookup.
    let coeffs = if n_t > 1 && !reader.temperatures.is_empty() {
        let target_temp = reader.temperatures[temp_idx.min(n_t - 1)];
        kernel.temp_coeffs_ducru(&reader.temperatures, target_temp)
    } else {
        let t_idx = temp_idx.min(n_t - 1);
        kernel.temp_coeffs(t_idx)
    };
    Some(ReactionKernel { kernel, coeffs })
}

/// Build an SVD kernel from pre-computed XS values on the shared grid.
fn build_kernel_from_xs(
    xs: &[f64],
    shared_grid: &Arc<[f64]>,
    svd_rank: usize,
) -> Option<ReactionKernel> {
    let n_e = xs.len();
    if n_e == 0 { return None; }

    let log_xs: Vec<f64> = xs.iter().map(|&v| (v.max(1e-30)).log10()).collect();

    let rank = svd_rank;
    let mut basis = vec![0.0_f64; n_e * rank];
    for i in 0..n_e {
        basis[i * rank] = log_xs[i];
    }

    let kernel = SvdKernel::new(basis, vec![1.0; rank], Arc::clone(shared_grid), rank, n_e, 1);
    let mut coeffs = vec![0.0_f64; rank];
    coeffs[0] = 1.0;
    Some(ReactionKernel { kernel, coeffs })
}

/// Load a complete nuclide from HDF5 in a single pass.
///
/// Opens the file once, reads energy grids once, then reads all reactions
/// and metadata without redundant I/O.
pub fn load_nuclide(
    h5_path: &std::path::Path,
    svd_rank: usize,
    temp_idx: usize,
    awr_fallback: f64,
    nu_bar_fallback: f64,
) -> NuclideKernels {
    println!("  Loading {} (rank={svd_rank})...", h5_path.display());

    // Open file ONCE and cache energy grids
    let reader = match NuclideFileReader::open(h5_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    WARNING: failed to open {}: {e}", h5_path.display());
            return NuclideKernels {
                elastic: None, total_table: None, total_xs_raw: None, missing_xs: None, inelastic: None, n2n: None, n3n: None,
                fission: None, capture: None, awr: awr_fallback,
                nu_bar_const: nu_bar_fallback, nu_bar_table: None,
                discrete_levels: vec![], has_continuum_inelastic: false,
                elastic_angle: None, fission_energy_dist: None, urr_tables: None,
            };
        }
    };

    let awr = reader.awr().unwrap_or(awr_fallback);
    println!("    AWR = {awr:.3} ({} temps, {} energy pts)",
             reader.temp_labels.len(), reader.union_grid.len());

    // Create shared energy grid — one Arc for all reactions in this nuclide
    let shared_grid: Arc<[f64]> = reader.union_grid.clone().into();

    let nu_bar_table = reader.nu_bar().ok();
    if let Some(ref t) = nu_bar_table {
        if !t.energies.is_empty() {
            println!("    nu-bar(E): {} pts, {:.3} @ thermal, {:.3} @ 1 MeV",
                     t.energies.len(), t.lookup(0.0253), t.lookup(1.0e6));
        }
    }

    // Discrete levels — all read from the same open file
    let level_infos = reader.discrete_levels(awr);
    let has_continuum = level_infos.iter().any(|l| l.mt == 91);
    let n_levels = level_infos.len();
    let mut discrete_levels: Vec<DiscreteLevel> = Vec::with_capacity(n_levels);
    for info in level_infos {
        let kernel = build_kernel_from_reader(&reader, info.mt, svd_rank.min(2), temp_idx, &shared_grid);
        discrete_levels.push(DiscreteLevel { info, kernel });
    }
    let loaded_count = discrete_levels.iter().filter(|l| l.kernel.is_some()).count();
    if n_levels > 0 {
        println!("    Discrete levels: {loaded_count}/{n_levels} (continuum={has_continuum})");
    }

    let fission_energy_dist = reader.fission_energy_dist();
    if let Some(ref d) = fission_energy_dist {
        println!("    Fission spectrum: {} energies", d.energies.len());
    }

    let elastic_angle = reader.angular_distribution(2);
    if let Some(ref a) = elastic_angle {
        println!("    Elastic angular dist: {} energies, CM={}", a.energies.len(), a.center_of_mass);
    }

    let urr_temp = reader.temp_labels.get(temp_idx).cloned().unwrap_or_else(|| "294K".to_string());
    let urr_tables = reader.urr_tables(&urr_temp);
    if let Some(ref u) = urr_tables {
        println!("    URR: {} energies, {} bands, {:.0}–{:.0} eV",
                 u.energies.len(), u.n_bands,
                 u.energies.first().unwrap_or(&0.0), u.energies.last().unwrap_or(&0.0));
    }

    let total_xs_vec = reader.compute_total_xs(temp_idx);
    let total_table = total_xs_vec.as_ref().map(|xs| {
        PointwiseTable::from_shared_grid(shared_grid.clone(), xs.clone())
    });
    let elastic = build_kernel_from_reader(&reader, 2, svd_rank, temp_idx, &shared_grid);
    if elastic.is_some() { println!("    MT=2 (elastic)"); }
    let inelastic = build_kernel_from_reader(&reader, 4, svd_rank, temp_idx, &shared_grid);
    if inelastic.is_some() { println!("    MT=4 (inelastic)"); }
    else if !discrete_levels.is_empty() { println!("    MT=4 (inelastic) — synthesized from {} discrete levels", discrete_levels.len()); }
    let n2n = build_kernel_from_reader(&reader, 16, svd_rank, temp_idx, &shared_grid);
    if n2n.is_some() { println!("    MT=16 (n,2n)"); }
    let n3n = build_kernel_from_reader(&reader, 17, svd_rank, temp_idx, &shared_grid);
    if n3n.is_some() { println!("    MT=17 (n,3n)"); }
    let fission = build_kernel_from_reader(&reader, 18, svd_rank, temp_idx, &shared_grid);
    if fission.is_some() { println!("    MT=18 (fission)"); }
    let capture = build_kernel_from_reader(&reader, 102, svd_rank, temp_idx, &shared_grid);
    if capture.is_some() { println!("    MT=102 (capture)"); }

    let missing_xs = total_xs_vec.as_ref().map(|total| {
        let n = shared_grid.len();
        let mut partial_sum = vec![0.0_f64; n];
        for mt in [2_u32, 16, 17, 18, 102] {
            if let Ok(data) = reader.read_reaction(mt) {
                if let Some(xs) = data.xs_per_temp.get(temp_idx) {
                    for i in 0..n.min(xs.len()) {
                        partial_sum[i] += xs[i].max(0.0);
                    }
                }
            }
        }
        if let Ok(data) = reader.read_reaction(4) {
            if let Some(xs) = data.xs_per_temp.get(temp_idx) {
                for i in 0..n.min(xs.len()) { partial_sum[i] += xs[i].max(0.0); }
            }
        } else {
            for mt in 51..=91 {
                if let Ok(data) = reader.read_reaction(mt) {
                    if let Some(xs) = data.xs_per_temp.get(temp_idx) {
                        for i in 0..n.min(xs.len()) { partial_sum[i] += xs[i].max(0.0); }
                    }
                }
            }
        }
        let mut missing = vec![0.0_f64; n];
        for i in 0..n {
            missing[i] = (total[i] - partial_sum[i]).max(0.0);
        }
        let max_missing = missing.iter().copied().fold(0.0_f64, f64::max);
        if max_missing > 0.001 {
            println!("    Missing channels: max={:.4} b", max_missing);
        }
        missing
    });

    NuclideKernels {
        elastic, total_table, total_xs_raw: total_xs_vec, missing_xs, inelastic, n2n, n3n, fission, capture,
        awr,
        nu_bar_const: nu_bar_fallback,
        nu_bar_table,
        discrete_levels,
        has_continuum_inelastic: has_continuum,
        elastic_angle,
        fission_energy_dist,
        urr_tables,
    }
}

// ── Pointwise Table XS Provider (OpenMC-style baseline) ─────────────

/// A discrete inelastic level backed by a pointwise table.
pub struct TableDiscreteLevel {
    pub info: DiscreteLevelInfo,
    pub table: Option<PointwiseTable>,
}

/// Per-nuclide pointwise cross-section tables — the OpenMC baseline.
///
/// Stores the same physics data as `NuclideKernels` but uses raw
/// pointwise tables instead of SVD-compressed kernels for XS lookup.
pub struct NuclideTableData {
    pub elastic: Option<PointwiseTable>,
    pub total_table: Option<PointwiseTable>,
    pub inelastic: Option<PointwiseTable>,
    pub n2n: Option<PointwiseTable>,
    pub n3n: Option<PointwiseTable>,
    pub fission: Option<PointwiseTable>,
    pub capture: Option<PointwiseTable>,
    pub awr: f64,
    pub nu_bar_const: f64,
    pub nu_bar_table: Option<NuBarTable>,
    pub discrete_levels: Vec<TableDiscreteLevel>,
    pub has_continuum_inelastic: bool,
    pub elastic_angle: Option<AngularDistribution>,
    pub fission_energy_dist: Option<EnergyDistribution>,
    pub urr_tables: Option<UrrProbabilityTables>,
}

impl NuclideTableData {
    pub fn nu_bar_at(&self, energy: f64) -> f64 {
        self.nu_bar_table
            .as_ref()
            .map_or(self.nu_bar_const, |t| t.lookup(energy))
    }

    pub fn discrete_level_xs(&self, energy: f64) -> Vec<f64> {
        self.discrete_levels
            .iter()
            .map(|lvl| {
                if energy < lvl.info.threshold {
                    0.0
                } else {
                    lvl.table.as_ref().map_or(0.0, |t| t.lookup(energy))
                }
            })
            .collect()
    }

    pub fn discrete_level_info(&self) -> Vec<DiscreteLevelInfo> {
        self.discrete_levels.iter().map(|l| l.info.clone()).collect()
    }

    pub fn apply_urr(&self, xs: &mut MicroXs, energy: f64, xi: f64) {
        let urr = match &self.urr_tables {
            Some(u) if u.in_range(energy) => u,
            _ => return,
        };
        let factors = urr.sample(energy, xi);
        if urr.multiply_smooth {
            xs.elastic *= factors.elastic;
            xs.fission *= factors.fission;
            xs.capture *= factors.capture;
        } else {
            xs.elastic = factors.elastic;
            xs.fission = factors.fission;
            xs.capture = factors.capture;
        }
        xs.total = xs.elastic + xs.inelastic + xs.n2n + xs.n3n + xs.fission + xs.capture;
    }

    /// Total memory of pointwise tables (bytes), excluding metadata.
    pub fn table_memory_bytes(&self) -> usize {
        let mut total = 0;
        if let Some(t) = &self.elastic { total += t.memory_bytes(); }
        if let Some(t) = &self.inelastic { total += t.memory_bytes(); }
        if let Some(t) = &self.n2n { total += t.memory_bytes(); }
        if let Some(t) = &self.n3n { total += t.memory_bytes(); }
        if let Some(t) = &self.fission { total += t.memory_bytes(); }
        if let Some(t) = &self.capture { total += t.memory_bytes(); }
        for lvl in &self.discrete_levels {
            if let Some(t) = &lvl.table { total += t.memory_bytes(); }
        }
        total
    }
}

/// Cross-section provider using OpenMC-style pointwise table lookup.
///
/// This is the baseline: binary search + log-log interpolation per energy point.
/// Used for the "honesty test" comparison against SVD reconstruction.
pub struct TableXsProvider {
    pub nuclides: Vec<NuclideTableData>,
    /// Thermal scattering data per nuclide (None if no S(α,β) for this nuclide).
    pub thermal: Vec<Option<Arc<ThermalScatteringData>>>,
}

impl XsProvider for TableXsProvider {
    fn lookup(&self, nuclide_idx: usize, energy: f64) -> MicroXs {
        let nuc = &self.nuclides[nuclide_idx];

        let elastic = nuc.elastic.as_ref().map_or(0.0, |t| t.lookup(energy));
        let inelastic = match &nuc.inelastic {
            Some(t) => t.lookup(energy),
            None if !nuc.discrete_levels.is_empty() => {
                nuc.discrete_levels.iter()
                    .filter(|lvl| energy >= lvl.info.threshold)
                    .filter_map(|lvl| lvl.table.as_ref())
                    .map(|t| t.lookup(energy).max(0.0))
                    .sum::<f64>()
            }
            None => 0.0,
        };
        let n2n = nuc.n2n.as_ref().map_or(0.0, |t| t.lookup(energy));
        let n3n = nuc.n3n.as_ref().map_or(0.0, |t| t.lookup(energy));
        let fission = nuc.fission.as_ref().map_or(0.0, |t| t.lookup(energy));
        let mut capture = nuc.capture.as_ref().map_or(0.0, |t| t.lookup(energy));

        let total = match &nuc.total_table {
            Some(tt) => {
                let tot = tt.lookup(energy);
                capture = (tot - elastic - inelastic - n2n - n3n - fission).max(0.0);
                tot
            }
            None => elastic + inelastic + n2n + n3n + fission + capture,
        };
        let nu_bar = nuc.nu_bar_at(energy);

        MicroXs {
            total, elastic, inelastic, n2n, n3n, fission, capture, nu_bar, awr: nuc.awr,
        }
    }

    fn discrete_level_info(&self, nuclide_idx: usize) -> Vec<DiscreteLevelInfo> {
        self.nuclides[nuclide_idx].discrete_level_info()
    }

    fn discrete_level_xs(&self, nuclide_idx: usize, energy: f64) -> Vec<f64> {
        self.nuclides[nuclide_idx].discrete_level_xs(energy)
    }

    fn has_continuum_inelastic(&self, nuclide_idx: usize) -> bool {
        self.nuclides[nuclide_idx].has_continuum_inelastic
    }

    fn elastic_angular_dist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::AngularDistribution> {
        self.nuclides[nuclide_idx].elastic_angle.as_ref()
    }

    fn fission_energy_dist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx].fission_energy_dist.as_ref()
    }

    fn apply_urr(&self, nuclide_idx: usize, xs: &mut MicroXs, energy: f64, xi: f64) {
        self.nuclides[nuclide_idx].apply_urr(xs, energy, xi);
    }

    fn thermal_scattering(&self, nuclide_idx: usize) -> Option<&ThermalScatteringData> {
        self.thermal.get(nuclide_idx)?.as_deref()
    }
}

/// Build a pointwise table for one reaction using a shared energy grid.
fn build_table_from_reader(
    reader: &NuclideFileReader,
    mt: u32,
    temp_idx: usize,
    shared_grid: &Arc<[f64]>,
) -> Option<PointwiseTable> {
    let data = reader.read_reaction(mt).ok()?;
    if data.n_energy() == 0 || data.n_temp() == 0 {
        return None;
    }
    let t = temp_idx.min(data.n_temp() - 1);
    Some(PointwiseTable::from_shared_grid(Arc::clone(shared_grid), data.xs_per_temp[t].clone()))
}

/// Load a complete nuclide from HDF5 as pointwise tables (OpenMC-style).
///
/// Opens the file once, reads the unionized energy grid and cross-sections
/// at the specified temperature. All physics metadata (nu-bar, angular
/// distributions, fission spectrum, URR tables) is loaded identically to
/// the SVD path — only the XS lookup mechanism differs.
pub fn load_nuclide_table(
    h5_path: &std::path::Path,
    temp_idx: usize,
    awr_fallback: f64,
    nu_bar_fallback: f64,
) -> NuclideTableData {
    println!("  Loading {} (pointwise table)...", h5_path.display());

    let reader = match NuclideFileReader::open(h5_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    WARNING: failed to open {}: {e}", h5_path.display());
            return NuclideTableData {
                elastic: None, total_table: None, inelastic: None, n2n: None, n3n: None,
                fission: None, capture: None, awr: awr_fallback,
                nu_bar_const: nu_bar_fallback, nu_bar_table: None,
                discrete_levels: vec![], has_continuum_inelastic: false,
                elastic_angle: None, fission_energy_dist: None, urr_tables: None,
            };
        }
    };

    let awr = reader.awr().unwrap_or(awr_fallback);
    println!("    AWR = {awr:.3} ({} temps, {} energy pts)",
             reader.temp_labels.len(), reader.union_grid.len());

    // Create shared energy grid for all tables in this nuclide
    let shared_grid: Arc<[f64]> = reader.union_grid.clone().into();

    let nu_bar_table = reader.nu_bar().ok();
    if let Some(ref t) = nu_bar_table {
        if !t.energies.is_empty() {
            println!("    nu-bar(E): {} pts, {:.3} @ thermal, {:.3} @ 1 MeV",
                     t.energies.len(), t.lookup(0.0253), t.lookup(1.0e6));
        }
    }

    let level_infos = reader.discrete_levels(awr);
    let has_continuum = level_infos.iter().any(|l| l.mt == 91);
    let n_levels = level_infos.len();
    let mut discrete_levels: Vec<TableDiscreteLevel> = Vec::with_capacity(n_levels);
    for info in level_infos {
        let table = build_table_from_reader(&reader, info.mt, temp_idx, &shared_grid);
        discrete_levels.push(TableDiscreteLevel { info, table });
    }
    let loaded_count = discrete_levels.iter().filter(|l| l.table.is_some()).count();
    if n_levels > 0 {
        println!("    Discrete levels: {loaded_count}/{n_levels} (continuum={has_continuum})");
    }

    let fission_energy_dist = reader.fission_energy_dist();
    if let Some(ref d) = fission_energy_dist {
        println!("    Fission spectrum: {} energies", d.energies.len());
    }

    let elastic_angle = reader.angular_distribution(2);
    if let Some(ref a) = elastic_angle {
        println!("    Elastic angular dist: {} energies, CM={}", a.energies.len(), a.center_of_mass);
    }

    let urr_temp = reader.temp_labels.get(temp_idx).cloned().unwrap_or_else(|| "294K".to_string());
    let urr_tables = reader.urr_tables(&urr_temp);
    if let Some(ref u) = urr_tables {
        println!("    URR: {} energies, {} bands, {:.0}–{:.0} eV",
                 u.energies.len(), u.n_bands,
                 u.energies.first().unwrap_or(&0.0), u.energies.last().unwrap_or(&0.0));
    }

    let total_table = reader.compute_total_xs(temp_idx).map(|xs| {
        PointwiseTable::from_shared_grid(shared_grid.clone(), xs)
    });
    let elastic = build_table_from_reader(&reader, 2, temp_idx, &shared_grid);
    if elastic.is_some() { println!("    MT=2 (elastic)"); }
    let inelastic = build_table_from_reader(&reader, 4, temp_idx, &shared_grid);
    if inelastic.is_some() { println!("    MT=4 (inelastic)"); }
    else if !discrete_levels.is_empty() { println!("    MT=4 (inelastic) — synthesized from {} discrete levels", discrete_levels.len()); }
    let n2n = build_table_from_reader(&reader, 16, temp_idx, &shared_grid);
    if n2n.is_some() { println!("    MT=16 (n,2n)"); }
    let n3n = build_table_from_reader(&reader, 17, temp_idx, &shared_grid);
    if n3n.is_some() { println!("    MT=17 (n,3n)"); }
    let fission = build_table_from_reader(&reader, 18, temp_idx, &shared_grid);
    if fission.is_some() { println!("    MT=18 (fission)"); }
    let capture = build_table_from_reader(&reader, 102, temp_idx, &shared_grid);
    if capture.is_some() { println!("    MT=102 (capture)"); }

    NuclideTableData {
        elastic, total_table, inelastic, n2n, n3n, fission, capture,
        awr,
        nu_bar_const: nu_bar_fallback,
        nu_bar_table,
        discrete_levels,
        has_continuum_inelastic: has_continuum,
        elastic_angle,
        fission_energy_dist,
        urr_tables,
    }
}
