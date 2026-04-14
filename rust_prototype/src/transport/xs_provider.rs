//! SVD-based cross-section provider — connects the SVD reconstruction
//! kernel to the transport loop.
//!
//! For each nuclide, stores SVD kernels for key reactions (elastic,
//! fission, capture). At lookup time, reconstructs sigma(E) via a dot
//! product instead of binary-searching a table.

use crate::decompose;
use crate::hdf5_reader::{self, AngularDistribution, DiscreteLevelInfo, EnergyDistribution, NuBarTable, NuclideData, NuclideFileReader, UrrProbabilityTables};
use crate::kernel::SvdKernel;
use crate::physics::collision::MicroXs;
use crate::transport::simulate::XsProvider;

/// Per-nuclide SVD-compressed cross-section data.
pub struct NuclideKernels {
    /// SVD kernel for elastic scattering (MT=2).
    pub elastic: Option<ReactionKernel>,
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
    #[inline]
    pub fn lookup(&self, energy: f64) -> f64 {
        let energies = self.kernel.energies();
        let n = energies.len();

        let idx = match energies.binary_search_by(|e| {
            e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 { 0 }
                else if i >= n { n - 1 }
                else { i }
            }
        };

        let log_val = self.kernel.reconstruct_single_log(idx, &self.coeffs);
        10.0_f64.powf(log_val)
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
}

impl XsProvider for SvdXsProvider {
    fn lookup(&self, nuclide_idx: usize, energy: f64) -> MicroXs {
        let nuc = &self.nuclides[nuclide_idx];

        let elastic = nuc.elastic.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let inelastic = nuc.inelastic.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let n2n = nuc.n2n.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let n3n = nuc.n3n.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let fission = nuc.fission.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let capture = nuc.capture.as_ref()
            .map_or(0.0, |k| k.lookup(energy));

        let total = elastic + inelastic + n2n + n3n + fission + capture;
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

    let kernel = SvdKernel::new(basis, vt_coeffs, data.energies.clone(), rank, n_e, n_t);

    let t_idx = temp_idx.min(n_t - 1);
    let coeffs = kernel.temp_coeffs(t_idx);

    Some(ReactionKernel { kernel, coeffs })
}

/// Build an SVD kernel from a `NuclideFileReader` (single-pass, no re-read).
fn build_kernel_from_reader(
    reader: &NuclideFileReader,
    mt: u32,
    svd_rank: usize,
    temp_idx: usize,
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

    let kernel = SvdKernel::new(basis, vt_coeffs, data.energies.clone(), rank, n_e, n_t);
    let t_idx = temp_idx.min(n_t - 1);
    let coeffs = kernel.temp_coeffs(t_idx);
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
                elastic: None, inelastic: None, n2n: None, n3n: None,
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
        let kernel = build_kernel_from_reader(&reader, info.mt, svd_rank.min(2), temp_idx);
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

    let elastic = build_kernel_from_reader(&reader, 2, svd_rank, temp_idx);
    if elastic.is_some() { println!("    MT=2 (elastic)"); }
    let inelastic = build_kernel_from_reader(&reader, 4, svd_rank, temp_idx);
    if inelastic.is_some() { println!("    MT=4 (inelastic)"); }
    let n2n = build_kernel_from_reader(&reader, 16, svd_rank, temp_idx);
    if n2n.is_some() { println!("    MT=16 (n,2n)"); }
    let n3n = build_kernel_from_reader(&reader, 17, svd_rank, temp_idx);
    if n3n.is_some() { println!("    MT=17 (n,3n)"); }
    let fission = build_kernel_from_reader(&reader, 18, svd_rank, temp_idx);
    if fission.is_some() { println!("    MT=18 (fission)"); }
    let capture = build_kernel_from_reader(&reader, 102, svd_rank, temp_idx);
    if capture.is_some() { println!("    MT=102 (capture)"); }

    NuclideKernels {
        elastic, inelastic, n2n, n3n, fission, capture,
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
