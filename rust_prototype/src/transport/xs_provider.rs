//! SVD-based cross-section provider — connects the SVD reconstruction
//! kernel to the transport loop.
//!
//! For each nuclide, stores SVD kernels for key reactions (elastic,
//! fission, capture). At lookup time, reconstructs sigma(E) via a dot
//! product instead of binary-searching a table.

use std::sync::{Arc, OnceLock};

/// Read `OPEN_RUST_MC_NO_URR` exactly once and cache the result. URR
/// apply is called per-nuclide per-collision in the hot transport
/// loop; calling `std::env::var_os` there serialises on the process-
/// wide env lock (esp. on Windows, which holds `ENV_LOCK` + issues
/// `GetEnvironmentVariableW`) and produces a ~4x slowdown on PWR
/// under rayon. Caching the bool drops that cost to a single atomic
/// load per call.
#[inline]
fn urr_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("OPEN_RUST_MC_NO_URR").is_some())
}

use crate::decompose;
use crate::hdf5_reader::{
    self, AngularDistribution, DiscreteLevelInfo, EnergyDistribution, NuBarTable, NuclideData,
    NuclideFileReader, UrrProbabilityTables,
};
use crate::kernel::SvdKernel;
use crate::physics::collision::MicroXs;
use crate::table::{PointwiseTable, StochTempTable};
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
    /// Pointwise XS [n_energy * 7]: el, inel, n2n, n3n, fis, cap, total — for GPU upload.
    pub pointwise_xs: Option<Vec<f64>>,
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
    /// CM-frame angular distribution per discrete level, aligned with
    /// `discrete_levels`. Entry is `None` when the evaluation does not
    /// tabulate an angular distribution for that MT — isotropic fallback.
    pub discrete_level_angles: Vec<Option<AngularDistribution>>,
    /// Whether continuum inelastic (MT=91) is present.
    pub has_continuum_inelastic: bool,
    /// Angular distribution for elastic scattering (MT=2).
    pub elastic_angle: Option<AngularDistribution>,
    /// Fission energy distribution for prompt neutrons.
    pub fission_energy_dist: Option<EnergyDistribution>,
    /// ENDF MT=91 continuum inelastic outgoing-energy distribution.
    /// When present, replaces the evaporation-spectrum approximation in
    /// the continuum branch of `sample_inelastic_level`.
    pub inelastic_continuum_edist: Option<EnergyDistribution>,
    /// ENDF MT=16 (n,2n) outgoing-energy distribution. Used for both
    /// the continuing primary and the emitted secondary in
    /// `process_collision`; replaces the evaporation approximation.
    pub n2n_edist: Option<EnergyDistribution>,
    /// ENDF MT=17 (n,3n) outgoing-energy distribution.
    pub n3n_edist: Option<EnergyDistribution>,
    /// URR probability tables.
    pub urr_tables: Option<UrrProbabilityTables>,
    /// Photon products keyed by ENDF MT. Populated from HDF5
    /// `reactions/reaction_{mt}/product_N` groups with
    /// `particle="photon"`. Used by the transport loop to sample the
    /// capture / fission / inelastic γ spectrum at each neutron
    /// reaction site for coupled neutron-photon tallies.
    pub photon_products: Vec<(u32, hdf5_reader::PhotonProduct)>,
}

/// A discrete inelastic level with its cross-section kernel and Q-value.
///
/// CM-frame angular distributions live on `NuclideKernels::discrete_level_angles`
/// (parallel vec), so the trait can return a contiguous slice without
/// allocating per collision.
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
        self.discrete_levels
            .iter()
            .map(|l| l.info.clone())
            .collect()
    }

    /// Total memory of SVD kernels (bytes), excluding metadata.
    pub fn svd_memory_bytes(&self) -> usize {
        let mut total = 0;
        if let Some(k) = &self.elastic {
            total += k.kernel.memory_bytes();
        }
        if let Some(t) = &self.total_table {
            total += t.memory_bytes();
        }
        if let Some(k) = &self.inelastic {
            total += k.kernel.memory_bytes();
        }
        if let Some(k) = &self.n2n {
            total += k.kernel.memory_bytes();
        }
        if let Some(k) = &self.n3n {
            total += k.kernel.memory_bytes();
        }
        if let Some(k) = &self.fission {
            total += k.kernel.memory_bytes();
        }
        if let Some(k) = &self.capture {
            total += k.kernel.memory_bytes();
        }
        for lvl in &self.discrete_levels {
            if let Some(k) = &lvl.kernel {
                total += k.kernel.memory_bytes();
            }
        }
        total
    }

    /// Apply URR probability table factors to a MicroXs if the energy is in the URR.
    ///
    /// When `multiply_smooth=true`, the URR factors multiply the smooth XS values.
    /// A random number `xi` selects the probability band for consistent sampling.
    pub fn apply_urr(&self, xs: &mut MicroXs, energy: f64, xi: f64) {
        // Ablation knob: `OPEN_RUST_MC_NO_URR=1` disables URR sampling so
        // Godiva/PWR offsets can be attributed to URR vs other engine
        // effects. Cached once — see `urr_disabled`.
        if urr_disabled() {
            return;
        }
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
        let any_kernel = nuc
            .elastic
            .as_ref()
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
        let elastic = nuc
            .elastic
            .as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let inelastic = match &nuc.inelastic {
            Some(k) => k.reconstruct_interp(idx, log_frac),
            None if !nuc.discrete_levels.is_empty() => nuc
                .discrete_levels
                .iter()
                .filter(|lvl| energy >= lvl.info.threshold)
                .filter_map(|lvl| lvl.kernel.as_ref())
                .map(|k| k.reconstruct_interp(idx, log_frac).max(0.0))
                .sum::<f64>(),
            None => 0.0,
        };
        let n2n = nuc
            .n2n
            .as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let n3n = nuc
            .n3n
            .as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let fission = nuc
            .fission
            .as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let mut capture = nuc
            .capture
            .as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));

        let total = match &nuc.total_table {
            // The total_table lives on the same shared energy grid that we
            // already searched for the SVD reactions — reuse `idx` instead
            // of doing a second binary/hash search here. `lookup_at_idx`
            // produces bit-identical output to `lookup(energy)`.
            Some(t) => {
                let tot = t.lookup_at_idx(energy, idx);
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

    fn elastic_angular_dist(
        &self,
        nuclide_idx: usize,
    ) -> Option<&hdf5_reader::AngularDistribution> {
        self.nuclides[nuclide_idx].elastic_angle.as_ref()
    }

    fn discrete_level_angles(
        &self,
        nuclide_idx: usize,
    ) -> &[Option<hdf5_reader::AngularDistribution>] {
        &self.nuclides[nuclide_idx].discrete_level_angles
    }

    fn fission_energy_dist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx].fission_energy_dist.as_ref()
    }

    fn inelastic_continuum_edist(
        &self,
        nuclide_idx: usize,
    ) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx]
            .inelastic_continuum_edist
            .as_ref()
    }

    fn n2n_edist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx].n2n_edist.as_ref()
    }

    fn n3n_edist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx].n3n_edist.as_ref()
    }

    fn apply_urr(&self, nuclide_idx: usize, xs: &mut MicroXs, energy: f64, xi: f64) {
        self.nuclides[nuclide_idx].apply_urr(xs, energy, xi);
    }

    fn thermal_scattering(&self, nuclide_idx: usize) -> Option<&ThermalScatteringData> {
        self.thermal.get(nuclide_idx)?.as_deref()
    }

    fn photon_products(&self, nuclide_idx: usize) -> &[(u32, hdf5_reader::PhotonProduct)] {
        &self.nuclides[nuclide_idx].photon_products
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
    build_kernel_from_data(&data, svd_rank, temp_idx, shared_grid, &reader.temperatures)
}

/// Build a ReactionKernel from a (possibly synthetic) NuclideData.
/// Same pipeline as `build_kernel_from_reader` but works off in-memory data
/// so the caller can pre-process xs_per_temp (e.g. sum across discrete levels
/// to synthesize MT=4 for nuclides whose ENDF/B-VII.1 evaluation omits the
/// total-inelastic block — Zr-90/91/92/94 in this project).
fn build_kernel_from_data(
    data: &hdf5_reader::NuclideData,
    svd_rank: usize,
    temp_idx: usize,
    shared_grid: &Arc<[f64]>,
    temperatures: &[f64],
) -> Option<ReactionKernel> {
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
    let coeffs = if n_t > 1 && !temperatures.is_empty() {
        let target_temp = temperatures[temp_idx.min(n_t - 1)];
        kernel.temp_coeffs_ducru(temperatures, target_temp)
    } else {
        let t_idx = temp_idx.min(n_t - 1);
        kernel.temp_coeffs(t_idx)
    };
    Some(ReactionKernel { kernel, coeffs })
}

/// Synthesize MT=4 (total inelastic) by summing the raw per-level
/// HDF5 cross sections across all discrete inelastic MTs (51..91).
/// Returns `None` if no levels are loadable. Used for Zr-90/91/92/94
/// and any other nuclide whose ENDF/B-VII.1 evaluation omits MT=4 —
/// avoids the GPU's 13× svd_reconstruct loop on the inelastic hot path
/// (paper §gpu engineering item).
fn synthesize_inelastic_mt4(
    reader: &NuclideFileReader,
    level_mts: &[u32],
    svd_rank: usize,
    temp_idx: usize,
    shared_grid: &Arc<[f64]>,
) -> Option<ReactionKernel> {
    let mut summed: Option<hdf5_reader::NuclideData> = None;
    for &mt in level_mts {
        if mt == 91 {
            // Continuum inelastic shares the inelastic channel but has its
            // own kinematics path (evaporation). Including its xs in the
            // synthetic MT=4 sum is correct for the macroscopic XS — the
            // sampling-side selection still distinguishes 91 from discrete.
        }
        let Ok(data) = reader.read_reaction(mt) else {
            continue;
        };
        if data.n_energy() == 0 || data.n_temp() == 0 {
            continue;
        }
        summed = Some(match summed.take() {
            None => data,
            Some(mut acc) => {
                if acc.n_energy() != data.n_energy() || acc.n_temp() != data.n_temp() {
                    // Different grids — should not happen within one nuclide
                    // (per-nuclide union grid is shared across all MTs in
                    // this loader), but guard anyway.
                    return None;
                }
                for t in 0..acc.n_temp() {
                    let dst = &mut acc.xs_per_temp[t];
                    let src = &data.xs_per_temp[t];
                    for i in 0..dst.len() {
                        dst[i] += src[i].max(0.0);
                    }
                }
                acc.mt = 4;
                acc
            }
        });
    }
    let summed = summed?;
    build_kernel_from_data(&summed, svd_rank, temp_idx, shared_grid, &reader.temperatures)
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
    load_nuclide_with_policy(
        h5_path,
        &RankPolicy::new(svd_rank),
        temp_idx,
        awr_fallback,
        nu_bar_fallback,
    )
}

/// Per-reaction SVD rank policy. Use `RankPolicy::new(default)` for a
/// uniform rank, then `with_mt(mt, rank)` to override. The default is
/// applied to any MT not explicitly overridden, plus to all discrete
/// inelastic levels (MT=51..91) for GPU stride-compatibility — the GPU
/// kernel reads discrete-level basis with a single fixed `P_RANK`.
///
/// Recommended defaults from `scripts/phase5_*.py` analysis on U-235:
///   - 47 of 52 reactions are rank-1 (machine-epsilon)
///   - smooth reactions (MT 2 elastic / 18 fission / 102 capture):
///     rank-1 sufficient when paired with WMP for the resonance window
///   - inelastic (MT 4) and continuum: rank 3-5
#[derive(Clone, Debug)]
pub struct RankPolicy {
    pub default: usize,
    pub per_mt: std::collections::HashMap<u32, usize>,
}

impl RankPolicy {
    pub fn new(default: usize) -> Self {
        Self {
            default: default.max(1),
            per_mt: std::collections::HashMap::new(),
        }
    }
    pub fn with_mt(mut self, mt: u32, rank: usize) -> Self {
        if rank > 0 {
            self.per_mt.insert(mt, rank);
        }
        self
    }
    pub fn rank_for(&self, mt: u32) -> usize {
        *self.per_mt.get(&mt).unwrap_or(&self.default)
    }
}

/// Adaptive-rank loader. Uses `policy.rank_for(MT)` for the six smooth
/// reactions (MT 2/4/16/17/18/102) and `policy.default` for all
/// discrete inelastic levels (GPU compatibility constraint, see comment
/// inside).
pub fn load_nuclide_with_policy(
    h5_path: &std::path::Path,
    policy: &RankPolicy,
    temp_idx: usize,
    awr_fallback: f64,
    nu_bar_fallback: f64,
) -> NuclideKernels {
    let svd_rank = policy.default;
    if policy.per_mt.is_empty() {
        println!("  Loading {} (rank={svd_rank})...", h5_path.display());
    } else {
        let mut overrides: Vec<_> = policy.per_mt.iter().collect();
        overrides.sort_by_key(|(mt, _)| **mt);
        let s: Vec<String> = overrides
            .iter()
            .map(|(m, r)| format!("MT={m}:{r}"))
            .collect();
        println!(
            "  Loading {} (rank={svd_rank}, overrides: {})...",
            h5_path.display(),
            s.join(", ")
        );
    }

    // Open file ONCE and cache energy grids
    let reader = match NuclideFileReader::open(h5_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    WARNING: failed to open {}: {e}", h5_path.display());
            return NuclideKernels {
                elastic: None,
                total_table: None,
                total_xs_raw: None,
                missing_xs: None,
                pointwise_xs: None,
                inelastic: None,
                n2n: None,
                n3n: None,
                fission: None,
                capture: None,
                awr: awr_fallback,
                nu_bar_const: nu_bar_fallback,
                nu_bar_table: None,
                discrete_levels: vec![],
                discrete_level_angles: vec![],
                has_continuum_inelastic: false,
                elastic_angle: None,
                fission_energy_dist: None,
                inelastic_continuum_edist: None,
                n2n_edist: None,
                n3n_edist: None,
                urr_tables: None,
                photon_products: Vec::new(),
            };
        }
    };

    let awr = reader.awr().unwrap_or(awr_fallback);
    println!(
        "    AWR = {awr:.3} ({} temps, {} energy pts)",
        reader.temp_labels.len(),
        reader.union_grid.len()
    );

    // Create shared energy grid — one Arc for all reactions in this nuclide
    let shared_grid: Arc<[f64]> = reader.union_grid.clone().into();

    let nu_bar_table = reader.nu_bar().ok();
    if let Some(ref t) = nu_bar_table
        && !t.energies.is_empty()
    {
        println!(
            "    nu-bar(E): {} pts, {:.3} @ thermal, {:.3} @ 1 MeV",
            t.energies.len(),
            t.lookup(0.0253),
            t.lookup(1.0e6)
        );
    }

    // Discrete levels — all read from the same open file
    let level_infos = reader.discrete_levels(awr);
    let has_continuum = level_infos.iter().any(|l| l.mt == 91);
    let n_levels = level_infos.len();
    let mut discrete_levels: Vec<DiscreteLevel> = Vec::with_capacity(n_levels);
    let mut discrete_level_angles: Vec<Option<AngularDistribution>> = Vec::with_capacity(n_levels);
    for info in level_infos {
        // Must match top-level svd_rank so the GPU kernel can use a single
        // rank value for basis stride (P_RANK). Using a different per-level
        // rank causes the GPU to read garbage data from wrong basis offsets.
        let kernel = build_kernel_from_reader(&reader, info.mt, svd_rank, temp_idx, &shared_grid);
        let angle = reader.angular_distribution(info.mt);
        discrete_level_angles.push(angle);
        discrete_levels.push(DiscreteLevel { info, kernel });
    }
    let loaded_count = discrete_levels
        .iter()
        .filter(|l| l.kernel.is_some())
        .count();
    let angles_count = discrete_level_angles.iter().filter(|a| a.is_some()).count();
    if n_levels > 0 {
        println!(
            "    Discrete levels: {loaded_count}/{n_levels} (continuum={has_continuum}, angle_dists={angles_count})"
        );
    }

    let fission_energy_dist = reader.fission_energy_dist();
    let inelastic_continuum_edist = reader.reaction_energy_dist(91);
    let n2n_edist = reader.reaction_energy_dist(16);
    let n3n_edist = reader.reaction_energy_dist(17);
    if let Some(ref d) = fission_energy_dist {
        println!("    Fission spectrum: {} energies", d.energies.len());
    }

    let elastic_angle = reader.angular_distribution(2);
    if let Some(ref a) = elastic_angle {
        println!(
            "    Elastic angular dist: {} energies, CM={}",
            a.energies.len(),
            a.center_of_mass
        );
    }

    let urr_temp = reader
        .temp_labels
        .get(temp_idx)
        .cloned()
        .unwrap_or_else(|| "294K".to_string());
    let urr_tables = reader.urr_tables(&urr_temp);
    if let Some(ref u) = urr_tables {
        println!(
            "    URR: {} energies, {} bands, {:.0}–{:.0} eV",
            u.energies.len(),
            u.n_bands,
            u.energies.first().unwrap_or(&0.0),
            u.energies.last().unwrap_or(&0.0)
        );
    }

    let total_xs_vec = reader.compute_total_xs(temp_idx);
    let total_table = total_xs_vec
        .as_ref()
        .map(|xs| PointwiseTable::from_shared_grid(shared_grid.clone(), xs.clone()));
    let elastic = build_kernel_from_reader(&reader, 2, policy.rank_for(2), temp_idx, &shared_grid);
    if elastic.is_some() {
        println!("    MT=2  (elastic)  rank={}", policy.rank_for(2));
    }
    let inelastic = build_kernel_from_reader(&reader, 4, policy.rank_for(4), temp_idx, &shared_grid)
        .or_else(|| {
            // ENDF/B-VII.1 omits the MT=4 total-inelastic block for some
            // nuclides (Zr-90/91/92/94 in this project). Synthesize it
            // by summing the raw per-level HDF5 cross sections across
            // MT=51..91, then SVD-decompose the result. Eliminates the
            // 13× svd_reconstruct loop that is the GPU SVD-vs-pointwise
            // gap driver on PWR (paper §gpu).
            if discrete_levels.is_empty() {
                return None;
            }
            let level_mts: Vec<u32> = discrete_levels.iter().map(|l| l.info.mt).collect();
            synthesize_inelastic_mt4(
                &reader,
                &level_mts,
                policy.rank_for(4),
                temp_idx,
                &shared_grid,
            )
        });
    if inelastic.is_some() {
        let label = if reader.read_reaction(4).is_ok() {
            "rank"
        } else {
            "rank, synthesized from {N} discrete levels"
        };
        println!(
            "    MT=4  (inelastic) {}={}{}",
            if label.contains("synthesized") {
                "synthetic rank"
            } else {
                "rank"
            },
            policy.rank_for(4),
            if label.contains("synthesized") {
                format!(" (sum over {} levels)", discrete_levels.len())
            } else {
                String::new()
            }
        );
    }
    let n2n = build_kernel_from_reader(&reader, 16, policy.rank_for(16), temp_idx, &shared_grid);
    if n2n.is_some() {
        println!("    MT=16 (n,2n)     rank={}", policy.rank_for(16));
    }
    let n3n = build_kernel_from_reader(&reader, 17, policy.rank_for(17), temp_idx, &shared_grid);
    if n3n.is_some() {
        println!("    MT=17 (n,3n)     rank={}", policy.rank_for(17));
    }
    let fission =
        build_kernel_from_reader(&reader, 18, policy.rank_for(18), temp_idx, &shared_grid);
    if fission.is_some() {
        println!("    MT=18 (fission)  rank={}", policy.rank_for(18));
    }
    let capture =
        build_kernel_from_reader(&reader, 102, policy.rank_for(102), temp_idx, &shared_grid);
    if capture.is_some() {
        println!("    MT=102 (capture) rank={}", policy.rank_for(102));
    }

    let missing_xs = total_xs_vec.as_ref().map(|total| {
        let n = shared_grid.len();
        let mut partial_sum = vec![0.0_f64; n];
        for mt in [2_u32, 16, 17, 18, 102] {
            if let Ok(data) = reader.read_reaction(mt)
                && let Some(xs) = data.xs_per_temp.get(temp_idx)
            {
                for i in 0..n.min(xs.len()) {
                    partial_sum[i] += xs[i].max(0.0);
                }
            }
        }
        if let Ok(data) = reader.read_reaction(4) {
            if let Some(xs) = data.xs_per_temp.get(temp_idx) {
                for i in 0..n.min(xs.len()) {
                    partial_sum[i] += xs[i].max(0.0);
                }
            }
        } else {
            for mt in 51..=91 {
                if let Ok(data) = reader.read_reaction(mt)
                    && let Some(xs) = data.xs_per_temp.get(temp_idx)
                {
                    for i in 0..n.min(xs.len()) {
                        partial_sum[i] += xs[i].max(0.0);
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

    let pointwise_xs = reader.compute_pointwise_xs(temp_idx);

    NuclideKernels {
        elastic,
        total_table,
        total_xs_raw: total_xs_vec,
        missing_xs,
        pointwise_xs,
        inelastic,
        n2n,
        n3n,
        fission,
        capture,
        awr,
        nu_bar_const: nu_bar_fallback,
        nu_bar_table,
        discrete_levels,
        discrete_level_angles,
        has_continuum_inelastic: has_continuum,
        elastic_angle,
        fission_energy_dist,
        inelastic_continuum_edist,
        n2n_edist,
        n3n_edist,
        urr_tables,
        photon_products: load_photon_products(&reader),
    }
}

// ── Pointwise Table XS Provider (OpenMC-style baseline) ─────────────

/// A discrete inelastic level backed by a pointwise table.
pub struct TableDiscreteLevel {
    pub info: DiscreteLevelInfo,
    pub table: Option<StochTempTable>,
}

/// Per-nuclide pointwise cross-section tables — the OpenMC baseline.
///
/// Stores the same physics data as `NuclideKernels` but uses raw
/// pointwise tables instead of SVD-compressed kernels for XS lookup.
/// Each reaction holds a [`StochTempTable`] which is either single-temp
/// (on-library) or two-temp (off-library, stochastic pseudo-interpolation).
pub struct NuclideTableData {
    pub elastic: Option<StochTempTable>,
    pub total_table: Option<StochTempTable>,
    pub inelastic: Option<StochTempTable>,
    pub n2n: Option<StochTempTable>,
    pub n3n: Option<StochTempTable>,
    pub fission: Option<StochTempTable>,
    pub capture: Option<StochTempTable>,
    pub awr: f64,
    pub nu_bar_const: f64,
    pub nu_bar_table: Option<NuBarTable>,
    pub discrete_levels: Vec<TableDiscreteLevel>,
    pub discrete_level_angles: Vec<Option<AngularDistribution>>,
    pub has_continuum_inelastic: bool,
    pub elastic_angle: Option<AngularDistribution>,
    pub fission_energy_dist: Option<EnergyDistribution>,
    pub inelastic_continuum_edist: Option<EnergyDistribution>,
    pub n2n_edist: Option<EnergyDistribution>,
    pub n3n_edist: Option<EnergyDistribution>,
    pub urr_tables: Option<UrrProbabilityTables>,
    /// Photon products keyed by ENDF MT. Populated from HDF5
    /// `reactions/reaction_{mt}/product_N` groups with
    /// `particle="photon"`. Used by the transport loop to sample the
    /// capture / fission / inelastic γ spectrum at each neutron
    /// reaction site for coupled neutron-photon tallies.
    pub photon_products: Vec<(u32, hdf5_reader::PhotonProduct)>,
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
        self.discrete_levels
            .iter()
            .map(|l| l.info.clone())
            .collect()
    }

    pub fn apply_urr(&self, xs: &mut MicroXs, energy: f64, xi: f64) {
        if urr_disabled() {
            return;
        }
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
        if let Some(t) = &self.elastic {
            total += t.memory_bytes();
        }
        if let Some(t) = &self.inelastic {
            total += t.memory_bytes();
        }
        if let Some(t) = &self.n2n {
            total += t.memory_bytes();
        }
        if let Some(t) = &self.n3n {
            total += t.memory_bytes();
        }
        if let Some(t) = &self.fission {
            total += t.memory_bytes();
        }
        if let Some(t) = &self.capture {
            total += t.memory_bytes();
        }
        for lvl in &self.discrete_levels {
            if let Some(t) = &lvl.table {
                total += t.memory_bytes();
            }
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

        // Every PointwiseTable in this nuclide was built with the same
        // shared_grid Arc, so one bracket search serves all reactions.
        // Falls back to per-call search only if every table is None.
        let any_table = nuc
            .total_table
            .as_ref()
            .or(nuc.elastic.as_ref())
            .or(nuc.fission.as_ref())
            .or(nuc.capture.as_ref())
            .or(nuc.inelastic.as_ref())
            .or(nuc.n2n.as_ref())
            .or(nuc.n3n.as_ref());
        let idx = any_table.map_or(0, |t| t.bracket_idx(energy));

        // Draw ONE stochastic-temperature pick per nuclide per collision
        // and reuse it for every channel lookup. This keeps partial
        // channels (el + in + fis + cap + ...) consistent with a single
        // library endpoint — otherwise per-channel independent picks
        // mix 600 K elastic with 900 K capture on the same collision,
        // which biases k_inf. Equivalent to what OpenMC does at the
        // particle-collision level.
        let use_hi = any_table.is_some_and(|t| t.draw_pick());

        let elastic = nuc
            .elastic
            .as_ref()
            .map_or(0.0, |t| t.lookup_at_idx_with_pick(energy, idx, use_hi));
        let inelastic = match &nuc.inelastic {
            Some(t) => t.lookup_at_idx_with_pick(energy, idx, use_hi),
            None if !nuc.discrete_levels.is_empty() => nuc
                .discrete_levels
                .iter()
                .filter(|lvl| energy >= lvl.info.threshold)
                .filter_map(|lvl| lvl.table.as_ref())
                .map(|t| t.lookup_at_idx_with_pick(energy, idx, use_hi).max(0.0))
                .sum::<f64>(),
            None => 0.0,
        };
        let n2n = nuc
            .n2n
            .as_ref()
            .map_or(0.0, |t| t.lookup_at_idx_with_pick(energy, idx, use_hi));
        let n3n = nuc
            .n3n
            .as_ref()
            .map_or(0.0, |t| t.lookup_at_idx_with_pick(energy, idx, use_hi));
        let fission = nuc
            .fission
            .as_ref()
            .map_or(0.0, |t| t.lookup_at_idx_with_pick(energy, idx, use_hi));
        let mut capture = nuc
            .capture
            .as_ref()
            .map_or(0.0, |t| t.lookup_at_idx_with_pick(energy, idx, use_hi));

        let total = match &nuc.total_table {
            Some(tt) => {
                let tot = tt.lookup_at_idx_with_pick(energy, idx, use_hi);
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

    fn elastic_angular_dist(
        &self,
        nuclide_idx: usize,
    ) -> Option<&hdf5_reader::AngularDistribution> {
        self.nuclides[nuclide_idx].elastic_angle.as_ref()
    }

    fn discrete_level_angles(
        &self,
        nuclide_idx: usize,
    ) -> &[Option<hdf5_reader::AngularDistribution>] {
        &self.nuclides[nuclide_idx].discrete_level_angles
    }

    fn fission_energy_dist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx].fission_energy_dist.as_ref()
    }

    fn inelastic_continuum_edist(
        &self,
        nuclide_idx: usize,
    ) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx]
            .inelastic_continuum_edist
            .as_ref()
    }

    fn n2n_edist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx].n2n_edist.as_ref()
    }

    fn n3n_edist(&self, nuclide_idx: usize) -> Option<&hdf5_reader::EnergyDistribution> {
        self.nuclides[nuclide_idx].n3n_edist.as_ref()
    }

    fn apply_urr(&self, nuclide_idx: usize, xs: &mut MicroXs, energy: f64, xi: f64) {
        self.nuclides[nuclide_idx].apply_urr(xs, energy, xi);
    }

    fn thermal_scattering(&self, nuclide_idx: usize) -> Option<&ThermalScatteringData> {
        self.thermal.get(nuclide_idx)?.as_deref()
    }

    fn photon_products(&self, nuclide_idx: usize) -> &[(u32, hdf5_reader::PhotonProduct)] {
        &self.nuclides[nuclide_idx].photon_products
    }
}

/// Build a pointwise table for one reaction using a shared energy grid.
/// Returns a single-temp `StochTempTable` (no stochastic pick).
fn build_table_from_reader(
    reader: &NuclideFileReader,
    mt: u32,
    temp_idx: usize,
    shared_grid: &Arc<[f64]>,
) -> Option<StochTempTable> {
    let data = reader.read_reaction(mt).ok()?;
    if data.n_energy() == 0 || data.n_temp() == 0 {
        return None;
    }
    let t = temp_idx.min(data.n_temp() - 1);
    let pt = PointwiseTable::from_shared_grid(Arc::clone(shared_grid), data.xs_per_temp[t].clone());
    Some(StochTempTable::single(pt))
}

/// Build a stochastic-temperature table for one reaction. Loads both
/// bracketing library temperatures so a per-lookup random pick forces
/// cache loads into two XS arrays (OpenMC-style pseudo-interpolation).
/// Falls back to a single-temp table when `target_temp` matches a
/// library endpoint exactly, or when only one library temperature exists.
fn build_stoch_table_from_reader(
    reader: &NuclideFileReader,
    mt: u32,
    target_temp: f64,
    shared_grid: &Arc<[f64]>,
) -> Option<StochTempTable> {
    let data = reader.read_reaction(mt).ok()?;
    if data.n_energy() == 0 || data.n_temp() == 0 {
        return None;
    }

    let (i_lo, i_hi) = bracket_temp_indices(&data.temperatures, target_temp);
    let pt_lo =
        PointwiseTable::from_shared_grid(Arc::clone(shared_grid), data.xs_per_temp[i_lo].clone());
    if i_lo == i_hi {
        return Some(StochTempTable::single(pt_lo));
    }
    let pt_hi =
        PointwiseTable::from_shared_grid(Arc::clone(shared_grid), data.xs_per_temp[i_hi].clone());
    Some(StochTempTable::stochastic(
        pt_lo,
        pt_hi,
        target_temp,
        data.temperatures[i_lo],
        data.temperatures[i_hi],
    ))
}

/// Find the bracketing temperature indices for `target_temp` in a
/// sorted-ascending temperature list. Returns `(i, i)` when the target
/// is at or outside an endpoint (single-temp fallback).
fn bracket_temp_indices(temps: &[f64], target: f64) -> (usize, usize) {
    if temps.len() < 2 {
        return (0, 0);
    }
    if target <= temps[0] {
        return (0, 0);
    }
    if target >= temps[temps.len() - 1] {
        let n = temps.len() - 1;
        return (n, n);
    }
    // Find i such that temps[i] <= target < temps[i+1]
    for i in 0..temps.len() - 1 {
        if target >= temps[i] && target < temps[i + 1] {
            // Exact match on lower endpoint → single-temp fallback
            if (target - temps[i]).abs() < 1e-6 {
                return (i, i);
            }
            return (i, i + 1);
        }
    }
    // Shouldn't reach here given the bounds checks above
    let n = temps.len() - 1;
    (n, n)
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
                elastic: None,
                total_table: None,
                inelastic: None,
                n2n: None,
                n3n: None,
                fission: None,
                capture: None,
                awr: awr_fallback,
                nu_bar_const: nu_bar_fallback,
                nu_bar_table: None,
                discrete_levels: vec![],
                discrete_level_angles: vec![],
                has_continuum_inelastic: false,
                elastic_angle: None,
                fission_energy_dist: None,
                inelastic_continuum_edist: None,
                n2n_edist: None,
                n3n_edist: None,
                urr_tables: None,
                photon_products: Vec::new(),
            };
        }
    };

    let awr = reader.awr().unwrap_or(awr_fallback);
    println!(
        "    AWR = {awr:.3} ({} temps, {} energy pts)",
        reader.temp_labels.len(),
        reader.union_grid.len()
    );

    // Create shared energy grid for all tables in this nuclide
    let shared_grid: Arc<[f64]> = reader.union_grid.clone().into();

    let nu_bar_table = reader.nu_bar().ok();
    if let Some(ref t) = nu_bar_table
        && !t.energies.is_empty()
    {
        println!(
            "    nu-bar(E): {} pts, {:.3} @ thermal, {:.3} @ 1 MeV",
            t.energies.len(),
            t.lookup(0.0253),
            t.lookup(1.0e6)
        );
    }

    let level_infos = reader.discrete_levels(awr);
    let has_continuum = level_infos.iter().any(|l| l.mt == 91);
    let n_levels = level_infos.len();
    let mut discrete_levels: Vec<TableDiscreteLevel> = Vec::with_capacity(n_levels);
    let mut discrete_level_angles: Vec<Option<AngularDistribution>> = Vec::with_capacity(n_levels);
    for info in level_infos {
        let table = build_table_from_reader(&reader, info.mt, temp_idx, &shared_grid);
        discrete_level_angles.push(reader.angular_distribution(info.mt));
        discrete_levels.push(TableDiscreteLevel { info, table });
    }
    let loaded_count = discrete_levels.iter().filter(|l| l.table.is_some()).count();
    let angles_count = discrete_level_angles.iter().filter(|a| a.is_some()).count();
    if n_levels > 0 {
        println!(
            "    Discrete levels: {loaded_count}/{n_levels} (continuum={has_continuum}, angle_dists={angles_count})"
        );
    }

    let fission_energy_dist = reader.fission_energy_dist();
    let inelastic_continuum_edist = reader.reaction_energy_dist(91);
    let n2n_edist = reader.reaction_energy_dist(16);
    let n3n_edist = reader.reaction_energy_dist(17);
    if let Some(ref d) = fission_energy_dist {
        println!("    Fission spectrum: {} energies", d.energies.len());
    }

    let elastic_angle = reader.angular_distribution(2);
    if let Some(ref a) = elastic_angle {
        println!(
            "    Elastic angular dist: {} energies, CM={}",
            a.energies.len(),
            a.center_of_mass
        );
    }

    let urr_temp = reader
        .temp_labels
        .get(temp_idx)
        .cloned()
        .unwrap_or_else(|| "294K".to_string());
    let urr_tables = reader.urr_tables(&urr_temp);
    if let Some(ref u) = urr_tables {
        println!(
            "    URR: {} energies, {} bands, {:.0}–{:.0} eV",
            u.energies.len(),
            u.n_bands,
            u.energies.first().unwrap_or(&0.0),
            u.energies.last().unwrap_or(&0.0)
        );
    }

    let total_table = reader.compute_total_xs(temp_idx).map(|xs| {
        StochTempTable::single(PointwiseTable::from_shared_grid(shared_grid.clone(), xs))
    });
    let elastic = build_table_from_reader(&reader, 2, temp_idx, &shared_grid);
    if elastic.is_some() {
        println!("    MT=2 (elastic)");
    }
    let inelastic = build_table_from_reader(&reader, 4, temp_idx, &shared_grid);
    if inelastic.is_some() {
        println!("    MT=4 (inelastic)");
    } else if !discrete_levels.is_empty() {
        println!(
            "    MT=4 (inelastic) — synthesized from {} discrete levels",
            discrete_levels.len()
        );
    }
    let n2n = build_table_from_reader(&reader, 16, temp_idx, &shared_grid);
    if n2n.is_some() {
        println!("    MT=16 (n,2n)");
    }
    let n3n = build_table_from_reader(&reader, 17, temp_idx, &shared_grid);
    if n3n.is_some() {
        println!("    MT=17 (n,3n)");
    }
    let fission = build_table_from_reader(&reader, 18, temp_idx, &shared_grid);
    if fission.is_some() {
        println!("    MT=18 (fission)");
    }
    let capture = build_table_from_reader(&reader, 102, temp_idx, &shared_grid);
    if capture.is_some() {
        println!("    MT=102 (capture)");
    }

    NuclideTableData {
        elastic,
        total_table,
        inelastic,
        n2n,
        n3n,
        fission,
        capture,
        awr,
        nu_bar_const: nu_bar_fallback,
        nu_bar_table,
        discrete_levels,
        discrete_level_angles,
        has_continuum_inelastic: has_continuum,
        elastic_angle,
        fission_energy_dist,
        inelastic_continuum_edist,
        n2n_edist,
        n3n_edist,
        urr_tables,
        photon_products: load_photon_products(&reader),
    }
}

/// Load a complete nuclide as stochastic-temperature pointwise tables.
///
/// When `target_temp` lies between two library endpoints, each reaction
/// holds both tables and picks one per lookup with probability
/// `(T_hi - T)/(T_hi - T_lo)` — OpenMC's "pseudo-interpolation". This
/// forces random memory loads into two XS arrays per collision and is
/// the realistic cache-pressure scenario for a real-world operating
/// temperature that is not in the library.
///
/// Auxiliary metadata (nu-bar, angular distributions, fission spectrum,
/// URR tables, total XS) is loaded at the nearest library temperature.
pub fn load_nuclide_table_at_temp(
    h5_path: &std::path::Path,
    target_temp: f64,
    awr_fallback: f64,
    nu_bar_fallback: f64,
) -> NuclideTableData {
    println!(
        "  Loading {} (pointwise table, target T = {:.1} K)...",
        h5_path.display(),
        target_temp
    );

    let reader = match NuclideFileReader::open(h5_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    WARNING: failed to open {}: {e}", h5_path.display());
            return NuclideTableData {
                elastic: None,
                total_table: None,
                inelastic: None,
                n2n: None,
                n3n: None,
                fission: None,
                capture: None,
                awr: awr_fallback,
                nu_bar_const: nu_bar_fallback,
                nu_bar_table: None,
                discrete_levels: vec![],
                discrete_level_angles: vec![],
                has_continuum_inelastic: false,
                elastic_angle: None,
                fission_energy_dist: None,
                inelastic_continuum_edist: None,
                n2n_edist: None,
                n3n_edist: None,
                urr_tables: None,
                photon_products: Vec::new(),
            };
        }
    };

    let awr = reader.awr().unwrap_or(awr_fallback);
    let (i_lo, i_hi) = bracket_temp_indices(&reader.temperatures, target_temp);
    let stoch = i_lo != i_hi;
    if stoch {
        println!(
            "    AWR = {awr:.3} | stochastic T between {:.1} K and {:.1} K ({} temps, {} energy pts)",
            reader.temperatures[i_lo],
            reader.temperatures[i_hi],
            reader.temp_labels.len(),
            reader.union_grid.len()
        );
    } else {
        println!(
            "    AWR = {awr:.3} | T = {:.1} K (on-library, no stochastic pick) ({} temps, {} energy pts)",
            reader.temperatures[i_lo],
            reader.temp_labels.len(),
            reader.union_grid.len()
        );
    }

    let aux_temp_idx = if stoch {
        // Use the lower endpoint for aux data (nu-bar/URR/total) — these are
        // weakly T-dependent at Godiva energies, and this keeps the memory
        // accounting honest (aux data not duplicated).
        i_lo
    } else {
        i_lo
    };

    let shared_grid: Arc<[f64]> = reader.union_grid.clone().into();

    let nu_bar_table = reader.nu_bar().ok();
    if let Some(ref t) = nu_bar_table
        && !t.energies.is_empty()
    {
        println!(
            "    nu-bar(E): {} pts, {:.3} @ thermal, {:.3} @ 1 MeV",
            t.energies.len(),
            t.lookup(0.0253),
            t.lookup(1.0e6)
        );
    }

    let level_infos = reader.discrete_levels(awr);
    let has_continuum = level_infos.iter().any(|l| l.mt == 91);
    let n_levels = level_infos.len();
    let mut discrete_levels: Vec<TableDiscreteLevel> = Vec::with_capacity(n_levels);
    let mut discrete_level_angles: Vec<Option<AngularDistribution>> = Vec::with_capacity(n_levels);
    for info in level_infos {
        let table = build_stoch_table_from_reader(&reader, info.mt, target_temp, &shared_grid);
        discrete_level_angles.push(reader.angular_distribution(info.mt));
        discrete_levels.push(TableDiscreteLevel { info, table });
    }
    let loaded_count = discrete_levels.iter().filter(|l| l.table.is_some()).count();
    if n_levels > 0 {
        println!(
            "    Discrete levels: {loaded_count}/{n_levels} (continuum={has_continuum}, stochastic={stoch})"
        );
    }

    let fission_energy_dist = reader.fission_energy_dist();
    let inelastic_continuum_edist = reader.reaction_energy_dist(91);
    let n2n_edist = reader.reaction_energy_dist(16);
    let n3n_edist = reader.reaction_energy_dist(17);
    let elastic_angle = reader.angular_distribution(2);

    let urr_temp = reader
        .temp_labels
        .get(aux_temp_idx)
        .cloned()
        .unwrap_or_else(|| "294K".to_string());
    let urr_tables = reader.urr_tables(&urr_temp);

    // Total XS: when stochastic, build two StochTempTable endpoints so the
    // macro-XS collision distance is also drawn from the picked library.
    let total_table = if stoch {
        let tot_lo = reader
            .compute_total_xs(i_lo)
            .map(|xs| PointwiseTable::from_shared_grid(Arc::clone(&shared_grid), xs));
        let tot_hi = reader
            .compute_total_xs(i_hi)
            .map(|xs| PointwiseTable::from_shared_grid(Arc::clone(&shared_grid), xs));
        match (tot_lo, tot_hi) {
            (Some(lo), Some(hi)) => Some(StochTempTable::stochastic(
                lo,
                hi,
                target_temp,
                reader.temperatures[i_lo],
                reader.temperatures[i_hi],
            )),
            (Some(lo), None) => Some(StochTempTable::single(lo)),
            _ => None,
        }
    } else {
        reader.compute_total_xs(aux_temp_idx).map(|xs| {
            StochTempTable::single(PointwiseTable::from_shared_grid(
                Arc::clone(&shared_grid),
                xs,
            ))
        })
    };

    let elastic = build_stoch_table_from_reader(&reader, 2, target_temp, &shared_grid);
    if elastic.is_some() {
        println!("    MT=2 (elastic)");
    }
    let inelastic = build_stoch_table_from_reader(&reader, 4, target_temp, &shared_grid);
    if inelastic.is_some() {
        println!("    MT=4 (inelastic)");
    }
    let n2n = build_stoch_table_from_reader(&reader, 16, target_temp, &shared_grid);
    let n3n = build_stoch_table_from_reader(&reader, 17, target_temp, &shared_grid);
    let fission = build_stoch_table_from_reader(&reader, 18, target_temp, &shared_grid);
    if fission.is_some() {
        println!("    MT=18 (fission)");
    }
    let capture = build_stoch_table_from_reader(&reader, 102, target_temp, &shared_grid);

    NuclideTableData {
        elastic,
        total_table,
        inelastic,
        n2n,
        n3n,
        fission,
        capture,
        awr,
        nu_bar_const: nu_bar_fallback,
        nu_bar_table,
        discrete_levels,
        discrete_level_angles,
        has_continuum_inelastic: has_continuum,
        elastic_angle,
        fission_energy_dist,
        inelastic_continuum_edist,
        n2n_edist,
        n3n_edist,
        urr_tables,
        photon_products: load_photon_products(&reader),
    }
}

/// Load a complete nuclide as SVD kernels for a specific target temperature.
///
/// Uses the Ducru kernel reconstruction (2017) to build continuous
/// temperature coefficients at `target_temp` — no library endpoint
/// snapping. Auxiliary pointwise data (total XS, URR, pointwise_xs for
/// GPU, missing channels) is loaded at the nearest library temperature.
/// The `total_table` is omitted when `target_temp` is off-library so the
/// SVD provider sums partials at its own reconstructed T, keeping the
/// comparison apples-to-apples.
pub fn load_nuclide_at_temp(
    h5_path: &std::path::Path,
    svd_rank: usize,
    target_temp: f64,
    awr_fallback: f64,
    nu_bar_fallback: f64,
    discrete_rank: Option<usize>,
) -> NuclideKernels {
    let d_rank = discrete_rank.unwrap_or(svd_rank);
    println!(
        "  Loading {} (rank={svd_rank}, discrete={d_rank}, target T = {:.1} K)...",
        h5_path.display(),
        target_temp
    );

    let reader = match NuclideFileReader::open(h5_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    WARNING: failed to open {}: {e}", h5_path.display());
            return NuclideKernels {
                elastic: None,
                total_table: None,
                total_xs_raw: None,
                missing_xs: None,
                pointwise_xs: None,
                inelastic: None,
                n2n: None,
                n3n: None,
                fission: None,
                capture: None,
                awr: awr_fallback,
                nu_bar_const: nu_bar_fallback,
                nu_bar_table: None,
                discrete_levels: vec![],
                discrete_level_angles: vec![],
                has_continuum_inelastic: false,
                elastic_angle: None,
                fission_energy_dist: None,
                inelastic_continuum_edist: None,
                n2n_edist: None,
                n3n_edist: None,
                urr_tables: None,
                photon_products: Vec::new(),
            };
        }
    };

    let awr = reader.awr().unwrap_or(awr_fallback);
    let (i_lo, i_hi) = bracket_temp_indices(&reader.temperatures, target_temp);
    let off_library = i_lo != i_hi;
    if off_library {
        println!(
            "    AWR = {awr:.3} | Ducru interp at {:.1} K (between {:.1}/{:.1} K) ({} temps, {} energy pts)",
            target_temp,
            reader.temperatures[i_lo],
            reader.temperatures[i_hi],
            reader.temp_labels.len(),
            reader.union_grid.len()
        );
    } else {
        println!(
            "    AWR = {awr:.3} | T = {:.1} K (on-library) ({} temps, {} energy pts)",
            reader.temperatures[i_lo],
            reader.temp_labels.len(),
            reader.union_grid.len()
        );
    }

    let aux_temp_idx = i_lo;
    let shared_grid: Arc<[f64]> = reader.union_grid.clone().into();

    let nu_bar_table = reader.nu_bar().ok();

    let level_infos = reader.discrete_levels(awr);
    let has_continuum = level_infos.iter().any(|l| l.mt == 91);
    let n_levels = level_infos.len();
    let mut discrete_levels: Vec<DiscreteLevel> = Vec::with_capacity(n_levels);
    let mut discrete_level_angles: Vec<Option<AngularDistribution>> = Vec::with_capacity(n_levels);
    for info in level_infos {
        let kernel = build_kernel_at_temp(&reader, info.mt, d_rank, target_temp, &shared_grid);
        discrete_level_angles.push(reader.angular_distribution(info.mt));
        discrete_levels.push(DiscreteLevel { info, kernel });
    }
    let loaded_count = discrete_levels
        .iter()
        .filter(|l| l.kernel.is_some())
        .count();
    if n_levels > 0 {
        println!("    Discrete levels: {loaded_count}/{n_levels} (continuum={has_continuum})");
    }

    let fission_energy_dist = reader.fission_energy_dist();
    let inelastic_continuum_edist = reader.reaction_energy_dist(91);
    let n2n_edist = reader.reaction_energy_dist(16);
    let n3n_edist = reader.reaction_energy_dist(17);
    let elastic_angle = reader.angular_distribution(2);

    let urr_temp = reader
        .temp_labels
        .get(aux_temp_idx)
        .cloned()
        .unwrap_or_else(|| "294K".to_string());
    let urr_tables = reader.urr_tables(&urr_temp);

    // Build total XS at target_temp. On-library: pick directly. Off-library:
    // Ducru-weighted sum of library totals (same weights as the SVD kernel
    // reconstruction), so the "total - partials → capture" calibration
    // trick in SvdXsProvider::lookup stays consistent with partial channels.
    let total_xs_vec = if off_library {
        interp_total_at_temp(&reader, target_temp, i_lo, i_hi)
    } else {
        reader.compute_total_xs(aux_temp_idx)
    };
    let total_table = total_xs_vec
        .as_ref()
        .map(|xs| PointwiseTable::from_shared_grid(Arc::clone(&shared_grid), xs.clone()));

    let elastic = build_kernel_at_temp(&reader, 2, svd_rank, target_temp, &shared_grid);
    let inelastic = build_kernel_at_temp(&reader, 4, svd_rank, target_temp, &shared_grid);
    let n2n = build_kernel_at_temp(&reader, 16, svd_rank, target_temp, &shared_grid);
    let n3n = build_kernel_at_temp(&reader, 17, svd_rank, target_temp, &shared_grid);
    let fission = build_kernel_at_temp(&reader, 18, svd_rank, target_temp, &shared_grid);
    let capture = build_kernel_at_temp(&reader, 102, svd_rank, target_temp, &shared_grid);

    // Auxiliary pointwise at nearest library temp (for GPU upload).
    let pointwise_xs = reader.compute_pointwise_xs(aux_temp_idx);
    let missing_xs = total_xs_vec.as_ref().map(|total| {
        let n = shared_grid.len();
        let mut partial_sum = vec![0.0_f64; n];
        for mt in [2_u32, 16, 17, 18, 102] {
            if let Ok(data) = reader.read_reaction(mt)
                && let Some(xs) = data.xs_per_temp.get(aux_temp_idx)
            {
                for i in 0..n.min(xs.len()) {
                    partial_sum[i] += xs[i].max(0.0);
                }
            }
        }
        if let Ok(data) = reader.read_reaction(4) {
            if let Some(xs) = data.xs_per_temp.get(aux_temp_idx) {
                for i in 0..n.min(xs.len()) {
                    partial_sum[i] += xs[i].max(0.0);
                }
            }
        } else {
            for mt in 51..=91 {
                if let Ok(data) = reader.read_reaction(mt)
                    && let Some(xs) = data.xs_per_temp.get(aux_temp_idx)
                {
                    for i in 0..n.min(xs.len()) {
                        partial_sum[i] += xs[i].max(0.0);
                    }
                }
            }
        }
        (0..n)
            .map(|i| (total[i] - partial_sum[i]).max(0.0))
            .collect::<Vec<f64>>()
    });

    NuclideKernels {
        elastic,
        total_table,
        total_xs_raw: total_xs_vec,
        missing_xs,
        pointwise_xs,
        inelastic,
        n2n,
        n3n,
        fission,
        capture,
        awr,
        nu_bar_const: nu_bar_fallback,
        nu_bar_table,
        discrete_levels,
        discrete_level_angles,
        has_continuum_inelastic: has_continuum,
        elastic_angle,
        fission_energy_dist,
        inelastic_continuum_edist,
        n2n_edist,
        n3n_edist,
        urr_tables,
        photon_products: load_photon_products(&reader),
    }
}

/// Linear-in-√T interpolation fraction: 0 at `t_lo`, 1 at `t_hi`.
/// Kept as a fallback; the default off-library path uses `ducru_unity_2temp`.
#[inline]
#[allow(dead_code)]
fn sqrt_temp_alpha(target: f64, t_lo: f64, t_hi: f64) -> f64 {
    let denom = t_hi.sqrt() - t_lo.sqrt();
    if denom.abs() < 1e-12 {
        0.0
    } else {
        ((target.sqrt() - t_lo.sqrt()) / denom).clamp(0.0, 1.0)
    }
}

/// Select the `k` library temperature indices nearest to `target`.
/// Returns indices sorted ascending (stable order for downstream code).
fn nearest_k_temps(temps: &[f64], target: f64, k: usize) -> Vec<usize> {
    let n = temps.len().min(k);
    if n == 0 {
        return vec![];
    }
    let mut idx: Vec<usize> = (0..temps.len()).collect();
    idx.sort_by(|&a, &b| {
        let da = (temps[a] - target).abs();
        let db = (temps[b] - target).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(n);
    idx.sort();
    idx
}

/// Partition-of-unity N-point Ducru weights on a temperature subset.
///
/// Raw Ducru (2017) Eq. 31 weights on `sub_temps` at `target`, then
/// normalized `w_k ← w_k / Σ w` so the reconstruction preserves peak
/// heights. For N = 2 this reduces to [`ducru_unity_2temp`]; for
/// N = 3 we pick the three library columns nearest `target` and solve
/// the 3x3 Ducru formula, which captures a quadratic correction to the
/// Faddeeva kernel that 2-point cannot. Higher N is numerically unstable
/// (product-of-ratios blowup).
///
/// Validated on U-238 MT=102 held-out 900 K (5-column training set):
/// N=2 peak ratio 1.009 → N=3 peak ratio 0.994; L2 1.48 % → 1.05 %.
/// Weights from the raw Ducru can go negative (the 294 K column earns
/// a -0.12 weight for the 600/294/1200 bracket at target 900 K) —
/// that's the signal for moving to QP-constrained weights next.
fn ducru_unity_weights(sub_temps: &[f64], target: f64) -> Vec<f64> {
    use crate::kernel::ducru_weights;
    let raw = ducru_weights(sub_temps, target);
    let s: f64 = raw.iter().sum();
    if s.abs() < 1e-12 {
        // Degenerate (near-collision); fall back to equal split.
        return vec![1.0 / sub_temps.len() as f64; sub_temps.len()];
    }
    raw.iter().map(|w| w / s).collect()
}

/// Partition-of-unity 2-point Ducru weights for interpolating a cross-
/// section at `target` K from library samples at `t_lo` and `t_hi`.
///
/// The raw Ducru (2017) Eq. 31 weights are L2-optimal in the free-Doppler
/// kernel approximation but do not sum to 1 — a quadrature-style scheme,
/// not a partition of unity. For resonance-dominated channels (U-238
/// 6.67 eV) this introduces a log-space gain error that breaks the peak
/// height. Normalizing `w_k ← w_k / Σ w` preserves the Faddeeva-derived
/// shape ratio `w_lo/w_hi` while restoring unity — exact at library
/// endpoints, peak-height-preserving between them.
///
/// Validated against U-238 MT=102 held-out 900 K reconstruction:
/// peak-height ratio at 6.67 eV improves from 1.022 (√T-linear) to
/// 1.009 (Ducru-unity); global L2 error drops from 2.4 % to 1.5 %.
/// Superseded as the default by 3-point `ducru_unity_weights` — kept
/// here for reference and possible regression A/B tests.
#[inline]
#[allow(dead_code)]
fn ducru_unity_2temp(target: f64, t_lo: f64, t_hi: f64) -> (f64, f64) {
    if (target - t_lo).abs() < 1e-6 {
        return (1.0, 0.0);
    }
    if (target - t_hi).abs() < 1e-6 {
        return (0.0, 1.0);
    }
    let w_lo = (t_lo * target).sqrt() / (t_lo + target) * (target - t_hi) / (target + t_hi)
        * (t_lo + t_hi)
        / (t_lo - t_hi);
    let w_hi = (t_hi * target).sqrt() / (t_hi + target) * (target - t_lo) / (target + t_lo)
        * (t_hi + t_lo)
        / (t_hi - t_lo);
    let s = w_lo + w_hi;
    if s.abs() < 1e-12 {
        // Degenerate case (near endpoint collision) — fall back to equal split.
        (0.5, 0.5)
    } else {
        (w_lo / s, w_hi / s)
    }
}

/// Interpolate total XS between two library endpoints using √T-linear.
/// Used to build the SVD-path `total_table` at an off-library temperature
/// so the "total − partials → capture" calibration stays consistent with
/// the SVD-reconstructed partial channels.
fn interp_total_at_temp(
    reader: &NuclideFileReader,
    target_temp: f64,
    i_lo: usize,
    i_hi: usize,
) -> Option<Vec<f64>> {
    let tot_lo = reader.compute_total_xs(i_lo)?;
    if i_lo == i_hi {
        return Some(tot_lo);
    }

    // 3-point unity Ducru: nearest three library columns to target_temp.
    // Captures a quadratic correction to the Faddeeva kernel that
    // 2-point cannot (measured on U-238 MT=102: halves peak-height
    // residual vs 2-point). Unchanged at on-library T (weights collapse
    // to one-hot via the exact-match shortcut in ducru_weights).
    let chosen = nearest_k_temps(&reader.temperatures, target_temp, 3);
    if chosen.is_empty() {
        return Some(tot_lo);
    }
    let sub_temps: Vec<f64> = chosen.iter().map(|&i| reader.temperatures[i]).collect();
    let weights = ducru_unity_weights(&sub_temps, target_temp);
    let totals: Vec<Vec<f64>> = chosen
        .iter()
        .map(|&i| reader.compute_total_xs(i).unwrap_or_else(|| tot_lo.clone()))
        .collect();
    let n = totals.iter().map(|v| v.len()).min().unwrap_or(0);
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut acc = 0.0_f64;
        for (k, tot) in totals.iter().enumerate() {
            acc += weights[k] * tot[i];
        }
        out[i] = acc.max(0.0);
    }
    Some(out)
}

/// Build an SVD reaction kernel with temperature coefficients computed at
/// an arbitrary `target_temp`.
///
/// For on-library temps, `temp_coeffs_ducru` returns a one-hot selection
/// (exact). For off-library temps, we use linear-in-√T interpolation of
/// the V^T columns between the two bracketing library indices — stable
/// and matches OpenMC's broadened-data interpolation convention. The
/// full-library Ducru Eq. 31 is numerically unstable with N > 3 temps
/// due to products of near-singular ratios.
fn build_kernel_at_temp(
    reader: &NuclideFileReader,
    mt: u32,
    svd_rank: usize,
    target_temp: f64,
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

    let coeffs = if n_t > 1 && !reader.temperatures.is_empty() {
        let (i_lo, i_hi) = bracket_temp_indices(&reader.temperatures, target_temp);
        if i_lo == i_hi {
            kernel.temp_coeffs(i_lo)
        } else {
            // Unity-normalized 3-point Ducru on V^T columns. Partition-
            // of-unity variant of Ducru (2017) Eq. 31 on the three
            // library temps nearest `target_temp`. Captures a quadratic
            // Faddeeva-kernel correction missed by 2-point: on U-238
            // MT=102 held-out at 900 K, rank-3 peak ratio improves from
            // 1.011 (2T) → 0.994 (3T); global L2 1.48 % → 1.05 %.
            // Caveat: raw Ducru weights can be negative in some brackets
            // (e.g. [294, 600, 1200] at 900 K gives the 294 K column a
            // -0.12 weight). Unity normalization keeps Σw = 1 but can
            // not enforce w ≥ 0 — QP-constrained reconstruction is the
            // next fix for physically monotonic interpolation.
            let chosen = nearest_k_temps(&reader.temperatures, target_temp, 3);
            let sub_temps: Vec<f64> = chosen.iter().map(|&i| reader.temperatures[i]).collect();
            let weights = ducru_unity_weights(&sub_temps, target_temp);
            let per_temp: Vec<Vec<f64>> = chosen.iter().map(|&i| kernel.temp_coeffs(i)).collect();
            let kdim = per_temp[0].len();
            let mut out = vec![0.0_f64; kdim];
            for (m, c) in per_temp.iter().enumerate() {
                for j in 0..kdim {
                    out[j] += weights[m] * c[j];
                }
            }
            out
        }
    } else {
        kernel.temp_coeffs(0)
    };
    Some(ReactionKernel { kernel, coeffs })
}

/// Load every prompt-photon product from every photon-emitting
/// reaction present in the nuclide file for use in coupled
/// neutron-photon tallies. Covers
///
///   * MT=102 — radiative capture `(n,γ)`,
///   * MT=18  — fission,
///   * MT=103 — `(n,p)` proton emission (threshold reaction,
///     important for O-16 in PWR fast spectrum),
///   * MT=107 — `(n,α)` alpha emission (likewise),
///   * MT=4   — lumped inelastic scatter, with fallback to
///     MT=51..91 discrete levels when MT=4 is absent.
///
/// For each MT we keep *all* tabulated photon products (a single MT
/// may list several when different excited states of the residual
/// nucleus emit different cascade lines — e.g. O-16 MT=107 has six
/// photon products). Absent reactions or products are silently
/// skipped; every nuclide ends up with whatever photon production
/// its HDF5 file actually tabulates.
fn load_photon_products(
    reader: &hdf5_reader::NuclideFileReader,
) -> Vec<(u32, hdf5_reader::PhotonProduct)> {
    let mut out = Vec::new();
    for mt in [102_u32, 18, 103, 107] {
        for pp in reader.photon_products(mt) {
            out.push((mt, pp));
        }
    }
    let mt4 = reader.photon_products(4);
    if !mt4.is_empty() {
        for pp in mt4 {
            out.push((4, pp));
        }
    } else {
        for mt in 51_u32..=91 {
            for pp in reader.photon_products(mt) {
                out.push((mt, pp));
            }
        }
    }
    out
}
