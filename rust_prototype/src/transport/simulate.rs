//! Eigenvalue simulation — power iteration for k_eff.
//!
//! Algorithm:
//!   1. Start with a source bank of neutrons
//!   2. Transport each neutron until absorption or leakage
//!   3. Collect fission sites into the fission bank
//!   4. k_eff = (fission bank size) / (source bank size)
//!   5. Normalize the fission bank -> new source bank
//!   6. Repeat

use std::io::Write;

use rayon::prelude::*;

use crate::geometry::cell::{Cell, CellFill};
use crate::geometry::surface::{BoundaryCondition, Surface};
use crate::geometry::{self, Vec3};
use crate::hdf5_reader::{AngularDistribution, DiscreteLevelInfo, EnergyDistribution};
use crate::physics::collision::{self, CollisionOutcome, InelasticData, MicroXs};
use crate::thermal::ThermalScatteringData;
use crate::transport::material::Material;
use crate::transport::particle::{FissionBank, FissionSite, Particle};
use crate::transport::rng::Rng;
use crate::transport::tally::{ParticleTallies, Tallies};

/// Maximum nuclides per material for stack-allocated XS buffers.
/// Godiva has 3, the basic PWR pin cell has 8, the actinides-chain
/// fuel material has 18 (U-235/238 + O-16 + Xe-135 + 14 chain
/// nuclides for actinide buildup + Sm/Pm/I/Cs poisoning). Avoids
/// per-collision heap allocation.
const MAX_NUCLIDES: usize = 32;

/// Configuration for a simulation.
pub struct SimConfig {
    pub batches: u32,
    /// Fixed number of inactive (settle) batches. Ignored when
    /// `auto_inactive == Some(_)`; used as a fallback otherwise.
    pub inactive: u32,
    pub particles_per_batch: u32,
    /// Global seed — different seeds produce independent runs for statistical benchmarking.
    pub seed: u64,
    /// Optional runtime source-convergence detector. When set, the
    /// simulator monitors Shannon entropy of the fission-site bank and
    /// promotes batches from inactive to active once the entropy
    /// plateaus. The fixed `inactive` count is replaced by this
    /// criterion, bounded by the policy's `min_inactive` / `max_inactive`.
    pub auto_inactive: Option<EntropyConvergence>,
    /// When `true` (default for CLI binaries), the engine prints
    /// per-batch `k_eff`, entropy, collision/fission/leak counters to
    /// stdout. When `false` (Python/FFI callers), the engine stays
    /// silent and the caller consumes the returned `BatchResult`s.
    /// Matters on Windows: locking stdout from a host process that
    /// also uses stdout (e.g. Python) can deadlock; setting this to
    /// `false` eliminates that risk.
    pub verbose: bool,
    /// When `true` (default), transport the per-batch particle bank
    /// in parallel via `rayon::par_iter`. When `false`, fall back to
    /// a sequential `iter().map().collect()`. The sequential path is
    /// slower but sidesteps rayon's first-use thread-pool creation,
    /// which on Windows can deadlock against Python's loader lock
    /// when called from a PyO3 extension.
    pub parallel: bool,
    /// Optional tallies (surface currents, mesh flux). When `None`,
    /// the transport loop's tally hooks are no-ops — zero hot-path
    /// cost. See `transport::tally` for shapes.
    pub tallies: Tallies,
    /// When set, write an HDF5 statepoint at the end of the run:
    /// per-batch arrays, tally arrays, and the post-normalize source
    /// bank ready for restart. See `transport::statepoint`.
    pub statepoint_path: Option<std::path::PathBuf>,
    /// Optional implicit-capture + Russian-roulette variance reduction.
    /// When `Some(_)`, the surface-tracking transport replaces analog
    /// absorption-as-kill with weight reduction `w *= σ_s/σ_t`, banks
    /// fission as stochastic-rounded `w·ν·σ_f/σ_t` sites, and rouletes
    /// once weight drops below `w_min`. Surface tracking + non-thermal
    /// collision branch only; thermal-scattering and delta-tracking
    /// paths fall back to analog regardless of this setting.
    pub survival_biasing: Option<SurvivalBiasing>,
    /// Optional initial source bank to resume from. When `None`, the
    /// engine rejection-samples a uniform initial source from the
    /// fissile cells (the default behavior). When `Some(bank)`, the
    /// bank is used as the source for batch 1 — typically loaded from
    /// a statepoint via `transport::statepoint::read_source_bank` to
    /// continue an earlier run.
    pub initial_source_bank: Option<Vec<FissionSite>>,
    /// Optional Cartesian-mesh weight window. When set, the transport
    /// loop applies splitting / Russian roulette at every advance to
    /// keep particle weight inside the per-voxel band. See
    /// `transport::weight_window` for shape and semantics.
    pub weight_window: Option<crate::transport::weight_window::WeightWindow>,
    /// When `true`, suppresses delayed-neutron emission entirely:
    /// every banked fission neutron draws its energy from the prompt
    /// fission spectrum, ignoring `ν_d(E)`. Used for ablation studies
    /// against the production path (which always samples ~0.65 % of
    /// fission neutrons from the soft-Watt delayed spectrum).
    pub disable_delayed_neutrons: bool,
    /// Optional URR equivalence-theory configuration. When set, the
    /// transport loop applies the Stoker-Weiss / NJOY rational
    /// equivalence correction to the per-nuclide URR sample for
    /// each `xs_kernel_idx` flagged on `UrrEquivalence::absorber_xs_idx`,
    /// using the cell-local Dancoff factor. When `None`, the URR
    /// path is infinite-medium-only (current default).
    pub urr_equivalence: Option<crate::transport::urr_equivalence::UrrEquivalence>,
}

/// Implicit-capture + Russian-roulette settings.
///
/// Common OpenMC-style defaults: `w_min = 0.25`, `w_survive = 1.0`.
/// Particles with weight below `w_min` are rouletted: with probability
/// `w/w_survive` they survive at weight `w_survive`; otherwise killed.
/// Net effect is unbiased — expected weight is preserved.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalBiasing {
    /// Weight threshold below which Russian roulette fires.
    pub w_min: f64,
    /// Weight that surviving rouletted particles are restored to.
    pub w_survive: f64,
}

impl Default for SurvivalBiasing {
    fn default() -> Self {
        Self {
            w_min: 0.25,
            w_survive: 1.0,
        }
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            batches: 50,
            inactive: 10,
            particles_per_batch: 5_000,
            seed: 1,
            auto_inactive: None,
            verbose: true,
            parallel: true,
            tallies: Tallies::default(),
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        }
    }
}

/// Policy for Shannon-entropy plateau detection. Defaults are tuned for
/// Godiva / PWR pin cell at $10^4$-$10^5$ particles / batch, where the
/// entropy equilibrates in $<50$ batches.
#[derive(Debug, Clone, Copy)]
pub struct EntropyConvergence {
    /// Never declare converged before this many batches have run.
    pub min_inactive: u32,
    /// Always start accumulating by this batch even if not converged
    /// (catches pathological long-transient sources).
    pub max_inactive: u32,
    /// Size of the sliding window over which the coefficient of
    /// variation (σ/μ) of entropy is computed.
    pub window: u32,
    /// Convergence threshold on the window's coefficient of variation.
    /// OpenMC uses values around 1e-3 for moderate meshes.
    pub cv_tol: f64,
}

impl Default for EntropyConvergence {
    fn default() -> Self {
        // cv_tol 5e-3 sits above the statistical noise floor for the
        // typical 10k-50k particles/batch used in paper benchmarks
        // (measured CV ≈ 2e-3 once settled) while still tight enough
        // that transient bias is well below the k_eff standard error.
        Self {
            min_inactive: 20,
            max_inactive: 200,
            window: 10,
            cv_tol: 5e-3,
        }
    }
}

impl EntropyConvergence {
    /// Return `true` if the last `window` entries of `history` have
    /// coefficient of variation below `cv_tol`. Requires at least
    /// `window` samples and `history.len() >= min_inactive`.
    pub fn has_converged(&self, history: &[f64]) -> bool {
        let n = history.len() as u32;
        if n < self.min_inactive || n < self.window {
            return false;
        }
        let start = history.len() - self.window as usize;
        let window = &history[start..];
        let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
        if mean.abs() < 1e-12 {
            return false;
        }
        let var: f64 = window.iter().map(|h| (h - mean).powi(2)).sum::<f64>() / window.len() as f64;
        let cv = var.sqrt() / mean.abs();
        cv < self.cv_tol
    }
}

// ── Delta tracking support ──────────────────────────────────────────

/// Coarse table of majorant macroscopic total XS, used for delta tracking.
/// Stores max(Σ_t) over all materials at log-spaced energy points.
pub struct MajorantTable {
    log_e_min: f64,
    inv_step: f64,
    values: Vec<f64>,
}

impl MajorantTable {
    /// Look up majorant Σ_t at a given energy (1/cm).
    #[inline]
    fn lookup(&self, energy: f64) -> f64 {
        let log_e = energy.max(1e-11).ln();
        let frac = (log_e - self.log_e_min) * self.inv_step;
        let idx = (frac as usize).min(self.values.len() - 2);
        let t = frac - idx as f64;
        self.values[idx] * (1.0 - t) + self.values[idx + 1] * t
    }
}

/// Transport algorithm selection — detected automatically from geometry.
enum TrackingMode {
    /// Standard surface tracking — for single-material or reflective-only geometries.
    Surface,
    /// Woodcock delta tracking — for heterogeneous geometries with transmission boundaries.
    /// Avoids surface intersection at internal material interfaces.
    Delta(MajorantTable),
}

/// Detect which tracking algorithm to use based on geometry and materials.
///
/// Returns `Delta` if multiple materials exist with moderate XS contrast,
/// `Surface` otherwise. Prints the decision to stdout.
fn detect_tracking_mode<XS: XsProvider>(
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
    verbose: bool,
) -> TrackingMode {
    // Count unique material indices
    let mut mat_indices: Vec<usize> = cells
        .iter()
        .filter_map(|c| match c.fill {
            CellFill::Material(m) => Some(m as usize),
            _ => None,
        })
        .collect();
    mat_indices.sort_unstable();
    mat_indices.dedup();

    if mat_indices.len() <= 1 {
        if verbose {
            println!("  Tracking: SURFACE (single material — delta tracking not beneficial)");
        }
        return TrackingMode::Surface;
    }

    // Multiple materials — compute majorant and contrast ratio
    let n_pts = 10_000;
    let e_min = 1e-5_f64;
    let e_max = 20e6_f64;
    let log_min = e_min.ln();
    let log_max = e_max.ln();
    let step = (log_max - log_min) / (n_pts - 1) as f64;

    let mut values = Vec::with_capacity(n_pts);
    let mut max_contrast = 0.0_f64;

    for i in 0..n_pts {
        let energy = (log_min + i as f64 * step).exp();

        let mut mat_totals = Vec::with_capacity(mat_indices.len());
        for &mi in &mat_indices {
            if mi >= materials.len() {
                continue;
            }
            let mat = &materials[mi];
            let mut macro_t = 0.0;
            for nuc in &mat.nuclides {
                let xs = xs_provider.lookup(nuc.xs_kernel_idx, energy);
                macro_t += nuc.atom_density * xs.total;
            }
            mat_totals.push(macro_t);
        }

        let sigma_max = mat_totals.iter().copied().fold(0.0_f64, f64::max);
        let sigma_min = mat_totals
            .iter()
            .copied()
            .filter(|&s| s > 1e-10)
            .fold(f64::INFINITY, f64::min);

        if sigma_min > 0.0 && sigma_min < f64::INFINITY {
            max_contrast = max_contrast.max(sigma_max / sigma_min);
        }

        // Add 5% safety margin to majorant
        values.push(sigma_max * 1.05);
    }

    // High contrast (>20x) means >95% virtual collisions — delta tracking is worse
    if max_contrast > 20.0 {
        if verbose {
            println!(
                "  Tracking: SURFACE (heterogeneous but contrast={:.1}x too high for delta tracking)",
                max_contrast
            );
        }
        return TrackingMode::Surface;
    }

    let avg_rejection = if max_contrast > 1.0 {
        1.0 - 1.0 / max_contrast
    } else {
        0.0
    };
    if verbose {
        println!(
            "  Tracking: DELTA (heterogeneous — {} materials, contrast={:.1}x, ~{:.0}% virtual collisions)",
            mat_indices.len(),
            max_contrast,
            avg_rejection * 100.0
        );
    }

    TrackingMode::Delta(MajorantTable {
        log_e_min: log_min,
        inv_step: 1.0 / step,
        values,
    })
}

/// Cross-section provider trait — abstracts over SVD kernel vs table lookup.
///
/// The transport loop doesn't care how cross-sections are obtained.
/// Must be Send + Sync for rayon parallel transport.
pub trait XsProvider: Send + Sync {
    /// Get microscopic cross-sections for a nuclide at a given energy.
    fn lookup(&self, nuclide_idx: usize, energy: f64) -> MicroXs;

    fn discrete_level_info(&self, _nuclide_idx: usize) -> Vec<DiscreteLevelInfo> {
        vec![]
    }

    fn discrete_level_xs(&self, _nuclide_idx: usize, _energy: f64) -> Vec<f64> {
        vec![]
    }

    fn has_continuum_inelastic(&self, _nuclide_idx: usize) -> bool {
        false
    }

    fn elastic_angular_dist(&self, _nuclide_idx: usize) -> Option<&AngularDistribution> {
        None
    }

    /// Per-discrete-level CM-frame angular distributions, aligned 1:1 with
    /// `discrete_level_info(nuclide_idx)`. Default empty slice = isotropic
    /// fallback everywhere. Providers that load ENDF MT=51-91 angular data
    /// from HDF5 override this.
    fn discrete_level_angles(&self, _nuclide_idx: usize) -> &[Option<AngularDistribution>] {
        &[]
    }

    fn fission_energy_dist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    /// ENDF MT=91 continuum inelastic outgoing-energy distribution.
    /// Default `None` → caller falls back to the evaporation spectrum
    /// (historical behaviour). Providers that load from HDF5 override.
    fn inelastic_continuum_edist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    /// ENDF MT=16 (n,2n) outgoing-energy distribution.
    fn n2n_edist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    /// ENDF MT=17 (n,3n) outgoing-energy distribution.
    fn n3n_edist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    fn apply_urr(&self, _nuclide_idx: usize, _xs: &mut MicroXs, _energy: f64, _xi: f64) {}

    /// True when `energy` falls inside the URR (unresolved-resonance
    /// region) probability-table range for `nuclide_idx`. Used by the
    /// equivalence-theory path to gate the spatial self-shielding
    /// correction — it's only valid inside the URR window. Default
    /// `false` for providers without URR data.
    fn is_urr(&self, _nuclide_idx: usize, _energy: f64) -> bool {
        false
    }

    /// Photon products for a nuclide, one entry per ENDF MT with a
    /// `particle="photon"` product in the HDF5 file. Used by coupled
    /// neutron-photon transport to sample `(multiplicity, E_γ)` at
    /// each capture / fission / inelastic site. Default empty slice =
    /// no photon production modelled.
    fn photon_products(&self, _nuclide_idx: usize) -> &[(u32, crate::hdf5_reader::PhotonProduct)] {
        &[]
    }

    /// Get thermal scattering data for a nuclide, if available.
    ///
    /// Returns `Some` if the nuclide has associated S(α,β) thermal scattering data
    /// (e.g., H1 in H₂O). The transport loop uses this to replace free-gas elastic
    /// scattering with thermal scattering below `energy_max` (~4 eV).
    fn thermal_scattering(&self, _nuclide_idx: usize) -> Option<&ThermalScatteringData> {
        None
    }

    /// Energy-dependent delayed-only ν̄ for a nuclide (sum of all
    /// delayed-product yields). Returns 0 when the nuclide has no
    /// delayed neutron data, or for non-fissile nuclides. Used by
    /// the fission-yield path to compute β(E) = ν_d / ν_total and
    /// pick prompt vs delayed for each banked fission neutron.
    fn delayed_nu_bar_at(&self, _nuclide_idx: usize, _energy: f64) -> f64 {
        0.0
    }
}

/// Simple constant cross-section provider for testing.
pub struct ConstantXs {
    pub xs: Vec<MicroXs>,
}

impl XsProvider for ConstantXs {
    fn lookup(&self, nuclide_idx: usize, _energy: f64) -> MicroXs {
        self.xs[nuclide_idx]
    }
}

/// Results from a batch.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub batch: u32,
    pub k_eff: f64,
    pub leakage: u32,
    pub absorptions: u32,
    pub fissions: u32,
    pub collisions: u32,
    /// Number of thermal scattering events (S(α,β)).
    pub thermal_scatters: u32,
    /// Number of surface crossings (reflections + transmissions).
    pub surface_crossings: u32,
    /// Shannon entropy (bits) of the fission site distribution on a
    /// coarse Cartesian mesh. Stabilises across inactive batches once
    /// the source has converged; used to gauge when active batches can
    /// start (OpenMC uses the same diagnostic).
    pub shannon_entropy: f64,
    /// True when this batch was counted towards the active tally.
    /// In fixed-inactive mode this is simply `batch > config.inactive`.
    /// In auto-inactive mode it reflects the entropy-plateau decision.
    pub active: bool,
    /// Per-cell count of non-fission absorption events (radiative
    /// capture and other `(n,X)` absorptions). Indexed by the cell's
    /// position in the `cells` slice passed to `run_eigenvalue`. Use
    /// this to build a photon source for coupled neutron-photon
    /// transport, or any other (n,γ)-rate-weighted tally.
    pub captures_by_cell: Vec<f64>,
    /// Photon emission events tallied at capture / fission / inelastic
    /// sites in this batch. Each event carries `(cell, pos, E_γ, MT)`
    /// — the photon driver consumes this directly as a fixed source.
    /// Empty when the XS provider has no photon-product data loaded.
    pub photon_events: Vec<PhotonSourceEvent>,
    /// Track-length k-eff estimator for this batch:
    ///   k_track = (1/N) Σ_segments  w · d · Σ_νf(E)
    /// summed across every flight segment of every particle, divided
    /// by the source size N. Equivalent in expectation to the
    /// fission-bank (collision) estimator `k_eff` but lower variance
    /// because every step contributes, not just collisions.
    /// Surface-tracking only — under delta tracking the path crosses
    /// material boundaries silently, so this field stays 0.
    pub k_track: f64,
    /// Per-bin J+ (forward) surface current — sum of `particle.weight`
    /// over crossings with `dir · normal ≥ 0`. Length matches the
    /// `SimConfig.tallies.surface_current` bin count, or empty when
    /// the tally is disabled. Net current = `pos - neg`; total = `pos + neg`.
    pub surface_current_pos: Vec<f64>,
    /// Per-bin J- (backward) surface current — `dir · normal < 0`.
    pub surface_current_neg: Vec<f64>,
    /// Per-voxel track-length flux: Σ_segments w · d (cm·source⁻¹).
    /// Length = `SimConfig.tallies.mesh_flux.n_voxels()`, or empty
    /// when the mesh tally is disabled.
    pub mesh_flux: Vec<f64>,
    /// Per-cell flux numerator for the reaction-rate tally:
    /// `Σ_segments w · d`. Length = `n_cells`, or empty when the
    /// reaction-rate tally is disabled.
    pub rr_flux: Vec<f64>,
    /// Per-(cell, xs_idx, mt_slot) rate numerator:
    /// `Σ_segments w · d · σ_micro,MT(E_local)`. Length =
    /// `n_cells × n_xs_idx × n_mts`, flat index
    /// `(c·n_xs_idx + n)·n_mts + m`. Empty when disabled.
    pub rr_rate: Vec<f64>,
}

/// Coarse Cartesian mesh used to bin fission sites for the Shannon
/// entropy source-convergence monitor. Grid bounds come from the AABB
/// of fissile cells; a fixed `N×N×N` resolution balances resolution
/// against per-bin statistics.
pub struct EntropyMesh {
    pub n: usize, // per axis
    pub lo: [f64; 3],
    pub hi: [f64; 3],
}

impl EntropyMesh {
    /// Build a mesh from a bounding box, clamped to `n = 8`--$16$ per
    /// axis depending on box size. For a PWR pin cell the fuel radius
    /// is ~0.41 cm and the pitch is ~1.26 cm, so `n = 8` gives ~1.6 mm
    /// bins, coarse enough for well-populated counts at $50\,000$
    /// particles / batch.
    pub fn from_aabb(aabb: &crate::geometry::Aabb, n: usize) -> Self {
        Self {
            n,
            lo: [aabb.min.x, aabb.min.y, aabb.min.z],
            hi: [aabb.max.x, aabb.max.y, aabb.max.z],
        }
    }

    /// Compute Shannon entropy (bits) of the fission-site spatial
    /// distribution on this mesh. Returns `0.0` if the bank is empty.
    ///
    /// H = -Σ p_i log_2 p_i, where p_i is the fraction of sites in
    /// bin i. Upper bound is log_2(n^3).
    pub fn entropy(&self, sites: &[FissionSite]) -> f64 {
        let total = sites.len();
        if total == 0 {
            return 0.0;
        }
        let n3 = self.n * self.n * self.n;
        let mut counts = vec![0u32; n3];
        let dx = (self.hi[0] - self.lo[0]) / self.n as f64;
        let dy = (self.hi[1] - self.lo[1]) / self.n as f64;
        let dz = (self.hi[2] - self.lo[2]) / self.n as f64;
        let inv_dx = if dx > 0.0 { 1.0 / dx } else { 0.0 };
        let inv_dy = if dy > 0.0 { 1.0 / dy } else { 0.0 };
        let inv_dz = if dz > 0.0 { 1.0 / dz } else { 0.0 };

        for s in sites {
            let ix =
                (((s.pos.x - self.lo[0]) * inv_dx) as isize).clamp(0, self.n as isize - 1) as usize;
            let iy =
                (((s.pos.y - self.lo[1]) * inv_dy) as isize).clamp(0, self.n as isize - 1) as usize;
            let iz =
                (((s.pos.z - self.lo[2]) * inv_dz) as isize).clamp(0, self.n as isize - 1) as usize;
            counts[ix * self.n * self.n + iy * self.n + iz] += 1;
        }

        let inv_n = 1.0 / total as f64;
        let mut h = 0.0_f64;
        for &c in &counts {
            if c > 0 {
                let p = c as f64 * inv_n;
                h -= p * p.log2();
            }
        }
        h
    }
}

/// A single photon emission event tallied during neutron transport,
/// suitable as a fixed source for a downstream photon driver.
///
/// Recorded at every collision that has a non-zero photon yield:
/// radiative capture (MT=102), fission (MT=18), and inelastic
/// scattering (MT=4 or MT=51..91). The event captures where the
/// reaction happened and what outgoing photon energy was sampled;
/// the photon phase then sources one history per event.
#[derive(Debug, Clone, Copy)]
pub struct PhotonSourceEvent {
    /// Cell index containing the reaction site.
    pub cell_idx: u32,
    /// World-frame position of the reaction (cm).
    pub pos: [f64; 3],
    /// Emitted photon energy (eV).
    pub energy: f64,
    /// Reaction-class tag for diagnostic binning: 102 = capture,
    /// 18 = fission, 4/51..91 = inelastic. Not used by the photon
    /// driver but useful for "fraction of γ-heat from fission vs
    /// capture" breakdowns.
    pub mt: u32,
}

/// Per-particle transport result for parallel reduction.
struct ParticleResult {
    fission_sites: Vec<FissionSite>,
    leakage: u32,
    absorptions: u32,
    thermal_scatters: u32,
    surface_crossings: u32,
    fissions: u32,
    collisions: u32,
    /// Cell indices where this particle was captured (n,γ-style
    /// non-fission absorption). Typical history captures at most
    /// once, so this vec is usually empty or a single element —
    /// allocation cost is amortised by the parallel reduction and the
    /// fact that capture events are rare relative to scatter events.
    capture_cells: Vec<usize>,
    /// Photon emission events sampled at each reaction site (capture,
    /// fission, inelastic). Reduced into `BatchResult::photon_events`.
    photon_events: Vec<PhotonSourceEvent>,
    /// Track-length tally accumulator: Σ_segments w · d · Σ_νf(E),
    /// summed over every advance the particle made through fuel-
    /// bearing material. Reduced into `BatchResult::k_track` as
    /// `total / N_source` after the batch completes.
    track_length_nu_sigf: f64,
    /// Per-particle tally accumulators (surface currents, mesh flux).
    /// Sized once at particle birth from the active `Tallies` config;
    /// empty Vec when the corresponding tally is disabled.
    tallies: ParticleTallies,
}

/// Sample photon products for a given reaction (MT) at the current
/// collision site and append the resulting emission events to the
/// per-particle photon tally. No-op when the XS provider has no
/// photon-product data for this nuclide/MT — which is the default
/// unless the user built the provider via `load_nuclide_table` or
/// `load_nuclide_kernels` on a file carrying `particle="photon"`
/// products.
fn sample_photon_products<XS: XsProvider>(
    xs_provider: &XS,
    xs_kernel_idx: usize,
    mts: &[u32],
    particle: &Particle,
    rng: &mut Rng,
    out: &mut Vec<PhotonSourceEvent>,
) {
    for (mt_pp, pp) in xs_provider.photon_products(xs_kernel_idx) {
        if !mts.contains(mt_pp) {
            continue;
        }
        let energies = pp.sample(particle.energy, rng);
        for e_gamma in energies {
            out.push(PhotonSourceEvent {
                cell_idx: particle.cell_idx as u32,
                pos: [particle.pos.x, particle.pos.y, particle.pos.z],
                energy: e_gamma,
                mt: *mt_pp,
            });
        }
    }
}

/// Non-fission absorption MTs whose photon products the transport
/// loop samples at every `CollisionOutcome::Absorption` event. The
/// yield tables for threshold reactions like MT=103 (n,p) and MT=107
/// (n,α) return 0 below threshold — no spurious events are emitted
/// for nuclides that can't reach those channels at the collision
/// energy.
const ABSORPTION_PHOTON_MTS: &[u32] = &[102, 103, 107];
/// Fission photon-production MT (prompt γs from MT=18).
const FISSION_PHOTON_MTS: &[u32] = &[18];

/// Apply URR equivalence-theory spatial self-shielding correction to
/// the per-nuclide MicroXs entries. For each nuclide flagged as an
/// absorber AND in its URR window at this energy:
///
///   σ_eff = σ_∞ · σ_0 / (σ_0 + σ_e)
///   σ_e = (1 − C) / (N_abs · l̄)
///   σ_0 = Σ_{j ≠ abs} N_j · σ_t,j / N_abs
///
/// Updates the absorber's `elastic`, `fission`, `capture`, and
/// `total` in-place; recomputes the entry's contribution to
/// `micro_totals`. Inelastic / (n,2n) / (n,3n) channels are not
/// corrected — equivalence only modulates the resonance-window
/// contribution, which is concentrated in elastic + capture +
/// fission for the U-238-class absorbers we care about.
#[allow(clippy::too_many_arguments)]
fn apply_urr_equivalence_correction<XS: XsProvider>(
    eq: &crate::transport::urr_equivalence::UrrEquivalence,
    material: &Material,
    xs_provider: &XS,
    energy: f64,
    dancoff: f64,
    mean_chord_cm: f64,
    micro_xs: &mut [MicroXs; MAX_NUCLIDES],
    micro_xs_smooth: &[MicroXs; MAX_NUCLIDES],
    micro_totals: &mut [f64; MAX_NUCLIDES],
    n_nuclides: usize,
) {
    use crate::transport::urr_equivalence::apply_equivalence_correction;
    for i in 0..n_nuclides {
        let nuc = &material.nuclides[i];
        if !eq.is_absorber(nuc.xs_kernel_idx) {
            continue;
        }
        if nuc.atom_density <= 0.0 {
            continue;
        }
        if !xs_provider.is_urr(nuc.xs_kernel_idx, energy) {
            continue;
        }
        // σ_0 = Σ_{j ≠ i} N_j · σ_t,j / N_i.
        let mut sigma_0 = 0.0_f64;
        for j in 0..n_nuclides {
            if j == i {
                continue;
            }
            sigma_0 += material.nuclides[j].atom_density * micro_xs[j].total;
        }
        sigma_0 /= nuc.atom_density;

        // Hwang superposition: shield only the resonance-fluctuation
        // part of each channel, leaving the smooth off-resonance
        // baseline (potential elastic, smooth s-wave capture)
        // unshielded. `apply_equivalence_correction` returns
        // `σ_smooth + (σ_URR − σ_smooth) · σ_0/(σ_0+σ_e)`.
        let smooth = &micro_xs_smooth[i];
        micro_xs[i].elastic = apply_equivalence_correction(
            micro_xs[i].elastic,
            smooth.elastic,
            sigma_0,
            nuc.atom_density,
            mean_chord_cm,
            dancoff,
        );
        micro_xs[i].fission = apply_equivalence_correction(
            micro_xs[i].fission,
            smooth.fission,
            sigma_0,
            nuc.atom_density,
            mean_chord_cm,
            dancoff,
        );
        micro_xs[i].capture = apply_equivalence_correction(
            micro_xs[i].capture,
            smooth.capture,
            sigma_0,
            nuc.atom_density,
            mean_chord_cm,
            dancoff,
        );
        micro_xs[i].total = micro_xs[i].elastic
            + micro_xs[i].inelastic
            + micro_xs[i].n2n
            + micro_xs[i].n3n
            + micro_xs[i].fission
            + micro_xs[i].capture;
        micro_totals[i] = micro_xs[i].total;
    }
}

/// Transport a single particle to completion.
#[allow(clippy::too_many_arguments)]
fn transport_particle<XS: XsProvider>(
    site: &FissionSite,
    batch: u64,
    particle_idx: u64,
    geometry: &crate::geometry::Geometry,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
    tallies: &Tallies,
    survival_biasing: Option<&SurvivalBiasing>,
    weight_window: Option<&crate::transport::weight_window::WeightWindow>,
    disable_delayed_neutrons: bool,
    urr_equivalence: Option<&crate::transport::urr_equivalence::UrrEquivalence>,
) -> ParticleResult {
    let mut rng = Rng::for_particle(batch, particle_idx);
    let mut result = ParticleResult {
        fission_sites: Vec::new(),
        leakage: 0,
        absorptions: 0,
        fissions: 0,
        collisions: 0,
        thermal_scatters: 0,
        surface_crossings: 0,
        capture_cells: Vec::new(),
        photon_events: Vec::new(),
        track_length_nu_sigf: 0.0,
        tallies: ParticleTallies::new(tallies),
    };

    let (u, v, w) = rng.isotropic_direction();
    let dir = Vec3::new(u, v, w);

    let mut particle = match geometry::ray::find_cell_recursive(site.pos, geometry) {
        Some(stack) => Particle::with_stack(site.pos, dir, site.energy, stack),
        // Match legacy behavior: if a fission site lands exactly on a
        // surface (float precision boundary case), default to cell 0
        // and let the transport loop sort it out. The OLD `find_cell`
        // call site used `.unwrap_or(0)` here.
        None => Particle::new(site.pos, dir, site.energy, 0),
    };

    // Current-generation secondary neutrons emitted by (n,2n)/(n,3n).
    // Drained in an outer loop: after the primary finishes transport,
    // pop the next pending secondary and transport it before moving on.
    let mut pending: Vec<Particle> = Vec::new();

    // OpenMC uses max_particle_events = 1,000,000 (any step: collision, surface, reflection).
    // For thermal systems, neutrons may undergo thousands of scattering events to thermalize.
    // Budget is shared across the primary and all multiplicity secondaries
    // so that pathological (n,xn) cascades cannot exceed the per-source bound.
    let max_events = 1_000_000_u32;
    let mut total_events = 0_u32;

    'history: loop {
        let mut void_crossings = 0_u32;
        while particle.is_alive() && total_events < max_events {
            total_events += 1;
            let cell = &cells[particle.cell_idx];

            // Effective fill: lattice override (if any) wins over the
            // cell's static `cell.fill`. Lets the same pin universe
            // be reused at many lattice positions with different
            // materials (different enrichments, burnup tiers, etc.).
            let effective = geometry.effective_material_idx(&particle.coord_stack);
            let mat_idx = match effective {
                crate::geometry::EffectiveFill::Material(m) => m as usize,
                crate::geometry::EffectiveFill::Void => {
                    // Void region — free-stream to next surface (no interactions).
                    // Safety limit prevents infinite loops in degenerate geometries.
                    void_crossings += 1;
                    if void_crossings > 100 {
                        particle.kill();
                        result.leakage += 1;
                        break;
                    }
                    let trace = geometry::ray::trace_step_recursive(
                        &particle.coord_stack,
                        particle.pos,
                        particle.dir,
                        geometry,
                    );
                    match trace {
                        Some(hit) => {
                            let nudge = (hit.distance * 1e-8).max(1e-8);
                            match hit.bc {
                                BoundaryCondition::Vacuum => {
                                    particle.advance(hit.distance);
                                    particle.kill();
                                    result.leakage += 1;
                                    break;
                                }
                                BoundaryCondition::Reflective => {
                                    let surf_idx = hit.surface_idx.unwrap_or(0);
                                    particle.advance(hit.distance);
                                    let n = surfaces[surf_idx].normal_at(particle.pos);
                                    let d = particle.dir;
                                    particle.dir = d - n * (2.0 * d.dot(n));
                                }
                                BoundaryCondition::Transmission => {
                                    particle.advance(hit.distance + nudge);
                                    match hit.next_stack {
                                        Some(stack) => {
                                            particle.cell_idx = stack
                                                .last()
                                                .map(|c| c.cell_idx as usize)
                                                .unwrap_or(0);
                                            particle.coord_stack = stack;
                                        }
                                        None => {
                                            particle.kill();
                                            result.leakage += 1;
                                            break;
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        None => {
                            particle.kill();
                            result.leakage += 1;
                            break;
                        }
                    }
                }
            };

            if mat_idx >= materials.len() {
                particle.kill();
                result.leakage += 1;
                break;
            }

            void_crossings = 0;
            let material = &materials[mat_idx];

            // Look up microscopic cross-sections with URR sampling.
            // Stack-allocated buffers — no heap allocation per collision.
            let urr_xi = rng.uniform();
            let n_nuclides = material.nuclides.len();
            let mut micro_xs = [MicroXs::default(); MAX_NUCLIDES];
            let mut micro_xs_smooth = [MicroXs::default(); MAX_NUCLIDES];
            let mut micro_totals = [0.0_f64; MAX_NUCLIDES];
            // Track thermal scattering XS addition per nuclide
            let mut thermal_xs_add = [0.0_f64; MAX_NUCLIDES];
            for (i, nuc) in material.nuclides.iter().enumerate() {
                let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, particle.energy);
                // Snapshot the smooth (pre-URR-PT) XS — Hwang
                // superposition needs this baseline so the URR
                // equivalence correction shields only the
                // resonance-fluctuation part of each channel.
                micro_xs_smooth[i] = xs;
                xs_provider.apply_urr(nuc.xs_kernel_idx, &mut xs, particle.energy, urr_xi);

                // S(α,β) thermal scattering: replace free-atom elastic XS
                // with thermal scattering XS below energy_max
                if let Some(tsl) = xs_provider.thermal_scattering(nuc.xs_kernel_idx)
                    && particle.energy < tsl.energy_max
                    && particle.energy > 0.0
                {
                    let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                    let thermal_total = tsl.total_xs(particle.energy, t_idx).max(0.0);
                    if thermal_total > 0.0 {
                        let delta = thermal_total - xs.elastic;
                        xs.total += delta;
                        thermal_xs_add[i] = thermal_total;
                        xs.elastic = 0.0;
                    }
                }

                if disable_delayed_neutrons {
                    xs.delayed_nu_bar = 0.0;
                }
                micro_totals[i] = xs.total;
                micro_xs[i] = xs;
            }

            // URR equivalence theory pass — applies the Stoker-Weiss
            // / NJOY rational self-shielding correction to URR samples
            // for nuclides flagged as resonance absorbers. Cell-local
            // Dancoff factor + mean chord drives the correction; the
            // hot path is gated on `is_urr` to avoid spurious damping
            // outside the URR window.
            if let Some(eq) = urr_equivalence {
                let dancoff = eq.dancoff.get(particle.cell_idx);
                let mean_chord = eq
                    .mean_chord_cm
                    .get(particle.cell_idx)
                    .copied()
                    .unwrap_or(0.0);
                if dancoff < 1.0 && mean_chord > 0.0 {
                    apply_urr_equivalence_correction(
                        eq,
                        material,
                        xs_provider,
                        particle.energy,
                        dancoff,
                        mean_chord,
                        &mut micro_xs,
                        &micro_xs_smooth,
                        &mut micro_totals,
                        n_nuclides,
                    );
                }
            }
            let macro_total = material.macro_total(&micro_totals[..n_nuclides]);

            if macro_total <= 0.0 {
                particle.kill();
                result.leakage += 1;
                break;
            }

            // Macroscopic ν·Σ_f for the track-length k-eff estimator.
            // Zero outside fuel; computed from the same MicroXs we just
            // looked up, so no extra XS evaluation.
            let mut macro_nu_sigma_f = 0.0_f64;
            for (i, nuc) in material.nuclides.iter().enumerate() {
                macro_nu_sigma_f += nuc.atom_density * micro_xs[i].nu_bar * micro_xs[i].fission;
            }

            let dist_collision = rng.exponential(macro_total);

            let trace = geometry::ray::trace_step_recursive(
                &particle.coord_stack,
                particle.pos,
                particle.dir,
                geometry,
            );

            // Tally track-length over the actual flight before any cell
            // change. The segment is straight-line through one cell, so
            // ν·Σ_f is constant along it.
            let advance_dist = match &trace {
                Some(hit) if hit.distance < dist_collision => hit.distance,
                _ => dist_collision,
            };
            if macro_nu_sigma_f > 0.0 {
                result.track_length_nu_sigf += particle.weight * advance_dist * macro_nu_sigma_f;
            }
            // Mesh flux tally: deposit w · d into every voxel the
            // segment intersects. Skipped when the tally is disabled.
            if let Some(mesh) = tallies.mesh_flux.as_ref() {
                mesh.deposit(
                    particle.pos,
                    particle.dir,
                    advance_dist,
                    particle.weight,
                    &mut result.tallies.mesh_flux,
                );
            }
            // Reaction-rate tally for chain-XS spectrum collapse:
            //   numerator   per (cell, xs_idx, MT): Σ w·d·σ_micro,MT
            //   denominator per cell:                Σ w·d
            // Track-length form is exact for the one-group XS
            // collapse; collision-estimator form would have higher
            // variance for non-rare reactions like (n,γ).
            if let Some(rr) = tallies.reaction_rate.as_ref() {
                let cell_idx = particle.cell_idx;
                if cell_idx < rr.n_cells {
                    let w_d = particle.weight * advance_dist;
                    result.tallies.rr_flux[cell_idx] += w_d;
                    let n_mts = rr.n_mts;
                    for (i, nuc) in material.nuclides.iter().take(n_nuclides).enumerate() {
                        let xs_idx = nuc.xs_kernel_idx;
                        if xs_idx >= rr.n_xs_idx {
                            continue;
                        }
                        let base = (cell_idx * rr.n_xs_idx + xs_idx) * n_mts;
                        for (m, &mt) in rr.mts.iter().enumerate() {
                            let sigma = match mt {
                                18 => micro_xs[i].fission,
                                102 => micro_xs[i].capture,
                                16 => micro_xs[i].n2n,
                                17 => micro_xs[i].n3n,
                                2 => micro_xs[i].elastic,
                                4 => micro_xs[i].inelastic,
                                _ => 0.0,
                            };
                            result.tallies.rr_rate[base + m] += w_d * sigma;
                        }
                    }
                }
            }

            match trace {
                Some(hit) if hit.distance < dist_collision => {
                    result.surface_crossings += 1;
                    // Surface current tally: split forward / backward
                    // crossings by sign(particle.dir · surface_normal).
                    if let (Some(sct), Some(surf_idx)) =
                        (tallies.surface_current.as_ref(), hit.surface_idx)
                        && let Some(bin) = sct.bin_for(surf_idx)
                    {
                        let crossing_pos = particle.pos + particle.dir * hit.distance;
                        let n = surfaces[surf_idx].normal_at(crossing_pos);
                        if particle.dir.dot(n) >= 0.0 {
                            result.tallies.surface_current_pos[bin] += particle.weight;
                        } else {
                            result.tallies.surface_current_neg[bin] += particle.weight;
                        }
                    }
                    match hit.bc {
                        BoundaryCondition::Vacuum => {
                            particle.advance(hit.distance);
                            particle.kill();
                            result.leakage += 1;
                        }
                        BoundaryCondition::Reflective => {
                            let surf_idx = hit.surface_idx.unwrap_or(0);
                            particle.advance(hit.distance);
                            let n = surfaces[surf_idx].normal_at(particle.pos);
                            let d = particle.dir;
                            particle.dir = d - n * (2.0 * d.dot(n));
                        }
                        BoundaryCondition::Transmission => {
                            particle.advance(hit.distance + (hit.distance * 1e-8).max(1e-8));
                            match hit.next_stack {
                                Some(stack) => {
                                    particle.cell_idx =
                                        stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                                    particle.coord_stack = stack;
                                }
                                None => {
                                    particle.kill();
                                    result.leakage += 1;
                                }
                            }
                        }
                    }
                }
                _ => {
                    particle.advance(dist_collision);
                    result.collisions += 1;

                    let nuc_idx = material.sample_nuclide(
                        &micro_totals[..n_nuclides],
                        macro_total,
                        rng.uniform(),
                    );

                    let xs_kernel_idx = material.nuclides[nuc_idx].xs_kernel_idx;

                    // Thermal scattering branch: roll once for thermal
                    // vs non-thermal reaction on the S(α,β) nuclide.
                    // Pre-fix: the non-thermal branch went through a
                    // separate `process_non_thermal_collision` →
                    // `process_collision` path that didn't honour
                    // survival biasing. Now both branches that don't
                    // resolve to a thermal scatter route through
                    // `dispatch_real_collision`, the single SB-aware
                    // entry point. Net result: implicit-capture +
                    // Russian roulette extends to PWR's H-1 capture
                    // events (small contribution but the variance
                    // reduction now applies uniformly).
                    let use_thermal = thermal_xs_add[nuc_idx] > 0.0;
                    let mut handled_as_thermal_scatter = false;
                    if use_thermal {
                        let tsl = xs_provider
                            .thermal_scattering(xs_kernel_idx)
                            .expect("thermal data");
                        let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                        let xi_reaction = rng.uniform() * micro_xs[nuc_idx].total;

                        if xi_reaction < thermal_xs_add[nuc_idx] {
                            // Thermal scattering event: sample E_out / μ
                            // directly from S(α,β); no fission/capture
                            // bookkeeping, so survival biasing doesn't
                            // apply (nothing to bias — analog scatter).
                            result.thermal_scatters += 1;
                            let (e_out, mu) = tsl.sample(particle.energy, t_idx, &mut rng);
                            particle.energy = e_out;
                            let phi = 2.0 * std::f64::consts::PI * rng.uniform();
                            let sin_mu = (1.0 - mu * mu).max(0.0).sqrt();
                            let d = particle.dir;
                            let w2 = d.z * d.z;
                            if w2 < 0.999 {
                                let inv_sq = 1.0 / (1.0 - w2).sqrt();
                                particle.dir = Vec3::new(
                                    mu * d.x
                                        + sin_mu
                                            * (d.x * d.z * phi.cos() - d.y * phi.sin())
                                            * inv_sq,
                                    mu * d.y
                                        + sin_mu
                                            * (d.y * d.z * phi.cos() + d.x * phi.sin())
                                            * inv_sq,
                                    mu * d.z - sin_mu * (1.0 - w2).sqrt() * phi.cos(),
                                );
                            } else {
                                let sign = if d.z > 0.0 { 1.0 } else { -1.0 };
                                particle.dir = Vec3::new(
                                    sin_mu * phi.cos(),
                                    sin_mu * phi.sin() * sign,
                                    mu * sign,
                                );
                            }
                            handled_as_thermal_scatter = true;
                        }
                        // else: fall through — non-thermal reaction on
                        // the thermal nuclide (capture / fission /
                        // inelastic). The MicroXs already has elastic
                        // zeroed (line ~910), so dispatch_real_collision
                        // sees the right channel weights.
                    }

                    if !handled_as_thermal_scatter {
                        // Single SB-aware dispatch — applies to both
                        // the !use_thermal nuclides and the
                        // use_thermal-but-non-thermal sub-branch above.
                        let level_info = xs_provider.discrete_level_info(xs_kernel_idx);
                        let level_xs =
                            xs_provider.discrete_level_xs(xs_kernel_idx, particle.energy);
                        let has_cont = xs_provider.has_continuum_inelastic(xs_kernel_idx);
                        let level_angles = xs_provider.discrete_level_angles(xs_kernel_idx);
                        let inelastic_data = if !level_info.is_empty() {
                            Some(InelasticData {
                                levels: &level_info,
                                level_xs: &level_xs,
                                has_continuum: has_cont,
                                level_angles,
                            })
                        } else {
                            None
                        };

                        let elastic_angle = xs_provider.elastic_angular_dist(xs_kernel_idx);
                        let fission_edist = xs_provider.fission_energy_dist(xs_kernel_idx);
                        let continuum_edist = xs_provider.inelastic_continuum_edist(xs_kernel_idx);
                        let n2n_edist = xs_provider.n2n_edist(xs_kernel_idx);
                        let n3n_edist = xs_provider.n3n_edist(xs_kernel_idx);

                        let micro = &micro_xs[nuc_idx];
                        dispatch_real_collision(
                            &mut particle,
                            micro,
                            xs_kernel_idx,
                            xs_provider,
                            inelastic_data.as_ref(),
                            elastic_angle,
                            fission_edist,
                            continuum_edist,
                            n2n_edist,
                            n3n_edist,
                            cell.temperature,
                            &mut rng,
                            survival_biasing,
                            &mut result,
                            &mut pending,
                        );
                    }
                }
            }

            // Weight-window splitting / roulette at the new position.
            // Inside the inner while loop so it fires after every step
            // (collision or surface crossing). When the window is None
            // this is a no-op; when active it walks the per-voxel
            // bounds and either splits, rouletes, or leaves the
            // particle alone.
            if let Some(ww) = weight_window {
                crate::transport::weight_window::apply(&mut particle, ww, &mut rng, &mut pending);
            }
        }

        // Current particle finished. If any (n,xn) secondaries are pending,
        // transport the next one in the same history. Otherwise the source
        // particle is done.
        match pending.pop() {
            Some(p) => {
                particle = p;
                continue 'history;
            }
            None => break 'history,
        }
    }

    result
}

/// Dispatch a real collision, branching between analog and survival-
/// biasing paths.
///
/// Updates `particle` (energy / direction / weight / status) and
/// appends to `result` (fission_sites, photon_events, capture_cells,
/// counters) and `pending` (secondaries from (n,xn) multiplicity).
/// Used by both `transport_particle` (surface tracking) and
/// `transport_particle_delta` so the implicit-capture path is shared.
#[allow(clippy::too_many_arguments)]
fn dispatch_real_collision<XS: XsProvider>(
    particle: &mut Particle,
    micro: &MicroXs,
    xs_kernel_idx: usize,
    xs_provider: &XS,
    inelastic_data: Option<&InelasticData<'_>>,
    elastic_angle: Option<&AngularDistribution>,
    fission_edist: Option<&EnergyDistribution>,
    continuum_edist: Option<&EnergyDistribution>,
    n2n_edist: Option<&EnergyDistribution>,
    n3n_edist: Option<&EnergyDistribution>,
    temperature: f64,
    rng: &mut Rng,
    survival_biasing: Option<&SurvivalBiasing>,
    result: &mut ParticleResult,
    pending: &mut Vec<Particle>,
) {
    if let Some(sb) = survival_biasing {
        // ── Implicit-capture + Russian roulette ─────────────────────
        let nu_sigf_over_sigt = if micro.total > 0.0 {
            micro.nu_bar * micro.fission / micro.total
        } else {
            0.0
        };
        let n_fiss_expected = particle.weight * nu_sigf_over_sigt;
        let n_fiss = n_fiss_expected.floor() as usize
            + if rng.uniform() < n_fiss_expected.fract() {
                1
            } else {
                0
            };
        if n_fiss > 0 {
            // β(E) = ν_delayed / ν_total — see process_collision for details.
            let beta = if micro.nu_bar > 0.0 {
                (micro.delayed_nu_bar / micro.nu_bar).clamp(0.0, 1.0)
            } else {
                0.0
            };
            for _ in 0..n_fiss {
                let e_f = if beta > 0.0 && rng.uniform() < beta {
                    collision::sample_delayed_energy(rng)
                } else {
                    match fission_edist {
                        Some(d) => d.sample(particle.energy, rng),
                        None => collision::sample_fission_energy(particle.energy, rng),
                    }
                };
                result.fission_sites.push(FissionSite {
                    pos: particle.pos,
                    energy: e_f,
                    weight: 1.0,
                });
            }
            result.fissions += 1;
            sample_photon_products(
                xs_provider,
                xs_kernel_idx,
                FISSION_PHOTON_MTS,
                particle,
                rng,
                &mut result.photon_events,
            );
        }

        let sigma_a = micro.capture + micro.fission;
        let sigma_s = (micro.total - sigma_a).max(0.0);
        if sigma_s <= 0.0 {
            result.absorptions += 1;
            result.capture_cells.push(particle.cell_idx);
            sample_photon_products(
                xs_provider,
                xs_kernel_idx,
                ABSORPTION_PHOTON_MTS,
                particle,
                rng,
                &mut result.photon_events,
            );
            particle.kill();
            return;
        }
        particle.weight *= sigma_s / micro.total;

        let outcome = collision::process_scatter_only(
            particle,
            micro,
            inelastic_data,
            elastic_angle,
            continuum_edist,
            n2n_edist,
            n3n_edist,
            temperature,
            rng,
        );
        match outcome {
            CollisionOutcome::Scatter => {}
            CollisionOutcome::InelasticScatter { q_value_ev } => {
                result.photon_events.push(PhotonSourceEvent {
                    cell_idx: particle.cell_idx as u32,
                    pos: [particle.pos.x, particle.pos.y, particle.pos.z],
                    energy: q_value_ev.abs(),
                    mt: 4,
                });
            }
            CollisionOutcome::Multiplicity { secondaries } => {
                for s in secondaries {
                    pending.push(Particle::new(s.pos, s.dir, s.energy, particle.cell_idx));
                }
            }
            CollisionOutcome::Absorption => {
                result.absorptions += 1;
            }
            CollisionOutcome::Fission { .. } => {}
        }

        if particle.is_alive() && particle.weight < sb.w_min {
            let p_survive = particle.weight / sb.w_survive;
            if rng.uniform() < p_survive {
                particle.weight = sb.w_survive;
            } else {
                particle.kill();
            }
        }
        return;
    }

    // ── Analog (legacy bit-exact) ───────────────────────────────────
    let outcome = collision::process_collision(
        particle,
        micro,
        inelastic_data,
        elastic_angle,
        fission_edist,
        continuum_edist,
        n2n_edist,
        n3n_edist,
        temperature,
        rng,
    );
    match outcome {
        CollisionOutcome::Scatter => {}
        CollisionOutcome::InelasticScatter { q_value_ev } => {
            result.photon_events.push(PhotonSourceEvent {
                cell_idx: particle.cell_idx as u32,
                pos: [particle.pos.x, particle.pos.y, particle.pos.z],
                energy: q_value_ev.abs(),
                mt: 4,
            });
        }
        CollisionOutcome::Absorption => {
            result.absorptions += 1;
            result.capture_cells.push(particle.cell_idx);
            sample_photon_products(
                xs_provider,
                xs_kernel_idx,
                ABSORPTION_PHOTON_MTS,
                particle,
                rng,
                &mut result.photon_events,
            );
        }
        CollisionOutcome::Fission { sites } => {
            result.fissions += 1;
            result.fission_sites.extend(sites);
            sample_photon_products(
                xs_provider,
                xs_kernel_idx,
                FISSION_PHOTON_MTS,
                particle,
                rng,
                &mut result.photon_events,
            );
        }
        CollisionOutcome::Multiplicity { secondaries } => {
            for s in secondaries {
                pending.push(Particle::new(s.pos, s.dir, s.energy, particle.cell_idx));
            }
        }
    }
}

/// Transport a single particle using Woodcock delta tracking.
///
/// Instead of tracing to surfaces to sample collision distance, uses a
/// pre-computed majorant XS for the entire geometry. At each potential
/// collision, accepts or rejects based on real/majorant XS ratio.
/// This avoids surface intersection at transmission boundaries.
#[allow(clippy::too_many_arguments)]
fn transport_particle_delta<XS: XsProvider>(
    site: &FissionSite,
    batch: u64,
    particle_idx: u64,
    geometry: &crate::geometry::Geometry,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
    disable_delayed_neutrons: bool,
    urr_equivalence: Option<&crate::transport::urr_equivalence::UrrEquivalence>,
    majorant: &MajorantTable,
    tallies: &Tallies,
    survival_biasing: Option<&SurvivalBiasing>,
    weight_window: Option<&crate::transport::weight_window::WeightWindow>,
) -> ParticleResult {
    let mut rng = Rng::for_particle(batch, particle_idx);
    let mut result = ParticleResult {
        fission_sites: Vec::new(),
        leakage: 0,
        absorptions: 0,
        fissions: 0,
        collisions: 0,
        thermal_scatters: 0,
        surface_crossings: 0,
        capture_cells: Vec::new(),
        photon_events: Vec::new(),
        track_length_nu_sigf: 0.0,
        tallies: ParticleTallies::new(tallies),
    };

    let (u, v, w) = rng.isotropic_direction();
    let dir = Vec3::new(u, v, w);
    let mut particle = match geometry::ray::find_cell_recursive(site.pos, geometry) {
        Some(stack) => Particle::with_stack(site.pos, dir, site.energy, stack),
        // Match legacy `unwrap_or(0)` behavior — see transport_particle.
        None => Particle::new(site.pos, dir, site.energy, 0),
    };

    let mut pending: Vec<Particle> = Vec::new();

    let max_steps = 10_000;
    let mut steps = 0;

    'history: loop {
        while particle.is_alive() && steps < max_steps {
            steps += 1;
            let sigma_maj = majorant.lookup(particle.energy);

            if sigma_maj <= 1e-20 {
                particle.kill();
                result.leakage += 1;
                break;
            }

            let d_collision = rng.exponential(sigma_maj);

            // Check for vacuum/reflective boundaries before advancing
            let trace = geometry::ray::trace_step_recursive(
                &particle.coord_stack,
                particle.pos,
                particle.dir,
                geometry,
            );

            // How far the particle actually moves this Woodcock step.
            // The vacuum branch advances to the surface, the reflective
            // branch up to the surface, the transmission branch through
            // the surface for the full Woodcock distance, and the
            // free-advance branch the full d_collision.
            let advance_dist = match &trace {
                Some(hit) if hit.distance < d_collision => match hit.bc {
                    BoundaryCondition::Vacuum | BoundaryCondition::Reflective => hit.distance,
                    BoundaryCondition::Transmission => d_collision,
                },
                _ => d_collision,
            };

            // Mesh flux tally — deposit `w · advance_dist` over the
            // segment the particle traverses this Woodcock step.
            // Amanatides-Woo deposit handles axis-aligned voxel walks;
            // the integrand is the same as in surface tracking even if
            // the segment crosses materials silently.
            if let Some(mesh) = tallies.mesh_flux.as_ref() {
                mesh.deposit(
                    particle.pos,
                    particle.dir,
                    advance_dist,
                    particle.weight,
                    &mut result.tallies.mesh_flux,
                );
            }

            // Surface current tally on the FIRST boundary the segment
            // hits. Reflective and transmission BCs both produce real
            // surface crossings; vacuum kills the history at the
            // boundary but the crossing still counts. Subsequent
            // surfaces beyond `hit.distance` along the same Woodcock
            // step (only possible for transmission) are not tallied —
            // standard pragmatic limitation of surface-current under
            // delta tracking; reflective/vacuum boundaries are exact
            // because the segment stops there.
            if let (Some(hit), Some(sct)) =
                (trace.as_ref(), tallies.surface_current.as_ref())
                && hit.distance < d_collision
                && let Some(surf_idx) = hit.surface_idx
                && let Some(bin) = sct.bin_for(surf_idx)
            {
                let crossing_pos = particle.pos + particle.dir * hit.distance;
                let n = surfaces[surf_idx].normal_at(crossing_pos);
                if particle.dir.dot(n) >= 0.0 {
                    result.tallies.surface_current_pos[bin] += particle.weight;
                } else {
                    result.tallies.surface_current_neg[bin] += particle.weight;
                }
            }

            match trace {
                Some(hit) if hit.distance < d_collision => {
                    result.surface_crossings += 1;
                    match hit.bc {
                        BoundaryCondition::Vacuum => {
                            particle.kill();
                            result.leakage += 1;
                            break;
                        }
                        BoundaryCondition::Reflective => {
                            let surf_idx = hit.surface_idx.unwrap_or(0);
                            particle.advance(hit.distance);
                            let n = surfaces[surf_idx].normal_at(particle.pos);
                            let d = particle.dir;
                            particle.dir = d - n * (2.0 * d.dot(n));
                            continue;
                        }
                        BoundaryCondition::Transmission => {
                            // Delta tracking: skip transmission boundaries, just advance.
                            particle.advance(d_collision);
                            match geometry::ray::find_cell_recursive(particle.pos, geometry) {
                                Some(stack) => {
                                    particle.cell_idx =
                                        stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                                    particle.coord_stack = stack;
                                }
                                None => {
                                    particle.kill();
                                    result.leakage += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
                _ => {
                    // No boundary before collision distance — advance freely.
                    particle.advance(d_collision);
                    match geometry::ray::find_cell_recursive(particle.pos, geometry) {
                        Some(stack) => {
                            particle.cell_idx =
                                stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                            particle.coord_stack = stack;
                        }
                        None => {
                            particle.kill();
                            result.leakage += 1;
                            break;
                        }
                    }
                }
            }

            // Get current material — handle void by free-streaming.
            // Effective fill applies any per-lattice-element override.
            let cell = &cells[particle.cell_idx];
            let _ = cell; // silence unused while we drop the static-fill match
            let effective = geometry.effective_material_idx(&particle.coord_stack);
            let mat_idx = match effective {
                crate::geometry::EffectiveFill::Material(m) => m as usize,
                crate::geometry::EffectiveFill::Void => {
                    // Void region — free-stream to next surface.
                    let trace = geometry::ray::trace_step_recursive(
                        &particle.coord_stack,
                        particle.pos,
                        particle.dir,
                        geometry,
                    );
                    match trace {
                        Some(hit) => {
                            match hit.bc {
                                BoundaryCondition::Vacuum => {
                                    particle.advance(hit.distance);
                                    particle.kill();
                                    result.leakage += 1;
                                    break;
                                }
                                BoundaryCondition::Reflective => {
                                    let surf_idx = hit.surface_idx.unwrap_or(0);
                                    particle.advance(hit.distance);
                                    let n = surfaces[surf_idx].normal_at(particle.pos);
                                    let d = particle.dir;
                                    particle.dir = d - n * (2.0 * d.dot(n));
                                }
                                BoundaryCondition::Transmission => {
                                    particle
                                        .advance(hit.distance + (hit.distance * 1e-8).max(1e-8));
                                    match hit.next_stack {
                                        Some(stack) => {
                                            particle.cell_idx = stack
                                                .last()
                                                .map(|c| c.cell_idx as usize)
                                                .unwrap_or(0);
                                            particle.coord_stack = stack;
                                        }
                                        None => {
                                            particle.kill();
                                            result.leakage += 1;
                                            break;
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        None => {
                            particle.kill();
                            result.leakage += 1;
                            break;
                        }
                    }
                }
            };
            if mat_idx >= materials.len() {
                particle.kill();
                result.leakage += 1;
                break;
            }

            let material = &materials[mat_idx];

            // Compute real Σ_t for acceptance test
            let urr_xi = rng.uniform();
            let n_nuclides = material.nuclides.len();
            let mut micro_xs = [MicroXs::default(); MAX_NUCLIDES];
            let mut micro_xs_smooth = [MicroXs::default(); MAX_NUCLIDES];
            let mut micro_totals = [0.0_f64; MAX_NUCLIDES];
            for (i, nuc) in material.nuclides.iter().enumerate() {
                let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, particle.energy);
                // Snapshot pre-URR-PT smooth XS for Hwang superposition.
                micro_xs_smooth[i] = xs;
                xs_provider.apply_urr(nuc.xs_kernel_idx, &mut xs, particle.energy, urr_xi);
                if disable_delayed_neutrons {
                    xs.delayed_nu_bar = 0.0;
                }
                micro_totals[i] = xs.total;
                micro_xs[i] = xs;
            }
            // URR equivalence pass — same gating + correction as
            // surface-tracking. Delta tracking's acceptance test
            // depends on the real Σ_t, so the correction has to be
            // applied before that test.
            if let Some(eq) = urr_equivalence {
                let dancoff = eq.dancoff.get(particle.cell_idx);
                let mean_chord = eq
                    .mean_chord_cm
                    .get(particle.cell_idx)
                    .copied()
                    .unwrap_or(0.0);
                if dancoff < 1.0 && mean_chord > 0.0 {
                    apply_urr_equivalence_correction(
                        eq,
                        material,
                        xs_provider,
                        particle.energy,
                        dancoff,
                        mean_chord,
                        &mut micro_xs,
                        &micro_xs_smooth,
                        &mut micro_totals,
                        n_nuclides,
                    );
                }
            }
            let sigma_real = material.macro_total(&micro_totals[..n_nuclides]);

            // Acceptance test: real collision with probability Σ_t / Σ_maj
            if rng.uniform() >= sigma_real / sigma_maj {
                continue; // Virtual collision — keep tracking
            }

            // Real collision — process exactly as surface tracking
            result.collisions += 1;

            // Sutton-Brown-style track-length k-eff estimator under
            // delta tracking. Surface tracking can score `w·d·ν·Σ_f`
            // per cell-residence segment because each segment is in
            // one cell; under Woodcock, segments cross materials
            // silently and the per-cell-segment integrand isn't
            // available. The unbiased equivalent is to score
            // `w · ν·Σ_f(m,E) / Σ_t(m,E)` at each *real* collision:
            //
            //   E[score per unit length in material m] =
            //     Σ_t(m,E) · [ν·Σ_f(m,E)/Σ_t(m,E)]  =  ν·Σ_f(m,E)
            //
            // which is the same integrand the surface-tracking k_track
            // accumulates. Variance is comparable to the collision
            // estimator (k_eff itself); the value is the cross-check
            // it gives against k_eff. Closes the documented gap
            // (k_track was 0 under delta tracking pre this change).
            if sigma_real > 0.0 {
                let mut macro_nu_sigma_f = 0.0_f64;
                for (i, nuc) in material.nuclides.iter().enumerate() {
                    macro_nu_sigma_f +=
                        nuc.atom_density * micro_xs[i].nu_bar * micro_xs[i].fission;
                }
                if macro_nu_sigma_f > 0.0 {
                    result.track_length_nu_sigf +=
                        particle.weight * macro_nu_sigma_f / sigma_real;
                }
            }

            let macro_total = sigma_real;
            if macro_total <= 0.0 {
                particle.kill();
                result.leakage += 1;
                break;
            }

            let nuc_idx =
                material.sample_nuclide(&micro_totals[..n_nuclides], macro_total, rng.uniform());

            let xs_kernel_idx = material.nuclides[nuc_idx].xs_kernel_idx;
            let level_info = xs_provider.discrete_level_info(xs_kernel_idx);
            let level_xs = xs_provider.discrete_level_xs(xs_kernel_idx, particle.energy);
            let has_cont = xs_provider.has_continuum_inelastic(xs_kernel_idx);
            let level_angles = xs_provider.discrete_level_angles(xs_kernel_idx);

            let inelastic_data = if !level_info.is_empty() {
                Some(InelasticData {
                    levels: &level_info,
                    level_xs: &level_xs,
                    has_continuum: has_cont,
                    level_angles,
                })
            } else {
                None
            };

            let elastic_angle = xs_provider.elastic_angular_dist(xs_kernel_idx);
            let fission_edist = xs_provider.fission_energy_dist(xs_kernel_idx);
            let continuum_edist = xs_provider.inelastic_continuum_edist(xs_kernel_idx);
            let n2n_edist = xs_provider.n2n_edist(xs_kernel_idx);
            let n3n_edist = xs_provider.n3n_edist(xs_kernel_idx);

            dispatch_real_collision(
                &mut particle,
                &micro_xs[nuc_idx],
                xs_kernel_idx,
                xs_provider,
                inelastic_data.as_ref(),
                elastic_angle,
                fission_edist,
                continuum_edist,
                n2n_edist,
                n3n_edist,
                cell.temperature,
                &mut rng,
                survival_biasing,
                &mut result,
                &mut pending,
            );

            // Weight-window splitting / roulette at the new position.
            // Inside the inner while loop so it fires per step.
            if let Some(ww) = weight_window {
                crate::transport::weight_window::apply(&mut particle, ww, &mut rng, &mut pending);
            }
        }

        match pending.pop() {
            Some(p) => {
                particle = p;
                continue 'history;
            }
            None => break 'history,
        }
    }

    result
}

/// Run a k-eigenvalue simulation with rayon parallel transport.
///
/// Automatically detects the optimal tracking algorithm based on geometry:
/// - Single material → surface tracking
/// - Multiple materials → Woodcock delta tracking
///
/// Returns per-batch results and the final k_eff estimate.
pub fn run_eigenvalue<XS: XsProvider>(
    config: &SimConfig,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
) -> (Vec<BatchResult>, f64) {
    // Wrap the flat surfaces+cells in a single-universe Geometry, then
    // delegate to the geometry-aware variant. Existing binaries that
    // pass slices keep working unchanged.
    let geometry =
        crate::geometry::Geometry::from_slices(surfaces, cells).expect("geometry must validate");
    run_eigenvalue_with_geometry(config, &geometry, materials, xs_provider)
}

/// Run a k-eigenvalue simulation with an explicit recursive `Geometry`.
///
/// Use this entry point for problems with nested universes or
/// lattices. Flat-geometry callers should keep using `run_eigenvalue`,
/// which wraps slices into a single-universe Geometry and then calls
/// this function.
pub fn run_eigenvalue_with_geometry<XS: XsProvider>(
    config: &SimConfig,
    geometry: &crate::geometry::Geometry,
    materials: &[Material],
    xs_provider: &XS,
) -> (Vec<BatchResult>, f64) {
    let n = config.particles_per_batch as usize;
    let surfaces = geometry.surfaces.as_slice();
    let cells = geometry.cells.as_slice();

    let tracking = detect_tracking_mode(cells, materials, xs_provider, config.verbose);

    let seed = config.seed;
    let mut source_bank = match config.initial_source_bank.as_ref() {
        Some(bank) if !bank.is_empty() => {
            // Resume mode: use the provided bank as the source for
            // batch 1. Resample (with replacement) to the requested
            // particles_per_batch so the population size is consistent.
            let mut rng = Rng::new(seed * 100_000, 1);
            (0..n)
                .map(|_| {
                    let idx = (rng.uniform() * bank.len() as f64) as usize;
                    bank[idx.min(bank.len() - 1)].clone()
                })
                .collect()
        }
        _ => initial_source(n, geometry, cells, seed),
    };

    // Build the Shannon-entropy mesh from the AABB of the first fissile cell.
    let aabb = cells
        .iter()
        .find_map(|c| match c.fill {
            CellFill::Material(_) => Some(c.aabb),
            _ => None,
        })
        .unwrap_or(crate::geometry::Aabb::new(
            crate::geometry::Vec3::new(-1.0, -1.0, -1.0),
            crate::geometry::Vec3::new(1.0, 1.0, 1.0),
        ));
    let entropy_mesh = EntropyMesh::from_aabb(&aabb, 8);

    let mut results = Vec::with_capacity(config.batches as usize);
    let mut k_sum = 0.0;
    let mut k_count = 0_u32;

    // For auto-inactive: running entropy history and effective inactive count.
    let mut entropy_history: Vec<f64> = Vec::with_capacity(config.batches as usize);
    // When `auto_inactive` is set, start with effective_inactive = batches so
    // NO batch is counted active until the plateau detector fires. The fixed
    // `config.inactive` is used only in non-auto mode.
    let mut effective_inactive = if config.auto_inactive.is_some() {
        config.batches
    } else {
        config.inactive
    };
    let mut auto_converged_at: Option<u32> = None;

    for batch in 1..=config.batches {
        // Parallel transport: dispatch based on tracking mode.
        // Seed offsets batch number to make each seed produce independent streams.
        let batch_seed = batch as u64 + seed * 100_000;
        let transport_one = |(i, site): (usize, &FissionSite)| match &tracking {
            TrackingMode::Surface => transport_particle(
                site,
                batch_seed,
                i as u64,
                geometry,
                surfaces,
                cells,
                materials,
                xs_provider,
                &config.tallies,
                config.survival_biasing.as_ref(),
                config.weight_window.as_ref(),
                config.disable_delayed_neutrons,
                config.urr_equivalence.as_ref(),
            ),
            TrackingMode::Delta(majorant) => transport_particle_delta(
                site,
                batch_seed,
                i as u64,
                geometry,
                surfaces,
                cells,
                materials,
                xs_provider,
                config.disable_delayed_neutrons,
                config.urr_equivalence.as_ref(),
                majorant,
                &config.tallies,
                config.survival_biasing.as_ref(),
                config.weight_window.as_ref(),
            ),
        };
        let particle_results: Vec<ParticleResult> = if config.parallel {
            source_bank
                .par_iter()
                .enumerate()
                .map(transport_one)
                .collect()
        } else {
            source_bank.iter().enumerate().map(transport_one).collect()
        };

        // Reduce: merge per-particle results
        let mut fission_bank = FissionBank::new();
        let mut leakage = 0_u32;
        let mut absorptions = 0_u32;
        let mut fissions = 0_u32;
        let mut collisions = 0_u32;
        let mut thermal_scatters = 0_u32;
        let mut surface_crossings = 0_u32;
        let mut captures_by_cell = vec![0.0_f64; cells.len()];
        let mut photon_events: Vec<PhotonSourceEvent> = Vec::new();
        let mut track_length_sum = 0.0_f64;
        let n_surf_bins = config.tallies.n_surface_bins();
        let n_mesh_voxels = config.tallies.n_mesh_voxels();
        let (rr_flux_size, rr_rate_size) = config.tallies.reaction_rate_size();
        let mut surface_current_pos = vec![0.0_f64; n_surf_bins];
        let mut surface_current_neg = vec![0.0_f64; n_surf_bins];
        let mut mesh_flux = vec![0.0_f64; n_mesh_voxels];
        let mut rr_flux_acc = vec![0.0_f64; rr_flux_size];
        let mut rr_rate_acc = vec![0.0_f64; rr_rate_size];

        for pr in particle_results {
            fission_bank.sites.extend(pr.fission_sites);
            leakage += pr.leakage;
            absorptions += pr.absorptions;
            fissions += pr.fissions;
            collisions += pr.collisions;
            thermal_scatters += pr.thermal_scatters;
            surface_crossings += pr.surface_crossings;
            track_length_sum += pr.track_length_nu_sigf;
            for c in pr.capture_cells {
                if c < captures_by_cell.len() {
                    captures_by_cell[c] += 1.0;
                }
            }
            photon_events.extend(pr.photon_events);
            // Reduce per-particle tallies into batch totals. When the
            // tally is disabled both vecs are empty so `zip` is a no-op.
            for (b, v) in surface_current_pos
                .iter_mut()
                .zip(pr.tallies.surface_current_pos.iter())
            {
                *b += v;
            }
            for (b, v) in surface_current_neg
                .iter_mut()
                .zip(pr.tallies.surface_current_neg.iter())
            {
                *b += v;
            }
            for (b, v) in mesh_flux.iter_mut().zip(pr.tallies.mesh_flux.iter()) {
                *b += v;
            }
            for (b, v) in rr_flux_acc.iter_mut().zip(pr.tallies.rr_flux.iter()) {
                *b += v;
            }
            for (b, v) in rr_rate_acc.iter_mut().zip(pr.tallies.rr_rate.iter()) {
                *b += v;
            }
        }

        let k_batch = fission_bank.len() as f64 / n as f64;
        let k_track = track_length_sum / n as f64;
        let entropy = entropy_mesh.entropy(&fission_bank.sites);

        let mut result = BatchResult {
            batch,
            k_eff: k_batch,
            leakage,
            absorptions,
            fissions,
            collisions,
            thermal_scatters,
            surface_crossings,
            shannon_entropy: entropy,
            active: false,
            captures_by_cell,
            photon_events,
            k_track,
            surface_current_pos,
            surface_current_neg,
            mesh_flux,
            rr_flux: rr_flux_acc,
            rr_rate: rr_rate_acc,
        };

        // Auto-inactive: promote this batch to active if entropy has plateaued.
        // Still honors the user's fixed `inactive` as a minimum unless auto
        // explicitly decides earlier.
        entropy_history.push(entropy);
        if let Some(policy) = config.auto_inactive
            && auto_converged_at.is_none()
            && batch >= policy.min_inactive
            && (policy.has_converged(&entropy_history) || batch >= policy.max_inactive)
        {
            // Convergence fires at end of `batch`; first active is batch+1.
            effective_inactive = batch;
            auto_converged_at = Some(batch);
            let reason = if batch >= policy.max_inactive && !policy.has_converged(&entropy_history)
            {
                "max_inactive"
            } else {
                "plateau"
            };
            if config.verbose {
                println!("  [auto-inactive] entropy converged at batch {batch} ({reason})");
                let _ = std::io::stdout().flush();
            }
        }

        let is_active = batch > effective_inactive;
        result.active = is_active;
        if is_active {
            k_sum += k_batch;
            k_count += 1;
        }

        if config.verbose {
            let active_str = if is_active { " *" } else { "" };
            println!(
                "  Batch {batch:>4}: k={k_batch:.5}  k_t={k_track:.5}  H={entropy:.4}  \
                 coll={collisions}  fiss={fissions}  leak={leakage}  \
                 therm={thermal_scatters}  surf={surface_crossings}{active_str}"
            );
            let _ = std::io::stdout().flush();
        }

        results.push(result);

        source_bank = normalize_fission_bank(&fission_bank, n, batch + seed as u32 * 100_000);
    }

    let k_final = if k_count > 0 {
        k_sum / k_count as f64
    } else {
        0.0
    };

    // Optional statepoint write at end of run.
    if let Some(path) = config.statepoint_path.as_ref() {
        let n_active = results.iter().filter(|r| r.active).count() as u32;
        let inputs = crate::transport::statepoint::StatepointInputs {
            batches: &results,
            source_bank: &source_bank,
            n_active,
            particles_per_batch: config.particles_per_batch,
            seed: config.seed,
            k_eff_mean: k_final,
            n_surface_bins: config.tallies.n_surface_bins(),
            n_mesh_voxels: config.tallies.n_mesh_voxels(),
        };
        match crate::transport::statepoint::write_statepoint(path, &inputs) {
            Ok(()) => {
                if config.verbose {
                    println!("  Statepoint written: {}", path.display());
                }
            }
            Err(e) => eprintln!("  Statepoint write FAILED: {e}"),
        }
    }

    (results, k_final)
}

/// Create an initial source uniformly distributed in fissile material cells.
///
/// Rejection-samples points in the bounding box of fissile cells, accepting
/// only those that land inside a cell containing material. For Godiva, this
/// is the single fuel sphere. For PWR pin cell, this is the cylindrical
/// fuel region (rejects corners of the bounding box that fall in gap/clad/water).
fn initial_source(
    n: usize,
    geometry: &crate::geometry::Geometry,
    cells: &[Cell],
    seed: u64,
) -> Vec<FissionSite> {
    let mut rng = Rng::new(seed * 100_000, 0);
    let mut sites = Vec::with_capacity(n);

    // Find the first material cell (assumed fissile for eigenvalue problems).
    let target_idx = cells
        .iter()
        .position(|c| matches!(c.fill, CellFill::Material(_)));

    // Sampling AABB. For flat geometries the target Material cell
    // typically has a tight AABB (set by the caller) and we use that.
    // For nested geometries the Material cell lives in element-local
    // coords — its AABB is meaningless in world coords. In that case
    // fall back to the union of every cell that has a finite world
    // AABB; that's the smallest world-coord box guaranteed to contain
    // the geometry.
    let target_aabb = target_idx.map(|i| cells[i].aabb);
    let aabb = match target_aabb {
        Some(a) if a.surface_area().is_finite() && a.surface_area() > 0.0 => a,
        _ => union_finite_world_aabbs(cells).unwrap_or(crate::geometry::Aabb::new(
            Vec3::new(-10.0, -10.0, -10.0),
            Vec3::new(10.0, 10.0, 10.0),
        )),
    };

    let mut attempts: u64 = 0;
    let max_attempts: u64 = (n as u64).saturating_mul(10_000).max(1_000_000);

    while sites.len() < n {
        attempts += 1;
        if attempts > max_attempts {
            panic!(
                "initial_source: rejection sampling failed to find {} fissile points \
                 in {} attempts inside AABB {:?}. Either the geometry has no Material \
                 cells, or the target AABB doesn't intersect any fissile region.",
                n, max_attempts, aabb
            );
        }
        let x = aabb.min.x + rng.uniform() * (aabb.max.x - aabb.min.x);
        let y = aabb.min.y + rng.uniform() * (aabb.max.y - aabb.min.y);
        let z = aabb.min.z + rng.uniform() * (aabb.max.z - aabb.min.z);
        let pos = Vec3::new(x, y, z);

        if let Some(stack) = geometry::ray::find_cell_recursive(pos, geometry) {
            let deepest = stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
            // Accept if the leaf is the target Material cell (flat case)
            // OR any Material cell (nested case where the leaf is inside
            // a universe/lattice fill).
            let accept = match cells.get(deepest).map(|c| c.fill) {
                Some(CellFill::Material(_)) => {
                    Some(deepest) == target_idx || target_idx.is_none() || stack.len() > 1
                }
                _ => false,
            };
            if accept {
                sites.push(FissionSite {
                    pos,
                    energy: 1.0e6,
                    weight: 1.0,
                });
            }
        }
    }

    sites
}

/// Union of every cell's AABB that's finite (i.e. set by the caller —
/// the default `Aabb::INFINITE` is excluded). Used as a sampling-box
/// fallback in `initial_source` for nested geometries where the
/// fissile cell's AABB is element-local and meaningless in world
/// coordinates.
fn union_finite_world_aabbs(cells: &[Cell]) -> Option<crate::geometry::Aabb> {
    cells
        .iter()
        .map(|c| c.aabb)
        .filter(|a| a.surface_area().is_finite() && a.surface_area() > 0.0)
        .reduce(crate::geometry::Aabb::union)
}

/// Normalize fission bank to N particles for the next generation.
fn normalize_fission_bank(bank: &FissionBank, n: usize, batch: u32) -> Vec<FissionSite> {
    if bank.is_empty() {
        return (0..n)
            .map(|_| FissionSite {
                pos: Vec3::new(0.0, 0.0, 0.0),
                energy: 1.0e6,
                weight: 1.0,
            })
            .collect();
    }

    let mut rng = Rng::new(batch as u64, 0);
    (0..n)
        .map(|_| {
            let idx = (rng.uniform() * bank.len() as f64) as usize;
            let idx = idx.min(bank.len() - 1);
            bank.sites[idx].clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, CellId};

    // ── EntropyConvergence ────────────────────────────────────────────

    #[test]
    fn entropy_convergence_rejects_empty_history() {
        let p = EntropyConvergence::default();
        assert!(!p.has_converged(&[]));
    }

    #[test]
    fn entropy_convergence_respects_min_inactive() {
        let p = EntropyConvergence {
            min_inactive: 10,
            max_inactive: 100,
            window: 5,
            cv_tol: 1e-2,
        };
        // First 9 entries are extremely flat but should NOT trigger because
        // we have not reached min_inactive.
        let history: Vec<f64> = (0..9).map(|_| 7.5).collect();
        assert!(!p.has_converged(&history), "fired before min_inactive");
    }

    #[test]
    fn entropy_convergence_fires_on_flat_window() {
        let p = EntropyConvergence {
            min_inactive: 5,
            max_inactive: 100,
            window: 10,
            cv_tol: 1e-3,
        };
        // Flat 12 samples at 8.0 → CV = 0 < tol.
        let history: Vec<f64> = vec![8.0; 12];
        assert!(p.has_converged(&history));
    }

    #[test]
    fn entropy_convergence_does_not_fire_on_noisy_window() {
        let p = EntropyConvergence {
            min_inactive: 5,
            max_inactive: 100,
            window: 10,
            cv_tol: 1e-3,
        };
        // 10% oscillation on top of 8.0 → CV ≈ 5e-2 ≫ tol.
        let history: Vec<f64> = (0..20)
            .map(|i| 8.0 + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        assert!(!p.has_converged(&history));
    }

    #[test]
    fn entropy_convergence_window_only_looks_at_tail() {
        let p = EntropyConvergence {
            min_inactive: 5,
            max_inactive: 100,
            window: 5,
            cv_tol: 1e-3,
        };
        // Early noise, recent flat tail → should fire (only tail matters).
        let mut history: Vec<f64> = (0..10)
            .map(|i| if i < 5 { 5.0 + i as f64 } else { 8.0 })
            .collect();
        history.push(8.0);
        assert!(p.has_converged(&history));
    }

    #[test]
    fn entropy_convergence_handles_near_zero_mean() {
        let p = EntropyConvergence {
            min_inactive: 5,
            max_inactive: 100,
            window: 5,
            cv_tol: 1e-3,
        };
        // Mean near zero — guard against divide-by-near-zero in CV.
        // Must not falsely trigger.
        let history: Vec<f64> = vec![1e-15, -1e-15, 1e-15, -1e-15, 1e-15, -1e-15, 1e-15];
        assert!(!p.has_converged(&history));
    }

    #[test]
    fn entropy_convergence_cv_threshold_is_honoured() {
        // With cv_tol = 1e-2, a 0.5% coefficient of variation should fire
        // but a 2% coefficient should not.
        let p = EntropyConvergence {
            min_inactive: 5,
            max_inactive: 100,
            window: 10,
            cv_tol: 1e-2,
        };
        let flat: Vec<f64> = (0..20)
            .map(|i| 8.0 + if i % 2 == 0 { 0.04 } else { -0.04 })
            .collect();
        // CV ≈ 0.04/8.0 = 5e-3 < 1e-2 → should fire
        assert!(
            p.has_converged(&flat),
            "0.5% CV should be below 1e-2 threshold"
        );

        let noisy: Vec<f64> = (0..20)
            .map(|i| 8.0 + if i % 2 == 0 { 0.16 } else { -0.16 })
            .collect();
        // CV = 0.16/8.0 = 2e-2 > 1e-2 → should not fire
        assert!(
            !p.has_converged(&noisy),
            "2% CV should exceed 1e-2 threshold"
        );
    }

    // ── EntropyMesh ───────────────────────────────────────────────────

    #[test]
    fn entropy_mesh_empty_bank_is_zero() {
        use crate::geometry::{Aabb, Vec3};
        let mesh = EntropyMesh::from_aabb(
            &Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            4,
        );
        assert_eq!(mesh.entropy(&[]), 0.0);
    }

    #[test]
    fn entropy_mesh_single_bin_is_zero() {
        use crate::geometry::{Aabb, Vec3};
        let mesh = EntropyMesh::from_aabb(
            &Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            4,
        );
        // All sites in the same bin: p = 1 → H = 0.
        let sites: Vec<FissionSite> = (0..100)
            .map(|_| FissionSite {
                pos: Vec3::new(0.1, 0.1, 0.1),
                energy: 1e6,
                weight: 1.0,
            })
            .collect();
        assert!(
            mesh.entropy(&sites).abs() < 1e-12,
            "concentrated bank should have H ≈ 0"
        );
    }

    #[test]
    fn entropy_mesh_uniform_bank_saturates() {
        use crate::geometry::{Aabb, Vec3};
        let n = 4;
        let mesh = EntropyMesh::from_aabb(
            &Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 4.0, 4.0)),
            n,
        );
        // One site per bin = perfectly uniform → H = log2(n^3) = log2(64) = 6.
        let mut sites = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    sites.push(FissionSite {
                        pos: Vec3::new(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5),
                        energy: 1e6,
                        weight: 1.0,
                    });
                }
            }
        }
        let h = mesh.entropy(&sites);
        let upper = ((n * n * n) as f64).log2();
        assert!(
            (h - upper).abs() < 1e-10,
            "uniform bank should saturate at log2(64): got {h}"
        );
    }

    #[test]
    fn godiva_eigenvalue_smoke_test() {
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 8.7407,
            bc: BoundaryCondition::Vacuum,
        }];

        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)).with_aabb(
                crate::geometry::Aabb::new(
                    Vec3::new(-8.7407, -8.7407, -8.7407),
                    Vec3::new(8.7407, 8.7407, 8.7407),
                ),
            ),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];

        let mut heu = Material::new("HEU", 294.0);
        heu.add_nuclide(0.048, 0);

        let materials = vec![heu];

        let xs_provider = ConstantXs {
            xs: vec![MicroXs {
                total: 7.0,
                elastic: 4.0,
                inelastic: 0.0,
                n2n: 0.0,
                n3n: 0.0,
                fission: 1.2,
                capture: 0.1,
                nu_bar: 2.43,
                delayed_nu_bar: 0.0,
                awr: 235.0,
            }],
        };

        let config = SimConfig {
            batches: 10,
            inactive: 3,
            particles_per_batch: 500,
            seed: 0,
            auto_inactive: None,
            verbose: false,
            parallel: false,
            tallies: Default::default(),
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        };

        let (results, k_final) =
            run_eigenvalue(&config, &surfaces, &cells, &materials, &xs_provider);

        assert_eq!(results.len(), 10);
        assert!(
            k_final > 0.3 && k_final < 3.0,
            "k_final = {k_final} — out of reasonable range"
        );
        println!("\n  Godiva smoke test: k_final = {k_final:.4}");
    }

    #[test]
    fn void_streaming_pincell_geometry() {
        // Particle born in fuel → crosses void gap → enters clad.
        // Verifies that void cells don't kill the particle.
        use crate::geometry::surface::BoundaryCondition;

        let fuel_r = 0.4096;
        let clad_ir = 0.4180;
        let clad_or = 0.4750;

        let surfaces = vec![
            Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: fuel_r,
                bc: BoundaryCondition::Transmission,
            },
            Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: clad_ir,
                bc: BoundaryCondition::Transmission,
            },
            Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: clad_or,
                bc: BoundaryCondition::Vacuum,
            },
        ];

        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)).with_aabb(
                crate::geometry::Aabb::new(
                    Vec3::new(-fuel_r, -fuel_r, -fuel_r),
                    Vec3::new(fuel_r, fuel_r, fuel_r),
                ),
            ),
            Cell::new(
                CellId(1),
                cell::intersect_all(vec![cell::outside(0), cell::inside(1)]),
                CellFill::Void,
            )
            .with_aabb(crate::geometry::Aabb::new(
                Vec3::new(-clad_ir, -clad_ir, -clad_ir),
                Vec3::new(clad_ir, clad_ir, clad_ir),
            )),
            Cell::new(
                CellId(2),
                cell::intersect_all(vec![cell::outside(1), cell::inside(2)]),
                CellFill::Material(1),
            )
            .with_aabb(crate::geometry::Aabb::new(
                Vec3::new(-clad_or, -clad_or, -clad_or),
                Vec3::new(clad_or, clad_or, clad_or),
            )),
            Cell::new(CellId(3), cell::outside(2), CellFill::Void),
        ];

        // Fuel: high fission to keep neutrons alive
        let mut fuel = Material::new("fuel", 294.0);
        fuel.add_nuclide(0.048, 0);
        // Clad: pure absorber (kills all neutrons)
        let mut clad = Material::new("clad", 294.0);
        clad.add_nuclide(0.04, 1);

        let materials = vec![fuel, clad];

        let xs_provider = ConstantXs {
            xs: vec![
                MicroXs {
                    total: 7.0,
                    elastic: 4.0,
                    inelastic: 0.0,
                    n2n: 0.0,
                    n3n: 0.0,
                    fission: 2.0,
                    capture: 0.1,
                    nu_bar: 2.43,
                    delayed_nu_bar: 0.0,
                    awr: 235.0,
                },
                MicroXs {
                    total: 5.0,
                    elastic: 1.0,
                    inelastic: 0.0,
                    n2n: 0.0,
                    n3n: 0.0,
                    fission: 0.0,
                    capture: 4.0,
                    nu_bar: 0.0,
                    delayed_nu_bar: 0.0,
                    awr: 91.0,
                },
            ],
        };

        let config = SimConfig {
            batches: 5,
            inactive: 1,
            particles_per_batch: 200,
            seed: 0,
            auto_inactive: None,
            verbose: false,
            parallel: false,
            tallies: Default::default(),
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        };
        let (results, _k) = run_eigenvalue(&config, &surfaces, &cells, &materials, &xs_provider);

        // Key check: simulation runs to completion. If void streaming is broken,
        // particles die in the gap and we get k=0 and no collisions.
        let total_collisions: u32 = results.iter().map(|r| r.collisions).sum();
        assert!(
            total_collisions > 100,
            "Too few collisions ({total_collisions}) — void streaming may be broken"
        );
        println!(
            "\n  Void streaming test: {total_collisions} collisions across {} batches",
            results.len()
        );
    }

    #[test]
    fn tracking_mode_single_material_is_surface() {
        // Single material → surface tracking
        let _surfaces = [Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 5.0,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];
        let mut mat = Material::new("test", 294.0);
        mat.add_nuclide(0.04, 0);
        let materials = vec![mat];

        let xs = ConstantXs {
            xs: vec![MicroXs {
                total: 5.0,
                elastic: 3.0,
                inelastic: 0.0,
                n2n: 0.0,
                n3n: 0.0,
                fission: 1.0,
                capture: 1.0,
                nu_bar: 2.43,
                delayed_nu_bar: 0.0,
                awr: 235.0,
            }],
        };

        let mode = detect_tracking_mode(&cells, &materials, &xs, false);
        assert!(
            matches!(mode, TrackingMode::Surface),
            "Single material should use surface tracking"
        );
    }

    #[test]
    fn tracking_mode_high_contrast_falls_back() {
        // Two materials with high XS contrast → should fall back to surface
        let _surfaces = [
            Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 5.0,
                bc: BoundaryCondition::Transmission,
            },
            Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 10.0,
                bc: BoundaryCondition::Vacuum,
            },
        ];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(
                CellId(1),
                cell::intersect_all(vec![cell::outside(0), cell::inside(1)]),
                CellFill::Material(1),
            ),
            Cell::new(CellId(2), cell::outside(1), CellFill::Void),
        ];
        let mut mat_strong = Material::new("strong", 294.0);
        mat_strong.add_nuclide(0.05, 0);
        let mut mat_weak = Material::new("weak", 294.0);
        mat_weak.add_nuclide(0.0001, 1);
        let materials = vec![mat_strong, mat_weak];

        let xs = ConstantXs {
            xs: vec![
                MicroXs {
                    total: 100.0,
                    elastic: 90.0,
                    inelastic: 0.0,
                    n2n: 0.0,
                    n3n: 0.0,
                    fission: 5.0,
                    capture: 5.0,
                    nu_bar: 2.43,
                    delayed_nu_bar: 0.0,
                    awr: 235.0,
                },
                MicroXs {
                    total: 1.0,
                    elastic: 0.9,
                    inelastic: 0.0,
                    n2n: 0.0,
                    n3n: 0.0,
                    fission: 0.0,
                    capture: 0.1,
                    nu_bar: 0.0,
                    delayed_nu_bar: 0.0,
                    awr: 56.0,
                },
            ],
        };

        let mode = detect_tracking_mode(&cells, &materials, &xs, false);
        assert!(
            matches!(mode, TrackingMode::Surface),
            "High contrast should fall back to surface tracking"
        );
    }

    #[test]
    fn different_seeds_produce_different_results() {
        // Two runs with different seeds should give different per-batch k values
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 8.7407,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)).with_aabb(
                crate::geometry::Aabb::new(
                    Vec3::new(-8.7407, -8.7407, -8.7407),
                    Vec3::new(8.7407, 8.7407, 8.7407),
                ),
            ),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];
        let mut mat = Material::new("HEU", 294.0);
        mat.add_nuclide(0.048, 0);
        let materials = vec![mat];
        let xs = ConstantXs {
            xs: vec![MicroXs {
                total: 7.0,
                elastic: 4.0,
                inelastic: 0.0,
                n2n: 0.0,
                n3n: 0.0,
                fission: 1.2,
                capture: 0.1,
                nu_bar: 2.43,
                delayed_nu_bar: 0.0,
                awr: 235.0,
            }],
        };

        let config0 = SimConfig {
            batches: 5,
            inactive: 1,
            particles_per_batch: 500,
            seed: 0,
            auto_inactive: None,
            verbose: false,
            parallel: false,
            tallies: Default::default(),
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        };
        let config1 = SimConfig {
            batches: 5,
            inactive: 1,
            particles_per_batch: 500,
            seed: 1,
            auto_inactive: None,
            verbose: false,
            parallel: false,
            tallies: Default::default(),
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        };

        let (r0, _) = run_eigenvalue(&config0, &surfaces, &cells, &materials, &xs);
        let (r1, _) = run_eigenvalue(&config1, &surfaces, &cells, &materials, &xs);

        // Per-batch k_eff values should differ (stochastic independence)
        let k0: Vec<f64> = r0.iter().map(|r| r.k_eff).collect();
        let k1: Vec<f64> = r1.iter().map(|r| r.k_eff).collect();
        assert_ne!(
            k0, k1,
            "Different seeds must produce different batch sequences"
        );
    }

    /// Two-material low-contrast test geometry that auto-selects
    /// delta tracking, with both materials fissile so the
    /// track-length estimator has fission events to score.
    fn delta_tracking_two_material_problem() -> (
        Vec<Surface>,
        Vec<Cell>,
        Vec<Material>,
        ConstantXs,
    ) {
        let surfaces = vec![
            Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 5.0,
                bc: BoundaryCondition::Transmission,
            },
            Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 8.7407,
                bc: BoundaryCondition::Vacuum,
            },
        ];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)).with_aabb(
                crate::geometry::Aabb::new(
                    Vec3::new(-5.0, -5.0, -5.0),
                    Vec3::new(5.0, 5.0, 5.0),
                ),
            ),
            Cell::new(
                CellId(1),
                cell::intersect_all(vec![cell::outside(0), cell::inside(1)]),
                CellFill::Material(1),
            )
            .with_aabb(crate::geometry::Aabb::new(
                Vec3::new(-8.7407, -8.7407, -8.7407),
                Vec3::new(8.7407, 8.7407, 8.7407),
            )),
            Cell::new(CellId(2), cell::outside(1), CellFill::Void),
        ];
        let mut mat_inner = Material::new("inner", 294.0);
        mat_inner.add_nuclide(0.04, 0);
        let mut mat_outer = Material::new("outer", 294.0);
        // Same nuclide index (0), different density → contrast 4×, in
        // delta-tracking band.
        mat_outer.add_nuclide(0.01, 0);
        let materials = vec![mat_inner, mat_outer];
        let xs = ConstantXs {
            xs: vec![MicroXs {
                total: 7.0,
                elastic: 4.0,
                inelastic: 0.0,
                n2n: 0.0,
                n3n: 0.0,
                fission: 1.5,
                capture: 1.5,
                nu_bar: 2.43,
                delayed_nu_bar: 0.0,
                awr: 235.0,
            }],
        };
        (surfaces, cells, materials, xs)
    }

    /// Verify the geometry above does pick delta tracking.
    #[test]
    fn delta_tracking_two_material_problem_picks_delta() {
        let (_, cells, materials, xs) = delta_tracking_two_material_problem();
        let mode = detect_tracking_mode(&cells, &materials, &xs, false);
        assert!(
            matches!(mode, TrackingMode::Delta(_)),
            "low-contrast two-material problem must auto-select delta tracking",
        );
    }

    /// Mesh-flux tally must populate non-trivially under delta
    /// tracking (pre-fix the helper only fired on the surface-tracking
    /// path; voxels stayed at zero on PWR pin / heterogeneous
    /// geometries that auto-select Woodcock). Sanity check: total
    /// flux summed over voxels equals the total tracked path length
    /// times the source weight, modulo whatever path lies outside the
    /// mesh (the test geometry's source sits inside the mesh, so the
    /// equality is approximate but tight).
    #[test]
    fn delta_tracking_mesh_flux_populates() {
        let (surfaces, cells, materials, xs) = delta_tracking_two_material_problem();
        let mesh = crate::transport::tally::MeshFluxTally::new(
            [-5.0, -5.0, -5.0],
            [2.5, 2.5, 2.5],
            [4, 4, 4],
        );
        let mut tallies = Tallies::default();
        tallies.mesh_flux = Some(mesh);

        let config = SimConfig {
            batches: 8,
            inactive: 2,
            particles_per_batch: 500,
            seed: 11,
            auto_inactive: None,
            verbose: false,
            parallel: false,
            tallies,
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        };
        let (results, _) = run_eigenvalue(&config, &surfaces, &cells, &materials, &xs);

        let total: f64 = results
            .iter()
            .filter(|r| r.active)
            .map(|r| r.mesh_flux.iter().sum::<f64>())
            .sum();
        assert!(
            total > 0.0,
            "mesh flux is zero under delta tracking — tally helper isn't wired",
        );

        // At least half the voxels should have non-zero flux given a
        // 5 cm radius source inside a 10 cm cube mesh and 6 active
        // batches × 500 particles. (Pre-fix this would be 0 voxels
        // populated, so the threshold is forgiving but conclusive.)
        let mut populated = 0;
        for r in results.iter().filter(|r| r.active) {
            for &v in &r.mesh_flux {
                if v > 0.0 {
                    populated += 1;
                    break;
                }
            }
        }
        assert!(populated > 0, "no active batch deposited mesh flux");
    }

    /// Surface-current tally must score boundary crossings under delta
    /// tracking. Vacuum / reflective boundaries hit before the
    /// Woodcock collision distance are real crossings; pre-fix the
    /// surface-current helper was silently no-op on the delta path.
    #[test]
    fn delta_tracking_surface_currents_populate() {
        let (surfaces, cells, materials, xs) = delta_tracking_two_material_problem();
        // Tally the inner sphere (idx 0, the material interface) and
        // the outer sphere (idx 1, the vacuum boundary). Both are real
        // crossings under delta tracking even though the inner one is
        // transmission.
        let surf_tally = crate::transport::tally::SurfaceCurrentTally::new(vec![0, 1]);
        let mut tallies = Tallies::default();
        tallies.surface_current = Some(surf_tally);

        let config = SimConfig {
            batches: 8,
            inactive: 2,
            particles_per_batch: 500,
            seed: 13,
            auto_inactive: None,
            verbose: false,
            parallel: false,
            tallies,
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        };
        let (results, _) = run_eigenvalue(&config, &surfaces, &cells, &materials, &xs);

        let mut total_pos = 0.0_f64;
        let mut total_neg = 0.0_f64;
        for r in results.iter().filter(|r| r.active) {
            total_pos += r.surface_current_pos.iter().sum::<f64>();
            total_neg += r.surface_current_neg.iter().sum::<f64>();
        }
        assert!(
            total_pos + total_neg > 0.0,
            "surface currents are zero under delta tracking — pos+neg = {} + {}",
            total_pos,
            total_neg,
        );
    }

    /// k_track (track-length k-eff under delta tracking, Sutton-Brown
    /// form) should agree with k_eff (collision estimator) within MC
    /// noise. Pre-fix `k_track` was identically 0 under delta
    /// tracking — this test would have caught that.
    #[test]
    fn delta_tracking_k_track_matches_k_eff() {
        let (surfaces, cells, materials, xs) = delta_tracking_two_material_problem();
        let config = SimConfig {
            batches: 30,
            inactive: 5,
            particles_per_batch: 1000,
            seed: 7,
            auto_inactive: None,
            verbose: false,
            parallel: false,
            tallies: Default::default(),
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        };
        let (results, _) = run_eigenvalue(&config, &surfaces, &cells, &materials, &xs);

        let active: Vec<&BatchResult> = results.iter().filter(|r| r.active).collect();
        assert!(!active.is_empty(), "no active batches");
        let n = active.len() as f64;
        let k_eff_mean: f64 = active.iter().map(|r| r.k_eff).sum::<f64>() / n;
        let k_track_mean: f64 = active.iter().map(|r| r.k_track).sum::<f64>() / n;

        // k_track must not be identically zero — that was the bug.
        assert!(k_track_mean > 1e-3, "k_track is zero under delta tracking: {k_track_mean}");

        // Within combined MC noise of k_eff (couple-percent at 25
        // active batches × 1k particles).
        let rel = (k_track_mean - k_eff_mean).abs() / k_eff_mean.max(1e-30);
        assert!(
            rel < 0.05,
            "k_track {k_track_mean:.5} vs k_eff {k_eff_mean:.5} disagree by {:.2}% (>5%)",
            rel * 100.0,
        );
    }
}
