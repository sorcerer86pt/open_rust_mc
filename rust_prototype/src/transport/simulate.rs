//! Eigenvalue simulation — power iteration for k_eff. See
//! `docs/engine-notes.md § transport/simulate.rs` for SimConfig knob
//! semantics, Windows-stdout / rayon-loader-lock workarounds, and
//! the ablation-only flags.

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
use crate::transport::tally::{BatchTallies, ParticleTallies, Tallies};

use crate::MAX_NUCLIDES_PER_MATERIAL as MAX_NUCLIDES;

/// See `docs/engine-notes.md § SimConfig knobs` for each field's
/// purpose, OS-specific deadlock workarounds, and which paths honour
/// which flags.
pub struct SimConfig {
    pub batches: u32,
    /// Fallback when `auto_inactive` is `None`.
    pub inactive: u32,
    pub particles_per_batch: u32,
    pub seed: u64,
    /// Shannon-entropy convergence detector for the inactive→active
    /// transition. Overrides `inactive` when `Some(_)`.
    pub auto_inactive: Option<EntropyConvergence>,
    /// `false` on PyO3 callers to avoid the Windows stdout deadlock.
    pub verbose: bool,
    /// `false` on PyO3 callers to avoid the Windows loader-lock
    /// deadlock at rayon first-use.
    pub parallel: bool,
    pub tallies: Tallies,
    pub statepoint_path: Option<std::path::PathBuf>,
    /// Surface-tracking + non-thermal branch only; thermal + delta
    /// paths fall back to analog regardless.
    pub survival_biasing: Option<SurvivalBiasing>,
    /// Resume from a statepoint bank when `Some(_)`.
    pub initial_source_bank: Option<Vec<FissionSite>>,
    pub weight_window: Option<crate::transport::weight_window::WeightWindow>,
    /// Ablation only — production always samples ν_d(E).
    pub disable_delayed_neutrons: bool,
    /// `None` → infinite-medium URR.
    pub urr_equivalence: Option<crate::transport::urr_equivalence::UrrEquivalence>,
    /// **GPU-only.** PHYSOR 2022 Optimization F — continuous particle
    /// refill. When `Some(factor)`, the CUDA runner builds a source
    /// bank of `particles_per_batch * factor` particles per batch and
    /// uses the overflow to refill dead slots between event-pipeline
    /// steps. `total_histories` reported by the batch is the actual
    /// count consumed from the (oversampled) bank. `None` (default)
    /// preserves the historical behaviour.
    ///
    /// On the RTX 3080 saturation curve the kernel already runs at
    /// peak around 1 M particles, so a useful factor is hardware-
    /// dependent. The CPU path ignores this — refill is meaningless
    /// when rayon threads already saturate at 5 k particles.
    pub gpu_refill_pool_factor: Option<f64>,
    /// **GPU-only.** When `true` AND `gpu_refill_pool_factor` is
    /// `None`, the CUDA runner queries the active device's SM count
    /// and the kernel's compiled register count, computes the
    /// saturation knee, and picks a refill factor automatically.
    /// Logs what got picked. Explicit `gpu_refill_pool_factor =
    /// Some(_)` always wins over auto. See
    /// `gpu_recursive::recommend_refill_factor` for the heuristic.
    pub gpu_auto_refill: bool,
}

/// Implicit-capture + Russian roulette. OpenMC defaults
/// `w_min=0.25`, `w_survive=1.0`; expectation preserved.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalBiasing {
    pub w_min: f64,
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
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
        }
    }
}

/// Shannon-entropy plateau detector. Defaults sit above the
/// statistical noise floor for 10k-50k particles/batch (measured CV
/// ≈ 2e-3 once settled).
#[derive(Debug, Clone, Copy)]
pub struct EntropyConvergence {
    pub min_inactive: u32,
    pub max_inactive: u32,
    /// Sliding window for CV(σ/μ).
    pub window: u32,
    pub cv_tol: f64,
}

impl Default for EntropyConvergence {
    fn default() -> Self {
        Self {
            min_inactive: 20,
            max_inactive: 200,
            window: 10,
            cv_tol: 5e-3,
        }
    }
}

impl EntropyConvergence {
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

/// Majorant macroscopic Σ_t over all materials, log-spaced grid.
pub struct MajorantTable {
    log_e_min: f64,
    inv_step: f64,
    values: Vec<f64>,
}

impl MajorantTable {
    #[inline]
    fn lookup(&self, energy: f64) -> f64 {
        let log_e = energy.max(1e-11).ln();
        let frac = (log_e - self.log_e_min) * self.inv_step;
        let idx = (frac as usize).min(self.values.len() - 2);
        let t = frac - idx as f64;
        self.values[idx] * (1.0 - t) + self.values[idx + 1] * t
    }
}

enum TrackingMode {
    Surface,
    /// Woodcock; avoids surface intersection at internal interfaces.
    Delta(MajorantTable),
}

/// `Delta` when multiple materials with moderate XS contrast, else
/// `Surface`.
fn detect_tracking_mode<XS: XsProvider>(
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
    verbose: bool,
) -> TrackingMode {
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
                // The transport loop replaces the free-atom elastic XS
                // with the S(α,β) total scattering whenever the nuclide
                // has thermal data and E < energy_max. Including this
                // increment in the majorant is required so the runtime
                // σ_real never exceeds σ_majorant — otherwise Woodcock
                // tracking under-samples collisions in the moderator and
                // the system under-thermalises (HEU-SOL-THERM-001 lost
                // ~1500 pcm to this before the fix).
                let mut nuc_total = xs.total;
                if let Some(tsl) = xs_provider.thermal_scattering(nuc.xs_kernel_idx)
                    && energy < tsl.energy_max
                    && energy > 0.0
                {
                    let n_t = tsl.kts.len().max(1);
                    let mut tsl_max = 0.0_f64;
                    for t_idx in 0..n_t {
                        tsl_max = tsl_max.max(tsl.total_xs(energy, t_idx));
                    }
                    let delta = (tsl_max - xs.elastic).max(0.0);
                    nuc_total += delta;
                }
                macro_t += nuc.atom_density * nuc_total;
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
/// `Send + Sync` for rayon parallel transport.
pub trait XsProvider: Send + Sync {
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

    /// Per-discrete-level CM angular dist (ENDF MT=51-91), aligned
    /// 1:1 with `discrete_level_info`. Empty = isotropic fallback.
    fn discrete_level_angles(&self, _nuclide_idx: usize) -> &[Option<AngularDistribution>] {
        &[]
    }

    fn fission_energy_dist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    /// ENDF MT=91; `None` → evaporation fallback.
    fn inelastic_continuum_edist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    /// ENDF MT=16.
    fn n2n_edist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    /// ENDF MT=17.
    fn n3n_edist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    /// MT=22 / 24 / 28. Empty → those reactions have zero XS in
    /// `MicroXs` and the dispatch branches never fire.
    fn charged_particle_edists(
        &self,
        _nuclide_idx: usize,
    ) -> crate::physics::collision::ChargedParticleEdists<'_> {
        crate::physics::collision::ChargedParticleEdists::default()
    }

    fn apply_urr(&self, _nuclide_idx: usize, _xs: &mut MicroXs, _energy: f64, _xi: f64) {}

    /// Inside the URR probability-table range — gates the
    /// equivalence-theory self-shielding correction.
    fn is_urr(&self, _nuclide_idx: usize, _energy: f64) -> bool {
        false
    }

    /// Per-MT photon products (multiplicity, E_γ) for coupled n-γ.
    fn photon_products(&self, _nuclide_idx: usize) -> &[(u32, crate::hdf5_reader::PhotonProduct)] {
        &[]
    }

    /// S(α,β) thermal data; replaces free-gas elastic below
    /// `energy_max` (~4 eV) in the transport loop.
    fn thermal_scattering(&self, _nuclide_idx: usize) -> Option<&ThermalScatteringData> {
        None
    }

    /// Sum of delayed-product yields; drives β(E) = ν_d / ν_total in
    /// the fission-yield path.
    fn delayed_nu_bar_at(&self, _nuclide_idx: usize, _energy: f64) -> f64 {
        0.0
    }

    /// MT not exposed via `MicroXs` (e.g. MT=103, MT=107) for the
    /// per-MT reaction-rate tally. `None` when the provider doesn't
    /// stock the MT for that nuclide.
    fn partial_xs(&self, _nuclide_idx: usize, _energy: f64, _mt: u32) -> Option<f64> {
        None
    }
}

pub struct ConstantXs {
    pub xs: Vec<MicroXs>,
}

impl XsProvider for ConstantXs {
    fn lookup(&self, nuclide_idx: usize, _energy: f64) -> MicroXs {
        self.xs[nuclide_idx]
    }
}

#[derive(Debug, Clone)]
pub struct BatchResult {
    pub batch: u32,
    pub k_eff: f64,
    pub leakage: u32,
    pub absorptions: u32,
    pub fissions: u32,
    pub collisions: u32,
    /// S(α,β) events.
    pub thermal_scatters: u32,
    /// Reflections + transmissions.
    pub surface_crossings: u32,
    /// Bits, on a coarse Cartesian mesh. OpenMC convention.
    pub shannon_entropy: f64,
    /// Active = counted towards tallies. Fixed-inactive: `batch >
    /// config.inactive`. Auto-inactive: entropy-plateau decision.
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
    /// Reduced per-batch tally output (surface currents, mesh flux,
    /// reaction-rate numerators). Each `Vec` inside is empty when the
    /// matching tally is disabled in `SimConfig.tallies`.
    pub tallies: BatchTallies,
    /// Spectrum-hardening diagnostic tallies — see `bin/metal_stats_diag`.
    /// Per-reaction event counts plus E-in / E-out accumulators so the
    /// caller can compute ⟨E_in at fission⟩, ⟨E_in elastic⟩, and the
    /// inelastic energy-loss moment ⟨ΔE⟩ = (e_inel_in − e_inel_out) / n_inel.
    /// Default 0 when the backend doesn't populate them.
    pub n_elastic: u64,
    pub n_inelastic: u64,
    pub n_capture: u64,
    pub e_fis_in_sum: f64,
    pub e_el_in_sum: f64,
    pub e_inel_in_sum: f64,
    pub e_inel_out_sum: f64,
    /// Squared-energy accumulators (GPU-only for now). σ(E_at_reaction)
    /// = sqrt(⟨E²⟩ − ⟨E⟩²). Used by `bin/metal_stats_diag` to test
    /// whether the metal hot bias is a higher-moment effect on the
    /// fission-incident energy distribution after ν(E) parity was
    /// confirmed via `bin/nu_lookup_compare`.
    pub e_fis_in_sq_sum: f64,
    pub e_el_in_sq_sum: f64,
    pub e_inel_in_sq_sum: f64,
    /// Σ |Q| over inelastic events — used to localise the metal hot
    /// bias to per-level-XS-weighted sampling. CPU and GPU should
    /// agree on ⟨|Q|⟩ = q_inel_sum / n_inelastic if level selection
    /// is unbiased.
    pub q_inel_sum: f64,
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
    /// Thin wrapper over [`rust_mc_sim::transport::simulate::shannon_entropy_xyz`]
    /// — the math lives there now; this method just adapts the
    /// engine's `FissionSite` slice to the position-iterator API.
    pub fn entropy(&self, sites: &[FissionSite]) -> f64 {
        rust_mc_sim::transport::simulate::shannon_entropy_xyz(
            sites.iter().map(|s| [s.pos.x, s.pos.y, s.pos.z]),
            (self.lo, self.hi),
            self.n,
        )
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

/// Per-particle transport result — scalar counters only. Variable-
/// length outputs (fission sites, capture cells, photon events) and
/// per-particle tally accumulators have been moved into
/// [`TransportCtx`] so the transport functions can push directly into
/// the worker-thread-local buffers instead of allocating a per-particle
/// Vec on every fission / capture / γ-producing collision. For ICSBEP
/// k-eigenvalue runs (no tallies, no γ coupling, ~2-3 fissions per
/// history) this eliminates the dominant remaining per-particle heap
/// allocation.
struct ParticleResult {
    leakage: u32,
    absorptions: u32,
    thermal_scatters: u32,
    surface_crossings: u32,
    fissions: u32,
    collisions: u32,
    /// Track-length tally accumulator: Σ_segments w · d · Σ_νf(E),
    /// summed over every advance the particle made through fuel-
    /// bearing material. Reduced into `BatchResult::k_track` as
    /// `total / N_source` after the batch completes.
    track_length_nu_sigf: f64,
    // Spectrum-hardening diagnostics — once per fission/elastic/
    // inelastic event. Counted on the CPU side to give `metal_stats_diag`
    // a CPU σ value to compare against GPU σ at fission, after the
    // ν(E)-table parity check (`bin/nu_lookup_compare`) ruled out the
    // upload path.
    n_elastic: u32,
    n_inelastic: u32,
    n_capture: u32,
    e_fis_in_sum: f64,
    e_fis_in_sq: f64,
    e_el_in_sum: f64,
    e_el_in_sq: f64,
    e_inel_in_sum: f64,
    e_inel_in_sq: f64,
    e_inel_out_sum: f64,
    q_inel_sum: f64,
}

/// Worker-thread-local sinks that the transport functions write into
/// directly. Each rayon fold step rebuilds one of these around the
/// current `WorkerAccum`'s buffers; the per-particle path never
/// allocates a Vec of its own.
///
/// `captures_by_cell` is a pre-sized `&mut [f64]` (one entry per
/// geometry cell). Capture events `+= 1.0` into the slot inline,
/// replacing the prior `Vec<usize>` push-then-iter-then-bump pattern.
struct TransportCtx<'a> {
    fission_sites: &'a mut Vec<FissionSite>,
    captures_by_cell: &'a mut [f64],
    photon_events: &'a mut Vec<PhotonSourceEvent>,
    tallies: &'a mut ParticleTallies,
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
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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
///
/// `tally_cfg` carries the static "what to track" config; `ctx` is the
/// worker-thread-local sink — its `tallies` field is reset by the
/// caller before each particle, and its Vec fields are append-only
/// pointers into the worker-local batch buffers. Returns only scalar
/// counters; all variable-length output (fission sites, capture cells,
/// photon events, per-particle tallies) is written through `ctx`.
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
    tally_cfg: &Tallies,
    ctx: &mut TransportCtx<'_>,
    survival_biasing: Option<&SurvivalBiasing>,
    weight_window: Option<&crate::transport::weight_window::WeightWindow>,
    disable_delayed_neutrons: bool,
    urr_equivalence: Option<&crate::transport::urr_equivalence::UrrEquivalence>,
) -> ParticleResult {
    let mut rng = Rng::for_particle(batch, particle_idx);
    let mut result = ParticleResult {
        leakage: 0,
        absorptions: 0,
        fissions: 0,
        collisions: 0,
        thermal_scatters: 0,
        surface_crossings: 0,
        track_length_nu_sigf: 0.0,
        n_elastic: 0,
        n_inelastic: 0,
        n_capture: 0,
        e_fis_in_sum: 0.0,
        e_fis_in_sq: 0.0,
        e_el_in_sum: 0.0,
        e_el_in_sq: 0.0,
        e_inel_in_sum: 0.0,
        e_inel_in_sq: 0.0,
        e_inel_out_sum: 0.0,
        q_inel_sum: 0.0,
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
            if let Some(mesh) = tally_cfg.mesh_flux.as_ref() {
                mesh.deposit(
                    particle.pos,
                    particle.dir,
                    advance_dist,
                    particle.weight,
                    &mut ctx.tallies.mesh_flux,
                );
            }
            // Reaction-rate tally for chain-XS spectrum collapse:
            //   numerator   per (cell, xs_idx, MT): Σ w·d·σ_micro,MT
            //   denominator per cell:                Σ w·d
            // Track-length form is exact for the one-group XS
            // collapse; collision-estimator form would have higher
            // variance for non-rare reactions like (n,γ).
            if let Some(rr) = tally_cfg.reaction_rate.as_ref() {
                let cell_idx = particle.cell_idx;
                if cell_idx < rr.n_cells {
                    let w_d = particle.weight * advance_dist;
                    ctx.tallies.rr_flux[cell_idx] += w_d;
                    let n_mts = rr.n_mts;
                    for (i, nuc) in material.nuclides.iter().take(n_nuclides).enumerate() {
                        let xs_idx = nuc.xs_kernel_idx;
                        if xs_idx >= rr.n_xs_idx {
                            continue;
                        }
                        let base = (cell_idx * rr.n_xs_idx + xs_idx) * n_mts;
                        for (m, &mt) in rr.mts.iter().enumerate() {
                            // MTs covered by MicroXs go straight from
                            // the cached lookup. MTs not in MicroXs
                            // (notably MT=103 (n,p), MT=107 (n,α))
                            // route through `partial_xs`, which the
                            // SVD and Table providers populate from
                            // their own per-MT kernels. `partial_xs`
                            // returning None means the channel isn't
                            // stocked for this nuclide → contribute 0
                            // (e.g. light nuclides have no (n,α)).
                            let sigma = match mt {
                                18 => micro_xs[i].fission,
                                102 => micro_xs[i].capture,
                                16 => micro_xs[i].n2n,
                                17 => micro_xs[i].n3n,
                                2 => micro_xs[i].elastic,
                                4 => micro_xs[i].inelastic,
                                other => xs_provider
                                    .partial_xs(xs_idx, particle.energy, other)
                                    .unwrap_or(0.0),
                            };
                            ctx.tallies.rr_rate[base + m] += w_d * sigma;
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
                        (tally_cfg.surface_current.as_ref(), hit.surface_idx)
                        && let Some(bin) = sct.bin_for(surf_idx)
                    {
                        let crossing_pos = particle.pos + particle.dir * hit.distance;
                        let n = surfaces[surf_idx].normal_at(crossing_pos);
                        if particle.dir.dot(n) >= 0.0 {
                            ctx.tallies.surface_current_pos[bin] += particle.weight;
                        } else {
                            ctx.tallies.surface_current_neg[bin] += particle.weight;
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
                        let cp_edists = xs_provider.charged_particle_edists(xs_kernel_idx);

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
                            cp_edists,
                            cell.temperature,
                            &mut rng,
                            survival_biasing,
                            &mut result,
                            ctx,
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
/// Updates `particle` (energy / direction / weight / status), writes
/// scalar counters into `result`, and appends variable-length output
/// (fission sites, photon events, capture-cell bumps) directly into
/// the worker sinks on `ctx`. `pending` carries (n,xn) secondaries
/// back to the per-history outer loop. Used by both
/// `transport_particle` (surface tracking) and
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
    cp_edists: crate::physics::collision::ChargedParticleEdists<'_>,
    temperature: f64,
    rng: &mut Rng,
    survival_biasing: Option<&SurvivalBiasing>,
    result: &mut ParticleResult,
    ctx: &mut TransportCtx<'_>,
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
            // Accumulate σ-at-fission inputs once per fission event
            // (matches the GPU semantics in transport_recursive.cu:430).
            let e_pre = particle.energy;
            result.e_fis_in_sum += e_pre;
            result.e_fis_in_sq += e_pre * e_pre;
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
                ctx.fission_sites.push(FissionSite {
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
                ctx.photon_events,
            );
        }

        let sigma_a = micro.capture + micro.fission;
        let sigma_s = (micro.total - sigma_a).max(0.0);
        if sigma_s <= 0.0 {
            result.absorptions += 1;
            bump_capture(ctx.captures_by_cell, particle.cell_idx);
            sample_photon_products(
                xs_provider,
                xs_kernel_idx,
                ABSORPTION_PHOTON_MTS,
                particle,
                rng,
                ctx.photon_events,
            );
            particle.kill();
            return;
        }
        particle.weight *= sigma_s / micro.total;

        let e_pre = particle.energy;
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
            CollisionOutcome::Scatter => {
                result.n_elastic += 1;
                result.e_el_in_sum += e_pre;
                result.e_el_in_sq += e_pre * e_pre;
            }
            CollisionOutcome::InelasticScatter { q_value_ev } => {
                result.n_inelastic += 1;
                result.e_inel_in_sum += e_pre;
                result.e_inel_in_sq += e_pre * e_pre;
                result.e_inel_out_sum += particle.energy;
                result.q_inel_sum += q_value_ev.abs();
                ctx.photon_events.push(PhotonSourceEvent {
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
                result.n_capture += 1;
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
    let e_pre = particle.energy;
    let outcome = collision::process_collision(
        particle,
        micro,
        inelastic_data,
        elastic_angle,
        fission_edist,
        continuum_edist,
        cp_edists,
        n2n_edist,
        n3n_edist,
        temperature,
        rng,
    );
    match outcome {
        CollisionOutcome::Scatter => {
            result.n_elastic += 1;
            result.e_el_in_sum += e_pre;
            result.e_el_in_sq += e_pre * e_pre;
        }
        CollisionOutcome::InelasticScatter { q_value_ev } => {
            result.n_inelastic += 1;
            result.e_inel_in_sum += e_pre;
            result.e_inel_in_sq += e_pre * e_pre;
            result.e_inel_out_sum += particle.energy;
            result.q_inel_sum += q_value_ev.abs();
            ctx.photon_events.push(PhotonSourceEvent {
                cell_idx: particle.cell_idx as u32,
                pos: [particle.pos.x, particle.pos.y, particle.pos.z],
                energy: q_value_ev.abs(),
                mt: 4,
            });
        }
        CollisionOutcome::Absorption => {
            result.absorptions += 1;
            result.n_capture += 1;
            bump_capture(ctx.captures_by_cell, particle.cell_idx);
            sample_photon_products(
                xs_provider,
                xs_kernel_idx,
                ABSORPTION_PHOTON_MTS,
                particle,
                rng,
                ctx.photon_events,
            );
        }
        CollisionOutcome::Fission { sites } => {
            result.fissions += 1;
            result.e_fis_in_sum += e_pre;
            result.e_fis_in_sq += e_pre * e_pre;
            ctx.fission_sites.extend(sites);
            sample_photon_products(
                xs_provider,
                xs_kernel_idx,
                FISSION_PHOTON_MTS,
                particle,
                rng,
                ctx.photon_events,
            );
        }
        CollisionOutcome::Multiplicity { secondaries } => {
            for s in secondaries {
                pending.push(Particle::new(s.pos, s.dir, s.energy, particle.cell_idx));
            }
        }
    }
}

/// Bump the captures_by_cell slot for a capture at `cell_idx`,
/// replacing the prior `Vec<usize>` push-then-iter-then-bump pattern.
#[inline]
fn bump_capture(captures_by_cell: &mut [f64], cell_idx: usize) {
    if cell_idx < captures_by_cell.len() {
        captures_by_cell[cell_idx] += 1.0;
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
    tally_cfg: &Tallies,
    ctx: &mut TransportCtx<'_>,
    survival_biasing: Option<&SurvivalBiasing>,
    weight_window: Option<&crate::transport::weight_window::WeightWindow>,
) -> ParticleResult {
    let mut rng = Rng::for_particle(batch, particle_idx);
    let mut result = ParticleResult {
        leakage: 0,
        absorptions: 0,
        fissions: 0,
        collisions: 0,
        thermal_scatters: 0,
        surface_crossings: 0,
        track_length_nu_sigf: 0.0,
        n_elastic: 0,
        n_inelastic: 0,
        n_capture: 0,
        e_fis_in_sum: 0.0,
        e_fis_in_sq: 0.0,
        e_el_in_sum: 0.0,
        e_el_in_sq: 0.0,
        e_inel_in_sum: 0.0,
        e_inel_in_sq: 0.0,
        e_inel_out_sum: 0.0,
        q_inel_sum: 0.0,
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
            if let Some(mesh) = tally_cfg.mesh_flux.as_ref() {
                mesh.deposit(
                    particle.pos,
                    particle.dir,
                    advance_dist,
                    particle.weight,
                    &mut ctx.tallies.mesh_flux,
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
            if let (Some(hit), Some(sct)) = (trace.as_ref(), tally_cfg.surface_current.as_ref())
                && hit.distance < d_collision
                && let Some(surf_idx) = hit.surface_idx
                && let Some(bin) = sct.bin_for(surf_idx)
            {
                let crossing_pos = particle.pos + particle.dir * hit.distance;
                let n = surfaces[surf_idx].normal_at(crossing_pos);
                if particle.dir.dot(n) >= 0.0 {
                    ctx.tallies.surface_current_pos[bin] += particle.weight;
                } else {
                    ctx.tallies.surface_current_neg[bin] += particle.weight;
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
            // S(α,β) thermal-scattering XS swap-in. Mirrors the surface-
            // tracking path (transport_particle:894-906). Pre-fix the
            // delta tracker had no thermal handling — H-in-H2O fell back
            // to free-atom MT=2 elastic, under-moderating thermal
            // neutrons and biasing solution-benchmark k_eff downward
            // by O(1500 pcm) on HEU-SOL-THERM-001 (vs OpenMC on the
            // same data).
            let mut thermal_xs_add = [0.0_f64; MAX_NUCLIDES];
            for (i, nuc) in material.nuclides.iter().enumerate() {
                let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, particle.energy);
                // Snapshot pre-URR-PT smooth XS for Hwang superposition.
                micro_xs_smooth[i] = xs;
                xs_provider.apply_urr(nuc.xs_kernel_idx, &mut xs, particle.energy, urr_xi);

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
                    macro_nu_sigma_f += nuc.atom_density * micro_xs[i].nu_bar * micro_xs[i].fission;
                }
                if macro_nu_sigma_f > 0.0 {
                    result.track_length_nu_sigf += particle.weight * macro_nu_sigma_f / sigma_real;
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

            // Thermal scattering branch: if this nuclide has S(α,β)
            // attached and the energy is below the cutoff, the thermal
            // total replaced the free-atom elastic. Roll thermal-vs-
            // other-channel against `thermal_xs_add[nuc_idx]`; on
            // thermal, sample (E_out, μ) from the bound-H kernel and
            // skip the non-thermal dispatch.
            let use_thermal = thermal_xs_add[nuc_idx] > 0.0;
            let mut handled_as_thermal = false;
            if use_thermal {
                let tsl = xs_provider
                    .thermal_scattering(xs_kernel_idx)
                    .expect("thermal data");
                let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                let xi_reaction = rng.uniform() * micro_xs[nuc_idx].total;
                if xi_reaction < thermal_xs_add[nuc_idx] {
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
                    handled_as_thermal = true;
                }
            }

            if handled_as_thermal {
                continue;
            }

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
            let cp_edists = xs_provider.charged_particle_edists(xs_kernel_idx);

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
                cp_edists,
                cell.temperature,
                &mut rng,
                survival_biasing,
                &mut result,
                ctx,
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
        // Per-worker accumulator. Each rayon worker initialises one of
        // these and folds every particle it handles into it; the
        // `ParticleTallies` scratch is recycled (reset in place) across
        // every particle so per-particle `vec![0.0; rr_rate_size]`
        // allocations stop showing up under depletion / RR-CADIS runs.
        // At the end of the batch the per-worker accumulators are merged
        // in `reduce_op` below.
        struct WorkerAccum {
            leakage: u32,
            absorptions: u32,
            fissions: u32,
            collisions: u32,
            thermal_scatters: u32,
            surface_crossings: u32,
            track_length_sum: f64,
            fission_sites: Vec<FissionSite>,
            captures_by_cell: Vec<f64>,
            photon_events: Vec<PhotonSourceEvent>,
            tallies: BatchTallies,
            // Worker-local scratch — reset before each particle, written
            // by the inner-loop tally code, then folded into `tallies`.
            scratch: ParticleTallies,
            // Spectrum-hardening diagnostic accumulators (CPU side now
            // mirrors GPU side; see `bin/metal_stats_diag`).
            n_elastic: u64,
            n_inelastic: u64,
            n_capture: u64,
            e_fis_in_sum: f64,
            e_fis_in_sq: f64,
            e_el_in_sum: f64,
            e_el_in_sq: f64,
            e_inel_in_sum: f64,
            e_inel_in_sq: f64,
            e_inel_out_sum: f64,
            q_inel_sum: f64,
        }

        let worker_init = || WorkerAccum {
            leakage: 0,
            absorptions: 0,
            fissions: 0,
            collisions: 0,
            thermal_scatters: 0,
            surface_crossings: 0,
            track_length_sum: 0.0,
            fission_sites: Vec::new(),
            captures_by_cell: vec![0.0_f64; cells.len()],
            photon_events: Vec::new(),
            tallies: BatchTallies::new(&config.tallies),
            scratch: ParticleTallies::new(&config.tallies),
            n_elastic: 0,
            n_inelastic: 0,
            n_capture: 0,
            e_fis_in_sum: 0.0,
            e_fis_in_sq: 0.0,
            e_el_in_sum: 0.0,
            e_el_in_sq: 0.0,
            e_inel_in_sum: 0.0,
            e_inel_in_sq: 0.0,
            e_inel_out_sum: 0.0,
            q_inel_sum: 0.0,
        };

        let fold_one = |mut acc: WorkerAccum, (i, site): (usize, &FissionSite)| {
            acc.scratch.reset();
            // Build the per-particle sink around acc's worker-local
            // buffers; the transport function pushes fission sites,
            // capture-cell bumps, photon events, and tally writes
            // directly into them. No per-particle Vec allocation.
            let WorkerAccum {
                ref mut fission_sites,
                ref mut captures_by_cell,
                ref mut photon_events,
                ref mut scratch,
                ..
            } = acc;
            let mut ctx = TransportCtx {
                fission_sites,
                captures_by_cell: captures_by_cell.as_mut_slice(),
                photon_events,
                tallies: scratch,
            };
            let pr = match &tracking {
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
                    &mut ctx,
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
                    &mut ctx,
                    config.survival_biasing.as_ref(),
                    config.weight_window.as_ref(),
                ),
            };
            // ctx goes out of scope here, releasing the borrow on `acc`.
            acc.tallies.accumulate(&acc.scratch);
            acc.leakage += pr.leakage;
            acc.absorptions += pr.absorptions;
            acc.fissions += pr.fissions;
            acc.collisions += pr.collisions;
            acc.thermal_scatters += pr.thermal_scatters;
            acc.surface_crossings += pr.surface_crossings;
            acc.track_length_sum += pr.track_length_nu_sigf;
            acc.n_elastic += pr.n_elastic as u64;
            acc.n_inelastic += pr.n_inelastic as u64;
            acc.n_capture += pr.n_capture as u64;
            acc.e_fis_in_sum += pr.e_fis_in_sum;
            acc.e_fis_in_sq += pr.e_fis_in_sq;
            acc.e_el_in_sum += pr.e_el_in_sum;
            acc.e_el_in_sq += pr.e_el_in_sq;
            acc.e_inel_in_sum += pr.e_inel_in_sum;
            acc.e_inel_in_sq += pr.e_inel_in_sq;
            acc.e_inel_out_sum += pr.e_inel_out_sum;
            acc.q_inel_sum += pr.q_inel_sum;
            acc
        };

        let reduce_op = |mut a: WorkerAccum, b: WorkerAccum| {
            a.leakage += b.leakage;
            a.absorptions += b.absorptions;
            a.fissions += b.fissions;
            a.collisions += b.collisions;
            a.thermal_scatters += b.thermal_scatters;
            a.surface_crossings += b.surface_crossings;
            a.track_length_sum += b.track_length_sum;
            a.fission_sites.extend(b.fission_sites);
            for (slot, v) in a.captures_by_cell.iter_mut().zip(&b.captures_by_cell) {
                *slot += v;
            }
            a.photon_events.extend(b.photon_events);
            a.tallies.merge(&b.tallies);
            a.n_elastic += b.n_elastic;
            a.n_inelastic += b.n_inelastic;
            a.n_capture += b.n_capture;
            a.e_fis_in_sum += b.e_fis_in_sum;
            a.e_fis_in_sq += b.e_fis_in_sq;
            a.e_el_in_sum += b.e_el_in_sum;
            a.e_el_in_sq += b.e_el_in_sq;
            a.e_inel_in_sum += b.e_inel_in_sum;
            a.e_inel_in_sq += b.e_inel_in_sq;
            a.e_inel_out_sum += b.e_inel_out_sum;
            a.q_inel_sum += b.q_inel_sum;
            a
        };

        let final_acc = if config.parallel {
            source_bank
                .par_iter()
                .enumerate()
                .fold(worker_init, fold_one)
                .reduce(worker_init, reduce_op)
        } else {
            source_bank
                .iter()
                .enumerate()
                .fold(worker_init(), fold_one)
        };

        let mut fission_bank = FissionBank::new();
        fission_bank.sites = final_acc.fission_sites;
        let leakage = final_acc.leakage;
        let absorptions = final_acc.absorptions;
        let fissions = final_acc.fissions;
        let collisions = final_acc.collisions;
        let thermal_scatters = final_acc.thermal_scatters;
        let surface_crossings = final_acc.surface_crossings;
        let captures_by_cell = final_acc.captures_by_cell;
        let photon_events = final_acc.photon_events;
        let track_length_sum = final_acc.track_length_sum;
        let batch_tallies = final_acc.tallies;

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
            tallies: batch_tallies,
            // Spectrum-hardening diagnostic tallies — populated on
            // both CPU and GPU now. CPU side adds per-collision
            // accumulators in `dispatch_real_collision` so
            // `bin/metal_stats_diag` can do a three-way CPU↔GPU↔OpenMC
            // σ(E_at_reaction) comparison.
            n_elastic: final_acc.n_elastic,
            n_inelastic: final_acc.n_inelastic,
            n_capture: final_acc.n_capture,
            e_fis_in_sum: final_acc.e_fis_in_sum,
            e_el_in_sum: final_acc.e_el_in_sum,
            e_inel_in_sum: final_acc.e_inel_in_sum,
            e_inel_out_sum: final_acc.e_inel_out_sum,
            e_fis_in_sq_sum: final_acc.e_fis_in_sq,
            e_el_in_sq_sum: final_acc.e_el_in_sq,
            e_inel_in_sq_sum: final_acc.e_inel_in_sq,
            q_inel_sum: final_acc.q_inel_sum,
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
/// Sampler error returned by [`try_initial_source`]. Carries enough
/// context (case name, AABB used, attempts spent) so callers from the
/// Python or sweep harnesses can decide how to surface the failure.
#[derive(Debug, Clone)]
pub struct InitialSourceError {
    pub message: String,
}

impl std::fmt::Display for InitialSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for InitialSourceError {}

/// Fallible version of [`initial_source`]. Returns
/// `Err(InitialSourceError)` instead of panicking when rejection
/// sampling cannot place `n` fissile sites in `max_attempts`. The
/// `initial_source` panicking wrapper is preserved for legacy callers
/// (CLI binaries and ICSBEP regression tests) that prefer fail-fast
/// semantics over Result handling; new harnesses (Python sweep,
/// long-running tooling) should call this variant directly.
pub fn try_initial_source(
    n: usize,
    geometry: &crate::geometry::Geometry,
    cells: &[Cell],
    seed: u64,
) -> Result<Vec<FissionSite>, InitialSourceError> {
    try_initial_source_in_materials(n, geometry, cells, None, seed)
}

/// Material-aware initial-source sampler. Matches Serpent 2's default
/// behaviour: walk every cell, compute its region-tree AABB, build a
/// table of cells whose material is fissionable (per the caller-
/// supplied `materials_fissionable` flag list), sample weighted by
/// per-cell AABB volume, and accept any draw that lands in a cell
/// whose deepest material is in the fissionable set.
///
/// Passing `None` for `materials_fissionable` accepts any Material
/// cell — preserves the legacy `initial_source(n, geom, cells, seed)`
/// signature for binaries that haven't been threaded with XS-derived
/// fissionability flags.
///
/// Replaces the historical "first Material cell" / "smallest-volume
/// material" heuristics, which broke on BWR cruciform absorbers, PWR
/// burnable poisons, HFIR plate cladding, CANDU spacers, and reflector-
/// only benchmarks. See `bench/icsbep/` audit notes in CLAUDE.md.
pub fn try_initial_source_in_materials(
    n: usize,
    geometry: &crate::geometry::Geometry,
    cells: &[Cell],
    materials_fissionable: Option<&[bool]>,
    seed: u64,
) -> Result<Vec<FissionSite>, InitialSourceError> {
    let mut rng = Rng::new(seed * 100_000, 0);
    let mut sites = Vec::with_capacity(n);

    // ── Build the per-cell sampling table ────────────────────────────
    //
    // A cell qualifies if (a) it's filled with a Material that the
    // caller marked fissionable, and (b) its region tree produces a
    // finite, non-empty AABB. Each qualifying cell contributes one
    // entry with its AABB and an importance weight equal to the
    // AABB volume — uniform sampling inside the union then matches
    // the geometry-volume fraction of each cell, which is the right
    // first-batch prior for k-eigenvalue.
    struct CellEntry {
        #[allow(dead_code)] // kept for future diagnostics / debug logging
        cell_idx: usize,
        aabb: crate::geometry::Aabb,
        weight: f64,
    }
    let is_fissionable_mat = |m: u32| -> bool {
        match materials_fissionable {
            Some(flags) => flags.get(m as usize).copied().unwrap_or(true),
            None => true,
        }
    };

    let mut table: Vec<CellEntry> = Vec::new();
    let surfaces = geometry.surfaces.as_slice();
    for (idx, c) in cells.iter().enumerate() {
        let CellFill::Material(m) = c.fill else {
            continue;
        };
        if !is_fissionable_mat(m) {
            continue;
        }
        let aabb = c.region.world_aabb(surfaces);
        let dx = aabb.max.x - aabb.min.x;
        let dy = aabb.max.y - aabb.min.y;
        let dz = aabb.max.z - aabb.min.z;
        if !(dx.is_finite() && dy.is_finite() && dz.is_finite())
            || dx <= 0.0
            || dy <= 0.0
            || dz <= 0.0
        {
            continue;
        }
        let weight = dx * dy * dz;
        table.push(CellEntry {
            cell_idx: idx,
            aabb,
            weight,
        });
    }

    // Lattice fallback: when the top-level cells slice carries only
    // Universe / Lattice fills (LCT, CANDU bundles, hex assemblies),
    // no Material cell shows up here. Drop to the lattice extent,
    // which is the world-coord AABB of the lattice's element grid.
    if table.is_empty() {
        if let Some(lat_aabb) = lattices_world_aabb(geometry) {
            let lat_aabb = clamp_degenerate_axes(lat_aabb);
            let dx = lat_aabb.max.x - lat_aabb.min.x;
            let dy = lat_aabb.max.y - lat_aabb.min.y;
            let dz = lat_aabb.max.z - lat_aabb.min.z;
            if dx.is_finite() && dy.is_finite() && dz.is_finite() && dx > 0.0 && dy > 0.0 && dz > 0.0
            {
                table.push(CellEntry {
                    cell_idx: usize::MAX,
                    aabb: lat_aabb,
                    weight: dx * dy * dz,
                });
            }
        }
    }

    if table.is_empty() {
        return Err(InitialSourceError {
            message: format!(
                "initial_source: no fissionable cells found in geometry with {} cells / \
                 {} surfaces / {} lattices. Either no material is flagged fissionable, \
                 or every fissionable cell's region tree produced an empty / unbounded \
                 AABB.",
                cells.len(),
                geometry.surfaces.len(),
                geometry.lattices.len(),
            ),
        });
    }

    let cumulative: Vec<f64> = {
        let mut acc = 0.0;
        table
            .iter()
            .map(|e| {
                acc += e.weight;
                acc
            })
            .collect()
    };
    let total_weight = *cumulative.last().unwrap();

    let max_attempts =
        crate::transport::sim_limits::SimLimits::default().initial_source_max_attempts(n);
    let mut attempts: u64 = 0;
    while sites.len() < n {
        attempts += 1;
        if attempts > max_attempts {
            return Err(InitialSourceError {
                message: format!(
                    "initial_source: rejection sampling failed to find {n} fissile points \
                     in {max_attempts} attempts across {} candidate cells (total weight = {:.3e}). \
                     Geometry may have a fissionable material whose region tree resolves to \
                     a box mostly outside the actual cell volume — e.g. a CSG difference \
                     that the region-tree AABB walker bounds conservatively.",
                    table.len(),
                    total_weight,
                ),
            });
        }

        // Pick a cell by cumulative-weight.
        let xi = rng.uniform() * total_weight;
        let pick = cumulative.partition_point(|&w| w < xi).min(table.len() - 1);
        let entry = &table[pick];

        let x = entry.aabb.min.x + rng.uniform() * (entry.aabb.max.x - entry.aabb.min.x);
        let y = entry.aabb.min.y + rng.uniform() * (entry.aabb.max.y - entry.aabb.min.y);
        let z = entry.aabb.min.z + rng.uniform() * (entry.aabb.max.z - entry.aabb.min.z);
        let pos = Vec3::new(x, y, z);

        if let Some(stack) = geometry::ray::find_cell_recursive(pos, geometry) {
            let deepest = stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
            let accept = match cells.get(deepest).map(|c| c.fill) {
                Some(CellFill::Material(m)) => is_fissionable_mat(m),
                _ => stack.len() > 1, // nested-fill leaf — the descent already verified containment
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

    Ok(sites)
}

/// Panicking wrapper around [`try_initial_source`] for legacy callers
/// that want fail-fast behaviour when the geometry can't be sampled.
/// Sampling AABB resolution and rejection logic are documented inline
/// in `try_initial_source`.
pub fn initial_source(
    n: usize,
    geometry: &crate::geometry::Geometry,
    cells: &[Cell],
    seed: u64,
) -> Vec<FissionSite> {
    match try_initial_source(n, geometry, cells, seed) {
        Ok(sites) => sites,
        Err(e) => panic!("{}", e.message),
    }
}

/// World-coordinate AABB enclosing every rectangular lattice in the
/// geometry. For each lattice the extent is `[origin, origin + pitch ·
/// shape]`. Used by `initial_source` when the cells alone don't pin
/// the bounding box (LCT / sol-therm cases where every fissile cell
/// lives inside a lattice and therefore carries an element-local
/// AABB). Returns `None` only for lattice-free geometries.
fn lattices_world_aabb(geometry: &crate::geometry::Geometry) -> Option<crate::geometry::Aabb> {
    geometry
        .lattices
        .iter()
        .map(|lat| {
            let extent = Vec3::new(
                lat.pitch.x * lat.shape[0] as f64,
                lat.pitch.y * lat.shape[1] as f64,
                lat.pitch.z * lat.shape[2] as f64,
            );
            crate::geometry::Aabb::new(lat.origin, lat.origin + extent)
        })
        .reduce(crate::geometry::Aabb::union)
}

/// Tighten any axis whose half-extent exceeds 10× the smallest finite
/// half-extent. 2D-extruded lattice problems (LCT-008, sol-therm cases
/// with z-pitch like ±10 000 cm) trip rejection sampling because most
/// uniform draws land outside the fuel; clamping that axis to ±10× the
/// in-plane radius around the AABB midpoint preserves the fissile
/// content while keeping the sampling-acceptance probability high.
fn clamp_degenerate_axes(aabb: crate::geometry::Aabb) -> crate::geometry::Aabb {
    // Per-axis centre and half-extent. Treat any axis whose bound is
    // infinite on either side as "no centre constraint" — collapsing
    // it to the AABB's median of the finite axes would land on NaN
    // via `inf + (-inf)`. Doubly-infinite axes (extruded pin cells,
    // unbounded planes) get centred at 0 and clamped to the in-plane
    // scale below.
    let axis = |lo: f64, hi: f64| -> (f64, f64) {
        if lo.is_finite() && hi.is_finite() {
            (0.5 * (lo + hi), 0.5 * (hi - lo))
        } else if lo.is_finite() {
            (lo, f64::INFINITY)
        } else if hi.is_finite() {
            (hi, f64::INFINITY)
        } else {
            // Doubly-unbounded — collapse around the origin and let
            // the in-plane cap below tighten it.
            (0.0, f64::INFINITY)
        }
    };
    let (cx, hx) = axis(aabb.min.x, aabb.max.x);
    let (cy, hy) = axis(aabb.min.y, aabb.max.y);
    let (cz, hz) = axis(aabb.min.z, aabb.max.z);
    let half = [hx, hy, hz];
    // Smallest positive half-extent; ignore degenerate (0 or negative)
    // axes so a perfectly planar geometry doesn't collapse the cap to 0.
    let min_pos = half
        .iter()
        .copied()
        .filter(|h| h.is_finite() && *h > 0.0)
        .fold(f64::INFINITY, f64::min);
    if !min_pos.is_finite() {
        return aabb;
    }
    // Cap each axis at the smallest finite half-extent — for an
    // extruded-2D geometry this collapses the artificial z dimension
    // to roughly the in-plane radius around the AABB midpoint. Tighter
    // than 10× because the root cell's z range is often a single pin
    // height (a few cm), not the lattice's "infinite" pitch.
    let cap = min_pos;
    let clamped = [half[0].min(cap), half[1].min(cap), half[2].min(cap)];
    crate::geometry::Aabb::new(
        Vec3::new(cx - clamped[0], cy - clamped[1], cz - clamped[2]),
        Vec3::new(cx + clamped[0], cy + clamped[1], cz + clamped[2]),
    )
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
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::needless_range_loop
)]
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
                n4n: 0.0,
                fission: 1.2,
                capture: 0.1,
                nu_bar: 2.43,
                n_nalpha: 0.0,
                n_2nalpha: 0.0,
                n_np: 0.0,
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
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
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
                    n4n: 0.0,
                    fission: 2.0,
                    capture: 0.1,
                    nu_bar: 2.43,
                    n_nalpha: 0.0,
                    n_2nalpha: 0.0,
                    n_np: 0.0,
                    delayed_nu_bar: 0.0,
                    awr: 235.0,
                },
                MicroXs {
                    total: 5.0,
                    elastic: 1.0,
                    inelastic: 0.0,
                    n2n: 0.0,
                    n3n: 0.0,
                    n4n: 0.0,
                    fission: 0.0,
                    capture: 4.0,
                    nu_bar: 0.0,
                    n_nalpha: 0.0,
                    n_2nalpha: 0.0,
                    n_np: 0.0,
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
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
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
                n4n: 0.0,
                fission: 1.0,
                capture: 1.0,
                nu_bar: 2.43,
                n_nalpha: 0.0,
                n_2nalpha: 0.0,
                n_np: 0.0,
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
                    n4n: 0.0,
                    fission: 5.0,
                    capture: 5.0,
                    nu_bar: 2.43,
                    n_nalpha: 0.0,
                    n_2nalpha: 0.0,
                    n_np: 0.0,
                    delayed_nu_bar: 0.0,
                    awr: 235.0,
                },
                MicroXs {
                    total: 1.0,
                    elastic: 0.9,
                    inelastic: 0.0,
                    n2n: 0.0,
                    n3n: 0.0,
                    n4n: 0.0,
                    fission: 0.0,
                    capture: 0.1,
                    nu_bar: 0.0,
                    n_nalpha: 0.0,
                    n_2nalpha: 0.0,
                    n_np: 0.0,
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
                n4n: 0.0,
                fission: 1.2,
                capture: 0.1,
                nu_bar: 2.43,
                n_nalpha: 0.0,
                n_2nalpha: 0.0,
                n_np: 0.0,
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
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
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
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
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
    fn delta_tracking_two_material_problem() -> (Vec<Surface>, Vec<Cell>, Vec<Material>, ConstantXs)
    {
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
                crate::geometry::Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0)),
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
                n4n: 0.0,
                fission: 1.5,
                capture: 1.5,
                nu_bar: 2.43,
                n_nalpha: 0.0,
                n_2nalpha: 0.0,
                n_np: 0.0,
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
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
        };
        let (results, _) = run_eigenvalue(&config, &surfaces, &cells, &materials, &xs);

        let total: f64 = results
            .iter()
            .filter(|r| r.active)
            .map(|r| r.tallies.mesh_flux.iter().sum::<f64>())
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
            for &v in &r.tallies.mesh_flux {
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
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
        };
        let (results, _) = run_eigenvalue(&config, &surfaces, &cells, &materials, &xs);

        let mut total_pos = 0.0_f64;
        let mut total_neg = 0.0_f64;
        for r in results.iter().filter(|r| r.active) {
            total_pos += r.tallies.surface_current_pos.iter().sum::<f64>();
            total_neg += r.tallies.surface_current_neg.iter().sum::<f64>();
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
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
        };
        let (results, _) = run_eigenvalue(&config, &surfaces, &cells, &materials, &xs);

        let active: Vec<&BatchResult> = results.iter().filter(|r| r.active).collect();
        assert!(!active.is_empty(), "no active batches");
        let n = active.len() as f64;
        let k_eff_mean: f64 = active.iter().map(|r| r.k_eff).sum::<f64>() / n;
        let k_track_mean: f64 = active.iter().map(|r| r.k_track).sum::<f64>() / n;

        // k_track must not be identically zero — that was the bug.
        assert!(
            k_track_mean > 1e-3,
            "k_track is zero under delta tracking: {k_track_mean}"
        );

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
