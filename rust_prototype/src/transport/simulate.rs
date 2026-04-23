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

/// Maximum nuclides per material for stack-allocated XS buffers.
/// Godiva has 3, most materials have < 8. Avoids per-collision heap allocation.
const MAX_NUCLIDES: usize = 16;

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
        println!("  Tracking: SURFACE (single material — delta tracking not beneficial)");
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
        println!(
            "  Tracking: SURFACE (heterogeneous but contrast={:.1}x too high for delta tracking)",
            max_contrast
        );
        return TrackingMode::Surface;
    }

    let avg_rejection = if max_contrast > 1.0 {
        1.0 - 1.0 / max_contrast
    } else {
        0.0
    };
    println!(
        "  Tracking: DELTA (heterogeneous — {} materials, contrast={:.1}x, ~{:.0}% virtual collisions)",
        mat_indices.len(),
        max_contrast,
        avg_rejection * 100.0
    );

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

/// Transport a single particle to completion.
fn transport_particle<XS: XsProvider>(
    site: &FissionSite,
    batch: u64,
    particle_idx: u64,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
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
    };

    let (u, v, w) = rng.isotropic_direction();
    let dir = Vec3::new(u, v, w);

    let cell_idx = geometry::ray::find_cell(site.pos, surfaces, cells).unwrap_or(0);
    let mut particle = Particle::new(site.pos, dir, site.energy, cell_idx);

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

            let mat_idx = match cell.fill {
                CellFill::Material(m) => m as usize,
                CellFill::Void => {
                    // Void region — free-stream to next surface (no interactions).
                    // Safety limit prevents infinite loops in degenerate geometries.
                    void_crossings += 1;
                    if void_crossings > 100 {
                        particle.kill();
                        result.leakage += 1;
                        break;
                    }
                    let trace = geometry::ray::trace_step(
                        particle.pos,
                        particle.dir,
                        particle.cell_idx,
                        surfaces,
                        cells,
                    );
                    match trace {
                        Some(hit) => {
                            // Nudge proportional to distance — ensures clean surface crossing
                            let nudge = (hit.distance * 1e-8).max(1e-8);
                            let bc = surfaces[hit.surface_idx].boundary_condition();
                            match bc {
                                BoundaryCondition::Vacuum => {
                                    particle.advance(hit.distance);
                                    particle.kill();
                                    result.leakage += 1;
                                    break;
                                }
                                BoundaryCondition::Reflective => {
                                    particle.advance(hit.distance);
                                    let n = surfaces[hit.surface_idx].normal_at(particle.pos);
                                    let d = particle.dir;
                                    particle.dir = d - n * (2.0 * d.dot(n));
                                }
                                BoundaryCondition::Transmission => {
                                    particle.advance(hit.distance + nudge);
                                    if let Some(next) = hit.next_cell_idx {
                                        particle.cell_idx = next;
                                    } else {
                                        particle.kill();
                                        result.leakage += 1;
                                        break;
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
                CellFill::Universe(_) => {
                    particle.kill();
                    result.leakage += 1;
                    break;
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
            let mut micro_totals = [0.0_f64; MAX_NUCLIDES];
            // Track thermal scattering XS addition per nuclide
            let mut thermal_xs_add = [0.0_f64; MAX_NUCLIDES];
            for (i, nuc) in material.nuclides.iter().enumerate() {
                let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, particle.energy);
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

                micro_totals[i] = xs.total;
                micro_xs[i] = xs;
            }
            let macro_total = material.macro_total(&micro_totals[..n_nuclides]);

            if macro_total <= 0.0 {
                particle.kill();
                result.leakage += 1;
                break;
            }

            let dist_collision = rng.exponential(macro_total);

            let trace = geometry::ray::trace_step(
                particle.pos,
                particle.dir,
                particle.cell_idx,
                surfaces,
                cells,
            );

            match trace {
                Some(hit) if hit.distance < dist_collision => {
                    result.surface_crossings += 1;
                    let bc = surfaces[hit.surface_idx].boundary_condition();
                    match bc {
                        BoundaryCondition::Vacuum => {
                            particle.advance(hit.distance);
                            particle.kill();
                            result.leakage += 1;
                        }
                        BoundaryCondition::Reflective => {
                            // Advance exactly to the surface (no overshoot), then reflect.
                            // COINCIDENCE_TOL in Surface::distance() filters the t≈0
                            // re-intersection, preventing infinite bounce loops.
                            particle.advance(hit.distance);
                            let n = surfaces[hit.surface_idx].normal_at(particle.pos);
                            let d = particle.dir;
                            particle.dir = d - n * (2.0 * d.dot(n));
                        }
                        BoundaryCondition::Transmission => {
                            // Overshoot slightly to land clearly inside the next cell.
                            particle.advance(hit.distance + (hit.distance * 1e-8).max(1e-8));
                            if let Some(next) = hit.next_cell_idx {
                                particle.cell_idx = next;
                            } else {
                                particle.kill();
                                result.leakage += 1;
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

                    // Check if this nuclide+energy qualifies for thermal scattering
                    let use_thermal = thermal_xs_add[nuc_idx] > 0.0;
                    if use_thermal {
                        // Thermal scattering: sample from S(α,β)
                        let tsl = xs_provider
                            .thermal_scattering(xs_kernel_idx)
                            .expect("thermal data");
                        let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                        let xi_reaction = rng.uniform() * micro_xs[nuc_idx].total;

                        if xi_reaction < thermal_xs_add[nuc_idx] {
                            // Thermal scattering event
                            result.thermal_scatters += 1;
                            let (e_out, mu) = tsl.sample(particle.energy, t_idx, &mut rng);
                            particle.energy = e_out;
                            // Apply scattering angle
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
                            // Continue — this was a scatter
                        } else {
                            // Non-thermal reaction (fission, capture, inelastic, etc.)
                            // Process normally but with elastic = 0
                            let outcome = process_non_thermal_collision(
                                &mut particle,
                                &micro_xs[nuc_idx],
                                xs_kernel_idx,
                                xs_provider,
                                cell.temperature,
                                &mut rng,
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
                                        &particle,
                                        &mut rng,
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
                                        &particle,
                                        &mut rng,
                                        &mut result.photon_events,
                                    );
                                }
                                CollisionOutcome::Multiplicity { secondaries } => {
                                    for s in secondaries {
                                        pending.push(Particle::new(
                                            s.pos,
                                            s.dir,
                                            s.energy,
                                            particle.cell_idx,
                                        ));
                                    }
                                }
                            }
                        }
                    } else {
                        // Standard collision processing (no thermal scattering)
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

                        let outcome = collision::process_collision(
                            &mut particle,
                            &micro_xs[nuc_idx],
                            inelastic_data.as_ref(),
                            elastic_angle,
                            fission_edist,
                            continuum_edist,
                            n2n_edist,
                            n3n_edist,
                            cell.temperature,
                            &mut rng,
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
                                    &particle,
                                    &mut rng,
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
                                    &particle,
                                    &mut rng,
                                    &mut result.photon_events,
                                );
                            }
                            CollisionOutcome::Multiplicity { secondaries } => {
                                for s in secondaries {
                                    pending.push(Particle::new(
                                        s.pos,
                                        s.dir,
                                        s.energy,
                                        particle.cell_idx,
                                    ));
                                }
                            }
                        }
                    }
                }
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

/// Process a non-thermal collision for a nuclide where thermal scattering
/// replaced elastic. The elastic channel is zero, so only inelastic/fission/capture remain.
fn process_non_thermal_collision<XS: XsProvider>(
    particle: &mut Particle,
    xs: &MicroXs,
    xs_kernel_idx: usize,
    xs_provider: &XS,
    temperature: f64,
    rng: &mut Rng,
) -> CollisionOutcome {
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

    collision::process_collision(
        particle,
        xs,
        inelastic_data.as_ref(),
        elastic_angle,
        fission_edist,
        continuum_edist,
        n2n_edist,
        n3n_edist,
        temperature,
        rng,
    )
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
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
    majorant: &MajorantTable,
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
    };

    let (u, v, w) = rng.isotropic_direction();
    let dir = Vec3::new(u, v, w);
    let cell_idx = geometry::ray::find_cell(site.pos, surfaces, cells).unwrap_or(0);
    let mut particle = Particle::new(site.pos, dir, site.energy, cell_idx);

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
            let trace = geometry::ray::trace_step(
                particle.pos,
                particle.dir,
                particle.cell_idx,
                surfaces,
                cells,
            );

            match trace {
                Some(hit) if hit.distance < d_collision => {
                    let bc = surfaces[hit.surface_idx].boundary_condition();
                    match bc {
                        BoundaryCondition::Vacuum => {
                            particle.kill();
                            result.leakage += 1;
                            break;
                        }
                        BoundaryCondition::Reflective => {
                            particle.advance(hit.distance);
                            let n = surfaces[hit.surface_idx].normal_at(particle.pos);
                            let d = particle.dir;
                            particle.dir = d - n * (2.0 * d.dot(n));
                            continue;
                        }
                        BoundaryCondition::Transmission => {
                            // Delta tracking: skip transmission boundaries, just advance
                            particle.advance(d_collision);
                            // Find which cell we ended up in
                            match geometry::ray::find_cell(particle.pos, surfaces, cells) {
                                Some(idx) => particle.cell_idx = idx,
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
                    // No boundary before collision distance — advance freely
                    particle.advance(d_collision);
                    // Verify we're still in a valid cell
                    if let Some(idx) = geometry::ray::find_cell(particle.pos, surfaces, cells) {
                        particle.cell_idx = idx;
                    } else {
                        particle.kill();
                        result.leakage += 1;
                        break;
                    }
                }
            }

            // Get current material — handle void by free-streaming
            let cell = &cells[particle.cell_idx];
            let mat_idx = match cell.fill {
                CellFill::Material(m) => m as usize,
                CellFill::Void => {
                    // Void region — free-stream to next surface
                    let trace = geometry::ray::trace_step(
                        particle.pos,
                        particle.dir,
                        particle.cell_idx,
                        surfaces,
                        cells,
                    );
                    match trace {
                        Some(hit) => {
                            let bc = surfaces[hit.surface_idx].boundary_condition();
                            match bc {
                                BoundaryCondition::Vacuum => {
                                    particle.advance(hit.distance);
                                    particle.kill();
                                    result.leakage += 1;
                                    break;
                                }
                                BoundaryCondition::Reflective => {
                                    particle.advance(hit.distance);
                                    let n = surfaces[hit.surface_idx].normal_at(particle.pos);
                                    let d = particle.dir;
                                    particle.dir = d - n * (2.0 * d.dot(n));
                                }
                                BoundaryCondition::Transmission => {
                                    particle
                                        .advance(hit.distance + (hit.distance * 1e-8).max(1e-8));
                                    if let Some(next) = hit.next_cell_idx {
                                        particle.cell_idx = next;
                                    } else {
                                        particle.kill();
                                        result.leakage += 1;
                                        break;
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
                CellFill::Universe(_) => {
                    particle.kill();
                    result.leakage += 1;
                    break;
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
            let mut micro_totals = [0.0_f64; MAX_NUCLIDES];
            for (i, nuc) in material.nuclides.iter().enumerate() {
                let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, particle.energy);
                xs_provider.apply_urr(nuc.xs_kernel_idx, &mut xs, particle.energy, urr_xi);
                micro_totals[i] = xs.total;
                micro_xs[i] = xs;
            }
            let sigma_real = material.macro_total(&micro_totals[..n_nuclides]);

            // Acceptance test: real collision with probability Σ_t / Σ_maj
            if rng.uniform() >= sigma_real / sigma_maj {
                continue; // Virtual collision — keep tracking
            }

            // Real collision — process exactly as surface tracking
            result.collisions += 1;

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

            let outcome = collision::process_collision(
                &mut particle,
                &micro_xs[nuc_idx],
                inelastic_data.as_ref(),
                elastic_angle,
                fission_edist,
                continuum_edist,
                n2n_edist,
                n3n_edist,
                cell.temperature,
                &mut rng,
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
                        &particle,
                        &mut rng,
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
                        &particle,
                        &mut rng,
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
    let n = config.particles_per_batch as usize;

    // Auto-detect optimal tracking algorithm
    let tracking = detect_tracking_mode(cells, materials, xs_provider);

    let seed = config.seed;
    let mut source_bank = initial_source(n, surfaces, cells, seed);

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
        let particle_results: Vec<ParticleResult> = source_bank
            .par_iter()
            .enumerate()
            .map(|(i, site)| match &tracking {
                TrackingMode::Surface => transport_particle(
                    site,
                    batch_seed,
                    i as u64,
                    surfaces,
                    cells,
                    materials,
                    xs_provider,
                ),
                TrackingMode::Delta(majorant) => transport_particle_delta(
                    site,
                    batch_seed,
                    i as u64,
                    surfaces,
                    cells,
                    materials,
                    xs_provider,
                    majorant,
                ),
            })
            .collect();

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

        for pr in particle_results {
            fission_bank.sites.extend(pr.fission_sites);
            leakage += pr.leakage;
            absorptions += pr.absorptions;
            fissions += pr.fissions;
            collisions += pr.collisions;
            thermal_scatters += pr.thermal_scatters;
            surface_crossings += pr.surface_crossings;
            for c in pr.capture_cells {
                if c < captures_by_cell.len() {
                    captures_by_cell[c] += 1.0;
                }
            }
            photon_events.extend(pr.photon_events);
        }

        let k_batch = fission_bank.len() as f64 / n as f64;
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
            println!("  [auto-inactive] entropy converged at batch {batch} ({reason})");
            let _ = std::io::stdout().flush();
        }

        let is_active = batch > effective_inactive;
        result.active = is_active;
        if is_active {
            k_sum += k_batch;
            k_count += 1;
        }

        let active_str = if is_active { " *" } else { "" };
        println!(
            "  Batch {batch:>4}: k={k_batch:.5}  H={entropy:.4}  \
             coll={collisions}  fiss={fissions}  leak={leakage}  \
             therm={thermal_scatters}  surf={surface_crossings}{active_str}"
        );
        let _ = std::io::stdout().flush();

        results.push(result);

        source_bank = normalize_fission_bank(&fission_bank, n, batch + seed as u32 * 100_000);
    }

    let k_final = if k_count > 0 {
        k_sum / k_count as f64
    } else {
        0.0
    };
    (results, k_final)
}

/// Create an initial source uniformly distributed in fissile material cells.
///
/// Rejection-samples points in the bounding box of fissile cells, accepting
/// only those that land inside a cell containing material. For Godiva, this
/// is the single fuel sphere. For PWR pin cell, this is the cylindrical
/// fuel region (rejects corners of the bounding box that fall in gap/clad/water).
fn initial_source(n: usize, surfaces: &[Surface], cells: &[Cell], seed: u64) -> Vec<FissionSite> {
    let mut rng = Rng::new(seed * 100_000, 0);
    let mut sites = Vec::with_capacity(n);

    // Find the first material cell (assumed fissile for eigenvalue problems)
    let target_idx = cells
        .iter()
        .position(|c| matches!(c.fill, CellFill::Material(_)));
    let aabb = target_idx
        .map(|i| cells[i].aabb)
        .unwrap_or(crate::geometry::Aabb::new(
            Vec3::new(-10.0, -10.0, -10.0),
            Vec3::new(10.0, 10.0, 10.0),
        ));

    while sites.len() < n {
        let x = aabb.min.x + rng.uniform() * (aabb.max.x - aabb.min.x);
        let y = aabb.min.y + rng.uniform() * (aabb.max.y - aabb.min.y);
        let z = aabb.min.z + rng.uniform() * (aabb.max.z - aabb.min.z);
        let pos = Vec3::new(x, y, z);

        // Only accept if the point is actually in the target cell
        // (rejects points in the AABB corners that are outside the cylinder)
        if let Some(idx) = geometry::ray::find_cell(pos, surfaces, cells)
            && Some(idx) == target_idx
        {
            sites.push(FissionSite {
                pos,
                energy: 1.0e6,
                weight: 1.0,
            });
        }
    }

    sites
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
                awr: 235.0,
            }],
        };

        let config = SimConfig {
            batches: 10,
            inactive: 3,
            particles_per_batch: 500,
            seed: 0,
            auto_inactive: None,
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
                awr: 235.0,
            }],
        };

        let mode = detect_tracking_mode(&cells, &materials, &xs);
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
                    awr: 56.0,
                },
            ],
        };

        let mode = detect_tracking_mode(&cells, &materials, &xs);
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
                awr: 235.0,
            }],
        };

        let config0 = SimConfig {
            batches: 5,
            inactive: 1,
            particles_per_batch: 500,
            seed: 0,
            auto_inactive: None,
        };
        let config1 = SimConfig {
            batches: 5,
            inactive: 1,
            particles_per_batch: 500,
            seed: 1,
            auto_inactive: None,
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
}
