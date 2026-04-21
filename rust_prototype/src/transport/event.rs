//! Event-based Monte Carlo transport loop.
//!
//! Instead of tracking each particle birth-to-death (history-based),
//! processes all particles in batch operations by event type:
//!   1. Batch XS lookup (sorted by energy → cache/GPU friendly)
//!   2. Sample collision distance
//!   3. Advance / surface crossing
//!   4. Process collisions
//!   5. Compact (remove dead, bank fission sites)
//!
//! References:
//!   - Tramm et al. 2024: "Performance Portable MC on Intel, NVIDIA, AMD GPUs"
//!     Event-based is 6x faster than history-based on GPU.
//!   - Ridley 2024 (MIT PhD): GPU-oriented algorithms for CE MC transport
//!   - Hamilton & Evans 2019: event-based outperforms history-based on GPU

use std::io::Write;

use rayon::prelude::*;

use crate::geometry::cell::{Cell, CellFill};
use crate::geometry::surface::{BoundaryCondition, Surface};
use crate::geometry::{self, Vec3};
use crate::physics::collision::{self, CollisionOutcome, InelasticData, MicroXs};
use crate::transport::material::Material;
use crate::transport::particle::{FissionBank, FissionSite};
use crate::transport::rng::Rng;
use crate::transport::simulate::{self, BatchResult, SimConfig, XsProvider};

/// Maximum nuclides per material (stack-allocated XS buffers).
const MAX_NUCLIDES: usize = 16;

/// Structure-of-Arrays particle bank for batch processing.
///
/// Enables energy sorting for coalesced GPU memory access and
/// SIMD-friendly iteration on CPU.
pub struct ParticleBank {
    pub pos: Vec<Vec3>,
    pub dir: Vec<Vec3>,
    pub energy: Vec<f64>,
    pub cell_idx: Vec<usize>,
    pub alive: Vec<bool>,
    pub n_collisions: Vec<u32>,
    /// RNG state per particle (PCG-64).
    pub rng_state: Vec<u64>,
    pub rng_stream: Vec<u64>,
    /// Indices sorted by energy (for batch XS lookup).
    pub sorted_idx: Vec<usize>,
}

impl ParticleBank {
    /// Create a bank from fission sites.
    pub fn from_fission_sites(
        sites: &[FissionSite],
        batch_seed: u64,
        surfaces: &[Surface],
        cells: &[Cell],
    ) -> Self {
        let n = sites.len();
        let mut bank = Self {
            pos: Vec::with_capacity(n),
            dir: Vec::with_capacity(n),
            energy: Vec::with_capacity(n),
            cell_idx: Vec::with_capacity(n),
            alive: vec![true; n],
            n_collisions: vec![0; n],
            rng_state: Vec::with_capacity(n),
            rng_stream: Vec::with_capacity(n),
            sorted_idx: (0..n).collect(),
        };

        for (i, site) in sites.iter().enumerate() {
            let mut rng = Rng::for_particle(batch_seed, i as u64);
            let (u, v, w) = rng.isotropic_direction();
            let cell = geometry::ray::find_cell(site.pos, surfaces, cells).unwrap_or(0);

            bank.pos.push(site.pos);
            bank.dir.push(Vec3::new(u, v, w));
            bank.energy.push(site.energy);
            bank.cell_idx.push(cell);
            bank.rng_state.push(rng.state());
            bank.rng_stream.push(rng.stream());
        }

        bank
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    pub fn alive_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Sort particle indices by energy for coalesced memory access.
    /// This is THE critical optimization for GPU performance (Tramm 2024).
    pub fn sort_by_energy(&mut self) {
        let energies = &self.energy;
        self.sorted_idx.sort_unstable_by(|&a, &b| {
            energies[a]
                .partial_cmp(&energies[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Result of one event-based batch.
pub struct EventBatchResult {
    pub fission_sites: Vec<FissionSite>,
    pub leakage: u32,
    pub absorptions: u32,
    pub fissions: u32,
    pub collisions: u32,
}

/// Per-particle result of one event step. Produced by the parallel
/// map inside `transport_batch_event` and applied back to `ParticleBank`
/// by a serial reduce. Carries the particle's new state plus any
/// counter deltas and fission sites generated this step.
struct ParticleStepOutcome {
    pos: Vec3,
    dir: Vec3,
    energy: f64,
    cell_idx: usize,
    alive: bool,
    n_collisions: u32,
    rng_state: u64,
    rng_stream: u64,
    leakage: u32,
    absorptions: u32,
    fissions: u32,
    collisions: u32,
    fission_sites: Vec<FissionSite>,
}

/// Step one particle through one event cycle. Pure function over
/// immutable borrows of the bank and environment; returns a fresh
/// outcome describing the particle's new state.
#[allow(clippy::too_many_arguments)]
fn step_particle<XS: XsProvider>(
    pi: usize,
    bank: &ParticleBank,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
) -> ParticleStepOutcome {
    let mut out = ParticleStepOutcome {
        pos: bank.pos[pi],
        dir: bank.dir[pi],
        energy: bank.energy[pi],
        cell_idx: bank.cell_idx[pi],
        alive: bank.alive[pi],
        n_collisions: bank.n_collisions[pi],
        rng_state: bank.rng_state[pi],
        rng_stream: bank.rng_stream[pi],
        leakage: 0,
        absorptions: 0,
        fissions: 0,
        collisions: 0,
        fission_sites: Vec::new(),
    };

    if !out.alive {
        return out;
    }
    if out.n_collisions >= 1000 {
        out.alive = false;
        out.leakage = 1;
        return out;
    }

    let mut rng = Rng::from_state(out.rng_state, out.rng_stream);

    let cell = &cells[out.cell_idx];
    let mat_idx = match cell.fill {
        CellFill::Material(m) => m as usize,
        CellFill::Void => {
            let trace = geometry::ray::trace_step(out.pos, out.dir, out.cell_idx, surfaces, cells);
            match trace {
                Some(hit) => {
                    let nudge = (hit.distance * 1e-8).max(1e-8);
                    out.pos = out.pos + out.dir * (hit.distance + nudge);
                    match surfaces[hit.surface_idx].boundary_condition() {
                        BoundaryCondition::Vacuum => {
                            out.alive = false;
                            out.leakage = 1;
                        }
                        BoundaryCondition::Reflective => {
                            let n = surfaces[hit.surface_idx].normal_at(out.pos);
                            out.dir = out.dir - n * (2.0 * out.dir.dot(n));
                        }
                        BoundaryCondition::Transmission => {
                            if let Some(next) = hit.next_cell_idx {
                                out.cell_idx = next;
                            } else {
                                out.alive = false;
                                out.leakage = 1;
                            }
                        }
                    }
                }
                None => {
                    out.alive = false;
                    out.leakage = 1;
                }
            }
            out.rng_state = rng.state();
            out.rng_stream = rng.stream();
            return out;
        }
        CellFill::Universe(_) => {
            out.alive = false;
            out.leakage = 1;
            return out;
        }
    };

    if mat_idx >= materials.len() {
        out.alive = false;
        out.leakage = 1;
        return out;
    }

    let material = &materials[mat_idx];

    let urr_xi = rng.uniform();
    let n_nuclides = material.nuclides.len();
    let mut micro_xs = [MicroXs::default(); MAX_NUCLIDES];
    let mut micro_totals = [0.0_f64; MAX_NUCLIDES];

    for (i, nuc) in material.nuclides.iter().enumerate() {
        let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, out.energy);
        xs_provider.apply_urr(nuc.xs_kernel_idx, &mut xs, out.energy, urr_xi);

        if let Some(tsl) = xs_provider.thermal_scattering(nuc.xs_kernel_idx)
            && out.energy < tsl.energy_max
        {
            let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
            let thermal_total = tsl.total_xs(out.energy, t_idx);
            let delta = thermal_total - xs.elastic;
            xs.total += delta;
            xs.elastic = 0.0;
        }

        micro_totals[i] = xs.total;
        micro_xs[i] = xs;
    }

    let macro_total = material.macro_total(&micro_totals[..n_nuclides]);
    if macro_total <= 0.0 {
        out.alive = false;
        out.leakage = 1;
        out.rng_state = rng.state();
        out.rng_stream = rng.stream();
        return out;
    }

    let dist_collision = rng.exponential(macro_total);
    let trace = geometry::ray::trace_step(out.pos, out.dir, out.cell_idx, surfaces, cells);

    match trace {
        Some(hit) if hit.distance < dist_collision => {
            let nudge = (hit.distance * 1e-8).max(1e-8);
            out.pos = out.pos + out.dir * (hit.distance + nudge);
            match surfaces[hit.surface_idx].boundary_condition() {
                BoundaryCondition::Vacuum => {
                    out.alive = false;
                    out.leakage = 1;
                }
                BoundaryCondition::Reflective => {
                    let normal = surfaces[hit.surface_idx].normal_at(out.pos);
                    out.dir = out.dir - normal * (2.0 * out.dir.dot(normal));
                }
                BoundaryCondition::Transmission => {
                    if let Some(next) = hit.next_cell_idx {
                        out.cell_idx = next;
                    } else {
                        out.alive = false;
                        out.leakage = 1;
                    }
                }
            }
        }
        _ => {
            out.pos = out.pos + out.dir * dist_collision;
            out.collisions = 1;
            out.n_collisions += 1;

            let nuc_idx =
                material.sample_nuclide(&micro_totals[..n_nuclides], macro_total, rng.uniform());
            let xs_kernel_idx = material.nuclides[nuc_idx].xs_kernel_idx;

            let thermal_active = xs_provider
                .thermal_scattering(xs_kernel_idx)
                .is_some_and(|tsl| out.energy < tsl.energy_max);

            let outcome = if thermal_active {
                let tsl = xs_provider
                    .thermal_scattering(xs_kernel_idx)
                    .expect("thermal");
                let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                let thermal_xs = tsl.total_xs(out.energy, t_idx);
                let xi = rng.uniform() * micro_xs[nuc_idx].total;

                if xi < thermal_xs {
                    let (e_out, mu) = tsl.sample(out.energy, t_idx, &mut rng);
                    out.energy = e_out;
                    rotate_direction(&mut out.dir, mu, &mut rng);
                    None
                } else {
                    let mut particle =
                        make_temp_particle(out.pos, out.dir, out.energy, out.cell_idx);
                    let o = process_standard_collision(
                        &mut particle,
                        &micro_xs[nuc_idx],
                        xs_kernel_idx,
                        xs_provider,
                        cell.temperature,
                        &mut rng,
                    );
                    out.energy = particle.energy;
                    out.dir = particle.dir;
                    Some(o)
                }
            } else {
                let mut particle = make_temp_particle(out.pos, out.dir, out.energy, out.cell_idx);
                let o = process_standard_collision(
                    &mut particle,
                    &micro_xs[nuc_idx],
                    xs_kernel_idx,
                    xs_provider,
                    cell.temperature,
                    &mut rng,
                );
                out.energy = particle.energy;
                out.dir = particle.dir;
                Some(o)
            };

            if let Some(o) = outcome {
                match o {
                    CollisionOutcome::Scatter => {}
                    CollisionOutcome::Absorption => {
                        out.alive = false;
                        out.absorptions = 1;
                    }
                    CollisionOutcome::Fission { sites } => {
                        out.alive = false;
                        out.fissions = 1;
                        out.fission_sites = sites;
                    }
                }
            }
        }
    }

    out.rng_state = rng.state();
    out.rng_stream = rng.stream();
    out
}

/// Run one batch of event-based transport.
///
/// Processes all particles through repeated event cycles until all die or leak.
/// Returns fission sites for the next generation.
pub fn transport_batch_event<XS: XsProvider>(
    sites: &[FissionSite],
    batch_seed: u64,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
) -> EventBatchResult {
    let mut bank = ParticleBank::from_fission_sites(sites, batch_seed, surfaces, cells);
    let mut result = EventBatchResult {
        fission_sites: Vec::new(),
        leakage: 0,
        absorptions: 0,
        fissions: 0,
        collisions: 0,
    };

    let max_events = 1000;
    let mut event_count = 0;

    while bank.alive_count() > 0 && event_count < max_events {
        event_count += 1;

        // ── EVENT 1: Sort by energy (scaffold — not yet exploited by
        // batched XS lookup; retained to keep the event-loop shape
        // future-proof for GPU/SIMD batching work). ──
        bank.sort_by_energy();

        // ── EVENT 2-5: process all particles in parallel, then
        // serial-apply the updates. Each `step_particle` call is
        // independent (reads an immutable snapshot of the bank),
        // so rayon's work-stealing scheduler parallelises freely.
        let n = bank.len();
        let outcomes: Vec<ParticleStepOutcome> = (0..n)
            .into_par_iter()
            .map(|pi| step_particle(pi, &bank, surfaces, cells, materials, xs_provider))
            .collect();

        for (pi, o) in outcomes.into_iter().enumerate() {
            bank.pos[pi] = o.pos;
            bank.dir[pi] = o.dir;
            bank.energy[pi] = o.energy;
            bank.cell_idx[pi] = o.cell_idx;
            bank.alive[pi] = o.alive;
            bank.n_collisions[pi] = o.n_collisions;
            bank.rng_state[pi] = o.rng_state;
            bank.rng_stream[pi] = o.rng_stream;
            result.leakage += o.leakage;
            result.absorptions += o.absorptions;
            result.fissions += o.fissions;
            result.collisions += o.collisions;
            if !o.fission_sites.is_empty() {
                result.fission_sites.extend(o.fission_sites);
            }
        }
    }

    result
}

// ── Helpers ────────────────────────────────────────────────────────────

fn rotate_direction(dir: &mut Vec3, mu: f64, rng: &mut Rng) {
    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
    let sin_mu = (1.0 - mu * mu).max(0.0).sqrt();
    let d = *dir;
    let w2 = d.z * d.z;
    if w2 < 0.999 {
        let inv_sq = 1.0 / (1.0 - w2).sqrt();
        *dir = Vec3::new(
            mu * d.x + sin_mu * (d.x * d.z * phi.cos() - d.y * phi.sin()) * inv_sq,
            mu * d.y + sin_mu * (d.y * d.z * phi.cos() + d.x * phi.sin()) * inv_sq,
            mu * d.z - sin_mu * (1.0 - w2).sqrt() * phi.cos(),
        );
    } else {
        let sign = if d.z > 0.0 { 1.0 } else { -1.0 };
        *dir = Vec3::new(sin_mu * phi.cos(), sin_mu * phi.sin() * sign, mu * sign);
    }
}

fn make_temp_particle(
    pos: Vec3,
    dir: Vec3,
    energy: f64,
    cell_idx: usize,
) -> crate::transport::particle::Particle {
    crate::transport::particle::Particle::new(pos, dir, energy, cell_idx)
}

fn process_standard_collision<XS: XsProvider>(
    particle: &mut crate::transport::particle::Particle,
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

    collision::process_collision(
        particle,
        xs,
        inelastic_data.as_ref(),
        elastic_angle,
        fission_edist,
        temperature,
        rng,
    )
}

/// Run k-eigenvalue power iteration using the event-based transport loop.
///
/// Mirrors `simulate::run_eigenvalue` but calls `transport_batch_event`
/// per batch. The batch kernel is currently serial (no rayon) — this
/// driver is the honest head-to-head baseline for the history-based
/// parallel driver. `thermal_scatters` and `surface_crossings` are not
/// tracked by the event kernel and are reported as 0 in `BatchResult`.
pub fn run_eigenvalue_event<XS: XsProvider>(
    config: &SimConfig,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
) -> (Vec<BatchResult>, f64) {
    let n = config.particles_per_batch as usize;
    let seed = config.seed;
    let mut source_bank = simulate::initial_source(n, surfaces, cells, seed);

    let mut results = Vec::with_capacity(config.batches as usize);
    let mut k_sum = 0.0;
    let mut k_count = 0_u32;

    for batch in 1..=config.batches {
        let batch_seed = batch as u64 + seed * 100_000;
        let batch_result = transport_batch_event(
            &source_bank,
            batch_seed,
            surfaces,
            cells,
            materials,
            xs_provider,
        );

        let k_batch = batch_result.fission_sites.len() as f64 / n as f64;
        let is_active = batch > config.inactive;
        if is_active {
            k_sum += k_batch;
            k_count += 1;
        }

        let active_str = if is_active { " *" } else { "" };
        println!(
            "  Batch {batch:>4}: k={k_batch:.5}  coll={collisions}  \
             fiss={fissions}  leak={leakage}  abs={absorptions}{active_str}",
            collisions = batch_result.collisions,
            fissions = batch_result.fissions,
            leakage = batch_result.leakage,
            absorptions = batch_result.absorptions,
        );
        let _ = std::io::stdout().flush();

        results.push(BatchResult {
            batch,
            k_eff: k_batch,
            leakage: batch_result.leakage,
            absorptions: batch_result.absorptions,
            fissions: batch_result.fissions,
            collisions: batch_result.collisions,
            thermal_scatters: 0,
            surface_crossings: 0,
            shannon_entropy: 0.0,
            active: is_active,
        });

        // Rebuild a normalized fission bank for the next generation.
        let mut fission_bank = FissionBank::new();
        fission_bank.sites.extend(batch_result.fission_sites);
        source_bank = simulate::normalize_fission_bank(
            &fission_bank,
            n,
            batch + seed as u32 * 100_000,
        );
    }

    let k_final = if k_count > 0 {
        k_sum / k_count as f64
    } else {
        0.0
    };
    (results, k_final)
}
