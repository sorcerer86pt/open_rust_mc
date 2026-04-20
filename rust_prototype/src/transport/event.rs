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

use crate::geometry::cell::{Cell, CellFill};
use crate::geometry::surface::{BoundaryCondition, Surface};
use crate::geometry::{self, Vec3};
use crate::physics::collision::{self, CollisionOutcome, InelasticData, MicroXs};
use crate::transport::material::Material;
use crate::transport::particle::FissionSite;
use crate::transport::rng::Rng;
use crate::transport::simulate::XsProvider;

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

        // ── EVENT 1: Sort by energy (critical for cache/GPU) ──
        bank.sort_by_energy();

        // ── EVENT 2-5: Process all particles ──
        // For now, process serially per particle for correctness.
        // GPU acceleration: move XS lookup to GPU kernel.
        let n = bank.len();
        for pi in 0..n {
            if !bank.alive[pi] {
                continue;
            }
            if bank.n_collisions[pi] >= 1000 {
                bank.alive[pi] = false;
                result.leakage += 1;
                continue;
            }

            let mut rng = Rng::from_state(bank.rng_state[pi], bank.rng_stream[pi]);

            // Get cell and material
            let cell = &cells[bank.cell_idx[pi]];
            let mat_idx = match cell.fill {
                CellFill::Material(m) => m as usize,
                CellFill::Void => {
                    // Free-stream through void
                    let trace = geometry::ray::trace_step(
                        bank.pos[pi],
                        bank.dir[pi],
                        bank.cell_idx[pi],
                        surfaces,
                        cells,
                    );
                    match trace {
                        Some(hit) => {
                            let nudge = (hit.distance * 1e-8).max(1e-8);
                            bank.pos[pi] = bank.pos[pi] + bank.dir[pi] * (hit.distance + nudge);
                            let bc = surfaces[hit.surface_idx].boundary_condition();
                            match bc {
                                BoundaryCondition::Vacuum => {
                                    bank.alive[pi] = false;
                                    result.leakage += 1;
                                }
                                BoundaryCondition::Reflective => {
                                    let n = surfaces[hit.surface_idx].normal_at(bank.pos[pi]);
                                    bank.dir[pi] = bank.dir[pi] - n * (2.0 * bank.dir[pi].dot(n));
                                }
                                BoundaryCondition::Transmission => {
                                    if let Some(next) = hit.next_cell_idx {
                                        bank.cell_idx[pi] = next;
                                    } else {
                                        bank.alive[pi] = false;
                                        result.leakage += 1;
                                    }
                                }
                            }
                        }
                        None => {
                            bank.alive[pi] = false;
                            result.leakage += 1;
                        }
                    }
                    bank.rng_state[pi] = rng.state();
                    bank.rng_stream[pi] = rng.stream();
                    continue;
                }
                CellFill::Universe(_) => {
                    bank.alive[pi] = false;
                    result.leakage += 1;
                    continue;
                }
            };

            if mat_idx >= materials.len() {
                bank.alive[pi] = false;
                result.leakage += 1;
                continue;
            }

            let material = &materials[mat_idx];

            // ── XS lookup (sorted particles → coalesced access) ──
            let urr_xi = rng.uniform();
            let n_nuclides = material.nuclides.len();
            let mut micro_xs = [MicroXs::default(); MAX_NUCLIDES];
            let mut micro_totals = [0.0_f64; MAX_NUCLIDES];

            for (i, nuc) in material.nuclides.iter().enumerate() {
                let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, bank.energy[pi]);
                xs_provider.apply_urr(nuc.xs_kernel_idx, &mut xs, bank.energy[pi], urr_xi);

                // S(α,β) thermal scattering adjustment
                if let Some(tsl) = xs_provider.thermal_scattering(nuc.xs_kernel_idx)
                    && bank.energy[pi] < tsl.energy_max
                {
                    let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                    let thermal_total = tsl.total_xs(bank.energy[pi], t_idx);
                    let delta = thermal_total - xs.elastic;
                    xs.total += delta;
                    xs.elastic = 0.0;
                }

                micro_totals[i] = xs.total;
                micro_xs[i] = xs;
            }

            let macro_total = material.macro_total(&micro_totals[..n_nuclides]);
            if macro_total <= 0.0 {
                bank.alive[pi] = false;
                result.leakage += 1;
                bank.rng_state[pi] = rng.state();
                bank.rng_stream[pi] = rng.stream();
                continue;
            }

            // ── Sample collision distance ──
            let dist_collision = rng.exponential(macro_total);

            // ── Trace to nearest surface ──
            let trace = geometry::ray::trace_step(
                bank.pos[pi],
                bank.dir[pi],
                bank.cell_idx[pi],
                surfaces,
                cells,
            );

            match trace {
                Some(hit) if hit.distance < dist_collision => {
                    // Surface crossing
                    let nudge = (hit.distance * 1e-8).max(1e-8);
                    bank.pos[pi] = bank.pos[pi] + bank.dir[pi] * (hit.distance + nudge);

                    let bc = surfaces[hit.surface_idx].boundary_condition();
                    match bc {
                        BoundaryCondition::Vacuum => {
                            bank.alive[pi] = false;
                            result.leakage += 1;
                        }
                        BoundaryCondition::Reflective => {
                            let normal = surfaces[hit.surface_idx].normal_at(bank.pos[pi]);
                            bank.dir[pi] = bank.dir[pi] - normal * (2.0 * bank.dir[pi].dot(normal));
                        }
                        BoundaryCondition::Transmission => {
                            if let Some(next) = hit.next_cell_idx {
                                bank.cell_idx[pi] = next;
                            } else {
                                bank.alive[pi] = false;
                                result.leakage += 1;
                            }
                        }
                    }
                }
                _ => {
                    // Collision
                    bank.pos[pi] = bank.pos[pi] + bank.dir[pi] * dist_collision;
                    result.collisions += 1;
                    bank.n_collisions[pi] += 1;

                    let nuc_idx = material.sample_nuclide(
                        &micro_totals[..n_nuclides],
                        macro_total,
                        rng.uniform(),
                    );
                    let xs_kernel_idx = material.nuclides[nuc_idx].xs_kernel_idx;

                    // Check for thermal scattering
                    let thermal_active = xs_provider
                        .thermal_scattering(xs_kernel_idx)
                        .is_some_and(|tsl| bank.energy[pi] < tsl.energy_max);

                    if thermal_active {
                        let tsl = xs_provider
                            .thermal_scattering(xs_kernel_idx)
                            .expect("thermal");
                        let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                        let thermal_xs = tsl.total_xs(bank.energy[pi], t_idx);
                        let xi = rng.uniform() * micro_xs[nuc_idx].total;

                        if xi < thermal_xs {
                            // Thermal scattering
                            let (e_out, mu) = tsl.sample(bank.energy[pi], t_idx, &mut rng);
                            bank.energy[pi] = e_out;
                            rotate_direction(&mut bank.dir[pi], mu, &mut rng);
                        } else {
                            // Non-thermal collision
                            let mut particle = make_temp_particle(
                                bank.pos[pi],
                                bank.dir[pi],
                                bank.energy[pi],
                                bank.cell_idx[pi],
                            );
                            let outcome = process_standard_collision(
                                &mut particle,
                                &micro_xs[nuc_idx],
                                xs_kernel_idx,
                                xs_provider,
                                cell.temperature,
                                &mut rng,
                            );
                            bank.energy[pi] = particle.energy;
                            bank.dir[pi] = particle.dir;
                            handle_outcome(outcome, pi, &mut bank, &mut result);
                        }
                    } else {
                        // Standard collision
                        let mut particle = make_temp_particle(
                            bank.pos[pi],
                            bank.dir[pi],
                            bank.energy[pi],
                            bank.cell_idx[pi],
                        );
                        let outcome = process_standard_collision(
                            &mut particle,
                            &micro_xs[nuc_idx],
                            xs_kernel_idx,
                            xs_provider,
                            cell.temperature,
                            &mut rng,
                        );
                        bank.energy[pi] = particle.energy;
                        bank.dir[pi] = particle.dir;
                        handle_outcome(outcome, pi, &mut bank, &mut result);
                    }
                }
            }

            bank.rng_state[pi] = rng.state();
            bank.rng_stream[pi] = rng.stream();
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

fn handle_outcome(
    outcome: CollisionOutcome,
    pi: usize,
    bank: &mut ParticleBank,
    result: &mut EventBatchResult,
) {
    match outcome {
        CollisionOutcome::Scatter => {}
        CollisionOutcome::Absorption => {
            bank.alive[pi] = false;
            result.absorptions += 1;
        }
        CollisionOutcome::Fission { sites } => {
            bank.alive[pi] = false;
            result.fissions += 1;
            result.fission_sites.extend(sites);
        }
    }
}
