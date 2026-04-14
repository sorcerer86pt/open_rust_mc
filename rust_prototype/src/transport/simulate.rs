//! Eigenvalue simulation — power iteration for k_eff.
//!
//! Algorithm:
//!   1. Start with a source bank of neutrons
//!   2. Transport each neutron until absorption or leakage
//!   3. Collect fission sites into the fission bank
//!   4. k_eff = (fission bank size) / (source bank size)
//!   5. Normalize the fission bank -> new source bank
//!   6. Repeat

use rayon::prelude::*;

use crate::geometry::{self, Vec3};
use crate::geometry::cell::{Cell, CellFill};
use crate::geometry::surface::{BoundaryCondition, Surface};
use crate::hdf5_reader::{AngularDistribution, DiscreteLevelInfo, EnergyDistribution};
use crate::physics::collision::{self, CollisionOutcome, InelasticData, MicroXs};
use crate::transport::material::Material;
use crate::transport::particle::{FissionBank, FissionSite, Particle};
use crate::transport::rng::Rng;

/// Configuration for a simulation.
pub struct SimConfig {
    pub batches: u32,
    pub inactive: u32,
    pub particles_per_batch: u32,
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

    fn fission_energy_dist(&self, _nuclide_idx: usize) -> Option<&EnergyDistribution> {
        None
    }

    fn apply_urr(&self, _nuclide_idx: usize, _xs: &mut MicroXs, _energy: f64, _xi: f64) {}
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
}

/// Per-particle transport result for parallel reduction.
struct ParticleResult {
    fission_sites: Vec<FissionSite>,
    leakage: u32,
    absorptions: u32,
    fissions: u32,
    collisions: u32,
}

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
    };

    let (u, v, w) = rng.isotropic_direction();
    let dir = Vec3::new(u, v, w);

    let cell_idx = geometry::ray::find_cell(site.pos, surfaces, cells).unwrap_or(0);
    let mut particle = Particle::new(site.pos, dir, site.energy, cell_idx);

    let max_collisions = 1000;
    while particle.is_alive() && particle.n_collisions < max_collisions {
        let cell = &cells[particle.cell_idx];

        let mat_idx = match cell.fill {
            CellFill::Material(m) => m as usize,
            CellFill::Void | CellFill::Universe(_) => {
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

        // Look up microscopic cross-sections with URR sampling
        let urr_xi = rng.uniform();
        let micro_xs: Vec<MicroXs> = material.nuclides
            .iter()
            .map(|nuc| {
                let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, particle.energy);
                xs_provider.apply_urr(nuc.xs_kernel_idx, &mut xs, particle.energy, urr_xi);
                xs
            })
            .collect();

        let micro_totals: Vec<f64> = micro_xs.iter().map(|x| x.total).collect();
        let macro_total = material.macro_total(&micro_totals);

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
                particle.advance(hit.distance + 1e-10);

                let bc = surfaces[hit.surface_idx].boundary_condition();
                match bc {
                    BoundaryCondition::Vacuum => {
                        particle.kill();
                        result.leakage += 1;
                    }
                    BoundaryCondition::Reflective => {
                        let n = surfaces[hit.surface_idx].normal_at(particle.pos);
                        let d = particle.dir;
                        particle.dir = d - n * (2.0 * d.dot(n));
                    }
                    BoundaryCondition::Transmission => {
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
                    &micro_totals,
                    macro_total,
                    rng.uniform(),
                );

                let xs_kernel_idx = material.nuclides[nuc_idx].xs_kernel_idx;
                let level_info = xs_provider.discrete_level_info(xs_kernel_idx);
                let level_xs = xs_provider.discrete_level_xs(xs_kernel_idx, particle.energy);
                let has_cont = xs_provider.has_continuum_inelastic(xs_kernel_idx);

                let inelastic_data = if !level_info.is_empty() {
                    Some(InelasticData {
                        levels: &level_info,
                        level_xs: &level_xs,
                        has_continuum: has_cont,
                    })
                } else {
                    None
                };

                let elastic_angle = xs_provider.elastic_angular_dist(xs_kernel_idx);
                let fission_edist = xs_provider.fission_energy_dist(xs_kernel_idx);

                let outcome = collision::process_collision(
                    &mut particle,
                    &micro_xs[nuc_idx],
                    inelastic_data.as_ref(),
                    elastic_angle,
                    fission_edist,
                    cell.temperature,
                    &mut rng,
                );

                match outcome {
                    CollisionOutcome::Scatter => {}
                    CollisionOutcome::Absorption => {
                        result.absorptions += 1;
                    }
                    CollisionOutcome::Fission { sites } => {
                        result.fissions += 1;
                        result.fission_sites.extend(sites);
                    }
                }
            }
        }
    }

    result
}

/// Run a k-eigenvalue simulation with rayon parallel transport.
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

    let mut source_bank = initial_source(n, surfaces, cells);

    let mut results = Vec::with_capacity(config.batches as usize);
    let mut k_sum = 0.0;
    let mut k_count = 0_u32;

    for batch in 1..=config.batches {
        // Parallel transport: each particle runs independently
        let particle_results: Vec<ParticleResult> = source_bank
            .par_iter()
            .enumerate()
            .map(|(i, site)| {
                transport_particle(
                    site, batch as u64, i as u64,
                    surfaces, cells, materials, xs_provider,
                )
            })
            .collect();

        // Reduce: merge per-particle results
        let mut fission_bank = FissionBank::new();
        let mut leakage = 0_u32;
        let mut absorptions = 0_u32;
        let mut fissions = 0_u32;
        let mut collisions = 0_u32;

        for pr in particle_results {
            fission_bank.sites.extend(pr.fission_sites);
            leakage += pr.leakage;
            absorptions += pr.absorptions;
            fissions += pr.fissions;
            collisions += pr.collisions;
        }

        let k_batch = fission_bank.len() as f64 / n as f64;

        let result = BatchResult {
            batch,
            k_eff: k_batch,
            leakage,
            absorptions,
            fissions,
            collisions,
        };

        if batch > config.inactive {
            k_sum += k_batch;
            k_count += 1;
        }

        let active = if batch > config.inactive { " *" } else { "" };
        println!(
            "  Batch {batch:>4}: k={k_batch:.5}  collisions={collisions}  \
             fissions={fissions}  leakage={leakage}{active}"
        );

        results.push(result);

        source_bank = normalize_fission_bank(&fission_bank, n, batch);
    }

    let k_final = if k_count > 0 { k_sum / k_count as f64 } else { 0.0 };
    (results, k_final)
}

/// Create an initial source uniformly distributed in the first material cell.
fn initial_source(n: usize, surfaces: &[Surface], cells: &[Cell]) -> Vec<FissionSite> {
    let mut rng = Rng::new(0, 0);
    let mut sites = Vec::with_capacity(n);

    let cell = cells.iter().find(|c| matches!(c.fill, CellFill::Material(_)));
    let aabb = cell.map(|c| c.aabb).unwrap_or(crate::geometry::Aabb::new(
        Vec3::new(-10.0, -10.0, -10.0),
        Vec3::new(10.0, 10.0, 10.0),
    ));

    while sites.len() < n {
        let x = aabb.min.x + rng.uniform() * (aabb.max.x - aabb.min.x);
        let y = aabb.min.y + rng.uniform() * (aabb.max.y - aabb.min.y);
        let z = aabb.min.z + rng.uniform() * (aabb.max.z - aabb.min.z);
        let pos = Vec3::new(x, y, z);

        if geometry::ray::find_cell(pos, surfaces, cells).is_some() {
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

    #[test]
    fn godiva_eigenvalue_smoke_test() {
        let surfaces = vec![
            Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 8.7407,
                bc: BoundaryCondition::Vacuum,
            },
        ];

        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0))
                .with_aabb(crate::geometry::Aabb::new(
                    Vec3::new(-8.7407, -8.7407, -8.7407),
                    Vec3::new(8.7407, 8.7407, 8.7407),
                )),
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
        };

        let (results, k_final) = run_eigenvalue(
            &config, &surfaces, &cells, &materials, &xs_provider,
        );

        assert_eq!(results.len(), 10);
        assert!(k_final > 0.3 && k_final < 3.0,
                "k_final = {k_final} — out of reasonable range");
        println!("\n  Godiva smoke test: k_final = {k_final:.4}");
    }
}
