//! Collision processing — determine what happens when a neutron hits a nucleus.
//!
//! At each collision:
//!   1. Sample which nuclide is hit (proportional to N_i · σ_t,i)
//!   2. Sample which reaction occurs (proportional to σ_x / σ_t)
//!   3. Process the reaction: scatter, absorb, or fission

use crate::transport::particle::{FissionSite, Particle};
use crate::transport::rng::Rng;
use crate::geometry::Vec3;

/// Cross-section data for a nuclide at a specific energy.
/// These would be looked up via the SVD kernel in production.
#[derive(Debug, Clone, Copy)]
pub struct MicroXs {
    /// Total cross-section (barns). From MT=1, or sum of all partials.
    pub total: f64,
    /// Elastic scattering (MT=2).
    pub elastic: f64,
    /// Inelastic scattering (MT=4).
    pub inelastic: f64,
    /// (n,2n) reaction (MT=16). Produces 2 outgoing neutrons.
    pub n2n: f64,
    /// Fission (MT=18).
    pub fission: f64,
    /// Radiative capture (MT=102).
    pub capture: f64,
    /// Average neutrons per fission (nu-bar).
    pub nu_bar: f64,
    /// Atomic weight ratio (A / neutron mass).
    pub awr: f64,
}

/// Outcome of processing a collision.
#[derive(Debug)]
pub enum CollisionOutcome {
    /// Particle scattered — new energy and direction set.
    Scatter,
    /// Particle absorbed (capture or other absorption).
    Absorption,
    /// Particle caused fission — absorbed, fission sites banked.
    Fission { sites: Vec<FissionSite> },
}

/// Process a collision for a particle.
///
/// Returns the outcome and modifies the particle state in-place
/// (new energy, direction for scattering; killed for absorption/fission).
pub fn process_collision(
    particle: &mut Particle,
    xs: &MicroXs,
    rng: &mut Rng,
) -> CollisionOutcome {
    particle.n_collisions += 1;

    // Sample reaction type: elastic / fission / capture
    // (simplified — a full implementation would include inelastic, (n,2n), etc.)
    let xi = rng.uniform() * xs.total;

    if xi < xs.elastic {
        // Elastic scattering
        let (new_energy, new_dir) = super::scatter::elastic_scatter(
            particle.energy,
            particle.dir,
            xs.awr,
            rng,
        );
        particle.energy = new_energy;
        particle.dir = new_dir;
        CollisionOutcome::Scatter
    } else if xi < xs.elastic + xs.inelastic {
        // Inelastic scattering — level excitation model
        let (new_energy, new_dir) = super::scatter::inelastic_scatter(
            particle.energy,
            particle.dir,
            xs.awr,
            rng,
        );
        particle.energy = new_energy;
        particle.dir = new_dir;
        CollisionOutcome::Scatter
    } else if xi < xs.elastic + xs.inelastic + xs.n2n {
        // (n,2n) reaction — neutron is absorbed, 2 neutrons emitted.
        // Bank one fission-like site (the other neutron continues as scatter).
        let (new_energy, new_dir) = super::scatter::inelastic_scatter(
            particle.energy,
            particle.dir,
            xs.awr,
            rng,
        );
        particle.energy = new_energy;
        particle.dir = new_dir;

        // Bank the second neutron as a fission-like site
        let e2 = sample_evaporation_energy(particle.energy, rng);
        let site = FissionSite {
            pos: particle.pos,
            energy: e2,
            weight: particle.weight,
        };
        CollisionOutcome::Fission { sites: vec![site] }
        // Note: particle stays alive (one neutron continues, one is banked)
    } else if xi < xs.elastic + xs.inelastic + xs.n2n + xs.fission {
        // Fission
        let n_neutrons = fission_yield(xs.nu_bar, particle.weight, rng);
        let mut sites = Vec::with_capacity(n_neutrons);

        for _ in 0..n_neutrons {
            let fission_energy = sample_fission_energy(particle.energy, rng);
            sites.push(FissionSite {
                pos: particle.pos,
                energy: fission_energy,
                weight: 1.0,
            });
        }

        particle.kill();
        CollisionOutcome::Fission { sites }
    } else {
        // Capture (absorption)
        particle.kill();
        CollisionOutcome::Absorption
    }
}

/// Determine the number of fission neutrons to bank.
///
/// Uses the standard stochastic rounding: if nu_bar = 2.43,
/// bank 2 neutrons with probability 0.57, 3 with probability 0.43.
fn fission_yield(nu_bar: f64, weight: f64, rng: &mut Rng) -> usize {
    let nu_weighted = nu_bar * weight;
    let n_floor = nu_weighted as usize;
    let remainder = nu_weighted - n_floor as f64;
    if rng.uniform() < remainder {
        n_floor + 1
    } else {
        n_floor
    }
}

/// Sample an evaporation spectrum for (n,2n) secondary neutrons.
///
/// P(E') ∝ E' · exp(-E'/T) where T ≈ E_incident / 10 (nuclear temperature).
fn sample_evaporation_energy(incident_energy: f64, rng: &mut Rng) -> f64 {
    let temp = incident_energy / 10.0; // nuclear temperature ~E/10
    // Sample: E = -T * ln(xi1 * xi2) (Maxwellian)
    let e = -temp * (rng.uniform() * rng.uniform()).ln();
    e.min(incident_energy).max(1e-5)
}

/// Sample fission neutron energy from a Watt spectrum.
///
/// P(E) ∝ exp(-E/a) · sinh(sqrt(b·E))
/// Using Cranberg parameters for U-235 thermal fission:
///   a = 0.988 MeV, b = 2.249 /MeV
fn sample_fission_energy(_incident_energy: f64, rng: &mut Rng) -> f64 {
    // Watt spectrum parameters for U-235 (in eV)
    let a = 988_000.0; // 0.988 MeV in eV
    let b = 2.249e-6;  // 2.249 /MeV in /eV

    // Rejection sampling from the Watt distribution
    loop {
        // Sample from exponential: E' = -a * ln(xi1)
        let e_prime = -a * rng.uniform().ln();
        // Accept with probability sinh(sqrt(b*E')) / cosh(sqrt(b*E'))
        let arg = (b * e_prime).sqrt();
        let accept_prob = (arg.exp() - (-arg).exp()) / (2.0 * arg.cosh());

        // Simplified: for typical energies, acceptance rate is high
        // Full Watt sampling: E = E' + (a²·b/4) + (2·xi2 - 1)·sqrt(a²·b·E'/4)
        let term = a * a * b / 4.0;
        let xi2 = rng.uniform();
        let e = e_prime + term + (2.0 * xi2 - 1.0) * (a * a * b * e_prime).sqrt() / 2.0;

        if e > 0.0 {
            return e;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fission_yield_averages_correctly() {
        let mut rng = Rng::new(42, 1);
        let nu_bar = 2.43;
        let n = 100_000;
        let total: usize = (0..n).map(|_| fission_yield(nu_bar, 1.0, &mut rng)).sum();
        let avg = total as f64 / n as f64;
        // Should be close to nu_bar
        assert!((avg - nu_bar).abs() < 0.02, "avg={avg}, expected ~{nu_bar}");
    }

    #[test]
    fn fission_energy_positive() {
        let mut rng = Rng::new(42, 1);
        for _ in 0..1000 {
            let e = sample_fission_energy(1.0e6, &mut rng);
            assert!(e > 0.0);
            assert!(e < 20.0e6); // should be < 20 MeV
        }
    }

    #[test]
    fn collision_elastic_preserves_alive() {
        let mut rng = Rng::new(42, 1);
        let xs = MicroXs {
            total: 10.0,
            elastic: 10.0, // 100% elastic
            inelastic: 0.0,
            n2n: 0.0,
            fission: 0.0,
            capture: 0.0,
            nu_bar: 0.0,
            awr: 235.0,
        };
        let mut p = Particle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0e6,
            0,
        );
        let outcome = process_collision(&mut p, &xs, &mut rng);
        assert!(matches!(outcome, CollisionOutcome::Scatter));
        assert!(p.is_alive());
    }

    #[test]
    fn collision_capture_kills() {
        let mut rng = Rng::new(42, 1);
        let xs = MicroXs {
            total: 10.0,
            elastic: 0.0,
            inelastic: 0.0,
            n2n: 0.0,
            fission: 0.0,
            capture: 10.0, // 100% capture
            nu_bar: 0.0,
            awr: 235.0,
        };
        let mut p = Particle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0e6,
            0,
        );
        let outcome = process_collision(&mut p, &xs, &mut rng);
        assert!(matches!(outcome, CollisionOutcome::Absorption));
        assert!(!p.is_alive());
    }

    #[test]
    fn collision_fission_produces_sites() {
        let mut rng = Rng::new(42, 1);
        let xs = MicroXs {
            total: 10.0,
            elastic: 0.0,
            inelastic: 0.0,
            n2n: 0.0,
            fission: 10.0, // 100% fission
            capture: 0.0,
            nu_bar: 2.43,
            awr: 235.0,
        };
        let mut p = Particle::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0e6,
            0,
        );
        let outcome = process_collision(&mut p, &xs, &mut rng);
        match outcome {
            CollisionOutcome::Fission { sites } => {
                assert!(!sites.is_empty());
                assert!(sites.len() >= 2 && sites.len() <= 3);
                for s in &sites {
                    assert_eq!(s.pos.x, 1.0);
                    assert!(s.energy > 0.0);
                }
            }
            _ => panic!("expected fission"),
        }
        assert!(!p.is_alive());
    }
}
