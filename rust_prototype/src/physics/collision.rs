//! Collision processing — determine what happens when a neutron hits a nucleus.
//!
//! At each collision:
//!   1. Sample which nuclide is hit (proportional to N_i · sigma_t,i)
//!   2. Sample which reaction occurs (proportional to sigma_x / sigma_t)
//!   3. Process the reaction: scatter, absorb, or fission

use crate::hdf5_reader::{AngularDistribution, DiscreteLevelInfo, EnergyDistribution};
use crate::transport::particle::{FissionSite, Particle};
use crate::transport::rng::Rng;

/// Cross-section data for a nuclide at a specific energy.
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
    /// (n,3n) reaction (MT=17). Produces 3 outgoing neutrons.
    pub n3n: f64,
    /// Fission (MT=18).
    pub fission: f64,
    /// Radiative capture (MT=102).
    pub capture: f64,
    /// Average neutrons per fission (nu-bar), energy-dependent.
    pub nu_bar: f64,
    /// Atomic weight ratio (A / neutron mass).
    pub awr: f64,
}

/// Additional per-nuclide data needed for detailed inelastic scattering.
pub struct InelasticData<'a> {
    /// Discrete level info (MT=51-91) with Q-values and thresholds.
    pub levels: &'a [DiscreteLevelInfo],
    /// Cross-sections for each discrete level at the current energy.
    pub level_xs: &'a [f64],
    /// Whether continuum inelastic (MT=91) is included.
    pub has_continuum: bool,
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
/// `inelastic_data` provides discrete level information for proper inelastic
/// kinematics. If `None`, falls back to the simplified single-level model.
/// `elastic_angle` provides anisotropic scattering angular distribution.
/// `fission_edist` provides the fission outgoing energy spectrum from HDF5.
/// `temperature` is the cell temperature in Kelvin for free gas scattering.
pub fn process_collision(
    particle: &mut Particle,
    xs: &MicroXs,
    inelastic_data: Option<&InelasticData<'_>>,
    elastic_angle: Option<&AngularDistribution>,
    fission_edist: Option<&EnergyDistribution>,
    temperature: f64,
    rng: &mut Rng,
) -> CollisionOutcome {
    particle.n_collisions += 1;

    let xi = rng.uniform() * xs.total;
    let mut cum = 0.0;

    // Elastic scattering
    cum += xs.elastic;
    if xi < cum {
        let (new_energy, new_dir) = super::scatter::elastic_scatter_aniso(
            particle.energy,
            particle.dir,
            xs.awr,
            elastic_angle,
            temperature,
            rng,
        );
        particle.energy = new_energy;
        particle.dir = new_dir;
        return CollisionOutcome::Scatter;
    }

    // Inelastic scattering
    cum += xs.inelastic;
    if xi < cum {
        let q_value = sample_inelastic_level(particle.energy, xs.awr, inelastic_data, rng);
        let (new_energy, new_dir) = super::scatter::inelastic_scatter(
            particle.energy,
            particle.dir,
            xs.awr,
            q_value,
            rng,
        );
        particle.energy = new_energy;
        particle.dir = new_dir;
        return CollisionOutcome::Scatter;
    }

    // (n,2n) reaction
    cum += xs.n2n;
    if xi < cum {
        let e2 = sample_evaporation_energy(particle.energy, rng);
        let site = FissionSite {
            pos: particle.pos,
            energy: e2,
            weight: particle.weight,
        };
        // Primary neutron continues with reduced energy
        let (new_energy, new_dir) = super::scatter::inelastic_scatter(
            particle.energy,
            particle.dir,
            xs.awr,
            -particle.energy * 0.1, // approximate Q for (n,2n)
            rng,
        );
        particle.energy = new_energy;
        particle.dir = new_dir;
        return CollisionOutcome::Fission { sites: vec![site] };
    }

    // (n,3n) reaction
    cum += xs.n3n;
    if xi < cum {
        let e2 = sample_evaporation_energy(particle.energy, rng);
        let e3 = sample_evaporation_energy(particle.energy, rng);
        let sites = vec![
            FissionSite { pos: particle.pos, energy: e2, weight: particle.weight },
            FissionSite { pos: particle.pos, energy: e3, weight: particle.weight },
        ];
        // Primary neutron continues with reduced energy
        let (new_energy, new_dir) = super::scatter::inelastic_scatter(
            particle.energy,
            particle.dir,
            xs.awr,
            -particle.energy * 0.2, // approximate Q for (n,3n)
            rng,
        );
        particle.energy = new_energy;
        particle.dir = new_dir;
        return CollisionOutcome::Fission { sites };
    }

    // Fission
    cum += xs.fission;
    if xi < cum {
        let n_neutrons = fission_yield(xs.nu_bar, particle.weight, rng);
        let mut sites = Vec::with_capacity(n_neutrons);

        for _ in 0..n_neutrons {
            let fission_energy = match fission_edist {
                Some(dist) => dist.sample(particle.energy, rng),
                None => sample_fission_energy(particle.energy, rng),
            };
            sites.push(FissionSite {
                pos: particle.pos,
                energy: fission_energy,
                weight: 1.0,
            });
        }

        particle.kill();
        return CollisionOutcome::Fission { sites };
    }

    // Capture (absorption)
    particle.kill();
    CollisionOutcome::Absorption
}

/// Sample which discrete inelastic level is excited and return its Q-value.
///
/// If discrete level data is available, sample proportionally to each level's
/// cross-section. If the selected level is continuum (MT=91), sample from an
/// evaporation spectrum instead (handled by returning a special large-negative Q).
fn sample_inelastic_level(
    energy: f64,
    awr: f64,
    inelastic_data: Option<&InelasticData<'_>>,
    rng: &mut Rng,
) -> f64 {
    let data = match inelastic_data {
        Some(d) if !d.levels.is_empty() && !d.level_xs.is_empty() => d,
        _ => return -45_000.0, // fallback: ~45 keV excitation
    };

    // Sum cross-sections for levels that are energetically accessible
    let mut xs_sum = 0.0;
    let mut accessible = Vec::new();
    for (i, level) in data.levels.iter().enumerate() {
        if i < data.level_xs.len() && energy > level.threshold && data.level_xs[i] > 0.0 {
            xs_sum += data.level_xs[i];
            accessible.push((i, data.level_xs[i]));
        }
    }

    if accessible.is_empty() || xs_sum <= 0.0 {
        return -45_000.0; // fallback
    }

    // Sample proportionally to cross-section
    let xi = rng.uniform() * xs_sum;
    let mut cum = 0.0;
    for &(idx, xs) in &accessible {
        cum += xs;
        if xi < cum {
            let level = &data.levels[idx];
            if level.mt == 91 && data.has_continuum {
                // Continuum inelastic: compute effective Q from evaporation model
                // E* = E_cm - S_n (neutron separation energy)
                // Use evaporation temperature: T = sqrt(E*/a), a ~ A/8
                let a_param = awr / 8.0; // level density parameter (MeV^-1)
                let e_cm_mev = energy * awr / ((awr + 1.0) * 1.0e6); // CM energy in MeV
                let e_excitation = e_cm_mev.max(0.1); // minimum excitation
                let temp_mev = (e_excitation / a_param).sqrt();
                // Sample outgoing CM energy from Maxwellian: E = -T * ln(xi1*xi2)
                let e_out_mev = -temp_mev * (rng.uniform() * rng.uniform()).ln();
                let e_out_mev = e_out_mev.min(e_cm_mev * 0.9); // can't exceed available
                // Effective Q = -(E_cm - E_out) * (A+1)/A in eV
                let q_eff = -(e_cm_mev - e_out_mev) * 1.0e6;
                return q_eff;
            }
            return level.q_value;
        }
    }

    // Shouldn't reach here, but return last level's Q
    data.levels[accessible.last().map_or(0, |&(i, _)| i)].q_value
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

/// Sample an evaporation spectrum for (n,xn) secondary neutrons.
///
/// P(E') ~ E' * exp(-E'/T) where T ~ E_incident / 10 (nuclear temperature).
fn sample_evaporation_energy(incident_energy: f64, rng: &mut Rng) -> f64 {
    let temp = incident_energy / 10.0;
    let e = -temp * (rng.uniform() * rng.uniform()).ln();
    e.min(incident_energy).max(1e-5)
}

/// Sample fission neutron energy from a Watt spectrum.
///
/// P(E) ~ exp(-E/a) * sinh(sqrt(b*E))
/// Using Cranberg parameters for U-235 thermal fission:
///   a = 0.988 MeV, b = 2.249 /MeV
fn sample_fission_energy(_incident_energy: f64, rng: &mut Rng) -> f64 {
    let a = 988_000.0; // 0.988 MeV in eV
    let b = 2.249e-6;  // 2.249 /MeV in /eV

    loop {
        let e_prime = -a * rng.uniform().ln();
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
    use crate::geometry::Vec3;

    #[test]
    fn fission_yield_averages_correctly() {
        let mut rng = Rng::new(42, 1);
        let nu_bar = 2.43;
        let n = 100_000;
        let total: usize = (0..n).map(|_| fission_yield(nu_bar, 1.0, &mut rng)).sum();
        let avg = total as f64 / n as f64;
        assert!((avg - nu_bar).abs() < 0.02, "avg={avg}, expected ~{nu_bar}");
    }

    #[test]
    fn fission_energy_positive() {
        let mut rng = Rng::new(42, 1);
        for _ in 0..1000 {
            let e = sample_fission_energy(1.0e6, &mut rng);
            assert!(e > 0.0);
            assert!(e < 20.0e6);
        }
    }

    #[test]
    fn collision_elastic_preserves_alive() {
        let mut rng = Rng::new(42, 1);
        let xs = MicroXs {
            total: 10.0,
            elastic: 10.0,
            inelastic: 0.0,
            n2n: 0.0,
            n3n: 0.0,
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
        let outcome = process_collision(&mut p, &xs, None, None, None, 0.0, &mut rng);
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
            n3n: 0.0,
            fission: 0.0,
            capture: 10.0,
            nu_bar: 0.0,
            awr: 235.0,
        };
        let mut p = Particle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0e6,
            0,
        );
        let outcome = process_collision(&mut p, &xs, None, None, None, 0.0, &mut rng);
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
            n3n: 0.0,
            fission: 10.0,
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
        let outcome = process_collision(&mut p, &xs, None, None, None, 0.0, &mut rng);
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

    #[test]
    fn inelastic_with_discrete_levels() {
        let mut rng = Rng::new(42, 1);
        let xs = MicroXs {
            total: 10.0,
            elastic: 0.0,
            inelastic: 10.0,
            n2n: 0.0,
            n3n: 0.0,
            fission: 0.0,
            capture: 0.0,
            nu_bar: 0.0,
            awr: 235.0,
        };
        let levels = vec![
            DiscreteLevelInfo { mt: 51, q_value: -76.8, threshold: 77.1 },
            DiscreteLevelInfo { mt: 52, q_value: -13040.0, threshold: 13095.5 },
            DiscreteLevelInfo { mt: 53, q_value: -46200.0, threshold: 46396.6 },
        ];
        let level_xs = vec![0.5, 0.3, 0.2];
        let data = InelasticData {
            levels: &levels,
            level_xs: &level_xs,
            has_continuum: false,
        };

        let mut p = Particle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0e6,
            0,
        );
        let outcome = process_collision(&mut p, &xs, Some(&data), None, None, 0.0, &mut rng);
        assert!(matches!(outcome, CollisionOutcome::Scatter));
        assert!(p.is_alive());
        assert!(p.energy < 1.0e6); // should have lost energy
    }
}
