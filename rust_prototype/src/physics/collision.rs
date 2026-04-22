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
#[derive(Debug, Clone, Copy, Default)]
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
    /// Per-level CM-frame angular distribution. Empty slice or any `None`
    /// entry means "isotropic" — matches the old fallback. Aligns 1:1 with
    /// `levels` when populated.
    pub level_angles: &'a [Option<AngularDistribution>],
}

/// A neutron emitted into the current generation (not banked for the
/// next generation's fission source). Used for the non-fission
/// multiplicative channels (n,2n) and (n,3n).
#[derive(Debug, Clone)]
pub struct SecondaryNeutron {
    pub pos: crate::geometry::Vec3,
    pub dir: crate::geometry::Vec3,
    pub energy: f64,
}

/// Outcome of processing a collision.
#[derive(Debug)]
pub enum CollisionOutcome {
    /// Particle scattered — new energy and direction set.
    Scatter,
    /// Particle absorbed (capture or other absorption).
    Absorption,
    /// Particle caused fission — absorbed, fission sites banked for
    /// the NEXT generation's source.
    Fission { sites: Vec<FissionSite> },
    /// Non-fission multiplicative reaction: (n,2n) or (n,3n). The
    /// primary continues (energy/direction already updated in-place);
    /// `secondaries` are additional neutrons at the collision site that
    /// must transport in the CURRENT generation. They do NOT seed the
    /// next generation's fission bank. This mirrors OpenMC / MCNP
    /// convention: only true fission neutrons count toward k_eff.
    Multiplicity { secondaries: Vec<SecondaryNeutron> },
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
        let (q_value, level_idx) =
            sample_inelastic_level(particle.energy, xs.awr, inelastic_data, rng);
        // Discrete level (not continuum / fallback) may carry its own
        // CM-frame angular distribution; else scatter isotropically.
        let angle = level_idx.and_then(|i| {
            inelastic_data
                .and_then(|d| d.level_angles.get(i))
                .and_then(|o| o.as_ref())
        });
        let (new_energy, new_dir) = super::scatter::inelastic_scatter(
            particle.energy,
            particle.dir,
            xs.awr,
            q_value,
            angle,
            rng,
        );
        particle.energy = new_energy;
        particle.dir = new_dir;
        return CollisionOutcome::Scatter;
    }

    // (n,2n) — two neutrons emerge from a compound nucleus. Both are
    // sampled from the evaporation spectrum, isotropic in the LAB
    // frame (standard compound-nucleus approximation for (n,xn)).
    // The primary continues as one of the two; the other is emitted
    // as a CURRENT-generation secondary that the transport loop will
    // pick up from `secondaries`. Neither neutron seeds the next
    // generation's fission bank — that's reserved for MT=18 fission.
    cum += xs.n2n;
    if xi < cum {
        let e_primary = sample_evaporation_energy(particle.energy, rng);
        let e_secondary = sample_evaporation_energy(particle.energy, rng);
        let (u, v, w) = rng.isotropic_direction();
        particle.energy = e_primary;
        particle.dir = crate::geometry::Vec3::new(u, v, w);
        let (us, vs, ws) = rng.isotropic_direction();
        let secondary = SecondaryNeutron {
            pos: particle.pos,
            dir: crate::geometry::Vec3::new(us, vs, ws),
            energy: e_secondary,
        };
        return CollisionOutcome::Multiplicity {
            secondaries: vec![secondary],
        };
    }

    // (n,3n) — three neutrons emerge. Primary continues, two
    // secondaries transport in current generation. Same evaporation /
    // isotropic approximation as (n,2n).
    cum += xs.n3n;
    if xi < cum {
        let e_primary = sample_evaporation_energy(particle.energy, rng);
        let e_s1 = sample_evaporation_energy(particle.energy, rng);
        let e_s2 = sample_evaporation_energy(particle.energy, rng);
        let (u, v, w) = rng.isotropic_direction();
        particle.energy = e_primary;
        particle.dir = crate::geometry::Vec3::new(u, v, w);
        let (u1, v1, w1) = rng.isotropic_direction();
        let (u2, v2, w2) = rng.isotropic_direction();
        let secondaries = vec![
            SecondaryNeutron {
                pos: particle.pos,
                dir: crate::geometry::Vec3::new(u1, v1, w1),
                energy: e_s1,
            },
            SecondaryNeutron {
                pos: particle.pos,
                dir: crate::geometry::Vec3::new(u2, v2, w2),
                energy: e_s2,
            },
        ];
        return CollisionOutcome::Multiplicity { secondaries };
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
) -> (f64, Option<usize>) {
    let data = match inelastic_data {
        Some(d) if !d.levels.is_empty() && !d.level_xs.is_empty() => d,
        _ => return (-45_000.0, None), // fallback: ~45 keV excitation
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
        return (-45_000.0, None); // fallback
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
                let a_param = awr / 8.0; // level density parameter (MeV^-1)
                let e_cm_mev = energy * awr / ((awr + 1.0) * 1.0e6);
                let e_excitation = e_cm_mev.max(0.1);
                let temp_mev = (e_excitation / a_param).sqrt();
                let e_out_mev = -temp_mev * (rng.uniform() * rng.uniform()).ln();
                let e_out_mev = e_out_mev.min(e_cm_mev * 0.9);
                let q_eff = -(e_cm_mev - e_out_mev) * 1.0e6;
                // Continuum: no discrete angular distribution — isotropic.
                return (q_eff, None);
            }
            return (level.q_value, Some(idx));
        }
    }

    let last = accessible.last().map_or(0, |&(i, _)| i);
    (data.levels[last].q_value, Some(last))
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
    let b = 2.249e-6; // 2.249 /MeV in /eV

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
        let mut p = Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0e6, 0);
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
        let mut p = Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0e6, 0);
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
        let mut p = Particle::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 0.0, 0.0), 1.0e6, 0);
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
            DiscreteLevelInfo {
                mt: 51,
                q_value: -76.8,
                threshold: 77.1,
            },
            DiscreteLevelInfo {
                mt: 52,
                q_value: -13040.0,
                threshold: 13095.5,
            },
            DiscreteLevelInfo {
                mt: 53,
                q_value: -46200.0,
                threshold: 46396.6,
            },
        ];
        let level_xs = vec![0.5, 0.3, 0.2];
        let data = InelasticData {
            levels: &levels,
            level_xs: &level_xs,
            has_continuum: false,
            level_angles: &[],
        };

        let mut p = Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0e6, 0);
        let outcome = process_collision(&mut p, &xs, Some(&data), None, None, 0.0, &mut rng);
        assert!(matches!(outcome, CollisionOutcome::Scatter));
        assert!(p.is_alive());
        assert!(p.energy < 1.0e6); // should have lost energy
    }

    // ── Per-level MT=51-91 anisotropic angular distribution plumbing ──
    //
    // Sampling a level whose CM-frame distribution is forward-peaked should
    // yield ⟨mu_cm⟩ > 0 — the final lab-frame scatter direction shows a
    // clear forward preference against the incident axis. The old isotropic
    // behaviour gave ⟨mu_cm⟩ ≈ 0 and ⟨dir.x⟩ near a nuclide-specific value
    // dominated by the recoil kinematics (same for every incident).

    fn build_forward_peaked_angle() -> crate::hdf5_reader::AngularDistribution {
        use crate::hdf5_reader::{AngularDistribution, TabularMuDist};
        // Single-energy tab: essentially a delta at mu_cm = +1 (linear CDF
        // from 0.9 .. 1.0 over [0.9, 1.0], zero below).
        let d = TabularMuDist {
            mu: vec![-1.0, 0.9, 1.0],
            cdf: vec![0.0, 0.0, 1.0],
        };
        AngularDistribution {
            energies: vec![1.0, 2.0e7],
            distributions: vec![
                TabularMuDist {
                    mu: d.mu.clone(),
                    cdf: d.cdf.clone(),
                },
                d,
            ],
            center_of_mass: true,
        }
    }

    #[test]
    fn per_level_angular_dist_is_used_when_provided() {
        let mut rng = Rng::new(2026, 1);
        let xs = MicroXs {
            total: 1.0,
            elastic: 0.0,
            inelastic: 1.0,
            n2n: 0.0,
            n3n: 0.0,
            fission: 0.0,
            capture: 0.0,
            nu_bar: 0.0,
            awr: 235.0,
        };
        let levels = vec![DiscreteLevelInfo {
            mt: 51,
            q_value: -50_000.0,
            threshold: 50_213.0,
        }];
        let level_xs = vec![1.0];
        let angle = build_forward_peaked_angle();
        let angles = vec![Some(angle)];
        let data = InelasticData {
            levels: &levels,
            level_xs: &level_xs,
            has_continuum: false,
            level_angles: &angles,
        };

        let mut sum_x = 0.0;
        let trials = 4000;
        for _ in 0..trials {
            let mut p = Particle::new(
                crate::geometry::Vec3::new(0.0, 0.0, 0.0),
                crate::geometry::Vec3::new(1.0, 0.0, 0.0),
                1.0e6,
                0,
            );
            let outcome = process_collision(&mut p, &xs, Some(&data), None, None, 0.0, &mut rng);
            assert!(matches!(outcome, CollisionOutcome::Scatter));
            sum_x += p.dir.x;
        }
        // Forward-peaked CM distribution → lab direction biased forward:
        // ⟨dir.x⟩ must be well above 0.5; isotropic gave ~0.0.
        let mean_x = sum_x / trials as f64;
        assert!(
            mean_x > 0.5,
            "expected forward bias, got mean dir.x = {mean_x}"
        );
    }

    #[test]
    fn per_level_fallback_to_isotropic_when_angles_empty() {
        let mut rng = Rng::new(2027, 1);
        let xs = MicroXs {
            total: 1.0,
            elastic: 0.0,
            inelastic: 1.0,
            n2n: 0.0,
            n3n: 0.0,
            fission: 0.0,
            capture: 0.0,
            nu_bar: 0.0,
            awr: 235.0,
        };
        let levels = vec![DiscreteLevelInfo {
            mt: 51,
            q_value: -50_000.0,
            threshold: 50_213.0,
        }];
        let level_xs = vec![1.0];
        let data = InelasticData {
            levels: &levels,
            level_xs: &level_xs,
            has_continuum: false,
            level_angles: &[], // empty → isotropic path
        };
        let mut sum_x = 0.0;
        let trials = 4000;
        for _ in 0..trials {
            let mut p = Particle::new(
                crate::geometry::Vec3::new(0.0, 0.0, 0.0),
                crate::geometry::Vec3::new(1.0, 0.0, 0.0),
                1.0e6,
                0,
            );
            let _ = process_collision(&mut p, &xs, Some(&data), None, None, 0.0, &mut rng);
            sum_x += p.dir.x;
        }
        let mean_x = sum_x / trials as f64;
        // Isotropic in CM plus two-body kinematics on a heavy nucleus (A=235)
        // leaves a weak forward bias around 0.5. Must stay well below the
        // forward-peaked case (> 0.5 there, well over 0.8 in practice).
        assert!(
            mean_x < 0.6,
            "isotropic should not be strongly forward: {mean_x}"
        );
    }
}
