//! Scattering kinematics — elastic and inelastic.
//!
//! Elastic: free-gas model, isotropic in center-of-mass frame.
//! Inelastic: level excitation model — neutron loses excitation energy Q
//! to the nucleus, then scatters isotropically in CM frame.
//! Q-value is now passed in from the caller (read from nuclear data).

use crate::geometry::Vec3;
use crate::hdf5_reader::AngularDistribution;
use crate::transport::rng::Rng;

/// Elastic scattering: compute new energy and direction.
///
/// `awr` is the atomic weight ratio (A / neutron mass).
/// Scattering is isotropic in the center-of-mass frame.
pub fn elastic_scatter(
    energy: f64,
    dir: Vec3,
    awr: f64,
    rng: &mut Rng,
) -> (f64, Vec3) {
    // Sample cosine of scattering angle in center-of-mass frame
    let mu_cm = 2.0 * rng.uniform() - 1.0;

    // Convert to lab frame energy
    let alpha = ((awr - 1.0) / (awr + 1.0)).powi(2);
    let new_energy = energy * 0.5 * ((1.0 + alpha) + (1.0 - alpha) * mu_cm);

    // Lab-frame scattering cosine
    let mu_lab = if awr > 1.0 + 1e-10 {
        (1.0 + awr * mu_cm) / (1.0 + 2.0 * awr * mu_cm + awr * awr).sqrt()
    } else {
        // Hydrogen special case (A~1): mu_lab = sqrt((1+mu_cm)/2)
        ((1.0 + mu_cm) * 0.5).max(0.0).sqrt()
    };

    let new_dir = rotate_direction(dir, mu_lab, rng);
    (new_energy.max(1e-11), new_dir) // floor at ~0 eV
}

/// Elastic scattering with optional anisotropic angular distribution
/// and free gas thermal scattering correction.
///
/// If `angle_dist` is provided, samples mu from the tabulated distribution.
/// If `temperature > 0`, applies the free gas thermal model: sample target
/// velocity from Maxwell-Boltzmann, scatter in relative frame.
/// For fast neutrons (E >> kT), the cold target approximation is used.
pub fn elastic_scatter_aniso(
    energy: f64,
    dir: Vec3,
    awr: f64,
    angle_dist: Option<&AngularDistribution>,
    temperature: f64,
    rng: &mut Rng,
) -> (f64, Vec3) {
    // Boltzmann constant in eV/K
    const K_BOLTZMANN: f64 = 8.617_333e-5;
    let kt = K_BOLTZMANN * temperature;

    // Use free gas model when neutron energy is comparable to thermal energy
    // Threshold: E < 400 * kT (OpenMC uses a similar cutoff)
    if temperature > 0.0 && energy < 400.0 * kt {
        return free_gas_scatter(energy, dir, awr, kt, angle_dist, rng);
    }

    // Cold target approximation (fast neutrons)
    let mu_cm = match angle_dist {
        Some(dist) if dist.center_of_mass => dist.sample_mu(energy, rng),
        _ => 2.0 * rng.uniform() - 1.0,
    };

    let alpha = ((awr - 1.0) / (awr + 1.0)).powi(2);
    let new_energy = energy * 0.5 * ((1.0 + alpha) + (1.0 - alpha) * mu_cm);

    let mu_lab = if awr > 1.0 + 1e-10 {
        (1.0 + awr * mu_cm) / (1.0 + 2.0 * awr * mu_cm + awr * awr).sqrt()
    } else {
        ((1.0 + mu_cm) * 0.5).max(0.0).sqrt()
    };

    let new_dir = rotate_direction(dir, mu_lab, rng);
    (new_energy.max(1e-11), new_dir)
}

/// Free gas thermal scattering: target nucleus has thermal motion.
///
/// 1. Sample target velocity from Maxwell-Boltzmann
/// 2. Compute relative velocity
/// 3. Scatter in the center-of-mass frame of the relative motion
/// 4. Transform back to lab frame
fn free_gas_scatter(
    energy: f64,
    dir: Vec3,
    awr: f64,
    kt: f64,
    angle_dist: Option<&AngularDistribution>,
    rng: &mut Rng,
) -> (f64, Vec3) {
    // Neutron speed (proportional to sqrt(2*E/m), m=1)
    let v_n = (2.0 * energy).sqrt();

    // Target speed from Maxwell-Boltzmann: P(v) ~ v^2 * exp(-A*v^2/(2*kT))
    // Sample using: v_target = sqrt(2*kT/A) * chi(3) where chi(3) is chi-distribution
    // Simplified: v_t = sqrt(-2*kT/A * ln(xi1)) for the magnitude (Maxwellian speed)
    let sigma = (kt / awr).sqrt(); // thermal speed parameter
    // Sample speed from Maxwell distribution using Box-Muller for 3 components
    let vx = sigma * (-2.0 * rng.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * rng.uniform()).cos();
    let vy = sigma * (-2.0 * rng.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * rng.uniform()).cos();
    let vz = sigma * (-2.0 * rng.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * rng.uniform()).cos();

    // Target velocity vector (in the same units as neutron speed)
    let v_target = Vec3::new(vx, vy, vz);

    // Neutron velocity vector
    let v_neutron = dir * v_n;

    // Relative velocity
    let v_rel = v_neutron - v_target;
    let v_rel_mag = v_rel.length();

    if v_rel_mag < 1e-20 {
        return (energy, dir); // no relative motion
    }

    // Relative energy: E_rel = 0.5 * mu_reduced * v_rel^2
    // where mu_reduced = A/(A+1) (reduced mass in neutron mass units)
    let mu_reduced = awr / (awr + 1.0);
    let e_rel = 0.5 * mu_reduced * v_rel_mag * v_rel_mag;

    // Sample scattering angle in CM frame
    let mu_cm = match angle_dist {
        Some(dist) if dist.center_of_mass => dist.sample_mu(e_rel, rng),
        _ => 2.0 * rng.uniform() - 1.0,
    };

    // CM velocity (velocity of the center of mass in lab frame)
    let v_cm = (v_neutron + v_target * awr) * (1.0 / (1.0 + awr));

    // Neutron velocity in CM frame before collision
    let v_n_cm_dir = v_rel.normalized();
    let _v_n_cm_mag = v_rel_mag / (1.0 + awr);
    // Actually: v_n_cm = v_rel * A/(A+1), and after elastic collision magnitude is preserved

    // After elastic collision in CM: speed is preserved, direction changes
    let new_v_n_cm_dir = rotate_direction(v_n_cm_dir, mu_cm, rng);
    let v_n_cm_after = new_v_n_cm_dir * (v_rel_mag * awr / (1.0 + awr));

    // Transform back to lab frame
    let v_n_lab = v_n_cm_after + v_cm;
    let v_n_lab_mag = v_n_lab.length();

    if v_n_lab_mag < 1e-20 {
        return (1e-11, dir);
    }

    let new_energy = 0.5 * v_n_lab_mag * v_n_lab_mag; // E = 0.5 * m * v^2, m=1
    let new_dir = v_n_lab * (1.0 / v_n_lab_mag);

    (new_energy.max(1e-11), new_dir)
}

/// Inelastic scattering via the level excitation model.
///
/// The neutron excites the nucleus to a discrete energy level, losing
/// excitation energy |Q|. After excitation, scattering is isotropic in
/// the CM frame of the excited system.
///
/// `q_value` is in eV, negative for excitation (endothermic).
/// Proper two-body kinematics with the exact Q-value.
pub fn inelastic_scatter(
    energy: f64,
    dir: Vec3,
    awr: f64,
    q_value: f64,
    angle: Option<&crate::hdf5_reader::AngularDistribution>,
    rng: &mut Rng,
) -> (f64, Vec3) {
    // Threshold check: inelastic is only possible if E > |Q|*(A+1)/A
    let threshold = if q_value < 0.0 {
        (-q_value) * (awr + 1.0) / awr
    } else {
        0.0
    };

    if energy < threshold {
        return elastic_scatter(energy, dir, awr, rng);
    }

    // Two-body kinematics in the center-of-mass frame
    // E_cm = E_lab * A / (A+1)
    let e_cm = energy * awr / (awr + 1.0);

    // Available kinetic energy in CM after excitation: E_cm + Q
    let e_cm_out = e_cm + q_value;
    if e_cm_out <= 0.0 {
        return elastic_scatter(energy, dir, awr, rng);
    }

    // CM-frame scattering cosine: prefer the ENDF tabulated angular
    // distribution (OpenMC UncorrelatedAngleEnergy); fall back to isotropic
    // when the nuclide's evaluation does not store one for this level.
    let mu_cm = match angle {
        Some(dist) => dist.sample_mu(energy, rng),
        None => 2.0 * rng.uniform() - 1.0,
    };

    // CM velocity of the system (in sqrt-energy units)
    // v_cm_system = sqrt(E / (A+1)^2) * (A+1) ... simplified:
    // We use the standard two-body lab energy formula:
    // E_lab_out = E_cm_out * [(1+A*mu)^2 + A^2*(1-mu^2)] / (1+A)^2
    // More precisely: E' = E_cm_out/(A+1)^2 * (1 + A^2 + 2*A*mu_cm) + E/(A+1)^2
    // But the exact formula from OpenMC is:
    //
    // v_n = sqrt(2 * E_cm_out / A_cm)  (neutron speed in CM after collision)
    // v_cm = sqrt(2 * E / (A+1)^2)     (CM system speed in lab)
    // E_lab_out = 0.5 * (v_n^2 + v_cm^2 + 2*v_n*v_cm*mu_cm)

    let a_plus_1 = awr + 1.0;

    // Neutron speed in CM after collision (in energy-equivalent units)
    // KE_neutron_cm = E_cm_out * A/(A+1) for the neutron share
    // Actually for the outgoing channel with Q-value:
    // The neutron gets fraction A/(A+1) of the available CM energy
    let e_neutron_cm = e_cm_out * awr / a_plus_1;
    let v_n = (2.0 * e_neutron_cm).sqrt();

    // CM system speed in lab frame
    let v_cm_sys = (2.0 * energy / (a_plus_1 * a_plus_1)).sqrt();

    // Lab frame energy via velocity addition
    let e_lab_out = 0.5 * (v_n * v_n + v_cm_sys * v_cm_sys + 2.0 * v_n * v_cm_sys * mu_cm);

    // Clamp to physical bounds
    let e_lab_out = e_lab_out.max(1e-5);

    // Lab scattering cosine
    let mu_lab = if v_n + v_cm_sys > 1e-20 {
        (v_cm_sys + v_n * mu_cm) / (v_n * v_n + v_cm_sys * v_cm_sys + 2.0 * v_n * v_cm_sys * mu_cm).sqrt()
    } else {
        2.0 * rng.uniform() - 1.0
    };
    let mu_lab = mu_lab.clamp(-1.0, 1.0);

    let new_dir = rotate_direction(dir, mu_lab, rng);
    (e_lab_out, new_dir)
}

/// Rotate a direction vector by a polar angle (given by its cosine)
/// and a uniformly sampled azimuthal angle.
fn rotate_direction(dir: Vec3, mu: f64, rng: &mut Rng) -> Vec3 {
    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    let (u, v, w) = (dir.x, dir.y, dir.z);

    // Check if the direction is nearly along the z-axis
    if w.abs() > 0.999_999 {
        let sign = w.signum();
        return Vec3::new(
            sin_theta * cos_phi,
            sign * sin_theta * sin_phi,
            sign * mu,
        );
    }

    // General rotation formula
    let inv_sqrt = 1.0 / (1.0 - w * w).sqrt();
    Vec3::new(
        mu * u + sin_theta * (u * w * cos_phi - v * sin_phi) * inv_sqrt,
        mu * v + sin_theta * (v * w * cos_phi + u * sin_phi) * inv_sqrt,
        mu * w - sin_theta * cos_phi * (1.0 - w * w) * inv_sqrt,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_energy_bounds() {
        let mut rng = Rng::new(42, 1);
        let e0 = 1.0e6; // 1 MeV
        let awr = 1.0; // hydrogen

        for _ in 0..10_000 {
            let (e_new, _) = elastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, &mut rng);
            assert!(e_new >= 0.0);
            assert!(e_new <= e0 * 1.0001);
        }
    }

    #[test]
    fn elastic_heavy_nucleus_small_energy_loss() {
        let mut rng = Rng::new(42, 1);
        let e0 = 1.0e6;
        let awr = 238.0;

        let mut total_ratio = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let (e_new, _) = elastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, &mut rng);
            total_ratio += e_new / e0;
        }
        let avg_ratio = total_ratio / n as f64;
        assert!(avg_ratio > 0.99, "avg ratio = {avg_ratio}");
    }

    #[test]
    fn elastic_direction_unit_vector() {
        let mut rng = Rng::new(42, 1);
        for _ in 0..1000 {
            let dir = Vec3::new(0.5, 0.5, 1.0 / 2.0_f64.sqrt()).normalized();
            let (_, new_dir) = elastic_scatter(1.0e6, dir, 12.0, &mut rng);
            let len = new_dir.length();
            assert!((len - 1.0).abs() < 1e-6, "len = {len}");
        }
    }

    #[test]
    fn inelastic_loses_more_energy() {
        let mut rng = Rng::new(42, 1);
        let e0 = 1.0e6; // 1 MeV (above threshold)
        let awr = 235.0;
        let q = -45_000.0; // 45 keV excitation

        let mut elastic_sum = 0.0;
        let mut inelastic_sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let (e_el, _) = elastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, &mut Rng::new(42, 1));
            let (e_in, _) = inelastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, q, None, &mut rng);
            elastic_sum += e_el;
            inelastic_sum += e_in;
        }
        let avg_inelastic = inelastic_sum / n as f64;
        let avg_elastic = elastic_sum / n as f64;
        assert!(avg_inelastic < avg_elastic,
                "inelastic avg = {avg_inelastic}, elastic avg = {avg_elastic}");
    }

    #[test]
    fn inelastic_below_threshold_falls_back() {
        let mut rng = Rng::new(42, 1);
        let e0 = 1000.0; // 1 keV — below 45 keV threshold
        let awr = 235.0;
        let q = -45_000.0;
        let (e_new, _) = inelastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, q, None, &mut rng);
        assert!(e_new > e0 * 0.95, "e_new={e_new}, e0={e0}");
    }

    #[test]
    fn inelastic_with_real_q_values() {
        let mut rng = Rng::new(42, 1);
        let e0 = 1.0e6; // 1 MeV
        let awr = 235.0;

        // Test different Q-values (U-235 discrete levels)
        let q_values = [-76.8, -13_040.0, -46_200.0, -103_000.0, -293_000.0];
        let mut prev_avg = e0;
        for &q in &q_values {
            let mut sum = 0.0;
            let n = 1000;
            for _ in 0..n {
                let (e_out, _) = inelastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, q, None, &mut rng);
                sum += e_out;
            }
            let avg = sum / n as f64;
            // Larger |Q| should give more energy loss on average
            assert!(avg < prev_avg, "Q={q}: avg={avg} should be < {prev_avg}");
            prev_avg = avg;
        }
    }
}
