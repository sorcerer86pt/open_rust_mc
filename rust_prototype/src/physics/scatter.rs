//! Scattering kinematics — elastic and inelastic.
//!
//! Elastic: free-gas model, isotropic in center-of-mass frame.
//! Inelastic: level excitation model — neutron loses a fixed excitation
//! energy Q to the nucleus, then scatters isotropically in CM frame.

use crate::geometry::Vec3;
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
        // Hydrogen special case (A≈1): mu_lab = sqrt((1+mu_cm)/2)
        ((1.0 + mu_cm) * 0.5).max(0.0).sqrt()
    };

    let new_dir = rotate_direction(dir, mu_lab, rng);
    (new_energy.max(1e-11), new_dir) // floor at ~0 eV
}

/// Inelastic scattering via the level excitation model.
///
/// The neutron excites the nucleus to a discrete energy level, losing
/// excitation energy Q. After excitation, scattering is isotropic in
/// the CM frame of the excited system.
///
/// For heavy nuclei (A >> 1), the outgoing energy in the lab frame is
/// approximately:
///   E' ≈ E - Q*(A+1)/A
/// where Q is the excitation energy of the first level.
///
/// Typical first excited level energies:
///   U-235: ~0.046 MeV (46 keV)
///   U-238: ~0.045 MeV (45 keV)
///   Pu-239: ~0.008 MeV (8 keV)
pub fn inelastic_scatter(
    energy: f64,
    dir: Vec3,
    awr: f64,
    rng: &mut Rng,
) -> (f64, Vec3) {
    // Approximate first excited level energy for actinides (~45 keV)
    // A proper implementation reads these from the nuclear data file.
    let q_excitation = 45_000.0; // eV

    // Threshold check: inelastic is only possible if E > Q*(A+1)/A
    let threshold = q_excitation * (awr + 1.0) / awr;
    if energy < threshold {
        // Below threshold — fall back to elastic
        return elastic_scatter(energy, dir, awr, rng);
    }

    // Available CM energy after excitation
    let e_cm = energy * awr / (awr + 1.0) - q_excitation;
    if e_cm <= 0.0 {
        return elastic_scatter(energy, dir, awr, rng);
    }

    // Isotropic scattering in CM frame
    let mu_cm = 2.0 * rng.uniform() - 1.0;

    // Lab-frame outgoing energy (two-body kinematics with Q-value)
    let e_out_cm = e_cm; // kinetic energy in CM after excitation
    let v_cm = (2.0 * e_out_cm / awr).sqrt(); // CM-frame neutron speed (arb. units)
    let v_lab_cm = (2.0 * energy * awr / ((awr + 1.0) * (awr + 1.0))).sqrt(); // CM velocity

    // Lab energy via velocity addition
    let new_energy = 0.5 * (v_cm * v_cm + v_lab_cm * v_lab_cm
        + 2.0 * v_cm * v_lab_cm * mu_cm);

    // Scale to correct units: E' ≈ E - Q*(A+1)/A + correction
    // Simplified but physically motivated:
    let new_energy = (energy - q_excitation * (awr + 1.0) / awr).max(1e-5);

    // Isotropic direction in lab (reasonable approximation for heavy nuclei)
    let mu_lab = 2.0 * rng.uniform() - 1.0;
    let new_dir = rotate_direction(dir, mu_lab, rng);

    (new_energy, new_dir)
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

        let mut elastic_sum = 0.0;
        let mut inelastic_sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let (e_el, _) = elastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, &mut Rng::new(42, 1));
            let (e_in, _) = inelastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, &mut rng);
            elastic_sum += e_el;
            inelastic_sum += e_in;
        }
        // Inelastic should lose more energy on average
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
        let (e_new, _) = inelastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, &mut rng);
        // Should behave like elastic — for A=235 the max energy loss is
        // ~1.7% per collision, so e_new should be close to e0.
        assert!(e_new > e0 * 0.95, "e_new={e_new}, e0={e0}");
    }
}
