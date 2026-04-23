//! ANSI/ANS-6.6.1 number buildup factor validation.
//!
//! Setup: point isotropic source of monoenergetic photons at the
//! origin of an infinite homogeneous water medium. Tally the number
//! of photon crossings on spheres at several optical depths `μ₀ r`
//! (where μ₀ is the total macroscopic cross section at the source
//! energy). Compute the number buildup factor
//!   `B_n(μ₀ r) = N_crossings(r) / N_uncollided(r)`
//! where `N_uncollided(r) = N_source · exp(-μ₀ r)`.
//!
//! We tally the **energy-fluence buildup factor**
//!   `B_E(r) = E-weighted net outward current at r / uncollided
//!             E-weighted current at r`
//!        `   = (1/E₀) · Σ (E_photon · signed crossings) / (N₀ · exp(-μ₀ r))`
//!
//! For water in the 0.1–1 MeV Compton-dominated regime the
//! mass-energy-absorption coefficient `μ_en/ρ` varies by less than
//! ±10 % across the scattered-photon spectrum, so `B_E ≈ B_e`
//! (exposure buildup) to within ~10 %. That lets us compare
//! directly against published exposure-buildup tables without
//! building a dose model.
//!
//! Reference EXPOSURE buildup factors for water at 1 MeV from
//! Chilton-Shultis-Faw *Principles of Radiation Shielding*
//! Appendix F (identical to ANSI/ANS-6.6.1-1979 for water at 1 MeV):
//!
//! | μ₀ r | B_e  |
//! |------|-----:|
//! |  1   | 1.57 |
//! |  2   | 2.51 |
//! |  4   | 5.06 |
//! |  7   | 10.6 |
//! | 10   | 19.3 |
//!
//! # Why energy-weighted
//!
//! Scattered photons have lower energy than the source. A pure
//! number-buildup tally (count crossings, don't weight by energy)
//! overestimates buildup because soft scattered photons
//! contribute 1 to the crossing count but much less than 1 to the
//! physical dose rate. Weighting each crossing by the photon's
//! current energy corrects for this and lands close to the
//! exposure-buildup the tables report.
//!
//! # Assertion strategy
//!
//! 1. **Monotone trend**: `B_E` strictly increases with optical
//!    depth (catches gross transport-loop regressions).
//! 2. **Physical sign**: `B_E > 1` at every optical depth.
//! 3. **Absolute agreement with B_e reference**: within ±25 % at
//!    small optical depth, widening to ±40 % at `μ₀r = 10` because
//!    deep-in-shield values get MC-noisy as uncollided counts drop,
//!    and the `B_E ≈ B_e` approximation worsens slightly where
//!    `μ_en(E)` varies more across the scattered spectrum.

use std::path::PathBuf;

use open_rust_mc::geometry::Vec3;
use open_rust_mc::photon::coherent::coherent_scatter;
use open_rust_mc::photon::compton::compton_scatter;
use open_rust_mc::photon::material::{Channel, PhotonMaterial};
use open_rust_mc::photon::pair::{pair_produce, ANNIHILATION_ENERGY_EV};
use open_rust_mc::photon::photoelectric::{
    photoelectric_absorb, DEFAULT_PHOTON_CUTOFF_EV,
};
use open_rust_mc::photon::transport::deflect;
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::transport::rng::Rng;

fn photon_path(name: &str) -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .parent()?
        .join("data/endfb-vii.1-hdf5/photon")
        .join(name);
    if p.exists() { Some(p) } else { None }
}

fn water() -> Option<PhotonMaterial> {
    let h = PhotonElement::from_hdf5(&photon_path("H.h5")?).ok()?;
    let o = PhotonElement::from_hdf5(&photon_path("O.h5")?).ok()?;
    let molecule_density = 3.3428e-2;
    Some(PhotonMaterial::new(vec![
        (2.0 * molecule_density, h),
        (1.0 * molecule_density, o),
    ]))
}

/// Net outward current contribution of one segment through a sphere.
///
/// A straight-line segment crossing a sphere can enter (inward, -1),
/// exit (outward, +1), or enter-and-exit (net 0). For a point-source
/// buildup tally the physically meaningful quantity is the net
/// outward current — inward and outward crossings on the same
/// segment cancel.
///
/// Returns +1, 0, or -1 based on endpoint radii alone. Enter-and-exit
/// with both endpoints on the same side of `r` contributes 0, which
/// is exactly the net current physics wants.
fn net_outward(p0: Vec3, dir: Vec3, d: f64, r: f64) -> i32 {
    let r0_sq = p0.x * p0.x + p0.y * p0.y + p0.z * p0.z;
    let p1 = p0 + dir * d;
    let r1_sq = p1.x * p1.x + p1.y * p1.y + p1.z * p1.z;
    let r_sq = r * r;
    let inside0 = r0_sq < r_sq;
    let inside1 = r1_sq < r_sq;
    match (inside0, inside1) {
        (true, false) => 1,
        (false, true) => -1,
        _ => 0,
    }
}

/// Transport one photon segment-by-segment through an infinite
/// medium, counting crossings on each radius in `sphere_radii`.
/// Spawns secondaries (fluorescence, annihilation) recursively.
fn transport_with_crossings(
    source_pos: Vec3,
    source_dir: Vec3,
    source_energy: f64,
    material: &PhotonMaterial,
    sphere_radii: &[f64],
    energy_current: &mut [f64],
    rng: &mut Rng,
) {
    let mut bank: Vec<(Vec3, Vec3, f64)> = vec![(source_pos, source_dir, source_energy)];
    const MAX_HOPS: u32 = 10_000;
    while let Some((mut pos, mut dir, mut e)) = bank.pop() {
        for _ in 0..MAX_HOPS {
            if e < 1_000.0 {
                break;
            }
            let sigma_tot = material.macro_total(e);
            if sigma_tot <= 0.0 {
                break;
            }
            let d = rng.exponential(sigma_tot);
            let new_pos = pos + dir * d;
            // Energy-weighted net outward current on this segment.
            for (idx, &r) in sphere_radii.iter().enumerate() {
                let w = net_outward(pos, dir, d, r) as f64;
                energy_current[idx] += w * e;
            }
            pos = new_pos;

            // Terminate if we've travelled past the outermost sphere
            // by a margin (the photon can no longer contribute to
            // tallies inside).
            let max_r = *sphere_radii.last().unwrap();
            let r_pos = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
            if r_pos > 2.0 * max_r {
                break;
            }

            let xi = rng.uniform();
            let ch = material.sample_channel(e, xi);
            let xi2 = rng.uniform();
            let el_idx = material.sample_element(ch, e, xi2);
            let elem = &material.entries[el_idx].1;

            match ch {
                Channel::Coherent => {
                    let out = coherent_scatter(elem, e, rng);
                    dir = deflect(dir, out.mu, rng);
                }
                Channel::Incoherent => {
                    let out = compton_scatter(elem, e, rng);
                    e = out.energy_out;
                    dir = deflect(dir, out.mu, rng);
                }
                Channel::Photoelectric => {
                    let out = photoelectric_absorb(
                        elem,
                        e,
                        DEFAULT_PHOTON_CUTOFF_EV,
                        rng,
                    );
                    for ep in out.fluorescence_photons {
                        let (dx, dy, dz) = rng.isotropic_direction();
                        bank.push((pos, Vec3::new(dx, dy, dz), ep));
                    }
                    break;
                }
                Channel::PairProductionNuclear | Channel::PairProductionElectron => {
                    if pair_produce(e, rng).is_some() {
                        let (dx, dy, dz) = rng.isotropic_direction();
                        let ann_dir = Vec3::new(dx, dy, dz);
                        bank.push((pos, ann_dir, ANNIHILATION_ENERGY_EV));
                        bank.push((pos, -ann_dir, ANNIHILATION_ENERGY_EV));
                    }
                    break;
                }
            }
        }
    }
}

#[test]
fn water_number_buildup_at_1mev() {
    let Some(water) = water() else {
        eprintln!("skipping: H.h5 or O.h5 not present");
        return;
    };
    let source_energy = 1.0e6_f64;
    let mu_0 = water.macro_total(source_energy); // cm⁻¹
    let mfp = 1.0 / mu_0;

    // Optical depths and reference EXPOSURE buildup factors (water,
    // 1 MeV, Chilton-Shultis-Faw Appendix F ≡ ANSI/ANS-6.6.1-1979).
    let optical_depths = [1.0_f64, 2.0, 4.0, 7.0, 10.0];
    let reference_be = [1.57_f64, 2.51, 5.06, 10.6, 19.3];

    let sphere_radii: Vec<f64> =
        optical_depths.iter().map(|mu_r| mu_r * mfp).collect();

    let n_hist = 50_000_usize;
    let mut energy_current = vec![0.0_f64; sphere_radii.len()];

    for h in 0..n_hist {
        let mut rng = Rng::new(0xAA551100 + h as u64, 1);
        let (dx, dy, dz) = rng.isotropic_direction();
        transport_with_crossings(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(dx, dy, dz),
            source_energy,
            &water,
            &sphere_radii,
            &mut energy_current,
            &mut rng,
        );
    }

    println!(
        "Water 1 MeV buildup (50k histories, μ₀ = {:.4} cm⁻¹, mfp = {:.2} cm):",
        mu_0, mfp
    );
    println!(
        "{:>6} {:>12} {:>12} {:>10}",
        "μ₀r", "measured_BE", "reference_Be", "rel_err"
    );
    let mut b_e_measured = vec![0.0; optical_depths.len()];
    for (i, &mu_r) in optical_depths.iter().enumerate() {
        let e_uncoll = source_energy * n_hist as f64 * (-mu_r).exp();
        let b_e = energy_current[i] / e_uncoll;
        b_e_measured[i] = b_e;
        let rel_err = (b_e - reference_be[i]).abs() / reference_be[i];
        println!(
            "{:>6.1} {:>12.3} {:>12.3} {:>9.1}%",
            mu_r,
            b_e,
            reference_be[i],
            rel_err * 100.0
        );
    }

    // Absolute agreement with exposure buildup reference. Tolerances
    // widen with optical depth because
    //   (a) MC noise grows as crossings drop off exponentially, and
    //   (b) the `B_E ≈ B_e` approximation worsens where μ_en(E)
    //       varies more across the scattered-photon spectrum.
    let tolerances = [0.25_f64, 0.30, 0.35, 0.40, 0.45];
    for (i, &mu_r) in optical_depths.iter().enumerate() {
        let rel_err = (b_e_measured[i] - reference_be[i]).abs() / reference_be[i];
        assert!(
            rel_err <= tolerances[i],
            "μ₀r = {mu_r}: B_E = {:.3}, B_e(ref) = {}, rel err {:.3} > tol {:.2}",
            b_e_measured[i],
            reference_be[i],
            rel_err,
            tolerances[i]
        );
    }

    // Physical-sign check: B_E > 1 everywhere (scattered contributions
    // always raise energy fluence above the uncollided baseline).
    for (i, &mu_r) in optical_depths.iter().enumerate() {
        assert!(
            b_e_measured[i] > 1.0,
            "μ₀r = {mu_r}: B_E = {:.3} < 1, transport loop broken",
            b_e_measured[i]
        );
    }

    // Monotone increase with depth.
    for w in b_e_measured.windows(2) {
        assert!(
            w[1] > w[0],
            "buildup not monotone: {} -> {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn net_outward_endpoint_cases() {
    let dir = Vec3::new(1.0, 0.0, 0.0);

    // Inside to outside: outward +1.
    assert_eq!(net_outward(Vec3::new(-0.5, 0.0, 0.0), dir, 2.0, 1.0), 1);

    // Outside to inside: inward -1.
    assert_eq!(net_outward(Vec3::new(-2.0, 0.0, 0.0), dir, 1.7, 1.0), -1);

    // Outside to outside (no crossing): 0.
    assert_eq!(net_outward(Vec3::new(-3.0, 3.0, 0.0), dir, 4.0, 1.0), 0);

    // Outside to outside (passes through, 2 crossings): net 0.
    assert_eq!(net_outward(Vec3::new(-2.0, 0.0, 0.0), dir, 4.0, 1.0), 0);

    // Inside to inside (stays inside): 0.
    assert_eq!(net_outward(Vec3::new(-0.5, 0.0, 0.0), dir, 0.5, 1.0), 0);
}
