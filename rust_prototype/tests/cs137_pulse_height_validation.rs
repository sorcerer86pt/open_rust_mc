//! Integration test for the Cs-137 pulse-height spectrum benchmark.
//!
//! Runs a quick (50 k history) Monte Carlo of 661.657 keV photons on a
//! 3"-thick NaI detector and asserts that the three canonical
//! spectral features land at their analytically expected energies:
//!
//!   * Full-energy peak at 661.657 keV
//!   * Compton edge   at 2α/(1+2α) · E = 477.67 keV
//!   * Backscatter peak at E/(1+2α)   = 183.99 keV
//!
//! Tolerances are chosen to accept MC noise at 50 k histories with
//! 2 keV bins while rejecting gross physics bugs.

use std::path::PathBuf;

use open_rust_mc::geometry::Vec3;
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::photon::transport::transport_history;
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::transport::rng::Rng;

const CS137_ENERGY_EV: f64 = 661_657.0;
const NAI_MOLECULE_DENSITY: f64 = 1.4743e-2;

fn photon_path(name: &str) -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .parent()?
        .join("data/endfb-vii.1-hdf5/photon")
        .join(name);
    if p.exists() { Some(p) } else { None }
}

fn bin_kev_center(bin: usize, bin_width: f64) -> f64 {
    (bin as f64 + 0.5) * bin_width / 1_000.0
}

fn argmax_in(hist: &[u64], lo_kev: f64, hi_kev: f64, bin_width: f64) -> usize {
    let lo = (lo_kev * 1_000.0 / bin_width) as usize;
    let hi = ((hi_kev * 1_000.0 / bin_width) as usize).min(hist.len());
    let mut best = lo;
    let mut val = 0u64;
    for (i, &c) in hist.iter().enumerate().take(hi).skip(lo) {
        if c > val {
            val = c;
            best = i;
        }
    }
    best
}

fn steepest_drop_in(hist: &[u64], lo_kev: f64, hi_kev: f64, bin_width: f64) -> usize {
    let lo = (lo_kev * 1_000.0 / bin_width) as usize;
    let hi = ((hi_kev * 1_000.0 / bin_width) as usize).min(hist.len() - 1);
    let mut best = lo;
    let mut drop: i64 = 0;
    for i in lo..hi {
        let d = hist[i] as i64 - hist[i + 1] as i64;
        if d > drop {
            drop = d;
            best = i;
        }
    }
    best
}

#[test]
fn cs137_nai_spectral_features_land_where_expected() {
    let Some(na_path) = photon_path("Na.h5") else {
        eprintln!("skipping: Na.h5 not present");
        return;
    };
    let Some(i_path) = photon_path("I.h5") else {
        eprintln!("skipping: I.h5 not present");
        return;
    };

    let na = PhotonElement::from_hdf5(&na_path).expect("load Na");
    let i = PhotonElement::from_hdf5(&i_path).expect("load I");
    let nai = PhotonMaterial::new(vec![
        (NAI_MOLECULE_DENSITY, na),
        (NAI_MOLECULE_DENSITY, i),
    ]);

    let thickness = 7.62_f64;
    let is_inside = |p: Vec3| p.z >= 0.0 && p.z <= thickness;

    let n_hist = 50_000usize;
    let bin_width = 2_000.0_f64;
    let n_bins = 350usize;
    let mut hist = vec![0u64; n_bins];

    for h in 0..n_hist {
        let mut rng = Rng::new(0xC513_7000 + h as u64, 1);
        let r = transport_history(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            CS137_ENERGY_EV,
            &nai,
            is_inside,
            1_000.0,
            &mut rng,
        );
        let e = r.energy_deposited;
        if e >= 0.5 * bin_width {
            let b = ((e / bin_width) as usize).min(n_bins - 1);
            hist[b] += 1;
        }
    }

    // --- Full-energy peak (661.657 keV analytic) -----------------------
    let full_bin = argmax_in(&hist, 620.0, 680.0, bin_width);
    let full_kev = bin_kev_center(full_bin, bin_width);
    let analytic_full = CS137_ENERGY_EV / 1_000.0;
    assert!(
        (full_kev - analytic_full).abs() < 4.0,
        "full-energy peak at {full_kev:.2} keV (analytic {analytic_full:.2}, tol 4 keV)"
    );

    // --- Compton edge: T_max = 2α/(1+2α) · E --------------------------
    let alpha = CS137_ENERGY_EV / 510_998.95;
    let compton_edge_kev = 2.0 * alpha / (1.0 + 2.0 * alpha) * CS137_ENERGY_EV / 1_000.0;
    let edge_bin = steepest_drop_in(&hist, 400.0, 520.0, bin_width);
    let edge_kev = bin_kev_center(edge_bin, bin_width);
    assert!(
        (edge_kev - compton_edge_kev).abs() < 10.0,
        "compton edge found at {edge_kev:.2} keV (analytic {compton_edge_kev:.2}, tol 10 keV)"
    );

    // --- Backscatter peak: E' = E/(1+2α) -----------------------------
    let bs_kev = CS137_ENERGY_EV / (1.0 + 2.0 * alpha) / 1_000.0;
    // For an axial-beam slab geometry without surrounding backscattering
    // material, the "backscatter peak" appears as a soft shoulder near
    // E'/(1+2α). Accept within 20 keV.
    let bs_bin = argmax_in(&hist, 150.0, 220.0, bin_width);
    let bs_found_kev = bin_kev_center(bs_bin, bin_width);
    assert!(
        (bs_found_kev - bs_kev).abs() < 20.0,
        "backscatter feature at {bs_found_kev:.2} keV (analytic {bs_kev:.2}, tol 20 keV)"
    );

    // --- Detection efficiency sanity ---------------------------------
    let detected: u64 = hist.iter().sum();
    let detect_frac = detected as f64 / n_hist as f64;
    assert!(
        detect_frac > 0.6,
        "detection fraction {detect_frac:.3} too low (expected > 0.6 for 7.62 cm NaI)"
    );
}
