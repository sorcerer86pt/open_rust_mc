//! Cs-137 pulse-height spectrum benchmark in a 3"×3" NaI(Tl) detector.
//!
//! Injects mono-energetic 662 keV photons axially into a 7.62 cm thick
//! × 7.62 cm diameter (disk slab approximation) NaI detector,
//! histograms total energy deposited per source photon, and reports
//! the four canonical spectral features:
//!   - Full-energy peak at 662 keV (photoelectric absorption + Compton
//!     continuum folded back by fluorescence reabsorption)
//!   - Compton edge at 477.7 keV (`T_max = 2α/(1+2α)·E`, α = 1.296)
//!   - Compton continuum from 0 to T_max
//!   - Backscatter peak at 184.3 keV (`E' = E/(1+2α)` from a single
//!     π-scatter in detector material or surroundings)
//!
//! Usage:
//!   cargo run --release --bin cs137_pulse_height -- \
//!     data/endfb-vii.1-hdf5/photon --n 200000
//!
//! Outputs a CSV `outputs/cs137_spectrum.csv` with histogrammed
//! counts per 2 keV bin. Use `scripts/plot_cs137.py` (not included)
//! for visualisation.

use std::path::PathBuf;
use std::process::ExitCode;

use open_rust_mc::geometry::Vec3;
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::photon::transport::transport_history;
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::transport::rng::Rng;

/// Cs-137 characteristic gamma-ray energy (eV).
pub const CS137_ENERGY_EV: f64 = 661_657.0;

/// NaI density and atom-density derivation.
/// Density 3.67 g/cm³, molar mass Na(22.99) + I(126.904) = 149.894 g/mol,
/// N_A = 6.02214e23 molecules/mol
/// → molecules/cm³ = 3.67 × 6.02214e23 / 149.894 = 1.4743e22
/// → atoms/(barn·cm) = 1.4743e-2 per element (Na and I are 1:1)
pub const NAI_MOLECULE_DENSITY: f64 = 1.4743e-2;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(data_dir) = args.next() else {
        eprintln!(
            "usage: cs137_pulse_height <data_dir> [--n NUM_HISTORIES]\n\
             example: cs137_pulse_height data/endfb-vii.1-hdf5/photon --n 200000"
        );
        return ExitCode::from(2);
    };
    let data_dir = PathBuf::from(data_dir);

    let mut n_hist = 200_000_usize;
    while let Some(a) = args.next() {
        if a == "--n" {
            if let Some(v) = args.next() {
                n_hist = v.parse().unwrap_or(n_hist);
            }
        }
    }

    // Load photon data for Na and I.
    let na = match PhotonElement::from_hdf5(&data_dir.join("Na.h5")) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to load Na.h5 from {}: {e}", data_dir.display());
            return ExitCode::from(1);
        }
    };
    let i = match PhotonElement::from_hdf5(&data_dir.join("I.h5")) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to load I.h5 from {}: {e}", data_dir.display());
            return ExitCode::from(1);
        }
    };
    let nai = PhotonMaterial::new(vec![
        (NAI_MOLECULE_DENSITY, na),
        (NAI_MOLECULE_DENSITY, i),
    ]);

    // Detector geometry: slab 0 < z < 7.62 cm, infinite in x,y.
    let thickness_cm = 7.62;
    let is_inside = |p: Vec3| p.z >= 0.0 && p.z <= thickness_cm;

    // Histogram: 0 to 700 keV in 2 keV bins.
    let bin_width_ev = 2_000.0;
    let max_ev = 700_000.0;
    let n_bins = (max_ev / bin_width_ev) as usize;
    let mut histogram = vec![0_u64; n_bins];
    let mut n_detected = 0_u64;

    println!(
        "Cs-137 (662 keV) on NaI {} cm thick, {} histories, {} bins @ {} eV",
        thickness_cm, n_hist, n_bins, bin_width_ev
    );
    let start = std::time::Instant::now();

    for i_hist in 0..n_hist {
        let mut rng = Rng::new(0xC513_7000 + i_hist as u64, 1);
        let r = transport_history(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            CS137_ENERGY_EV,
            &nai,
            is_inside,
            1_000.0, // 1 keV cutoff
            &mut rng,
        );
        let e_dep = r.energy_deposited;
        if e_dep >= 0.5 * bin_width_ev {
            n_detected += 1;
            let bin = ((e_dep / bin_width_ev) as usize).min(n_bins - 1);
            histogram[bin] += 1;
        }
    }

    let elapsed = start.elapsed();
    let rate = n_hist as f64 / elapsed.as_secs_f64();
    println!(
        "{} detected / {} histories ({:.1} %) in {:.2} s ({:.0} hist/s)",
        n_detected,
        n_hist,
        100.0 * n_detected as f64 / n_hist as f64,
        elapsed.as_secs_f64(),
        rate
    );

    // Find peak positions.
    let full_peak_bin = argmax_bin_range(&histogram, 620_000.0, 680_000.0, bin_width_ev);
    let compton_edge_bin = find_compton_edge(&histogram, bin_width_ev);
    let backscatter_bin = argmax_bin_range(&histogram, 150_000.0, 250_000.0, bin_width_ev);

    println!(
        "Full-energy peak bin center:     {:.1} keV",
        bin_center_kev(full_peak_bin, bin_width_ev)
    );
    println!(
        "Compton edge (drop location):    {:.1} keV (analytic 477.7)",
        bin_center_kev(compton_edge_bin, bin_width_ev)
    );
    println!(
        "Backscatter peak bin center:     {:.1} keV (analytic 184.3)",
        bin_center_kev(backscatter_bin, bin_width_ev)
    );

    // Write CSV.
    let out_dir = data_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(&data_dir)
        .join("outputs");
    let _ = std::fs::create_dir_all(&out_dir);
    let csv_path = out_dir.join("cs137_spectrum.csv");
    let mut s = String::from("energy_kev,counts\n");
    for (bin, &c) in histogram.iter().enumerate() {
        let e_kev = (bin as f64 + 0.5) * bin_width_ev / 1_000.0;
        s.push_str(&format!("{e_kev:.2},{c}\n"));
    }
    if let Err(e) = std::fs::write(&csv_path, s) {
        eprintln!("failed to write {}: {e}", csv_path.display());
        return ExitCode::from(1);
    }
    println!("Wrote {}", csv_path.display());

    ExitCode::SUCCESS
}

fn bin_center_kev(bin: usize, bin_width_ev: f64) -> f64 {
    (bin as f64 + 0.5) * bin_width_ev / 1_000.0
}

fn argmax_bin_range(hist: &[u64], lo_ev: f64, hi_ev: f64, bin_width_ev: f64) -> usize {
    let lo = (lo_ev / bin_width_ev) as usize;
    let hi = ((hi_ev / bin_width_ev) as usize).min(hist.len());
    let mut best_bin = lo;
    let mut best_val = 0;
    for (i, &v) in hist.iter().enumerate().take(hi).skip(lo) {
        if v > best_val {
            best_val = v;
            best_bin = i;
        }
    }
    best_bin
}

/// The Compton edge is a sharp drop in the spectrum between the
/// continuum plateau (~300-450 keV) and the gap (~480-600 keV).
/// Find the bin where the count drops fastest within [400, 520] keV.
fn find_compton_edge(hist: &[u64], bin_width_ev: f64) -> usize {
    let lo = (400_000.0 / bin_width_ev) as usize;
    let hi = ((520_000.0 / bin_width_ev) as usize).min(hist.len() - 1);
    let mut best_bin = lo;
    let mut best_drop: i64 = 0;
    for i in lo..hi {
        let drop = hist[i] as i64 - hist[i + 1] as i64;
        if drop > best_drop {
            best_drop = drop;
            best_bin = i;
        }
    }
    best_bin
}
