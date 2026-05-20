#![allow(
// SPDX-License-Identifier: MIT
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::manual_is_multiple_of,
    clippy::needless_borrow
)]
//! Diff every XsProvider trait method between SvdXsProvider and
//! TableXsProvider for the 9 PWR nuclides at thermal/resonance/fast
//! energies. Prints the first method/energy where they differ
//! materially. Used to root-cause the ~19000 pcm CPU SVD vs Table
//! k_inf discrepancy.
//!
//! Usage:
//!   cargo run --release --bin xs_provider_diff -- <data_dir>

use std::path::PathBuf;
use std::sync::Arc;

use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::XsProvider;
use open_rust_mc::transport::xs_provider;

const NUCLIDE_SPECS: &[(&str, f64, f64, usize)] = &[
    ("U235.h5", 233.025, 2.43, 3),
    ("U238.h5", 236.006, 2.49, 3),
    ("O16.h5", 15.858, 0.0, 3),
    ("H1.h5", 0.999, 0.0, 2),
    ("Zr90.h5", 89.132, 0.0, 2),
    ("Zr91.h5", 90.130, 0.0, 2),
    ("Zr92.h5", 91.126, 0.0, 2),
    ("Zr94.h5", 93.120, 0.0, 2),
    ("O16.h5", 15.858, 0.0, 2),
];

const NUCLIDE_NAMES: &[&str] = &[
    "U235", "U238", "O16f", "H1", "Zr90", "Zr91", "Zr92", "Zr94", "O16w",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: xs_provider_diff <data_dir>");
        std::process::exit(2);
    }
    let data_dir = PathBuf::from(&args[1]);
    let rank = 5;

    println!("Loading SVD provider...");
    let mut svd_kernels = Vec::new();
    for &(file, awr, nu_bar, t_idx) in NUCLIDE_SPECS {
        let path = data_dir.join(file);
        svd_kernels.push(xs_provider::load_nuclide(&path, rank, t_idx, awr, nu_bar));
    }
    let svd = xs_provider::SvdXsProvider {
        nuclides: svd_kernels.into_iter().map(std::sync::Arc::new).collect(),
        thermal: vec![None; NUCLIDE_SPECS.len()], // skip SAB for diff
    };

    println!("Loading Table provider...");
    let mut tab_nuclides = Vec::new();
    for &(file, awr, nu_bar, t_idx) in NUCLIDE_SPECS {
        let path = data_dir.join(file);
        tab_nuclides.push(xs_provider::load_nuclide_table(&path, t_idx, awr, nu_bar));
    }
    let tab = xs_provider::TableXsProvider {
        nuclides: tab_nuclides.into_iter().map(std::sync::Arc::new).collect(),
        thermal: vec![None; NUCLIDE_SPECS.len()],
    };

    // Dense logspace to catch any energy where SVD ≠ Table.
    let mut energies: Vec<f64> = (0..200)
        .map(|i| {
            let f = i as f64 / 199.0;
            10f64.powf(-2.0 + 9.0 * f)
        })
        .collect();
    energies.extend([6.674_f64, 20.9, 36.7, 66.0, 80.7, 102.5]);
    let energies = &energies[..];

    println!("\n=== diff: lookup() ===");
    for ni in 0..NUCLIDE_SPECS.len() {
        for &e in energies {
            let s = svd.lookup(ni, e);
            let t = tab.lookup(ni, e);
            // Skip the placeholder-1e-30 noise from inelastic.
            // Compare each channel relative to the larger of the two
            // values (avoids one-sided blow-ups when one path uses 0).
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1e-3);
            let max_rel = [
                rel(s.elastic, t.elastic),
                rel(s.fission, t.fission),
                rel(s.capture, t.capture),
                rel(s.total, t.total),
            ]
            .into_iter()
            .fold(0_f64, f64::max);
            if max_rel > 5e-2 {
                println!(
                    "  {} @ {:.3e} eV: max_rel={:.3e}\n    SVD: el={:.3e} fis={:.3e} cap={:.3e} tot={:.3e}\n    Tab: el={:.3e} fis={:.3e} cap={:.3e} tot={:.3e}",
                    NUCLIDE_NAMES[ni],
                    e,
                    max_rel,
                    s.elastic,
                    s.fission,
                    s.capture,
                    s.total,
                    t.elastic,
                    t.fission,
                    t.capture,
                    t.total,
                );
            }
        }
    }

    println!("\n=== diff: discrete_level_info() ===");
    for ni in 0..NUCLIDE_SPECS.len() {
        let s_info = svd.discrete_level_info(ni);
        let t_info = tab.discrete_level_info(ni);
        if s_info.len() != t_info.len() {
            println!(
                "  {}: SVD has {} levels, Table has {}",
                NUCLIDE_NAMES[ni],
                s_info.len(),
                t_info.len()
            );
            continue;
        }
        for (i, (sl, tl)) in s_info.iter().zip(t_info.iter()).enumerate() {
            if sl.mt != tl.mt
                || (sl.q_value - tl.q_value).abs() > 1e-3
                || (sl.threshold - tl.threshold).abs() > 1e-3
            {
                println!(
                    "  {} level {}: SVD mt={} Q={} thr={}  Table mt={} Q={} thr={}",
                    NUCLIDE_NAMES[ni],
                    i,
                    sl.mt,
                    sl.q_value,
                    sl.threshold,
                    tl.mt,
                    tl.q_value,
                    tl.threshold
                );
            }
        }
    }

    println!("\n=== diff: discrete_level_xs() ===");
    for ni in 0..NUCLIDE_SPECS.len() {
        for &e in energies {
            let s_xs = svd.discrete_level_xs(ni, e);
            let t_xs = tab.discrete_level_xs(ni, e);
            if s_xs.len() != t_xs.len() {
                println!(
                    "  {} @ {:.3e}: lengths differ {} vs {}",
                    NUCLIDE_NAMES[ni],
                    e,
                    s_xs.len(),
                    t_xs.len()
                );
                continue;
            }
            let max_rel = s_xs
                .iter()
                .zip(t_xs.iter())
                .map(|(s, t)| (s - t).abs() / t.abs().max(1e-30))
                .fold(0_f64, f64::max);
            if max_rel > 1e-2 {
                let s_sum: f64 = s_xs.iter().sum();
                let t_sum: f64 = t_xs.iter().sum();
                println!(
                    "  {} @ {:.3e}: max_rel={:.3e} sum_svd={:.3e} sum_tab={:.3e}",
                    NUCLIDE_NAMES[ni], e, max_rel, s_sum, t_sum,
                );
            }
        }
    }

    println!("\n=== diff: has_continuum_inelastic() ===");
    for ni in 0..NUCLIDE_SPECS.len() {
        let s = svd.has_continuum_inelastic(ni);
        let t = tab.has_continuum_inelastic(ni);
        if s != t {
            println!("  {}: SVD={} Table={}", NUCLIDE_NAMES[ni], s, t);
        }
    }

    println!("\n=== diff: angular dist availability ===");
    for ni in 0..NUCLIDE_SPECS.len() {
        let s_e = svd.elastic_angular_dist(ni).is_some();
        let t_e = tab.elastic_angular_dist(ni).is_some();
        if s_e != t_e {
            println!(
                "  {} elastic_angle: SVD={} Table={}",
                NUCLIDE_NAMES[ni], s_e, t_e
            );
        }
        let s_f = svd.fission_energy_dist(ni).is_some();
        let t_f = tab.fission_energy_dist(ni).is_some();
        if s_f != t_f {
            println!(
                "  {} fission_edist: SVD={} Table={}",
                NUCLIDE_NAMES[ni], s_f, t_f
            );
        }
        let s_c = svd.inelastic_continuum_edist(ni).is_some();
        let t_c = tab.inelastic_continuum_edist(ni).is_some();
        if s_c != t_c {
            println!(
                "  {} inelastic_cont_edist: SVD={} Table={}",
                NUCLIDE_NAMES[ni], s_c, t_c
            );
        }
        let s2 = svd.n2n_edist(ni).is_some();
        let t2 = tab.n2n_edist(ni).is_some();
        if s2 != t2 {
            println!("  {} n2n_edist: SVD={} Table={}", NUCLIDE_NAMES[ni], s2, t2);
        }
        let s_a = svd.discrete_level_angles(ni).len();
        let t_a = tab.discrete_level_angles(ni).len();
        if s_a != t_a {
            println!(
                "  {} #level_angles: SVD={} Table={}",
                NUCLIDE_NAMES[ni], s_a, t_a
            );
        }
    }

    println!("\n=== explicit U238 @ 1000 eV ===");
    {
        let s = svd.lookup(1, 1000.0);
        let t = tab.lookup(1, 1000.0);
        println!(
            "  PRE-URR  SVD: el={:.6e} fis={:.6e} cap={:.6e} tot={:.6e} inel={:.6e}",
            s.elastic, s.fission, s.capture, s.total, s.inelastic
        );
        println!(
            "  PRE-URR  Tab: el={:.6e} fis={:.6e} cap={:.6e} tot={:.6e} inel={:.6e}",
            t.elastic, t.fission, t.capture, t.total, t.inelastic
        );
        let mut s2 = s;
        let mut t2 = t;
        svd.apply_urr(1, &mut s2, 1000.0, 0.5);
        tab.apply_urr(1, &mut t2, 1000.0, 0.5);
        println!(
            "  POST-URR SVD: el={:.6e} fis={:.6e} cap={:.6e} tot={:.6e}",
            s2.elastic, s2.fission, s2.capture, s2.total
        );
        println!(
            "  POST-URR Tab: el={:.6e} fis={:.6e} cap={:.6e} tot={:.6e}",
            t2.elastic, t2.fission, t2.capture, t2.total
        );
    }
    println!("\n=== diff: apply_urr() at 1 keV ===");
    use open_rust_mc::physics::collision::MicroXs;
    for ni in 0..NUCLIDE_SPECS.len() {
        for &e in &[100.0_f64, 1000.0, 1.0e4, 5.0e4, 1.0e5] {
            for xi in &[0.1_f64, 0.5, 0.9] {
                let mut s_xs = svd.lookup(ni, e);
                let mut t_xs = tab.lookup(ni, e);
                svd.apply_urr(ni, &mut s_xs, e, *xi);
                tab.apply_urr(ni, &mut t_xs, e, *xi);
                let max_rel = [
                    (s_xs.elastic - t_xs.elastic).abs() / t_xs.elastic.abs().max(1e-30),
                    (s_xs.fission - t_xs.fission).abs() / t_xs.fission.abs().max(1e-30),
                    (s_xs.capture - t_xs.capture).abs() / t_xs.capture.abs().max(1e-30),
                ]
                .into_iter()
                .fold(0_f64, f64::max);
                if max_rel > 1e-2 {
                    println!(
                        "  {} @ {:.3e} xi={}: max_rel={:.3e} S(el={:.3e} f={:.3e} c={:.3e}) T(el={:.3e} f={:.3e} c={:.3e})",
                        NUCLIDE_NAMES[ni],
                        e,
                        xi,
                        max_rel,
                        s_xs.elastic,
                        s_xs.fission,
                        s_xs.capture,
                        t_xs.elastic,
                        t_xs.fission,
                        t_xs.capture
                    );
                }
            }
        }
    }
    let _ = (Arc::new(()), Material::new("dummy", 0.0)); // suppress unused-warnings
    let _ = MicroXs {
        total: 0.0,
        elastic: 0.0,
        inelastic: 0.0,
        n2n: 0.0,
        n3n: 0.0,
        n4n: 0.0,
        fission: 0.0,
        capture: 0.0,
        n_nalpha: 0.0,
        n_2nalpha: 0.0,
        n_np: 0.0,
        nu_bar: 0.0,
        delayed_nu_bar: 0.0,
        awr: 0.0,
    };
    println!("\n(diff complete; missing sections = no divergence found at sampled energies)");
}
