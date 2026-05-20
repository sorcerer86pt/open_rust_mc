// SPDX-License-Identifier: MIT
//! Engine-side XS dump at fixed (nuclide, energy, temperature) triples.
//!
//! Loads the same nine PWR pin-cell nuclides as `pwr_pincell` (same
//! temperature indices, same SVD rank, same total-XS convention) and
//! dumps the per-nuclide MicroXs at a set of test energies.
//!
//! Used in the capture-residue audit: compare the "total" column to
//! OpenMC's total (via scripts/xs_dump_openmc.py) to find where the
//! engine-level offset comes from.
//!
//! Output: CSV to stdout (or a path passed as arg 2).
//!
//! Usage:
//!   xs_dump <data_dir> [output.csv] [--mode svd|table|hybrid] [--rank N]

use std::io::Write;
use std::path::PathBuf;

use open_rust_mc::physics::collision::MicroXs;
use open_rust_mc::transport::hybrid_xs::HybridSvdWmpXsProvider;
use open_rust_mc::transport::simulate::XsProvider;
use open_rust_mc::transport::xs_provider;
use open_rust_mc::wmp::WindowedMultipole;
use std::sync::Arc;

// Keep in sync with NUCLIDE_SPECS in src/bin/pwr_pincell.rs.
const NUCLIDE_SPECS: &[(&str, f64, f64, usize, u32)] = &[
    ("U235.h5", 233.025, 2.43, 3, 900),
    ("U238.h5", 236.006, 2.49, 3, 900),
    ("O16.h5", 15.858, 0.0, 3, 900),
    ("H1.h5", 0.999, 0.0, 2, 600),
    ("Zr90.h5", 89.132, 0.0, 2, 600),
    ("Zr91.h5", 90.130, 0.0, 2, 600),
    ("Zr92.h5", 91.126, 0.0, 2, 600),
    ("Zr94.h5", 93.120, 0.0, 2, 600),
    ("O16.h5", 15.858, 0.0, 2, 600),
];

const NUCLIDE_NAMES: &[&str] = &[
    "U235", "U238", "O16_fuel", "H1", "Zr90", "Zr91", "Zr92", "Zr94", "O16_mod",
];

const WMP_SPECS: &[(&str, f64)] = &[
    ("092235.h5", 900.0),
    ("092238.h5", 900.0),
    ("008016.h5", 900.0),
    ("001001.h5", 600.0),
    ("040090.h5", 600.0),
    ("040091.h5", 600.0),
    ("040092.h5", 600.0),
    ("040094.h5", 600.0),
    ("008016.h5", 600.0),
];

fn test_energies() -> Vec<f64> {
    // Same grid as scripts/xs_dump_openmc.py
    let mut e: Vec<f64> = (0..50)
        .map(|i| {
            let f = i as f64 / 49.0;
            10f64.powf(-2.0 + 9.0 * f)
        })
        .collect();
    e.extend([6.674_f64, 20.9, 36.7, 66.0, 80.7, 102.5]);
    e.sort_by(|a, b| a.partial_cmp(b).expect("energy grid contains NaN"));
    e.dedup_by(|a, b| (*a - *b).abs() < 1e-9 * a.abs().max(1.0));
    e
}

fn write_row<W: Write>(w: &mut W, nuclide: &str, target_k: u32, e: f64, xs: &MicroXs) {
    let _ = writeln!(
        w,
        "{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
        nuclide,
        target_k,
        e,
        xs.total,
        xs.elastic,
        xs.inelastic,
        xs.n2n,
        xs.n3n,
        xs.fission,
        xs.capture,
        xs.nu_bar,
    );
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: xs_dump <data_dir> [output.csv] [--mode svd|table|hybrid] [--rank N]");
        std::process::exit(1);
    }

    // Parse flags
    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "svd".to_string());
    let rank: usize = args
        .iter()
        .position(|a| a == "--rank")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse().ok()))
        .unwrap_or(5);
    // strip flag args
    args.retain(|a| {
        !matches!(a.as_str(), "--mode" | "--rank") && a.parse::<usize>().is_err()
            || a.ends_with(".csv")
            || a.contains("\\")
            || a.contains("/")
    });

    let data_dir = PathBuf::from(&args[1]);
    let out_path: Option<PathBuf> = args.get(2).map(PathBuf::from);

    let energies = test_energies();

    // Load per-nuclide kernels
    let (mut out_w, path_str): (Box<dyn Write>, String) = match out_path.as_ref() {
        Some(p) => {
            let f = std::fs::File::create(p).expect("create output");
            (
                Box::new(std::io::BufWriter::new(f)),
                p.display().to_string(),
            )
        }
        None => (Box::new(std::io::stdout()), "stdout".to_string()),
    };

    writeln!(
        out_w,
        "nuclide,target_K,E_eV,total,elastic,inelastic,n2n,n3n,fission,capture,nu_bar"
    )
    .expect("write to XS dump output failed");

    match mode.as_str() {
        "table" => {
            let mut kernels = Vec::new();
            for &(filename, awr, nu_bar, nuc_temp_idx) in NUCLIDE_SPECS
                .iter()
                .map(|&(a, b, c, d, _)| (a, b, c, d))
                .collect::<Vec<_>>()
                .as_slice()
            {
                let path = data_dir.join(filename);
                kernels.push(xs_provider::load_nuclide_table(
                    &path,
                    nuc_temp_idx,
                    awr,
                    nu_bar,
                ));
            }
            let thermal = vec![None; kernels.len()];
            let provider = xs_provider::TableXsProvider {
                nuclides: kernels.into_iter().map(std::sync::Arc::new).collect(),
                thermal,
            };
            for (i, (_, _, _, _, target_k)) in NUCLIDE_SPECS.iter().enumerate() {
                for &e in &energies {
                    let xs = provider.lookup(i, e);
                    write_row(&mut out_w, NUCLIDE_NAMES[i], *target_k, e, &xs);
                }
            }
        }
        "hybrid" => {
            let (inner, _, _) = build_svd_provider(&data_dir, rank);
            let wmps = load_wmps(&data_dir);
            let provider = HybridSvdWmpXsProvider::new(inner, wmps);
            for (i, (_, _, _, _, target_k)) in NUCLIDE_SPECS.iter().enumerate() {
                for &e in &energies {
                    let xs = provider.lookup(i, e);
                    write_row(&mut out_w, NUCLIDE_NAMES[i], *target_k, e, &xs);
                }
            }
        }
        _ => {
            let (provider, _, _) = build_svd_provider(&data_dir, rank);
            for (i, (_, _, _, _, target_k)) in NUCLIDE_SPECS.iter().enumerate() {
                for &e in &energies {
                    let xs = provider.lookup(i, e);
                    write_row(&mut out_w, NUCLIDE_NAMES[i], *target_k, e, &xs);
                }
            }
        }
    }

    eprintln!("wrote {path_str}  (mode={mode}, rank={rank})");
}

fn build_svd_provider(
    data_dir: &std::path::Path,
    rank: usize,
) -> (xs_provider::SvdXsProvider, usize, f64) {
    let mut kernels = Vec::new();
    for &(filename, awr, nu_bar, nuc_temp_idx, _) in NUCLIDE_SPECS {
        let path = data_dir.join(filename);
        kernels.push(xs_provider::load_nuclide(
            &path,
            rank,
            nuc_temp_idx,
            awr,
            nu_bar,
        ));
    }
    let thermal = vec![None; kernels.len()];
    let mem: usize = kernels.iter().map(|k| k.svd_memory_bytes()).sum();
    let p = xs_provider::SvdXsProvider {
        nuclides: kernels.into_iter().map(std::sync::Arc::new).collect(),
        thermal,
    };
    (p, mem, 0.0)
}

fn load_wmps(data_dir: &std::path::Path) -> Vec<Option<(Arc<WindowedMultipole>, f64)>> {
    let wmp_dir = data_dir.join("..").join("wmp");
    let mut out = Vec::with_capacity(WMP_SPECS.len());
    for &(wmp_file, t_k) in WMP_SPECS {
        let path = wmp_dir.join(wmp_file);
        if !path.exists() {
            out.push(None);
            continue;
        }
        match WindowedMultipole::from_hdf5(&path) {
            Ok(w) => out.push(Some((Arc::new(w), t_k))),
            Err(_) => out.push(None),
        }
    }
    out
}
