//! Engine-side XS dump at (U-234, U-235, U-238), 294 K, on a dense
//! energy grid. Matches the Godiva operating conditions. Used for
//! channel-level diff against scripts/xs_dump_godiva_openmc.py to
//! localise the Godiva fast-spectrum offset.
//!
//! Usage:
//!     xs_dump_godiva <data_dir> [output.csv] [--mode svd|table] [--rank N]

use std::io::Write;
use std::path::PathBuf;

use open_rust_mc::physics::collision::MicroXs;
use open_rust_mc::transport::simulate::XsProvider;
use open_rust_mc::transport::xs_provider;

// Godiva config: 3 U isotopes at 294 K (temp_idx=1 after numeric sort
// of library temps: 0=250, 1=294, 2=600, 3=900, 4=1200, 5=2500).
const NUCLIDE_SPECS: &[(&str, f64, f64, usize, u32)] = &[
    ("U234.h5", 232.029, 2.49, 1, 294),
    ("U235.h5", 233.025, 2.43, 1, 294),
    ("U238.h5", 236.006, 2.49, 1, 294),
];

const NUCLIDE_NAMES: &[&str] = &["U234", "U235", "U238"];

fn test_energies() -> Vec<f64> {
    // Dense grid in Godiva's active band (100 keV – 20 MeV) plus sub-
    // decades for completeness. 80 points total, log-spaced.
    (0..80)
        .map(|i| {
            let f = i as f64 / 79.0;
            10f64.powf(-1.0 + 9.0 * f) // 0.1 eV to 100 MeV in log space
        })
        .collect()
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
        eprintln!("usage: xs_dump_godiva <data_dir> [output.csv] [--mode svd|table] [--rank N]");
        std::process::exit(1);
    }

    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "table".to_string());
    let rank: usize = args
        .iter()
        .position(|a| a == "--rank")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse().ok()))
        .unwrap_or(5);
    args.retain(|a| {
        !matches!(a.as_str(), "--mode" | "--rank") && a.parse::<usize>().is_err()
            || a.ends_with(".csv")
            || a.contains("\\")
            || a.contains("/")
    });

    let data_dir = PathBuf::from(&args[1]);
    let out_path: Option<PathBuf> = args.get(2).map(PathBuf::from);

    let energies = test_energies();

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
    .expect("write header");

    match mode.as_str() {
        "table" => {
            let mut kernels = Vec::new();
            for &(filename, awr, nu_bar, nuc_temp_idx, _) in NUCLIDE_SPECS {
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
                nuclides: kernels,
                thermal,
            };
            for (i, (_, _, _, _, target_k)) in NUCLIDE_SPECS.iter().enumerate() {
                for &e in &energies {
                    let xs = provider.lookup(i, e);
                    write_row(&mut out_w, NUCLIDE_NAMES[i], *target_k, e, &xs);
                }
            }
        }
        _ => {
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
            let provider = xs_provider::SvdXsProvider {
                nuclides: kernels,
                thermal,
            };
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
