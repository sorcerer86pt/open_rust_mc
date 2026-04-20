//! Validate the Rust WMP evaluator against OpenMC's Python reference.
//!
//! Evaluates (scattering, absorption, fission) at a set of test energies
//! for U-238, U-235, Pu-239 at T=293.6 K and prints them. A companion
//! Python script reads the same WMP file via openmc.data and prints the
//! reference; the user (or a shell diff) compares the two.
//!
//! Usage:
//!   cargo run --release --bin wmp_validate -- /path/to/wmp/092238.h5

use std::path::PathBuf;
use open_rust_mc::wmp::WindowedMultipole;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: wmp_validate <path/to/wmp/ZZAAA.h5>");
        std::process::exit(1);
    }
    let path = PathBuf::from(&args[1]);
    let wmp = WindowedMultipole::from_hdf5(&path)?;

    println!("# WMP validation for {}", wmp.name);
    println!("#   E_min = {:.6e} eV, E_max = {:.6e} eV", wmp.e_min, wmp.e_max);
    println!("#   n_poles = {}, n_windows = {}, fit_order = {}",
             wmp.n_poles, wmp.n_windows, wmp.fit_order);
    println!("#   sqrtAWR = {:.6e}, fissionable = {}",
             wmp.sqrt_awr, wmp.fissionable);

    // Pick a range of test energies across the resolved resonance region.
    let energies: Vec<f64> = vec![
        0.025,  // thermal
        1.0,
        6.674,  // first U-238 resonance
        10.0,
        20.9,
        36.7,
        50.0,
        100.0,
        500.0,
        1_000.0,
        5_000.0,
        10_000.0,
        19_000.0,
    ];

    let t = 293.6;
    println!("#   T = {} K", t);
    println!("#");
    println!("{:>14} {:>16} {:>16} {:>16}",
             "E (eV)", "sigma_s (b)", "sigma_a (b)", "sigma_f (b)");

    for &e in &energies {
        if e < wmp.e_min || e > wmp.e_max {
            continue;
        }
        let (ss, sa, sf) = wmp.evaluate(e, t);
        println!("{:14.6e} {:16.8e} {:16.8e} {:16.8e}",
                 e, ss, sa, sf);
    }

    Ok(())
}
