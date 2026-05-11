//! U-235 thermal-XS reconstruction sanity check.
//!
//! Loads U235.h5 at T_idx = 0 (lowest, normally 250 K) and T_idx for
//! 294 K, builds an SVD provider at rank 15, and compares the
//! reconstructed σ_f, σ_g, σ_el, ν̄, and ν·σ_f at a thermal set
//! against the raw HDF5 values. Used to isolate whether the
//! HEU-SOL-THERM-001 −1846 pcm bias has a U-235 channel-XS root cause.

use open_rust_mc::hdf5_reader::NuclideFileReader;
use open_rust_mc::transport::simulate::XsProvider;
use open_rust_mc::transport::xs_provider;
use std::path::PathBuf;

fn main() {
    let data_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../data/endfb-vii.1-hdf5/neutron"));
    let path = data_dir.join("U235.h5");

    let reader = NuclideFileReader::open(&path).unwrap();
    println!(
        "U-235 — {} temperature columns: {:?}",
        reader.temperatures.len(),
        reader.temperatures
    );
    // Pick the 294K slot (or closest).
    let target_t = 294.0_f64;
    let t_idx = reader
        .temperatures
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - target_t)
                .abs()
                .partial_cmp(&(**b - target_t).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "selected T_idx = {}  ({:.1} K)\n",
        t_idx, reader.temperatures[t_idx]
    );

    // Raw HDF5 channel XS at thermal — read MT=2/18/102 reactions
    // directly via NuclideFileReader (bypasses SVD) for reference.
    let mt2 = reader.read_reaction(2).ok();
    let mt18 = reader.read_reaction(18).ok();
    let mt102 = reader.read_reaction(102).ok();

    // Build a rank-15 SVD provider for U-235 at the chosen temperature.
    let kernel = xs_provider::load_nuclide(&path, 15, t_idx, 233.025, 2.43);
    let provider = xs_provider::SvdXsProvider {
        nuclides: vec![kernel],
        thermal: vec![None],
    };

    let test_energies = [0.0253_f64, 0.1, 1.0, 6.674, 20.9, 80.7];
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "E (eV)", "σ_f raw", "σ_f SVD", "σ_g raw", "σ_g SVD", "σ_el raw", "σ_el SVD", "ν·σ_f"
    );
    println!("{}", "-".repeat(90));

    for &e in &test_energies {
        let xs = provider.lookup(0, e);
        // Linearly interpolate the raw MT XS on the reaction's
        // energy grid for an apples-to-apples reference value.
        let raw_2 = raw_xs(&mt2, e, t_idx);
        let raw_18 = raw_xs(&mt18, e, t_idx);
        let raw_102 = raw_xs(&mt102, e, t_idx);
        println!(
            "{:>10.4} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            e,
            raw_18,
            xs.fission,
            raw_102,
            xs.capture,
            raw_2,
            xs.elastic,
            xs.nu_bar * xs.fission
        );
    }

    println!();
    println!("Reference ENDF/B-VII.1 U-235 @ 0.0253 eV:");
    println!("  σ_f = 584.9 b   σ_g = 98.7 b   σ_el = 15.1 b   ν̄ = 2.437");
    println!("  → ν·σ_f = 1425 b   α = σ_g/σ_f = 0.169   η = 2.079");
}

fn raw_xs(
    rxn_opt: &Option<open_rust_mc::hdf5_reader::NuclideData>,
    energy: f64,
    t_idx: usize,
) -> f64 {
    let Some(rxn) = rxn_opt else {
        return 0.0;
    };
    if t_idx >= rxn.xs_per_temp.len() {
        return 0.0;
    }
    let e = &rxn.energies;
    let xs = &rxn.xs_per_temp[t_idx];
    if e.is_empty() {
        return 0.0;
    }
    if energy <= e[0] {
        return xs[0];
    }
    if energy >= *e.last().unwrap() {
        return *xs.last().unwrap();
    }
    let i = match e
        .binary_search_by(|x| x.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
    {
        Ok(i) => return xs[i],
        Err(i) => i - 1,
    };
    let frac = (energy - e[i]) / (e[i + 1] - e[i]);
    xs[i] + frac * (xs[i + 1] - xs[i])
}
