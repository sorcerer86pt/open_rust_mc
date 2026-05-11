//! S(α,β) sampler audit for c_H_in_H2O at T = 294 K.
//!
//! Probes the engine's `ThermalScatteringData::sample` outputs against
//! known-physics reference values to isolate the −1846 pcm bias on
//! HEU-SOL-THERM-001:
//!
//!   * Total XS at fixed E_in — at thermal (0.0253 eV) for H-in-H2O,
//!     ENDF/B-VII.1 gives ≈ 80 b. Free-atom σ_f (H1.h5 MT=2) is ≈ 20 b.
//!   * Mean lethargy gain per collision ξ = ⟨ln(E_in/E_out)⟩. For
//!     **free** H (A = 1), ξ = 1.0 exactly. For H **bound in H₂O**,
//!     the molecular vibrations soften the distribution at low E_in;
//!     ENDF/B-VII.1 ξ for c_H_in_H2O is typically around 0.92–1.05
//!     depending on E_in.
//!   * Mean cosine ⟨μ⟩_lab — for free H elastic, ⟨μ⟩ = 2/(3A) = 2/3.
//!     For bound H at thermal, ⟨μ⟩ is smaller (~0.4–0.6) because the
//!     bound recoil reduces forward peaking.
//!   * ⟨E_out⟩ / E_in — at incident energies ≫ kT, expect <1
//!     (down-scatter dominates). Near thermal, ratio ≈ 1 (equilibrium).
//!
//! Outputs a single table that's directly comparable to a matching
//! probe of OpenMC's `IncoherentInelasticAE::sample`.
//!
//! Usage:  thermal_audit [data_dir]

use open_rust_mc::hdf5_reader::load_thermal_scattering;
use open_rust_mc::transport::rng::Rng;
use std::path::PathBuf;

const N: usize = 500_000;
const ENERGIES_EV: &[f64] = &[
    0.0253, // thermal kT @ 293.6 K
    0.05, 0.1, 0.3, 0.625, 1.0, 2.0, 3.0,
];

fn main() {
    let data_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../data/endfb-vii.1-hdf5/neutron"));
    let path = data_dir.join("c_H_in_H2O.h5");
    let tsl = load_thermal_scattering(&path)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()));

    println!("thermal_audit — c_H_in_H2O sampler vs known physics");
    println!("file: {}", path.display());
    println!(
        "energy_max = {:.3} eV   AWR = {:.4}   temperatures: {:?}",
        tsl.energy_max, tsl.awr, tsl.temp_labels
    );
    // Pick T=294K (first index typically).
    let t_idx = tsl.select_temperature(293.6, 0.5);
    println!(
        "selected T idx = {}  ({})\n",
        t_idx, tsl.temp_labels[t_idx]
    );

    println!(
        "{:>10} {:>10} {:>12} {:>12} {:>12} {:>12} {:>14}",
        "E_in (eV)", "σ_tot (b)", "⟨E_out⟩", "⟨E_out/E_in⟩", "ξ (leth.)", "⟨μ⟩", "ratio_up_down"
    );
    println!("{}", "-".repeat(90));

    for &e_in in ENERGIES_EV {
        let sigma = tsl.total_xs(e_in, t_idx);
        let mut rng = Rng::new(0xC0DE_5A11 + (e_in * 1e6) as u64, 0);
        let mut sum_eout = 0.0_f64;
        let mut sum_ratio = 0.0_f64;
        let mut sum_lethargy = 0.0_f64;
        let mut sum_mu = 0.0_f64;
        let mut n_up = 0usize;
        let mut n_down = 0usize;
        for _ in 0..N {
            let (e_out, mu) = tsl.sample(e_in, t_idx, &mut rng);
            sum_eout += e_out;
            let ratio = e_out / e_in;
            sum_ratio += ratio;
            sum_lethargy += -ratio.max(1e-30).ln(); // ξ = ⟨ln(E_in/E_out)⟩ = -⟨ln(ratio)⟩
            sum_mu += mu;
            if e_out > e_in {
                n_up += 1;
            } else {
                n_down += 1;
            }
        }
        let mean_eout = sum_eout / N as f64;
        let mean_ratio = sum_ratio / N as f64;
        let xi = sum_lethargy / N as f64;
        let mean_mu = sum_mu / N as f64;
        let up_dn = n_up as f64 / n_down.max(1) as f64;
        println!(
            "{:>10.4} {:>10.3} {:>12.4e} {:>12.4} {:>12.4} {:>12.4} {:>14.3}",
            e_in, sigma, mean_eout, mean_ratio, xi, mean_mu, up_dn
        );
    }

    println!();
    println!("Reference expectations (ENDF/B-VII.1 H in H2O @ 294 K, well-known):");
    println!("  σ_tot @ 0.0253 eV ≈ 80 b");
    println!("  ξ_free for H = 1.0;  for bound H ≈ 1.0 at E_in > 0.1 eV,");
    println!("    drops near thermal where up-scatter becomes significant.");
    println!("  ⟨μ⟩_lab for free H elastic = 2/3 ≈ 0.667.");
    println!("    For bound H at thermal: significantly smaller (≈ 0.4–0.6).");
    println!("  ratio_up_down should be ≪ 1 at E_in ≫ kT, ≈ 1 at E_in = kT.");
}
