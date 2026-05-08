//! CE adjoint-photon FW-CADIS importance map for `shield_slab`.
//!
//! Runs the continuous-energy adjoint photon slab walker
//! (`crate::transport::adjoint_photon::adjoint_slab_walk`) — which
//! composes the inverted Klein-Nishina kernel
//! (`adjoint_compton_scatter`), the self-adjoint Rayleigh kernel, and
//! a kill-on-absorption policy for photoelectric / pair — into the
//! 1D `(z, E)` track-length tally that is the exact adjoint flux for
//! a slab geometry. Sums over E to produce the same 1D
//! `CadisMap` JSON schema that `shield_slab --cadis-load` consumes.
//!
//! Sibling of `rr_cadis_slab`. The random-ray adjoint is fast and
//! energy-collapsed (1-group multigroup XS); this binary keeps the
//! full continuous-energy spectrum in the adjoint walk and only
//! collapses at the JSON-write step. For a monoenergetic detector
//! response at 1 MeV in water the two should agree to within stochastic
//! uncertainty — the CE walker carries broader applicability for
//! response functions that span multiple decades of energy where the
//! 1-group reduction breaks.

use std::path::PathBuf;

use clap::Parser;
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::transport::adjoint_photon::{AdjointSlabConfig, adjoint_slab_walk};
use open_rust_mc::transport::rng::Rng;

const N_A: f64 = 6.022_140_76e23;

#[derive(Parser, Debug)]
#[command(
    name = "adjoint_photon_cadis_slab",
    about = "Continuous-energy adjoint photon FW-CADIS importance map for shield_slab"
)]
struct Args {
    /// Directory holding ENDF/B-VII.1 photon HDF5 (e.g.
    /// `data/endfb-vii.1-hdf5/photon`).
    #[arg(long)]
    photon_data: PathBuf,

    /// Slab thickness in cm.
    #[arg(long, default_value_t = 100.0)]
    thickness_cm: f64,

    /// Material — only `water` is plumbed at present (matches what
    /// `shield_slab` defaults to). Add concrete / Pb / Fe / W in
    /// follow-up if needed; the walker is material-agnostic.
    #[arg(long, default_value = "water")]
    material: String,

    /// Detector response energy (eV). The adjoint walk is born at
    /// the detector face at this energy.
    #[arg(long, default_value_t = 1.0e6)]
    response_energy_ev: f64,

    /// Upper bound on adjoint Compton up-scatter (eV). Should bracket
    /// the source spectrum; 5 MeV is fine for any 1 MeV beam-on-water
    /// problem.
    #[arg(long, default_value_t = 5.0e6)]
    e_in_max: f64,

    /// Lower energy cutoff (eV). Below this the walker terminates.
    #[arg(long, default_value_t = 1.0e3)]
    e_cut_ev: f64,

    /// Number of adjoint histories. 100 k is enough to populate every
    /// z-bin to ≪10 % rel-err on a 25-bin mesh through 100 cm of water.
    #[arg(long, default_value_t = 100_000)]
    n_histories: usize,

    /// Number of z-bins in the importance map.
    #[arg(long, default_value_t = 25)]
    n_z_bins: usize,

    /// Number of log-spaced energy bins inside the walker. Matters
    /// only for the up-scatter sanity check the walker prints; the
    /// CadisMap output is energy-collapsed.
    #[arg(long, default_value_t = 30)]
    n_e_bins: usize,

    /// PCG-64 seed.
    #[arg(long, default_value_t = 0x4D70_u64)]
    seed: u64,

    /// Output JSON path. Schema is `shield_slab`'s `CadisMap`:
    ///   `{"thickness_cm":..., "n_z_bins":..., "counts":[...]}`.
    /// When omitted the JSON prints to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn build_water(photon_data: &std::path::Path) -> Result<PhotonMaterial, String> {
    let h =
        PhotonElement::from_hdf5(&photon_data.join("H.h5")).map_err(|e| format!("H.h5: {e}"))?;
    let o =
        PhotonElement::from_hdf5(&photon_data.join("O.h5")).map_err(|e| format!("O.h5: {e}"))?;
    let n_h2o = 1.00 * N_A / 18.0153 * 1.0e-24;
    Ok(PhotonMaterial::new(vec![(2.0 * n_h2o, h), (n_h2o, o)]).with_density(1.00))
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    println!(
        "adjoint_photon_cadis_slab — CE adjoint photon FW-CADIS for shield_slab\n\
         Thickness: {:.1} cm    n_z_bins: {}    response: {:.2} keV    histories: {}",
        args.thickness_cm,
        args.n_z_bins,
        args.response_energy_ev * 1.0e-3,
        args.n_histories,
    );

    let material = match args.material.as_str() {
        "water" => build_water(&args.photon_data)?,
        other => {
            return Err(format!(
                "unsupported material '{other}' (only 'water' is plumbed; add via build_<m>(...))",
            ));
        }
    };

    let cfg = AdjointSlabConfig {
        thickness_cm: args.thickness_cm,
        e_in_max: args.e_in_max,
        e_cut_ev: args.e_cut_ev,
        n_histories: args.n_histories,
        n_z_bins: args.n_z_bins,
        n_e_bins: args.n_e_bins,
        max_events_per_history: 5_000,
    };

    let mut rng = Rng::new(args.seed, 0);
    println!("Running adjoint walk...");
    let t0 = std::time::Instant::now();
    let map = adjoint_slab_walk(&cfg, &material, args.response_energy_ev, &mut rng);
    let dt = t0.elapsed().as_secs_f64();
    println!("Done in {dt:.2}s ({:.0} hist/s).", args.n_histories as f64 / dt);

    // Sum over E → ψ̂*(z). Walker stores `flux[iz * n_e_bins + ie]`.
    let mut z_profile = vec![0.0_f64; args.n_z_bins];
    for iz in 0..args.n_z_bins {
        for ie in 0..args.n_e_bins {
            z_profile[iz] += map.flux[iz * args.n_e_bins + ie];
        }
    }

    let phi_max = z_profile.iter().cloned().fold(0.0_f64, f64::max);
    if phi_max <= 0.0 {
        return Err("importance map collapsed to zero — check histories / σ_t".into());
    }

    // shield_slab's CadisMap stores u64 "counts" but treats them as
    // relative weights (the constant scale cancels in WW construction).
    // 1e6 keeps integer rounding error below 1 ppm.
    let scale = 1.0e6_f64 / phi_max;
    let counts: Vec<u64> = z_profile
        .iter()
        .map(|&v| (v * scale).max(0.0).round() as u64)
        .collect();

    println!("\nz-bin          ψ̂*(z) (norm.)   w_target ∝ 1/ψ̂*");
    let dz = args.thickness_cm / args.n_z_bins as f64;
    let print_every = (args.n_z_bins / 20).max(1);
    for (i, &c) in counts.iter().enumerate() {
        if i % print_every == 0 || i == args.n_z_bins - 1 {
            let z_lo = i as f64 * dz;
            let z_hi = (i + 1) as f64 * dz;
            let psi_norm = c as f64 / counts.iter().max().copied().unwrap_or(1).max(1) as f64;
            let w_target = if psi_norm > 0.0 {
                1.0 / psi_norm
            } else {
                f64::INFINITY
            };
            println!(
                "  {:>5.1}–{:<5.1}   {:>13.4}   {:>16.2e}",
                z_lo, z_hi, psi_norm, w_target
            );
        }
    }

    let json = format!(
        "{{\"thickness_cm\":{},\"n_z_bins\":{},\"counts\":[{}]}}",
        args.thickness_cm,
        args.n_z_bins,
        counts
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    if let Some(path) = &args.output {
        std::fs::write(path, &json).map_err(|e| format!("write {}: {e}", path.display()))?;
        println!(
            "\nSaved CE adjoint photon CADIS map → {} ({} bytes)",
            path.display(),
            json.len()
        );
        println!(
            "\nNext: shield_slab --cadis-load {} ...    \
             # uses CE adjoint ψ̂* instead of the random-ray 1-group reduction.",
            path.display()
        );
    } else {
        println!("\n--- JSON (no --output specified) ---\n{json}");
    }
    Ok(())
}
