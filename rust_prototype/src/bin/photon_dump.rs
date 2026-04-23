//! Diagnostic CLI: load an OpenMC photon HDF5 file and print a summary.
//!
//! Usage:
//!   cargo run --release --bin photon_dump -- <path/to/element.h5>
//!
//! Example:
//!   cargo run --release --bin photon_dump -- data/endfb-vii.1-hdf5/photon/C.h5

use std::path::PathBuf;
use std::process::ExitCode;

use open_rust_mc::photon::PhotonElement;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: photon_dump <path/to/element.h5>");
        return ExitCode::from(2);
    };
    let path = PathBuf::from(path);

    let elem = match PhotonElement::from_hdf5(&path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    let n = elem.n_energy();
    let e_lo = elem.energy[0];
    let e_hi = elem.energy[n - 1];

    println!("Element {:>2} (Z={}): {} energy points, {:.3e} to {:.3e} eV",
        elem.symbol, elem.z, n, e_lo, e_hi);

    // Sample a few probe energies to show XS magnitudes.
    let probes_ev = [1.0e3, 1.0e5, 1.0e6, 1.0e7];
    println!();
    println!("{:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "E (eV)", "coh", "incoh", "photoel", "pp_nuc", "pp_el", "total");
    for &e in &probes_ev {
        let idx = elem.energy.partition_point(|x| *x < e);
        if idx >= n {
            continue;
        }
        let total = elem.total_xs_at(idx);
        println!("{:>10.3e}  {:>10.3e}  {:>10.3e}  {:>10.3e}  {:>10.3e}  {:>10.3e}  {:>10.3e}",
            elem.energy[idx],
            elem.coherent_xs[idx],
            elem.incoherent_xs[idx],
            elem.photoelectric_xs[idx],
            elem.pair_production_nuclear_xs[idx],
            elem.pair_production_electron_xs[idx],
            total);
    }

    println!();
    println!("Subshells ({}):", elem.subshells.len());
    for s in &elem.subshells {
        println!("  {:<4}  binding {:>10.2} eV   n_e = {:>5.2}   xs_pts = {:>4}   transitions = {:>3}",
            s.designator, s.binding_energy, s.num_electrons, s.xs.len(), s.transitions.len());
    }

    println!();
    println!("Coherent form factor: {} points, x in [{:.3e}, {:.3e}] 1/Å",
        elem.coherent_form_factor.x.len(),
        elem.coherent_form_factor.x.first().copied().unwrap_or(0.0),
        elem.coherent_form_factor.x.last().copied().unwrap_or(0.0));
    println!("Incoherent scattering factor: {} points, x in [{:.3e}, {:.3e}] 1/Å",
        elem.incoherent_scattering_factor.x.len(),
        elem.incoherent_scattering_factor.x.first().copied().unwrap_or(0.0),
        elem.incoherent_scattering_factor.x.last().copied().unwrap_or(0.0));

    ExitCode::SUCCESS
}
