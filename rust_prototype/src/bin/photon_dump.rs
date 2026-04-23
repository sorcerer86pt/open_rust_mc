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
    println!("Coherent form factor F(x,Z):                {} points, x in [{:.3e}, {:.3e}] 1/Å",
        elem.coherent_form_factor.x.len(),
        elem.coherent_form_factor.x.first().copied().unwrap_or(0.0),
        elem.coherent_form_factor.x.last().copied().unwrap_or(0.0));
    println!("Coherent integrated F²(x,Z) cumulative:     {} points",
        elem.coherent_integrated_form_factor.x.len());
    println!("Coherent anomalous f'(E):                   {} points, E in [{:.3e}, {:.3e}] eV",
        elem.coherent_anomalous.real.grid.len(),
        elem.coherent_anomalous.real.grid.first().copied().unwrap_or(0.0),
        elem.coherent_anomalous.real.grid.last().copied().unwrap_or(0.0));
    println!("Coherent anomalous f''(E):                  {} points, E in [{:.3e}, {:.3e}] eV",
        elem.coherent_anomalous.imag.grid.len(),
        elem.coherent_anomalous.imag.grid.first().copied().unwrap_or(0.0),
        elem.coherent_anomalous.imag.grid.last().copied().unwrap_or(0.0));
    println!("Incoherent scattering factor S(x,Z):        {} points, x in [{:.3e}, {:.3e}] 1/Å",
        elem.incoherent_scattering_factor.x.len(),
        elem.incoherent_scattering_factor.x.first().copied().unwrap_or(0.0),
        elem.incoherent_scattering_factor.x.last().copied().unwrap_or(0.0));

    println!();
    let cp = &elem.compton_profiles;
    let total_occ: f64 = cp.num_electrons.iter().sum();
    println!("Compton profiles: {} shells, {} pz points, total occupancy = {:.2} (Z={})",
        cp.n_shells(), cp.n_pz(), total_occ, elem.z);
    println!("  pz grid [a.u.]: [{:.2}, {:.2}] in {} points",
        cp.pz.first().copied().unwrap_or(0.0),
        cp.pz.last().copied().unwrap_or(0.0),
        cp.n_pz());

    println!();
    let br = &elem.bremsstrahlung;
    println!("Bremsstrahlung DCS grid: {} electron-E × {} photon-E (I = {:.1} eV)",
        br.electron_energy.len(), br.photon_energy.len(), br.mean_excitation_energy);
    println!("  electron T_e in [{:.3e}, {:.3e}] eV",
        br.electron_energy.first().copied().unwrap_or(0.0),
        br.electron_energy.last().copied().unwrap_or(0.0));
    println!("  photon k/T_e in [{:.3e}, {:.3e}] (scaled)",
        br.photon_energy.first().copied().unwrap_or(0.0),
        br.photon_energy.last().copied().unwrap_or(0.0));
    println!("  Sternheimer oscillators: {}", br.ionization_energy.len());

    ExitCode::SUCCESS
}
