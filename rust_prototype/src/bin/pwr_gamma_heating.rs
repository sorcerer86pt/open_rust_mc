//! PWR pin cell gamma-heating estimate — coupled neutron + photon
//! transport on a shared CSG geometry, **using per-nuclide γ spectra
//! loaded directly from the HDF5 data library**.
//!
//! # Pipeline
//!
//!  1. Run a neutron k-eigenvalue simulation on the standard PWR pin
//!     cell (UO₂ fuel + Zr clad + H₂O, 1.26 cm pitch, reflective
//!     lattice). At every capture (MT=102), fission (MT=18), and
//!     inelastic-scatter (MT=4/51..91) site, sample the photon
//!     multiplicity from `reaction_{mt}/product_{photon}/yield` and
//!     the outgoing energies from the corresponding
//!     `distribution_0/energy` ContinuousTabular tree — exactly the
//!     same reader path the code already uses for fission-neutron
//!     outgoing spectra.
//!  2. Aggregate active-batch events into a single photon source
//!     bank `(cell_idx, pos, E_γ, MT)`, grouped by reaction class for
//!     diagnostic accounting.
//!  3. Transport each photon history from a uniformly-picked event in
//!     the bank through the same CSG with per-cell `PhotonMaterial`.
//!     Isotropic direction sampled at source (ENDF uncorrelated-angle
//!     product; anisotropic γ distributions are rare and small-effect
//!     for reactor heating).
//!  4. Bin per-collision photon deposits by containing cell and print
//!     the fraction of source energy landing in fuel / gap / clad /
//!     water, plus a capture-vs-fission-vs-inelastic breakdown of the
//!     source bank.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin pwr_gamma_heating -- \
//!     data/endfb-vii.1-hdf5/neutron \
//!     --photon-data data/endfb-vii.1-hdf5/photon
//! ```
//!
//! Defaults are tuned for a converged PWR pin: 150 batches × 50 k
//! neutrons (50 inactive) and 200 k photon histories. Expect ~1 min
//! wall time on a desktop CPU.
//!
//! # Caveats
//!
//! - Photon-product angular distributions are assumed isotropic.
//!   ENDF uncorrelated-angle products dominate for the MTs we sample;
//!   the handful of correlated-angle γ products in ENDF/B-VII.1
//!   contribute <1 % of reactor-γ energy, so this is a small source
//!   of residual bias (<100 pcm on heating fractions).
//! - Photon production from other `(n,X)` absorptions (α, p, d) is
//!   not modelled — amounts to <0.5 % of total γ energy in UO₂.
//! - The kerma approximation is still in place on the photon side
//!   (no electron transport, no secondary bremsstrahlung).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3, ray};
use open_rust_mc::hdf5_reader;
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::photon::transport::transport_history_csg;
use open_rust_mc::thermal::ThermalScatteringData;
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::rng::Rng;
use open_rust_mc::transport::simulate::{self, SimConfig};
use open_rust_mc::transport::xs_provider::{self, TableXsProvider};

// ── Pin cell dimensions (same as bin/pwr_pincell.rs) ────────────────
const FUEL_OR: f64 = 0.4096;
const CLAD_IR: f64 = 0.4180;
const CLAD_OR: f64 = 0.4750;
const PITCH: f64 = 1.2600;

// ── Photon material atom densities (per-element, photon XS) ─────────
const UO2_MOL_DENSITY: f64 = 2.319e-2;
const ZR_ATOM_DENSITY: f64 = 4.324e-2;
const H2O_MOL_DENSITY: f64 = 2.474e-2;

// Photon source spectra are now sampled at each neutron collision
// from the HDF5 `reaction_{MT}/product_{photon}/distribution_0/energy`
// tree, not from a notional two-line spectrum.

// ── Nuclide specs matching pwr_pincell's NUCLIDE_SPECS ──────────────
// (filename, AWR, fallback nu-bar, temp_idx after sort).
const NUCLIDE_SPECS: &[(&str, f64, f64, usize)] = &[
    ("U235.h5", 233.025, 2.43, 3), // 0  fuel  900K
    ("U238.h5", 236.006, 2.49, 3), // 1  fuel  900K
    ("O16.h5", 15.858, 0.0, 3),    // 2  fuel  900K
    ("H1.h5", 0.999, 0.0, 2),      // 3  water 600K
    ("Zr90.h5", 89.132, 0.0, 2),   // 4  clad  600K
    ("Zr91.h5", 90.130, 0.0, 2),   // 5  clad  600K
    ("Zr92.h5", 91.126, 0.0, 2),   // 6  clad  600K
    ("Zr94.h5", 93.120, 0.0, 2),   // 7  clad  600K
    ("O16.h5", 15.858, 0.0, 2),    // 8  water 600K
];

struct Args {
    neutron_data: PathBuf,
    photon_data: PathBuf,
    n_neutron_batches: u32,
    n_neutron_inactive: u32,
    n_neutron_particles: u32,
    n_photon: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        neutron_data: PathBuf::new(),
        photon_data: PathBuf::new(),
        n_neutron_batches: 150,
        n_neutron_inactive: 50,
        n_neutron_particles: 50_000,
        n_photon: 200_000,
    };
    let mut it = std::env::args().skip(1);
    let Some(neutron_data) = it.next() else {
        return Err(
            "usage: pwr_gamma_heating <neutron_data_dir> --photon-data <photon_dir> \
             [--n-neutron-batches N] [--n-neutron-inactive N] \
             [--n-neutron-particles N] [--n-photon N]"
                .to_string(),
        );
    };
    args.neutron_data = PathBuf::from(neutron_data);

    while let Some(a) = it.next() {
        match a.as_str() {
            "--photon-data" => {
                args.photon_data =
                    PathBuf::from(it.next().ok_or("missing value for --photon-data")?);
            }
            "--n-neutron-batches" => {
                args.n_neutron_batches = it
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--n-neutron-inactive" => {
                args.n_neutron_inactive = it
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--n-neutron-particles" => {
                args.n_neutron_particles = it
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--n-photon" => {
                args.n_photon = it
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    if args.photon_data.as_os_str().is_empty() {
        return Err("--photon-data is required".to_string());
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(m) => {
            eprintln!("{m}");
            return ExitCode::from(2);
        }
    };

    let (surfaces, cells) = setup_geometry();
    let materials_n = setup_neutron_materials();

    // ── Phase 1: neutron transport ─────────────────────────────────
    println!("── Phase 1: neutron (n,γ) capture tally ──");
    let t0 = Instant::now();
    let xs = match load_table_xs(&args.neutron_data) {
        Ok(p) => p,
        Err(m) => {
            eprintln!("{m}");
            return ExitCode::from(1);
        }
    };
    println!("  XS load: {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);

    let config = SimConfig {
        batches: args.n_neutron_batches,
        inactive: args.n_neutron_inactive,
        particles_per_batch: args.n_neutron_particles,
        seed: 1,
        auto_inactive: None,
    };
    let t1 = Instant::now();
    let (batch_results, k_eff) =
        simulate::run_eigenvalue(&config, &surfaces, &cells, &materials_n, &xs);
    let neutron_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Aggregate the photon source bank across ACTIVE batches only
    // (source-converged estimator). The neutron loop sampled every
    // (n,γ), fission, and inelastic γ spectrum directly from HDF5 —
    // no notional lines, no approximated mean energies.
    let mut photon_bank: Vec<simulate::PhotonSourceEvent> = Vec::new();
    let mut captures_per_cell = vec![0.0_f64; cells.len()];
    let mut active_batches = 0_usize;
    for br in &batch_results {
        if br.active {
            active_batches += 1;
            for (i, c) in br.captures_by_cell.iter().enumerate() {
                captures_per_cell[i] += *c;
            }
            photon_bank.extend(br.photon_events.iter().copied());
        }
    }

    let total_captures: f64 = captures_per_cell.iter().sum();
    println!(
        "  Neutron: k = {:.5}, {} active batches, {} captures, {} photon events, {:.1} s",
        k_eff,
        active_batches,
        total_captures as u64,
        photon_bank.len(),
        neutron_ms / 1000.0
    );
    println!("  Captures by cell (capture-tally proxy):");
    for (i, c) in captures_per_cell.iter().enumerate() {
        let label = cell_label(i);
        let frac = if total_captures > 0.0 {
            c / total_captures
        } else {
            0.0
        };
        println!(
            "    cell {i} [{label:<5}]: {:>8.0} captures  ({:>6.3} %)",
            c,
            100.0 * frac
        );
    }

    // MT breakdown on the sampled photon bank.
    let mut n_capture = 0_u64;
    let mut n_fission = 0_u64;
    let mut n_inelastic = 0_u64;
    let mut e_capture = 0.0_f64;
    let mut e_fission = 0.0_f64;
    let mut e_inelastic = 0.0_f64;
    for ev in &photon_bank {
        match ev.mt {
            102 => {
                n_capture += 1;
                e_capture += ev.energy;
            }
            18 => {
                n_fission += 1;
                e_fission += ev.energy;
            }
            _ => {
                n_inelastic += 1;
                e_inelastic += ev.energy;
            }
        }
    }
    println!("  Photon source bank by reaction class:");
    println!(
        "    capture (MT=102): {:>10} events, {:>10.3e} eV total",
        n_capture, e_capture
    );
    println!(
        "    fission (MT=18):  {:>10} events, {:>10.3e} eV total",
        n_fission, e_fission
    );
    println!(
        "    inelastic (MT=4/51-91): {:>10} events, {:>10.3e} eV total",
        n_inelastic, e_inelastic
    );

    if photon_bank.is_empty() {
        eprintln!("No photon events sampled — cannot seed photon phase.");
        return ExitCode::from(1);
    }

    // ── Phase 2: photon transport from real HDF5-sampled source ────
    println!("\n── Phase 2: photon transport from HDF5 γ spectra ──");
    let materials_p = match load_photon_materials(&args.photon_data) {
        Ok(m) => m,
        Err(m) => {
            eprintln!("{m}");
            return ExitCode::from(1);
        }
    };

    let t2 = Instant::now();
    let mut deposited_per_cell = vec![0.0_f64; cells.len()];
    let mut escaped_energy = 0.0_f64;
    let mut orphan_deposit = 0.0_f64;
    let mut total_source_energy = 0.0_f64;

    for i in 0..args.n_photon {
        let mut rng = Rng::new(0xB0F1_0000 + i as u64, 1);

        // Pick a photon source event uniformly — each event carries
        // its real (cell, position, energy, MT) from the neutron
        // collision that emitted it.
        let ev_idx = (rng.uniform() * photon_bank.len() as f64) as usize;
        let ev = photon_bank[ev_idx.min(photon_bank.len() - 1)];
        let pos = Vec3::new(ev.pos[0], ev.pos[1], ev.pos[2]);
        let (dx, dy, dz) = rng.isotropic_direction();
        let e_src = ev.energy;
        total_source_energy += e_src;

        let r = transport_history_csg(
            pos,
            Vec3::new(dx, dy, dz),
            e_src,
            &surfaces,
            &cells,
            &materials_p,
            1_000.0, // 1 keV cutoff
            &mut rng,
        );

        escaped_energy += r.energy_escaped;
        for (p, e) in &r.deposits {
            if let Some(idx) = ray::find_cell(*p, &surfaces, &cells) {
                deposited_per_cell[idx] += e;
            } else {
                orphan_deposit += e;
            }
        }
    }
    let photon_s = t2.elapsed().as_secs_f64();

    println!(
        "  Photon: {} histories, {:.2} s, {:.0} hist/s, total source {:.3e} eV",
        args.n_photon,
        photon_s,
        args.n_photon as f64 / photon_s,
        total_source_energy
    );

    // ── Final report ───────────────────────────────────────────────
    println!("\n── Energy deposition by region ──");
    println!(
        "  {:<6} {:>16} {:>10}",
        "region", "deposited (eV)", "fraction"
    );
    let mut total_dep = 0.0;
    for (i, e) in deposited_per_cell.iter().enumerate() {
        total_dep += e;
        println!(
            "  {:<6} {:>16.3e} {:>9.3} %",
            cell_label(i),
            e,
            100.0 * e / total_source_energy
        );
    }
    if orphan_deposit > 0.0 {
        println!(
            "  {:<6} {:>16.3e} {:>9.3} %",
            "orphan",
            orphan_deposit,
            100.0 * orphan_deposit / total_source_energy
        );
    }
    println!(
        "  {:<6} {:>16.3e} {:>9.3} %",
        "escape",
        escaped_energy,
        100.0 * escaped_energy / total_source_energy
    );
    let sum = total_dep + orphan_deposit + escaped_energy;
    println!(
        "  {:<6} {:>16.3e} {:>9.3} %",
        "sum",
        sum,
        100.0 * sum / total_source_energy
    );

    if escaped_energy / total_source_energy > 1.0e-3 {
        eprintln!(
            "WARNING: reflective lattice leaked {:.2} % of source energy",
            100.0 * escaped_energy / total_source_energy
        );
    }

    ExitCode::SUCCESS
}

// ── Geometry (same as pwr_pincell) ──────────────────────────────────

fn cell_label(idx: usize) -> &'static str {
    match idx {
        0 => "fuel",
        1 => "gap",
        2 => "clad",
        3 => "water",
        _ => "?",
    }
}

fn setup_geometry() -> (Vec<Surface>, Vec<Cell>) {
    let half = PITCH / 2.0;
    let z_half = half;
    let surfaces = vec![
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: FUEL_OR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_IR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_OR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::PlaneX {
            x0: -half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneX {
            x0: half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: -half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneZ {
            z0: -z_half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneZ {
            z0: z_half,
            bc: BoundaryCondition::Reflective,
        },
    ];
    let box_aabb = Aabb::new(
        Vec3::new(-half, -half, -z_half),
        Vec3::new(half, half, z_half),
    );
    let cells = vec![
        Cell::new(
            CellId(0),
            cell::intersect_all(vec![cell::inside(0), cell::outside(7), cell::inside(8)]),
            CellFill::Material(0),
        )
        .with_aabb(Aabb::new(
            Vec3::new(-FUEL_OR, -FUEL_OR, -z_half),
            Vec3::new(FUEL_OR, FUEL_OR, z_half),
        ))
        .with_temperature(900.0),
        Cell::new(
            CellId(1),
            cell::intersect_all(vec![
                cell::outside(0),
                cell::inside(1),
                cell::outside(7),
                cell::inside(8),
            ]),
            CellFill::Void,
        )
        .with_aabb(Aabb::new(
            Vec3::new(-CLAD_IR, -CLAD_IR, -z_half),
            Vec3::new(CLAD_IR, CLAD_IR, z_half),
        )),
        Cell::new(
            CellId(2),
            cell::intersect_all(vec![
                cell::outside(1),
                cell::inside(2),
                cell::outside(7),
                cell::inside(8),
            ]),
            CellFill::Material(1),
        )
        .with_aabb(Aabb::new(
            Vec3::new(-CLAD_OR, -CLAD_OR, -z_half),
            Vec3::new(CLAD_OR, CLAD_OR, z_half),
        ))
        .with_temperature(600.0),
        Cell::new(
            CellId(3),
            cell::intersect_all(vec![
                cell::outside(2),
                cell::outside(3),
                cell::inside(4),
                cell::outside(5),
                cell::inside(6),
                cell::outside(7),
                cell::inside(8),
            ]),
            CellFill::Material(2),
        )
        .with_aabb(box_aabb)
        .with_temperature(600.0),
    ];
    (surfaces, cells)
}

fn setup_neutron_materials() -> Vec<Material> {
    let mut fuel = Material::new("UO2", 900.0);
    fuel.add_nuclide(0.000719, 0);
    fuel.add_nuclide(0.022482, 1);
    fuel.add_nuclide(0.046402, 2);

    let mut clad = Material::new("Zircaloy", 600.0);
    clad.add_nuclide(0.022932, 4);
    clad.add_nuclide(0.004996, 5);
    clad.add_nuclide(0.007636, 6);
    clad.add_nuclide(0.007740, 7);

    let mut water = Material::new("H2O", 600.0);
    water.add_nuclide(0.049486, 3);
    water.add_nuclide(0.024743, 8);

    vec![fuel, clad, water]
}

fn load_table_xs(data_dir: &Path) -> Result<TableXsProvider, String> {
    let mut tables = Vec::with_capacity(NUCLIDE_SPECS.len());
    for &(filename, awr, nu_bar, nuc_temp_idx) in NUCLIDE_SPECS {
        let path = data_dir.join(filename);
        if !path.exists() {
            return Err(format!("missing neutron file: {}", path.display()));
        }
        tables.push(xs_provider::load_nuclide_table(
            &path,
            nuc_temp_idx,
            awr,
            nu_bar,
        ));
    }
    let thermal = load_thermal(data_dir);
    Ok(TableXsProvider {
        nuclides: tables,
        thermal,
    })
}

fn load_thermal(data_dir: &Path) -> Vec<Option<Arc<ThermalScatteringData>>> {
    let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; NUCLIDE_SPECS.len()];
    let h2o_path = data_dir.join("c_H_in_H2O.h5");
    if h2o_path.exists()
        && let Ok(tsl) = hdf5_reader::load_thermal_scattering(&h2o_path)
    {
        thermal[3] = Some(Arc::new(tsl));
    }
    thermal
}

fn load_photon_materials(data_dir: &Path) -> Result<Vec<Option<PhotonMaterial>>, String> {
    let load = |name: &str| -> Result<PhotonElement, String> {
        PhotonElement::from_hdf5(&data_dir.join(name))
            .map_err(|e| format!("failed to load {name}: {e}"))
    };
    let h = load("H.h5")?;
    let o1 = load("O.h5")?;
    let o2 = load("O.h5")?;
    let zr = load("Zr.h5")?;
    let u = load("U.h5")?;

    let uo2 = PhotonMaterial::new(vec![(UO2_MOL_DENSITY, u), (2.0 * UO2_MOL_DENSITY, o1)]);
    let clad = PhotonMaterial::mono(ZR_ATOM_DENSITY, zr);
    let h2o = PhotonMaterial::new(vec![(2.0 * H2O_MOL_DENSITY, h), (H2O_MOL_DENSITY, o2)]);

    Ok(vec![Some(uo2), Some(clad), Some(h2o)])
}

#[allow(dead_code)]
fn _flush() {
    let _ = std::io::stdout().flush();
}
