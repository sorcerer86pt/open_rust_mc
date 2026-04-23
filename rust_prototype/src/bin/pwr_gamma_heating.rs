//! PWR pin cell gamma-heating estimate via photon transport on the
//! same CSG geometry the neutron binary uses.
//!
//! # What this binary does
//! 1. Builds the standard PWR pin cell (fuel OR = 0.4096 cm, clad IR
//!    = 0.4180 cm, clad OR = 0.4750 cm, pitch 1.26 cm) — bit-for-bit
//!    the same geometry as `pwr_pincell`, with the outer box
//!    Reflective (infinite lattice) so no energy leaks.
//! 2. Loads photon cross-section data (ENDF/B-VII.1 EPICS HDF5) for
//!    H, O, Zr, U and composes a per-cell `PhotonMaterial`:
//!      - fuel cell: UO2 at 10.4 g/cm³,
//!      - gap cell: void (He fill approximated as vacuum),
//!      - clad cell: pure Zr at 6.55 g/cm³,
//!      - water cell: H2O at 0.74 g/cm³.
//! 3. Samples source photons uniformly inside the fuel cylinder —
//!    this is where ~95 % of thermal-neutron captures occur in fresh
//!    UO2, so it is a reasonable stand-in for the full `(n,γ)` source
//!    distribution until the neutron tally is wired through (future
//!    work). Energies are sampled from a two-line notional capture-γ
//!    distribution (70 % × 1 MeV soft, 30 % × 5 MeV hard; mean ≈
//!    2.2 MeV, typical for U-dominated capture).
//! 4. Transports each photon through the CSG via
//!    `transport_history_csg` with the four kernels already in the
//!    repo (coherent / Compton+Doppler / photoelectric+EADL / pair).
//! 5. Bins per-collision deposition into the cell containing the
//!    deposit position and reports fraction of source energy landing
//!    in fuel / gap / clad / water / (cutoff-escape).
//!
//! # Caveats (documented honestly — this is a scaffold, not a
//! # published benchmark)
//! - Capture source distribution is uniform-in-fuel, not spatially
//!   self-shielded. A real (n,γ) tally pushes more of the source to
//!   the fuel rim (higher thermal flux there); the ~96 %-in-fuel
//!   number VERA Problem 1 quotes is for that spatial distribution.
//! - Source energy is a two-line notional spectrum, not the per-
//!   nuclide cascade HDF5 spectra. Mean energy is realistic; shape
//!   differs from a real capture-γ line-and-continuum distribution.
//! - Photon source only; fission γs and inelastic-scatter γs are not
//!   emitted. These add another ~4 % of the reactor power budget and
//!   go mostly into fuel.
//!
//! # Usage
//!   cargo run --release --bin pwr_gamma_heating -- \
//!     data/endfb-vii.1-hdf5/photon --n 50000
//!
//! Output: fraction of deposited energy per region + timing.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{ray, Aabb, Vec3};
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::photon::transport::transport_history_csg;
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::transport::rng::Rng;

// ── Pin cell dimensions (same as bin/pwr_pincell.rs) ────────────────
const FUEL_OR: f64 = 0.4096;
const CLAD_IR: f64 = 0.4180;
const CLAD_OR: f64 = 0.4750;
const PITCH: f64 = 1.2600;

// ── Material atom densities (photon transport is per-element) ───────
// UO2 at 10.4 g/cm³, M = 270.03 g/mol → 2.319e22 molecules/cm³
//   = 2.319e-2 molecules/(barn·cm). One U and two O per molecule.
const UO2_MOL_DENSITY: f64 = 2.319e-2;

// Pure Zr at 6.55 g/cm³, M = 91.224 g/mol → 4.324e22 atoms/cm³
//   = 4.324e-2 atoms/(barn·cm).
const ZR_ATOM_DENSITY: f64 = 4.324e-2;

// Water at 0.74 g/cm³, M = 18.0153 g/mol → 2.474e22 molecules/cm³.
// Two H and one O per molecule.
const H2O_MOL_DENSITY: f64 = 2.474e-2;

// Notional capture-gamma source: 70 % at 1 MeV, 30 % at 5 MeV.
const SOURCE_E_SOFT_EV: f64 = 1.0e6;
const SOURCE_E_HARD_EV: f64 = 5.0e6;
const SOURCE_HARD_FRACTION: f64 = 0.30;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(data_dir) = args.next() else {
        eprintln!(
            "usage: pwr_gamma_heating <photon_data_dir> [--n N_HISTORIES]\n\
             example: pwr_gamma_heating data/endfb-vii.1-hdf5/photon --n 50000"
        );
        return ExitCode::from(2);
    };
    let data_dir = PathBuf::from(data_dir);

    let mut n_hist: usize = 50_000;
    while let Some(a) = args.next() {
        if a == "--n"
            && let Some(v) = args.next()
            && let Ok(v) = v.parse::<usize>()
        {
            n_hist = v;
        }
    }

    // ── Load photon elements ───────────────────────────────────────
    // Load twice where the same element appears in two materials —
    // `PhotonElement` is deliberately non-`Clone` (holds big owned
    // cross-section arrays).
    let h = match load_element(&data_dir, "H.h5") {
        Ok(e) => e,
        Err(m) => return bail(m),
    };
    let o_for_uo2 = match load_element(&data_dir, "O.h5") {
        Ok(e) => e,
        Err(m) => return bail(m),
    };
    let o_for_water = match load_element(&data_dir, "O.h5") {
        Ok(e) => e,
        Err(m) => return bail(m),
    };
    let zr = match load_element(&data_dir, "Zr.h5") {
        Ok(e) => e,
        Err(m) => return bail(m),
    };
    let u = match load_element(&data_dir, "U.h5") {
        Ok(e) => e,
        Err(m) => return bail(m),
    };

    // UO2: 1 × U, 2 × O per molecule.
    let uo2 = PhotonMaterial::new(vec![
        (UO2_MOL_DENSITY, u),
        (2.0 * UO2_MOL_DENSITY, o_for_uo2),
    ]);
    // Zr clad.
    let clad = PhotonMaterial::mono(ZR_ATOM_DENSITY, zr);
    // H2O: 2 × H, 1 × O per molecule.
    let h2o = PhotonMaterial::new(vec![
        (2.0 * H2O_MOL_DENSITY, h),
        (H2O_MOL_DENSITY, o_for_water),
    ]);

    let (surfaces, cells) = setup_geometry();

    // Per-cell photon material vector. Matches the CellFill::Material
    // indices in setup_geometry: 0 = fuel, 1 = clad, 2 = water.
    let materials: Vec<Option<PhotonMaterial>> = vec![Some(uo2), Some(clad), Some(h2o)];

    // Pre-resolve cell-id → human label for the report.
    let cell_labels = ["fuel", "gap", "clad", "water"];
    let mut deposited_per_cell = vec![0.0_f64; cells.len()];
    let mut escaped_energy = 0.0_f64;
    let mut cutoff_deposited = 0.0_f64; // subset of total for reference
    let mut total_source_energy = 0.0_f64;

    println!(
        "PWR pin gamma heating — {} histories, reflective lattice\n\
         fuel OR {:.4} cm, clad IR {:.4} cm, clad OR {:.4} cm, pitch {:.4} cm",
        n_hist, FUEL_OR, CLAD_IR, CLAD_OR, PITCH
    );
    println!(
        "Source: uniform in fuel cylinder; E = {:.1} MeV ({:.0} %) + {:.1} MeV ({:.0} %)",
        SOURCE_E_SOFT_EV * 1e-6,
        100.0 * (1.0 - SOURCE_HARD_FRACTION),
        SOURCE_E_HARD_EV * 1e-6,
        100.0 * SOURCE_HARD_FRACTION
    );

    let start = std::time::Instant::now();

    for i in 0..n_hist {
        let mut rng = Rng::new(0xB0F1_0000 + i as u64, 1);
        let pos = sample_uniform_in_fuel(&mut rng);
        let (dx, dy, dz) = rng.isotropic_direction();
        let e_src = if rng.uniform() < SOURCE_HARD_FRACTION {
            SOURCE_E_HARD_EV
        } else {
            SOURCE_E_SOFT_EV
        };
        total_source_energy += e_src;

        let r = transport_history_csg(
            pos,
            Vec3::new(dx, dy, dz),
            e_src,
            &surfaces,
            &cells,
            &materials,
            1_000.0, // 1 keV cutoff
            &mut rng,
        );

        escaped_energy += r.energy_escaped;
        for (p, e) in &r.deposits {
            // Bin the deposit into whichever cell contains the
            // deposition point. Reflective lattice means this always
            // resolves inside the box.
            if let Some(idx) = ray::find_cell(*p, &surfaces, &cells) {
                deposited_per_cell[idx] += e;
            } else {
                // Defensive: photons that deposited at an exact
                // surface. Attribute to the nearest neighbour cell.
                cutoff_deposited += e;
            }
        }
    }

    let elapsed = start.elapsed();

    println!(
        "\nResults (total source energy {:.3e} eV, {:.0} hist/s):",
        total_source_energy,
        n_hist as f64 / elapsed.as_secs_f64()
    );
    println!(
        "  {:<6} {:>16} {:>10}",
        "region", "deposited (eV)", "fraction"
    );
    let mut total_dep = 0.0;
    for (i, e) in deposited_per_cell.iter().enumerate() {
        total_dep += e;
        let label = cell_labels.get(i).copied().unwrap_or("?");
        println!(
            "  {:<6} {:>16.3e} {:>9.3} %",
            label,
            e,
            100.0 * e / total_source_energy
        );
    }
    if cutoff_deposited > 0.0 {
        println!(
            "  {:<6} {:>16.3e} {:>9.3} %",
            "orphan",
            cutoff_deposited,
            100.0 * cutoff_deposited / total_source_energy
        );
    }
    println!(
        "  {:<6} {:>16.3e} {:>9.3} %",
        "escape", escaped_energy, 100.0 * escaped_energy / total_source_energy
    );
    println!(
        "  {:<6} {:>16.3e} {:>9.3} %",
        "sum",
        total_dep + cutoff_deposited + escaped_energy,
        100.0 * (total_dep + cutoff_deposited + escaped_energy) / total_source_energy
    );

    println!(
        "\nElapsed: {:.2} s",
        elapsed.as_secs_f64()
    );

    // Sanity: reflective lattice must not leak.
    if escaped_energy / total_source_energy > 1.0e-3 {
        eprintln!(
            "WARNING: reflective lattice leaked {:.2} % of source energy",
            100.0 * escaped_energy / total_source_energy
        );
    }

    ExitCode::SUCCESS
}

fn bail(msg: String) -> ExitCode {
    eprintln!("{msg}");
    ExitCode::from(1)
}

fn load_element(dir: &Path, name: &str) -> Result<PhotonElement, String> {
    let p = dir.join(name);
    PhotonElement::from_hdf5(&p).map_err(|e| format!("failed to load {}: {e}", p.display()))
}

/// Sample a position uniformly inside the fuel cylinder (0 ≤ r ≤
/// FUEL_OR, −z_half ≤ z ≤ +z_half).
fn sample_uniform_in_fuel(rng: &mut Rng) -> Vec3 {
    let z_half = PITCH / 2.0;
    let r = (rng.uniform()).sqrt() * FUEL_OR;
    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
    let z = (rng.uniform() * 2.0 - 1.0) * z_half;
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

fn setup_geometry() -> (Vec<Surface>, Vec<Cell>) {
    let half = PITCH / 2.0;
    let z_half = half;

    let surfaces = vec![
        // 0: fuel outer cylinder
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: FUEL_OR,
            bc: BoundaryCondition::Transmission,
        },
        // 1: clad inner cylinder
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_IR,
            bc: BoundaryCondition::Transmission,
        },
        // 2: clad outer cylinder
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_OR,
            bc: BoundaryCondition::Transmission,
        },
        // 3-6: x/y reflective box
        Surface::PlaneX { x0: -half, bc: BoundaryCondition::Reflective },
        Surface::PlaneX { x0: half, bc: BoundaryCondition::Reflective },
        Surface::PlaneY { y0: -half, bc: BoundaryCondition::Reflective },
        Surface::PlaneY { y0: half, bc: BoundaryCondition::Reflective },
        // 7-8: z reflective (infinite axial lattice)
        Surface::PlaneZ { z0: -z_half, bc: BoundaryCondition::Reflective },
        Surface::PlaneZ { z0: z_half, bc: BoundaryCondition::Reflective },
    ];

    let box_aabb = Aabb::new(
        Vec3::new(-half, -half, -z_half),
        Vec3::new(half, half, z_half),
    );

    let cells = vec![
        // 0: Fuel
        Cell::new(
            CellId(0),
            cell::intersect_all(vec![cell::inside(0), cell::outside(7), cell::inside(8)]),
            CellFill::Material(0),
        )
        .with_aabb(Aabb::new(
            Vec3::new(-FUEL_OR, -FUEL_OR, -z_half),
            Vec3::new(FUEL_OR, FUEL_OR, z_half),
        )),
        // 1: Gap (He fill — treated as vacuum)
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
        // 2: Clad
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
        )),
        // 3: Water
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
        .with_aabb(box_aabb),
    ];

    (surfaces, cells)
}
