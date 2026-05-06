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
//! - **Absorption channel attribution**: at every
//!   `CollisionOutcome::Absorption`, the loop samples photon
//!   products from *all* non-fission absorption MTs simultaneously
//!   (MT=102 radiative capture, MT=103 `(n,p)`, MT=107 `(n,α)`).
//!   Threshold reactions return yield=0 below their kinematic
//!   threshold so no spurious photons come from sub-threshold MTs.
//!   Above threshold, all three channels emit in proportion to their
//!   tabulated yields — rather than being weighted by the actual
//!   per-MT cross-section fractions at the collision energy. This
//!   over-counts photon production at high energies by the ratio of
//!   yields-summed to yields-cross-weighted. For UO₂ at PWR
//!   spectrum, MT=103/107 contribute <1 % of total γ energy (they
//!   are dominated by MT=102 and MT=18), so the bias on heating
//!   fractions is <100 pcm.
//! - **Kerma approximation** on the photon side (no electron
//!   transport, no secondary bremsstrahlung). Standard ~5 % effect
//!   for shielding-type calculations.
//!
//! Photon-product angular distributions in ENDF/B-VII.1 are already
//! stored as isotropic for the MTs we sample on U-235/U-238/O-16/H-1/
//! Zr — the previously-stated "isotropic approximation" caveat does
//! not apply.

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
use open_rust_mc::photon::bremsstrahlung::MaterialBremss;
use open_rust_mc::photon::electron::{radiation_length_cm, track_integrate_electron_csg_with_ms};
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
        verbose: true,
        parallel: true,
        tallies: Default::default(),
        statepoint_path: None,
        survival_biasing: None,
        initial_source_bank: None,
        weight_window: None,
        disable_delayed_neutrons: false,
        urr_equivalence: None,
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

    // MT breakdown on the sampled photon bank. Keep an explicit map
    // so threshold reactions (MT=103 n,p and MT=107 n,α) are visible
    // even when their counts are small — if zero, it means no fast
    // neutrons reached the ~5-10 MeV threshold in this run.
    let mut by_mt: std::collections::BTreeMap<u32, (u64, f64)> = std::collections::BTreeMap::new();
    for ev in &photon_bank {
        let e = by_mt.entry(ev.mt).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ev.energy;
    }
    println!("  Photon source bank by MT:");
    for (mt, (n, e)) in &by_mt {
        let label = match *mt {
            102 => "(n,γ) capture",
            18 => "(n,f) fission",
            103 => "(n,p) proton",
            107 => "(n,α) alpha",
            4 => "(n,n') inelastic (lumped)",
            51..=91 => "(n,n') inelastic (level)",
            _ => "other",
        };
        println!(
            "    MT={:<3} [{:<24}]: {:>10} events, {:>10.3e} eV",
            mt, label, n, e
        );
    }

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
    // Per-cell Seltzer-Berger bremsstrahlung samplers, mirroring
    // `materials_p` one-to-one. Voids remain `None`, so a Compton
    // recoil electron born in the He gap (which cannot happen for
    // Compton but could for other edge cases) will cleanly skip the
    // brems step rather than panicking.
    let brems_p: Vec<Option<MaterialBremss>> = materials_p
        .iter()
        .map(|m| m.as_ref().map(MaterialBremss::from_photon_material))
        .collect();

    // Per-cell radiation length X₀ [cm] for Highland multiple-scattering.
    // Void cells get ∞ (no MS). Cells whose `cell.fill` is
    // `CellFill::Material(m)` resolve to `materials_p[m]` via Bragg
    // additivity of the element table; density is taken from the same
    // `PhotonMaterial`.
    let x0_per_cell: Vec<f64> = cells
        .iter()
        .map(|c| match c.fill {
            CellFill::Material(m) => materials_p
                .get(m as usize)
                .and_then(|o| o.as_ref())
                .map(radiation_length_cm)
                .unwrap_or(f64::INFINITY),
            _ => f64::INFINITY,
        })
        .collect();
    println!("  Radiation lengths per cell (cm):");
    for (i, x0) in x0_per_cell.iter().enumerate() {
        let label = cell_label(i);
        if x0.is_finite() {
            println!("    cell {i} [{label:<5}]: X₀ = {x0:.3} cm");
        } else {
            println!("    cell {i} [{label:<5}]: X₀ = ∞ (void / no MS)");
        }
    }

    let t2 = Instant::now();
    let mut deposited_per_cell = vec![0.0_f64; cells.len()];
    let mut escaped_energy = 0.0_f64;
    let mut orphan_deposit = 0.0_f64;
    let mut total_source_energy = 0.0_f64;
    let mut brems_photons_emitted = 0_u64;
    let mut brems_energy_emitted = 0.0_f64;

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

        // Secondary photon bank for this history — holds both photon
        // products from transport_history_csg and bremsstrahlung γs
        // born during electron track integration. Transported until
        // empty so one source neutron fully completes before moving on.
        let mut photon_bank_local: Vec<(Vec3, Vec3, f64, usize)> =
            vec![(pos, Vec3::new(dx, dy, dz), e_src, 0)];

        while let Some((p0, d0, e0, _c0)) = photon_bank_local.pop() {
            let r = transport_history_csg(
                p0,
                d0,
                e0,
                &surfaces,
                &cells,
                &materials_p,
                1_000.0,
                &mut rng,
            );

            escaped_energy += r.energy_escaped;
            // Non-electron photon-side deposits (low-E cutoff, sub-
            // threshold pair production).
            for (p, e) in &r.deposits {
                if let Some(idx) = ray::find_cell(*p, &surfaces, &cells) {
                    deposited_per_cell[idx] += e;
                } else {
                    orphan_deposit += e;
                }
            }
            // Recoil electrons — track-integrate through the CSG.
            // Before integrating, sample a single TTB bremsstrahlung
            // photon from the birth cell's material. Its energy is
            // deducted from the electron's CSDA budget and a photon is
            // added to the local bank for transport through the same
            // CSG, redistributing energy that otherwise would have been
            // pinned to the electron's immediate neighbourhood.
            for ele in &r.electrons {
                let mut csda_energy = ele.e_kin_ev;
                let mid = cell_material_id(&cells[ele.cell_idx]);
                // Brems emission is sampled as a single Poisson-like
                // event along the electron's CSDA track: probability
                // P = 1 − exp(−Σ_rad · R_e(E)). Only emit when the
                // sampled ξ falls inside that probability band so
                // total radiative yield matches physics (~1 % for
                // UO₂ at sub-MeV electrons, <0.1 % for water). Using
                // `sample_photon_energy` unconditionally as before
                // over-emitted by ~20-30×.
                if let (Some(Some(bs)), Some(Some(_mat))) = (brems_p.get(mid), materials_p.get(mid))
                {
                    // NIST ESTAR-calibrated radiation yield as the
                    // single-photon emission probability. See
                    // `MaterialBremss::radiative_yield_approx` for the
                    // fit; σ_rad is unreliable as an absolute value
                    // because the HDF5 DCS scaling convention varies.
                    let p_brems = bs.radiative_yield_approx(ele.e_kin_ev);
                    if p_brems > 0.0
                        && rng.uniform() < p_brems
                        && let Some(e_gamma) = bs.sample_photon_energy(ele.e_kin_ev, &mut rng)
                    {
                        let (gx, gy, gz) = rng.isotropic_direction();
                        photon_bank_local.push((
                            ele.pos,
                            Vec3::new(gx, gy, gz),
                            e_gamma,
                            ele.cell_idx,
                        ));
                        brems_photons_emitted += 1;
                        brems_energy_emitted += e_gamma;
                        csda_energy -= e_gamma;
                        if csda_energy < 0.0 {
                            csda_energy = 0.0;
                        }
                    }
                }
                // 0.005 cm sub-step: ~1/8 of UO₂ range at 1 MeV,
                // ~1/100 of H₂O range. Smaller steps → more Highland
                // scatters applied but also more geometry queries;
                // 0.005 balances fidelity and runtime.
                track_integrate_electron_csg_with_ms(
                    ele.pos,
                    ele.dir,
                    csda_energy,
                    ele.cell_idx,
                    &surfaces,
                    &cells,
                    &materials_p,
                    &x0_per_cell,
                    0.005,
                    &mut rng,
                    &mut deposited_per_cell,
                );
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
    if brems_photons_emitted > 0 {
        let brems_frac = brems_energy_emitted / total_source_energy;
        let n_electrons_rough = brems_photons_emitted; // one photon per electron in TTB
        println!(
            "  Bremsstrahlung: {} γ emitted, {:.3e} eV total ({:.3} % of source energy)",
            n_electrons_rough,
            brems_energy_emitted,
            100.0 * brems_frac
        );
    }

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

/// Material ID for a cell, used to index into `brems_p` / `materials_p`.
/// Returns `usize::MAX` for void/universe fills, which harmlessly indexes
/// outside the materials array so the caller's `.get()` yields `None`.
fn cell_material_id(cell: &Cell) -> usize {
    match cell.fill {
        CellFill::Material(m) => m as usize,
        _ => usize::MAX,
    }
}

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

    // Mass densities (g/cm³) used for CSDA electron-range
    // displacement. Match the atom-density choices:
    //   UO₂: 10.4 g/cm³ (fresh PWR fuel pellet)
    //   Zr: 6.55 g/cm³  (Zircaloy-4 approximation)
    //   H₂O: 0.74 g/cm³ (~600 K PWR moderator density)
    let uo2 = PhotonMaterial::new(vec![(UO2_MOL_DENSITY, u), (2.0 * UO2_MOL_DENSITY, o1)])
        .with_density(10.4);
    let clad = PhotonMaterial::mono(ZR_ATOM_DENSITY, zr).with_density(6.55);
    let h2o = PhotonMaterial::new(vec![(2.0 * H2O_MOL_DENSITY, h), (H2O_MOL_DENSITY, o2)])
        .with_density(0.74);

    Ok(vec![Some(uo2), Some(clad), Some(h2o)])
}

#[allow(dead_code)]
fn _flush() {
    let _ = std::io::stdout().flush();
}
