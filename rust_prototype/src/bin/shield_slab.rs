//! Photon shielding slab benchmark — fixed-source γ transmission
//! through a thick water (or pluggable) slab.
//!
//! The defining characteristic of a shielding benchmark is the
//! **deep-penetration regime**: a tally evaluated at a location
//! where the source-to-detector ratio is much smaller than 1, so
//! analog Monte Carlo spends most of its histories on neutrons
//! (or photons) that never reach the tally. The figure of merit
//! `FOM = 1 / (σ² · t)` is variance-limited rather than CPU-limited,
//! and any improvement comes from variance reduction (CADIS,
//! splitting / roulette weight windows, etc.) rather than from
//! more histories.
//!
//! This binary produces the **analog FOM baseline** that CADIS or
//! manual weight-window weight-reduction will be measured against.
//!
//! # Geometry
//!
//! - Slab in `z ∈ [0, T]`, infinite in `xy` via reflective `±H`
//!   walls (default `H = 100 cm`, far enough that forward-cone
//!   leakage is negligible).
//! - **Reflective** boundary on the back face (`z = 0`) and `xy`
//!   walls.
//! - **Vacuum** boundary on the front face (`z = T`) — this is the
//!   *only* way photons leave the geometry, so `energy_escaped` in
//!   the photon transport result is by construction the transmitted
//!   energy.
//! - Slab is filled with a single homogeneous photon material
//!   (default: water at 1.0 g/cm³).
//!
//! # Source
//!
//! - 1 MeV monodirectional photon beam, born at `(0, 0, ε)` with
//!   `dir = (0, 0, 1)`. (The `ε` nudge keeps the source unambiguously
//!   inside the slab — born-on-surface particles have undefined
//!   collision distances.)
//! - One source photon per history; total `histories` source events.
//!
//! # Tally
//!
//! The total transmitted energy fraction:
//!
//! ```text
//!   T = (1 / N_hist) · Σ_h (E_escaped_h / E_source)
//! ```
//!
//! Standard error estimated from per-history variance. FOM reported
//! as `1 / (σ_T² · t_wall)` in `(s · transmission²)⁻¹`.
//!
//! For 100 cm of water at 1 MeV, μ ≈ 0.0707 / cm → optical thickness
//! ≈ 7.07 mfp → uncollided transmission ≈ exp(−7.07) ≈ 8.5 × 10⁻⁴.
//! Buildup raises this to ~5 × 10⁻³ (an order of magnitude — the
//! classic "buildup factor" effect for thick water shields).
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin shield_slab -- \
//!     <photon_data_dir> [--thickness-cm 100] [--energy-mev 1.0] \
//!     [--histories 1_000_000]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::Vec3;
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::photon::transport::transport_history_csg;
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::transport::rng::Rng;

#[derive(Parser, Debug)]
#[command(
    name = "shield_slab",
    about = "Analog γ shielding-slab transmission benchmark — establishes the FOM baseline for CADIS variance reduction."
)]
struct Args {
    /// Photon-data HDF5 directory (e.g. `data/endfb-vii.1-hdf5/photon`).
    photon_data: PathBuf,

    /// Slab thickness in cm. 100 cm of water at 1 MeV ≈ 7 mfp.
    #[arg(long, default_value_t = 100.0)]
    thickness_cm: f64,

    /// Half-extent of the (reflective) `xy` walls. Should be large
    /// enough that the 1/r² spread of a beam source is captured by
    /// the forward-cone tally.
    #[arg(long, default_value_t = 100.0)]
    half_xy_cm: f64,

    /// Source photon energy in MeV.
    #[arg(long, default_value_t = 1.0)]
    energy_mev: f64,

    /// Number of source histories. With analog MC at ~10⁻³
    /// transmission, 10⁶ histories give ~10³ transmitted events
    /// and ~3 % relative error.
    #[arg(long, default_value_t = 1_000_000)]
    histories: u64,

    /// Energy cutoff (eV) below which photon transport stops.
    #[arg(long, default_value_t = 1.0e3)]
    cutoff_ev: f64,

    /// Random seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Material choice. "water" (1.0 g/cm³) and "concrete" (2.3 g/cm³)
    /// are built-in. Any other string assumes a single-element file
    /// `<name>.h5` in `photon_data`.
    #[arg(long, default_value = "water")]
    material: String,
}

const N_A: f64 = 6.022_140_76e23;

fn build_water(photon_data: &std::path::Path) -> Result<PhotonMaterial, String> {
    let h =
        PhotonElement::from_hdf5(&photon_data.join("H.h5")).map_err(|e| format!("H.h5: {e}"))?;
    let o =
        PhotonElement::from_hdf5(&photon_data.join("O.h5")).map_err(|e| format!("O.h5: {e}"))?;
    // ρ = 1.00 g/cm³, M(H₂O) = 18.0153 g/mol → N = 0.0334 atoms/(b·cm).
    let n_h2o = 1.00 * N_A / 18.0153 * 1.0e-24;
    Ok(PhotonMaterial::new(vec![(2.0 * n_h2o, h), (n_h2o, o)]).with_density(1.00))
}

fn build_concrete(photon_data: &std::path::Path) -> Result<PhotonMaterial, String> {
    // ANSI/ANS-6.4-1997 ordinary concrete (ρ = 2.3 g/cm³). Mass
    // fractions for the dominant elements; sub-percent contributions
    // (Mg, K, Na, Ti, ...) lumped into the average.
    let elements: [(&str, f64, f64); 6] = [
        // (file, atomic mass, mass fraction)
        ("H.h5", 1.00794, 0.0100),
        ("O.h5", 15.9994, 0.4983),
        ("Si.h5", 28.0855, 0.3158),
        ("Ca.h5", 40.078, 0.0826),
        ("Al.h5", 26.9815, 0.0456),
        ("Fe.h5", 55.845, 0.0122),
    ];
    let rho = 2.3_f64;
    let mut entries = Vec::with_capacity(elements.len());
    for &(file, m_g_mol, mass_frac) in &elements {
        let elem = PhotonElement::from_hdf5(&photon_data.join(file))
            .map_err(|e| format!("{file}: {e}"))?;
        // Atom density per element: ρ · w_i · N_A / M_i (× 1e-24 for /(b·cm)).
        let n = rho * mass_frac * N_A / m_g_mol * 1.0e-24;
        entries.push((n, elem));
    }
    Ok(PhotonMaterial::new(entries).with_density(rho))
}

fn build_single_element(
    photon_data: &std::path::Path,
    name: &str,
    rho: f64,
    m_g_mol: f64,
) -> Result<PhotonMaterial, String> {
    let file = format!("{name}.h5");
    let elem = PhotonElement::from_hdf5(&photon_data.join(&file))
        .map_err(|e| format!("{file}: {e}"))?;
    let n = rho * N_A / m_g_mol * 1.0e-24;
    Ok(PhotonMaterial::mono(n, elem).with_density(rho))
}

fn build_geometry(
    thickness_cm: f64,
    half_xy_cm: f64,
) -> (Vec<Surface>, Vec<Cell>) {
    // Surfaces: 0 z_back (refl), 1 z_front (vacuum), 2 x_min (refl),
    // 3 x_max (refl), 4 y_min (refl), 5 y_max (refl).
    let surfaces = vec![
        Surface::PlaneZ {
            z0: 0.0,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneZ {
            z0: thickness_cm,
            bc: BoundaryCondition::Vacuum,
        },
        Surface::PlaneX {
            x0: -half_xy_cm,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneX {
            x0: half_xy_cm,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: -half_xy_cm,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: half_xy_cm,
            bc: BoundaryCondition::Reflective,
        },
    ];
    let cells = vec![Cell::new(
        CellId(0),
        cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ]),
        CellFill::Material(0),
    )];
    (surfaces, cells)
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    println!("============================================================");
    println!("  Photon shielding slab — analog FOM baseline");
    println!("============================================================");
    println!(
        "  thickness  : {:.1} cm   xy half-extent: {:.1} cm",
        args.thickness_cm, args.half_xy_cm,
    );
    println!(
        "  source     : {:.3} MeV monodirectional, normal incidence on z = 0",
        args.energy_mev,
    );
    println!("  histories  : {}", args.histories);
    println!("  material   : {}", args.material);
    println!("  cutoff     : {:.0} eV", args.cutoff_ev);
    println!("  seed       : {}", args.seed);
    println!();

    let photon_material = match args.material.as_str() {
        "water" => build_water(&args.photon_data)?,
        "concrete" => build_concrete(&args.photon_data)?,
        // Single-element shorthand (defaults to typical metallic densities).
        "Pb" => build_single_element(&args.photon_data, "Pb", 11.35, 207.2)?,
        "Fe" => build_single_element(&args.photon_data, "Fe", 7.874, 55.845)?,
        "W" => build_single_element(&args.photon_data, "W", 19.3, 183.84)?,
        other => {
            return Err(format!(
                "unknown material {other:?}: try 'water', 'concrete', 'Pb', 'Fe', or 'W'"
            ));
        }
    };
    let materials = vec![Some(photon_material)];

    let (surfaces, cells) = build_geometry(args.thickness_cm, args.half_xy_cm);

    let source_energy_ev = args.energy_mev * 1.0e6;
    // Born just inside the slab so the find_cell_recursive lookup
    // sees us in cell 0 unambiguously.
    let source_pos = Vec3::new(0.0, 0.0, 1.0e-6);
    let source_dir = Vec3::new(0.0, 0.0, 1.0);

    println!("  ── Running analog photon transport ──");
    let t0 = Instant::now();

    // Per-history transmission fractions for variance estimation.
    // Stack-allocated batch sums would be more efficient for huge
    // history counts, but a Vec is fine up to ~10⁸ histories.
    let mut total_escaped_ev = 0.0_f64;
    let mut total_escaped_squared = 0.0_f64;
    let mut n_transmitted_histories: u64 = 0;
    let mut total_collisions: u64 = 0;

    for h in 0..args.histories {
        let mut rng = Rng::new(args.seed, h);
        let result = transport_history_csg(
            source_pos,
            source_dir,
            source_energy_ev,
            &surfaces,
            &cells,
            &materials,
            args.cutoff_ev,
            &mut rng,
        );
        total_escaped_ev += result.energy_escaped;
        total_escaped_squared += result.energy_escaped * result.energy_escaped;
        if result.energy_escaped > 0.0 {
            n_transmitted_histories += 1;
        }
        total_collisions += result.n_collisions as u64;
    }

    let wall_seconds = t0.elapsed().as_secs_f64();
    let n = args.histories as f64;
    let mean_escaped_ev = total_escaped_ev / n;
    let var_escaped = (total_escaped_squared / n) - mean_escaped_ev.powi(2);
    let stderr_escaped_ev = (var_escaped.max(0.0) / n).sqrt();

    let transmission = mean_escaped_ev / source_energy_ev;
    let transmission_stderr = stderr_escaped_ev / source_energy_ev;
    let rel_err = transmission_stderr / transmission.max(1e-30);

    // Figure of merit: 1 / (σ_rel² · t). Higher is better.
    let fom = if transmission_stderr > 0.0 && wall_seconds > 0.0 {
        1.0 / (rel_err * rel_err * wall_seconds)
    } else {
        0.0
    };

    println!();
    println!("  ── Result ──────────────────────────────────────────────");
    println!(
        "  transmission   = {:.4e} ± {:.4e}  ({:.2}% relative)",
        transmission,
        transmission_stderr,
        100.0 * rel_err,
    );
    println!(
        "  histories transmitted = {} / {} ({:.4}%)",
        n_transmitted_histories,
        args.histories,
        100.0 * n_transmitted_histories as f64 / n,
    );
    println!(
        "  total collisions      = {} ({:.1} per history)",
        total_collisions,
        total_collisions as f64 / n,
    );
    println!("  wall time             = {wall_seconds:.2} s");
    println!("  ns / history          = {:.1}", wall_seconds * 1e9 / n);
    println!();
    println!("  ── FOM (CADIS baseline) ───────────────────────────────");
    println!("  FOM = 1 / (σ_rel² · t)  = {fom:.3e}    [s⁻¹]");
    println!();
    println!("  Use this number as the analog reference. CADIS-driven");
    println!("  weight windows on the same problem must beat this FOM");
    println!("  for the variance reduction to be worthwhile.");
    Ok(())
}
