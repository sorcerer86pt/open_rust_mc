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
use open_rust_mc::geometry::{Aabb, Vec3};
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::photon::transport::{transport_history_csg, transport_history_csg_with_ww};
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::transport::rng::Rng;
use open_rust_mc::transport::weight_window::WeightWindow;

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

    /// Number of CADIS-lite calibration histories. When `> 0`, runs a
    /// pre-pass with photons born at `z = T` heading into the slab
    /// (`-z`), tallies their collision density per z-bin, and prints
    /// the resulting importance map. This is the input to the next-
    /// step CADIS weight-window translation; a non-zero value here
    /// gives a diagnostic readout without yet biasing the forward run.
    #[arg(long, default_value_t = 0)]
    cadis_calibration: u64,

    /// Number of z-bins for the CADIS-lite importance map.
    #[arg(long, default_value_t = 50)]
    cadis_z_bins: usize,

    /// Persist the calibration importance map to a JSON file. The
    /// next-step CADIS WW translator (`WeightWindow::from_flux` over
    /// a 1×1×n_z voxel mesh) will ingest this file. Schema:
    ///   {"thickness_cm": ..., "n_z_bins": ..., "counts": [...]}
    #[arg(long)]
    cadis_save: Option<PathBuf>,

    /// Load a previously-saved CADIS importance map (from --cadis-save)
    /// and translate it into a WeightWindow that biases the forward
    /// run. Splitting/roulette fires on every voxel transition so
    /// photon weight stays in the per-voxel band defined by ψ̂*. The
    /// transmission tally accumulates weight × escaped energy and
    /// remains an unbiased estimator. **This is the CADIS payoff
    /// step: FOM should jump 50-1000× over the analog baseline.**
    #[arg(long)]
    cadis_load: Option<PathBuf>,

    /// Width of the WW band as `w_upper / w_lower` ratio. Larger
    /// values widen the band (less aggressive splitting + roulette);
    /// 5.0 is the textbook default.
    #[arg(long, default_value_t = 5.0)]
    ww_ratio: f64,

    /// Voxels with importance below `ww_floor × ψ̂*_max` are flagged
    /// inactive (no splitting / roulette there). Default 1e-3.
    #[arg(long, default_value_t = 1.0e-3)]
    ww_floor: f64,
}

/// Loaded importance map from a `--cadis-save` JSON file.
#[derive(Debug, serde::Deserialize)]
struct CadisMap {
    thickness_cm: f64,
    n_z_bins: usize,
    counts: Vec<u64>,
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

/// CADIS-lite calibration pass.
///
/// Runs `n_histories` photons born at `(0, 0, T-ε)` heading `(0,0,-1)`
/// — i.e. into the slab from the detector face — and tallies the
/// collision density per z-bin.
///
/// The resulting density `ψ̂\*(z)` is a proxy for the adjoint flux:
/// voxels reached by these "detector-backward" photons are voxels
/// that contribute to the response at the detector. The CADIS
/// importance map is `w_target(z) ∝ ψ̂\*_max / ψ̂\*(z)` — high-
/// importance voxels (close to the detector) get small `w_target`
/// (split → finer sampling) and low-importance voxels (close to the
/// source) get large `w_target` (roulette → coarser sampling).
///
/// This is the "lite" form: not a true adjoint MC (no transposed
/// scatter kernels), just running the same forward physics from the
/// detector side. It gives the right qualitative gradient for slab
/// shielding because photon transport is dominated by Compton +
/// Rayleigh which are kinematically symmetric, and photoelectric
/// absorption breaks the symmetry only weakly above ~50 keV.
///
/// Returns the per-z-bin collision count (un-normalised; the WW
/// translator only cares about ratios).
fn cadis_calibration_pass(
    n_histories: u64,
    n_z_bins: usize,
    thickness_cm: f64,
    source_energy_ev: f64,
    cutoff_ev: f64,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Option<PhotonMaterial>],
    seed: u64,
) -> Vec<u64> {
    let mut counts = vec![0_u64; n_z_bins];
    let dz = thickness_cm / n_z_bins as f64;

    // Source: at z = T - ε, heading -z (into the slab from the detector).
    // The same `transport_history_csg` driver handles this — geometry
    // / physics don't care which way the source points.
    let source_pos = Vec3::new(0.0, 0.0, thickness_cm - 1.0e-6);
    let source_dir = Vec3::new(0.0, 0.0, -1.0);

    for h in 0..n_histories {
        let mut rng = Rng::new(seed.wrapping_add(0xC4D1_5_5EE_D), h);
        let result = transport_history_csg(
            source_pos,
            source_dir,
            source_energy_ev,
            surfaces,
            cells,
            materials,
            cutoff_ev,
            &mut rng,
        );
        // Bin every collision position by its z-coordinate.
        for &(pos, _e) in &result.collisions {
            let bin = ((pos.z / dz) as usize).min(n_z_bins.saturating_sub(1));
            counts[bin] += 1;
        }
    }
    counts
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

    // CADIS-lite calibration pass.
    if args.cadis_calibration > 0 {
        println!("  ── CADIS-lite calibration: {} histories, {} z-bins ──",
                 args.cadis_calibration, args.cadis_z_bins);
        let t_calib = Instant::now();
        let counts = cadis_calibration_pass(
            args.cadis_calibration,
            args.cadis_z_bins,
            args.thickness_cm,
            source_energy_ev,
            args.cutoff_ev,
            &surfaces,
            &cells,
            &materials,
            args.seed,
        );
        let calib_seconds = t_calib.elapsed().as_secs_f64();
        let total: u64 = counts.iter().sum();
        let max = *counts.iter().max().unwrap_or(&0);
        let dz = args.thickness_cm / args.cadis_z_bins as f64;
        println!(
            "  calibration: {} total collisions in {:.2} s ({:.1} ns/history)",
            total,
            calib_seconds,
            calib_seconds * 1e9 / args.cadis_calibration as f64,
        );
        println!();
        println!("  z [cm]      collisions     ψ̂*  (norm.)   w_target ∝ 1/ψ̂*");
        println!("  ----------  -------------  ------------  ------------------");
        for (i, &c) in counts.iter().enumerate() {
            let z_lo = i as f64 * dz;
            let z_hi = (i + 1) as f64 * dz;
            let psi_norm = c as f64 / max as f64;
            // Avoid div-by-zero for empty bins.
            let w_target = if c > 0 { max as f64 / c as f64 } else { f64::INFINITY };
            // Print a coarse subset (every n_z_bins/20 rows) so the
            // line count stays readable for any --cadis-z-bins.
            let stride = (args.cadis_z_bins / 20).max(1);
            if i % stride == 0 {
                println!(
                    "  {z_lo:>5.1}–{z_hi:<5.1}  {c:>13}  {psi_norm:>12.4}  {w_target:>16.2e}",
                );
            }
        }
        println!();
        println!("  note: this is the importance map ψ̂*(z) for the");
        println!("  next-step CADIS weight-window translator. Voxels");
        println!("  with low ψ̂* (near z=0, far from detector) are the");
        println!("  ones that will receive splitting via WW; voxels");
        println!("  with high ψ̂* (near z=T) get roulette to keep the");
        println!("  computational budget concentrated where it matters.");
        println!();

        // Optional persistence of the importance map for downstream
        // CADIS WW translation. Schema is the bare minimum needed by
        // the next-step `WeightWindow::from_flux` builder; extending
        // to xyz mesh / multi-energy is a follow-on.
        if let Some(path) = &args.cadis_save {
            let json = format!(
                "{{\"thickness_cm\":{},\"n_z_bins\":{},\"counts\":[{}]}}",
                args.thickness_cm,
                args.cadis_z_bins,
                counts.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
            );
            std::fs::write(path, json)
                .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
            println!("  importance map saved to {}", path.display());
            println!();
        }

        // === Next-step CADIS roadmap (in-source TODO) ===========
        // 1. Translate `counts` → `WeightWindow` via
        //    `WeightWindow::from_flux(aabb, [1,1,n_z], flux,
        //                             w_ref=1.0, ratio=5.0, floor=1e-3)`
        //    — the existing infrastructure already does the
        //    `w_target ∝ φ_max / φ` math; pass the calibration
        //    counts as the `flux` argument.
        // 2. Add a `weight: f64` field to the photon state in
        //    `photon::transport::transport_one_csg`. Initialise to
        //    1.0 at source; thread through the bank entries and
        //    secondaries so daughters inherit it.
        // 3. After each free-flight in `transport_one_csg`, query
        //    the WW for the new voxel: split if `weight > w_upper`
        //    (push N copies into the bank with weight w/N), roulette
        //    if `weight < w_lower` (kill with prob 1−w/w_survive
        //    else restore to w_survive).
        // 4. The transmission tally accumulates `result.energy_escaped`
        //    weighted by `weight` instead of unit-weight.
        // 5. Re-run shield_slab with --cadis-load <map.json>
        //    --use-cadis-ww and measure FOM gain on the existing
        //    analog 348/s baseline. Wagner-Haghighat 2003 reports
        //    50-1000× FOM gain for slab problems of this depth.
        // ========================================================
    }
    // Born just inside the slab so the find_cell_recursive lookup
    // sees us in cell 0 unambiguously.
    let source_pos = Vec3::new(0.0, 0.0, 1.0e-6);
    let source_dir = Vec3::new(0.0, 0.0, 1.0);

    // Load the CADIS importance map (from --cadis-save) and build a
    // `WeightWindow`. The 1D z-bin counts become a 1×1×n_z 3D mesh
    // aligned with the slab. We set `w_ref` so that `w_target` at
    // the source position equals 1.0 — that's the consistent-CADIS
    // normalization: source photons (weight 1.0) sit exactly at the
    // band centroid at birth, no roulette there. As they propagate
    // into higher-importance voxels, `w_target` shrinks and the
    // weight-1 photon ends up above `w_upper` → splitting fires,
    // pushing more samples toward the detector.
    let weight_window: Option<WeightWindow> = if let Some(path) = &args.cadis_load {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let map: CadisMap = serde_json::from_str(&text)
            .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
        let aabb = Aabb::new(
            Vec3::new(-args.half_xy_cm, -args.half_xy_cm, 0.0),
            Vec3::new(args.half_xy_cm, args.half_xy_cm, map.thickness_cm),
        );
        let flux: Vec<f64> = map.counts.iter().map(|&c| c as f64).collect();
        // Find the importance value at the source z-bin, normalize w_ref so
        // that w_target(source) ≈ 1.0.
        let dz = map.thickness_cm / map.n_z_bins as f64;
        let source_bin = ((source_pos.z / dz) as usize).min(map.n_z_bins - 1);
        let phi_source = flux[source_bin].max(1.0);
        let phi_max = flux.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
        let w_ref = phi_source / phi_max;
        let ww = WeightWindow::from_flux(
            &aabb,
            [1, 1, map.n_z_bins],
            &flux,
            w_ref,
            args.ww_ratio,
            args.ww_floor,
        );
        println!(
            "  ── CADIS WW loaded from {} ({} z-bins, ratio {}, floor {}) ──",
            path.display(),
            map.n_z_bins,
            args.ww_ratio,
            args.ww_floor,
        );
        println!(
            "  w_ref (calibrated so w_target(source)=1.0): {w_ref:.4e}",
        );
        Some(ww)
    } else {
        None
    };
    let source_weight: f64 = 1.0;

    println!(
        "  ── Running {} photon transport ──",
        if weight_window.is_some() { "CADIS-biased" } else { "analog" },
    );
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
        let result = transport_history_csg_with_ww(
            source_pos,
            source_dir,
            source_energy_ev,
            source_weight,
            &surfaces,
            &cells,
            &materials,
            args.cutoff_ev,
            weight_window.as_ref(),
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
