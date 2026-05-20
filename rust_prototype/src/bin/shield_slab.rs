// SPDX-License-Identifier: MIT
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
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::photon::transport::{NeeConfig, transport_history_csg_with_ww_nee};
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

    /// Load a CADIS importance map (JSON, schema
    /// `{"thickness_cm": ..., "n_z_bins": ..., "counts": [...]}`) and
    /// translate it into a `WeightWindow` that biases the forward run.
    /// Splitting/roulette fires on every voxel transition so photon
    /// weight stays in the per-voxel band defined by ψ̂*. The
    /// transmission tally accumulates weight × escaped energy and
    /// remains an unbiased estimator.
    ///
    /// The recommended generator for this map is the `rr_cadis_slab`
    /// binary, which solves the multigroup random-ray adjoint for the
    /// same slab geometry. See `outputs/random_ray_cadis_fom.txt` for
    /// the head-to-head measurement that motivated dropping the older
    /// in-binary "lite" collision-density proxy.
    #[arg(long)]
    cadis_load: Option<PathBuf>,

    /// Width of the WW band as `w_upper / w_lower` ratio. Larger
    /// values widen the band (less aggressive splitting + roulette);
    /// 5.0 is the textbook default.
    #[arg(long, default_value_t = 5.0)]
    ww_ratio: f64,

    /// Adaptive-ratio coefficient. When > 0, each voxel's band ratio
    /// scales as `ratio_v = ww_ratio · (1 + ww_growth · log10(φ_max/φ_v))`,
    /// widening the band at low-importance voxels (far from the
    /// detector). Set to 0 (default) for fixed `ww_ratio` everywhere.
    /// Empirically 0.5–2.0 helps at thick slabs (>10 mfp) where the
    /// importance gradient is steep enough that fixed ratio is either
    /// too tight near the source or too loose near the detector.
    #[arg(long, default_value_t = 0.0)]
    ww_growth: f64,

    /// Voxels with importance below `ww_floor × ψ̂*_max` are flagged
    /// inactive (no splitting / roulette there). Default 1e-3.
    #[arg(long, default_value_t = 1.0e-3)]
    ww_floor: f64,

    /// Enable the next-event (DXTRAN-style) estimator for the
    /// transmitted-energy tally. At every Compton collision the
    /// driver adds the expected energy that would scatter forward and
    /// arrive unscattered at the detector face — this fires at every
    /// collision regardless of whether the photon physically escapes,
    /// so the tally accumulates much faster than analog at deep
    /// penetration. Compatible with `--cadis-load`: NEE + WW combine
    /// for additional FOM gain. v1 limitations (Compton only, free-
    /// electron Klein-Nishina, no back-reflection contribution, plus
    /// a known systematic bias from per-collision summation that
    /// hasn't been fully diagnosed) are documented in
    /// `outputs/random_ray_cadis_fom.txt` and `src/photon/nee.rs`.
    #[arg(long, default_value_t = false)]
    next_event: bool,

    /// MCNP-style exclusion-zone thickness (cm) for the next-event
    /// estimator. When `> 0`, collisions inside the slab of thickness
    /// `nee_exclusion_cm` adjacent to the detector face score an
    /// analytically z-averaged contribution instead of the raw
    /// per-collision integrand. The regularisation is the standard
    /// fix for point-detector `1/r²` variance singularities; for the
    /// planar detector here the integrand is bounded so it's a
    /// parametric safety knob, not a known bias fix. Default 0.0 =
    /// disabled (numerically identical to v1 NEE).
    #[arg(long, default_value_t = 0.0)]
    nee_exclusion_cm: f64,

    /// Diagnostic: dump per-collision NEE contributions for every
    /// history. Use with very low `--histories` (≤ 20) — output is
    /// ~100 lines per history. Helps trace where the NEE accumulator
    /// diverges from analog expectations.
    #[arg(long, default_value_t = false)]
    nee_trace: bool,
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
    let elem =
        PhotonElement::from_hdf5(&photon_data.join(&file)).map_err(|e| format!("{file}: {e}"))?;
    let n = rho * N_A / m_g_mol * 1.0e-24;
    Ok(PhotonMaterial::mono(n, elem).with_density(rho))
}

fn build_geometry(thickness_cm: f64, half_xy_cm: f64) -> (Vec<Surface>, Vec<Cell>) {
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
        let ww = if args.ww_growth > 0.0 {
            WeightWindow::from_flux_adaptive(
                &aabb,
                [1, 1, map.n_z_bins],
                &flux,
                w_ref,
                args.ww_ratio,
                args.ww_growth,
                args.ww_floor,
            )
        } else {
            WeightWindow::from_flux(
                &aabb,
                [1, 1, map.n_z_bins],
                &flux,
                w_ref,
                args.ww_ratio,
                args.ww_floor,
            )
        };
        println!(
            "  ── CADIS WW loaded from {} ({} z-bins, ratio {} growth {}, floor {}) ──",
            path.display(),
            map.n_z_bins,
            args.ww_ratio,
            args.ww_growth,
            args.ww_floor,
        );
        println!("  w_ref (calibrated so w_target(source)=1.0): {w_ref:.4e}",);
        Some(ww)
    } else {
        None
    };
    let source_weight: f64 = 1.0;

    let nee_cfg = if args.next_event {
        // Force trace on whenever --next-event is on. The trace
        // capture has ~5–10% wall overhead but lets us compute the
        // first-only / last-only / full-sum estimator variants at the
        // same time from one run for clean head-to-head comparison.
        Some(NeeConfig {
            detector_z_cm: args.thickness_cm,
            exclusion_cm: args.nee_exclusion_cm.max(0.0),
            trace: true,
        })
    } else {
        None
    };

    let mode_label = match (weight_window.is_some(), args.next_event) {
        (true, true) => "CADIS+NEE",
        (true, false) => "CADIS-biased",
        (false, true) => "NEE",
        (false, false) => "analog",
    };
    println!("  ── Running {} photon transport ──", mode_label);
    let t0 = Instant::now();

    // Per-history transmission fractions for variance estimation.
    // In NEE mode the NEE tally is the unbiased estimator; otherwise
    // the analog escape tally is.
    let mut total_escaped_ev = 0.0_f64;
    let mut total_escaped_squared = 0.0_f64;
    let mut n_transmitted_histories: u64 = 0;
    let mut total_collisions: u64 = 0;
    // Three-way estimator accumulators, all measured from the same
    // transport run via the per-collision trace:
    //  - full: current (sum every collision's NEE_i + uncollided)
    //  - first: only the first collision's NEE + uncollided
    //  - last: only the last collision's NEE + uncollided
    let mut total_first_coll_nee = 0.0_f64;
    let mut total_uncollided_nee = 0.0_f64;
    let mut total_tally_first = 0.0_f64;
    let mut total_tally_first_sq = 0.0_f64;
    let mut total_tally_last = 0.0_f64;
    let mut total_tally_last_sq = 0.0_f64;

    for h in 0..args.histories {
        let mut rng = Rng::new(args.seed, h);
        let result = transport_history_csg_with_ww_nee(
            source_pos,
            source_dir,
            source_energy_ev,
            source_weight,
            &surfaces,
            &cells,
            &materials,
            args.cutoff_ev,
            weight_window.as_ref(),
            nee_cfg.as_ref(),
            &mut rng,
        );
        // Pick estimator: NEE tally if --next-event, else analog escape.
        let tally = if args.next_event {
            result.nee_tally
        } else {
            result.energy_escaped
        };
        total_escaped_ev += tally;
        total_escaped_squared += tally * tally;
        if tally > 0.0 {
            n_transmitted_histories += 1;
        }
        total_collisions += result.n_collisions as u64;
        // Diagnostic decomposition: first-collision-only NEE +
        // uncollided source contribution. Total NEE = uncollided +
        // Σ per-collision contributions; here we isolate the first
        // collision and the constant uncollided piece so we can see
        // which component carries the bias.
        let first_contrib = result.nee_trace.first().map(|t| t.2).unwrap_or(0.0);
        let last_contrib = result.nee_trace.last().map(|t| t.2).unwrap_or(0.0);
        total_first_coll_nee += first_contrib;
        // Uncollided contribution per history is the analog
        // uncollided E·exp(-Σ_t·T) — it's constant per source photon
        // for shield_slab. Reconstruct: total_NEE = uncoll + per-coll
        // sum; the per-collision sum is recoverable from the trace.
        let per_coll_sum: f64 = result.nee_trace.iter().map(|t| t.2).sum();
        let uncollided_part = result.nee_tally - per_coll_sum;
        total_uncollided_nee += uncollided_part;
        // First-only and last-only tally variants per history.
        let tally_first = uncollided_part + first_contrib;
        let tally_last = uncollided_part + last_contrib;
        total_tally_first += tally_first;
        total_tally_first_sq += tally_first * tally_first;
        total_tally_last += tally_last;
        total_tally_last_sq += tally_last * tally_last;

        // Per-history NEE diagnostic dump.
        if args.nee_trace && args.next_event {
            println!(
                "  history {h}: collisions={}, analog_escape={:.4e} eV, NEE_tally={:.4e} eV",
                result.n_collisions, result.energy_escaped, result.nee_tally
            );
            // Cumulative per-collision contribution.
            let mut cum = 0.0_f64;
            for (i, &(z, e_in, contrib)) in result.nee_trace.iter().enumerate() {
                cum += contrib;
                println!(
                    "    coll {:>3}: z={:>6.2} cm  E_in={:.3e} eV  Δ={:.3e} eV  cum={:.3e} eV  cum/E_src={:.3e}",
                    i,
                    z,
                    e_in,
                    contrib,
                    cum,
                    cum / source_energy_ev
                );
            }
            // Compare with analog: the photon either physically
            // escaped (analog tally > 0) or didn't.
            let analog_norm = result.energy_escaped / source_energy_ev;
            let nee_norm = result.nee_tally / source_energy_ev;
            println!(
                "    summary: analog/E={:.3e}  NEE/E={:.3e}  ratio={:.2}",
                analog_norm,
                nee_norm,
                if analog_norm > 0.0 {
                    nee_norm / analog_norm
                } else {
                    f64::INFINITY
                }
            );
        }
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
    if args.next_event {
        // Decomposition (per source photon, normalised).
        let uncoll_per_src = total_uncollided_nee / n / source_energy_ev;
        let first_coll_per_src = total_first_coll_nee / n / source_energy_ev;
        let total_per_src = total_escaped_ev / n / source_energy_ev;
        let per_coll_only = total_per_src - uncoll_per_src;
        let subsequent_coll = per_coll_only - first_coll_per_src;

        // First-only and last-only estimator variants. Statistics
        // computed from the same transport run for clean comparison.
        let mean_first = total_tally_first / n;
        let var_first = (total_tally_first_sq / n) - mean_first.powi(2);
        let stderr_first = (var_first.max(0.0) / n).sqrt();
        let t_first = mean_first / source_energy_ev;
        let rel_err_first = stderr_first / mean_first.max(1e-30);
        let fom_first = 1.0 / (rel_err_first * rel_err_first * wall_seconds.max(1e-9));

        let mean_last = total_tally_last / n;
        let var_last = (total_tally_last_sq / n) - mean_last.powi(2);
        let stderr_last = (var_last.max(0.0) / n).sqrt();
        let t_last = mean_last / source_energy_ev;
        let rel_err_last = stderr_last / mean_last.max(1e-30);
        let fom_last = 1.0 / (rel_err_last * rel_err_last * wall_seconds.max(1e-9));

        println!();
        println!("  ── NEE decomposition (per source photon, normalised) ──");
        println!("  uncollided contribution = {:.4e}", uncoll_per_src);
        println!("  first-collision NEE     = {:.4e}", first_coll_per_src);
        println!("  subsequent collisions   = {:.4e}", subsequent_coll);
        println!("  full-sum (= total)      = {:.4e}", total_per_src);
        println!();
        println!("  ── Three-way estimator comparison ──");
        println!("  Estimator           |  T          | σ_rel    | FOM (/s)");
        println!(
            "  full per-coll sum    | {:.4e} | {:>5.2}%  | {:.3e}",
            total_per_src,
            100.0 * rel_err,
            fom
        );
        println!(
            "  first-only + uncoll  | {:.4e} | {:>5.2}%  | {:.3e}",
            t_first,
            100.0 * rel_err_first,
            fom_first
        );
        println!(
            "  last-only + uncoll   | {:.4e} | {:>5.2}%  | {:.3e}",
            t_last,
            100.0 * rel_err_last,
            fom_last
        );
        println!();
        println!("  Reading:");
        println!("   - full overcounts if Σ NEE_i fires at multiple correlated states.");
        println!("   - first-only is a strict lower-bound estimator (single-Compton-from-first).");
        println!("   - last-only captures the photon's final-flight contribution; should be");
        println!("     close to analog scattered if photons mostly escape via single Compton");
        println!("     from their last collision.");
    }
    println!();
    println!("  Use this number as the analog reference. CADIS-driven");
    println!("  weight windows on the same problem must beat this FOM");
    println!("  for the variance reduction to be worthwhile.");
    Ok(())
}
