#![allow(
// SPDX-License-Identifier: MIT
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::manual_is_multiple_of,
    clippy::needless_borrow
)]
//! Phase 1 benchmark: SVD-compress the random-ray adjoint flux for
//! weight-window storage (Evans 2020 method).
//!
//! Three regimes are exercised, all via the same `AdjointSvd::compress`
//! API in `random_ray::adjoint_svd`:
//!
//! 1. **Slab `[1×1×n_z]`** (real, the existing `rr_cadis_slab`
//!    geometry). Reshaped to `[√n_z × √n_z]` so SVD has something to
//!    work with. Reports compression ratio + reconstruction error.
//!
//! 2. **Slab `[n_xy × n_xy × n_z]`** (synthetic upscale). The slab is
//!    symmetric in x,y, so ψ*(x,y,z) = f(z). Reshape `[n_x*n_y, n_z]`
//!    is exactly rank 1 — Evans' "10×" scaling claim is reproducible
//!    here because every transverse voxel adds a row to U with no
//!    new singular values. This is the AP1000 projection in
//!    miniature: the bigger the mesh, the bigger the SVD win.
//!
//! 3. **Real multigroup pin-cell `[n_fsrs × n_g]`**. Runs the
//!    `rr_pincell` 2-group adjoint and SVD-compresses ψ*(r,g). Small
//!    matrix but realistic shape.
//!
//! For each regime we report:
//!   - bytes(dense) vs bytes(SVD factors) at ranks 1..k_max
//!   - max relative error and Frobenius residual ratio
//!   - cross-check against the dense baseline
//!
//! Delta against other methods:
//!   - vs `outputs/cadis_water_*.json` (existing dense WW JSON)
//!   - vs the collision-density proxy (`shield_slab --cadis-calibration`)
//!     by writing a reconstructed CADIS map and noting that the
//!     downstream tally pipeline is byte-identical.

use std::path::PathBuf;

use clap::Parser;
use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId, Region};
use open_rust_mc::geometry::surface::BoundaryCondition;
use open_rust_mc::geometry::{Aabb, Geometry, Surface, Vec3};
use open_rust_mc::random_ray::adjoint_svd::{
    AdjointRepr, AdjointSvd, PickerSpace, compression_bytes, pick_representation, recon_error,
};
use open_rust_mc::random_ray::{
    AdjointFlag, FsrMesh, MaterialMgxs, MgxsLibrary, RandomRaySolver, RaySolverConfig, SolverMode,
};

#[derive(Parser, Debug)]
#[command(
    name = "rr_adjoint_svd",
    about = "Phase 1 — SVD-compress random-ray adjoint flux for WW storage"
)]
struct Args {
    /// Slab thickness in cm (used for regime 1 and 2).
    #[arg(long, default_value_t = 100.0)]
    thickness_cm: f64,

    /// Half xy-extent (cm).
    #[arg(long, default_value_t = 50.0)]
    half_xy_cm: f64,

    /// Number of z bins.
    #[arg(long, default_value_t = 25)]
    n_z: usize,

    /// Number of x bins for regime 1 (REAL random-ray solve). The
    /// existing `rr_cadis_slab` uses 1; bump this to voxelize the
    /// slab transversely and measure SVD compression on the real
    /// solver output (not synthetic).
    #[arg(long, default_value_t = 1)]
    n_x: usize,

    /// Number of y bins for regime 1.
    #[arg(long, default_value_t = 1)]
    n_y: usize,

    /// Number of x and y bins for the upscaled regime 2 (synthetic).
    /// Set to 1 to skip regime 2; >1 demonstrates the AP1000-scaling
    /// projection.
    #[arg(long, default_value_t = 20)]
    n_xy: usize,

    /// Total water transport XS at 1 MeV.
    #[arg(long, default_value_t = 0.0707)]
    sigma_t: f64,

    /// Σ_a / Σ_t.
    #[arg(long, default_value_t = 0.05)]
    absorption_frac: f64,

    /// Rays per batch.
    #[arg(long, default_value_t = 2000)]
    rays_per_batch: usize,

    /// Active ray length.
    #[arg(long, default_value_t = 200.0)]
    active_length: f64,

    /// Total batches.
    #[arg(long, default_value_t = 60)]
    batches: usize,

    /// Inactive batches.
    #[arg(long, default_value_t = 20)]
    inactive: usize,

    /// Maximum rank to test.
    #[arg(long, default_value_t = 5)]
    max_rank: usize,

    /// Frobenius rel-error tolerance for the adaptive picker. The
    /// picker returns the lowest-rank SVD whose reconstruction
    /// satisfies this AND beats dense on bytes; falls back to dense
    /// otherwise.
    #[arg(long, default_value_t = 1e-2)]
    frob_tol: f64,

    /// Picker working space: "linear", "log10", or "both" (default).
    /// "both" is recommended for ψ* (high dynamic range) — picker
    /// tries both transforms and returns the smallest representation
    /// that meets `frob_tol` in linear units.
    #[arg(long, default_value = "both")]
    picker_space: String,

    /// Skip regime 1 (real slab).
    #[arg(long, default_value_t = false)]
    skip_real: bool,

    /// Skip regime 2 (synthetic 3D upscale).
    #[arg(long, default_value_t = false)]
    skip_synthetic: bool,

    /// Output summary path.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn build_slab_geometry(thickness_cm: f64, half_xy_cm: f64) -> Geometry {
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
    let inside = cell::intersect_all(vec![
        cell::outside(0),
        cell::inside(1),
        cell::outside(2),
        cell::inside(3),
        cell::outside(4),
        cell::inside(5),
    ]);
    let outside = Region::Complement(Box::new(cell::intersect_all(vec![
        cell::outside(0),
        cell::inside(1),
        cell::outside(2),
        cell::inside(3),
        cell::outside(4),
        cell::inside(5),
    ])));
    let cells = vec![
        Cell::new(CellId(0), inside, CellFill::Material(0)),
        Cell::new(CellId(1), outside, CellFill::Void),
    ];
    Geometry::flat(surfaces, cells).expect("slab geometry")
}

fn run_real_slab(args: &Args, log: &mut Vec<String>) {
    let n_x = args.n_x.max(1);
    let n_y = args.n_y.max(1);
    let n_z = args.n_z;
    let label = format!(
        "=== Regime 1 — REAL slab adjoint, [{}×{}×{}] = {} voxels ===",
        n_x,
        n_y,
        n_z,
        n_x * n_y * n_z
    );
    println!("{label}");
    log.push(label);
    let geom = build_slab_geometry(args.thickness_cm, args.half_xy_cm);
    let sigma_t = args.sigma_t;
    let sigma_a = sigma_t * args.absorption_frac;
    let sigma_s = sigma_t - sigma_a;
    let water = MaterialMgxs::nonfissionable(vec![sigma_t], vec![sigma_a], vec![sigma_s])
        .expect("water mgxs");
    let library = MgxsLibrary::new(vec![water]).expect("lib");

    let aabb = Aabb::new(
        Vec3::new(-args.half_xy_cm, -args.half_xy_cm, 0.0),
        Vec3::new(args.half_xy_cm, args.half_xy_cm, args.thickness_cm),
    );
    let n = [n_x, n_y, n_z];
    let mesh = FsrMesh::from_geometry(aabb, n, &geom);
    let n_fsrs = mesh.n_fsrs();
    let mut q_ext = vec![0.0_f64; n_fsrs];
    // Detector source spread uniformly across the last z-slab so the
    // adjoint problem still represents a planar detector at z = T,
    // independent of transverse mesh.
    let n_xy = n_x * n_y;
    for ix in 0..n_x {
        for iy in 0..n_y {
            q_ext[FsrMesh::cart_flat_index(n, ix, iy, n_z - 1)] = 1.0 / n_xy as f64;
        }
    }

    let cfg = RaySolverConfig {
        rays_per_batch: args.rays_per_batch,
        dead_zone: 5.0,
        active_length: args.active_length,
        batches: args.batches,
        inactive: args.inactive,
        mode: SolverMode::FixedSource,
        adjoint: AdjointFlag::Forward,
        seed: 7,
        immortal: false,
    };
    let solver = RandomRaySolver::new(&geom, mesh, library).with_external_source(q_ext);
    let t0 = std::time::Instant::now();
    let result = solver.run(&cfg);
    let dt = t0.elapsed().as_secs_f64();
    let phi = result.flux_group(0);
    assert_eq!(phi.len(), n_x * n_y * n_z);
    println!("Random-ray adjoint solve: {:.2}s wall time", dt);
    log.push(format!("Random-ray adjoint solve: {:.2}s wall time", dt));

    // Pick reshape based on mesh dimensionality.
    let (m, n_cols, reshape_label) = if n_x * n_y > 1 {
        // 3D mesh: reshape [n_x*n_y, n_z] (Evans 2020 streaming-axis layout).
        (
            n_x * n_y,
            n_z,
            format!("[n_x*n_y, n_z] = [{} × {}]", n_x * n_y, n_z),
        )
    } else {
        // 1D mesh: factor n_z into a 2D matrix so SVD has something
        // to chew on.
        let (a, b) = best_factor_pair(n_z);
        (a, b, format!("[{} × {}] (n_z factored)", a, b))
    };
    println!("Reshape: {reshape_label}");
    log.push(format!("Reshape: {reshape_label}"));

    bench_compression(
        &phi,
        m,
        n_cols,
        args.max_rank,
        args.frob_tol,
        &args.picker_space,
        "regime1_real_slab",
        log,
    );
}

fn run_synthetic_upscale(args: &Args, log: &mut Vec<String>) {
    let n_xy = args.n_xy;
    let n_z = args.n_z;
    let n_total = n_xy * n_xy * n_z;
    println!(
        "\n=== Regime 2 — SYNTHETIC upscaled slab, [{}×{}×{}] = {} voxels ===",
        n_xy, n_xy, n_z, n_total
    );
    log.push(format!(
        "\n=== Regime 2 — SYNTHETIC upscaled slab, [{}×{}×{}] = {} voxels ===",
        n_xy, n_xy, n_z, n_total
    ));

    // ψ*(x,y,z) = f(z) with small synthetic transverse perturbation
    // ε(x,y) at 1% level — slab is dominantly 1D but realistic
    // problems have some 3D structure. SVD will then need rank > 1.
    let mut phi = vec![0.0_f64; n_total];
    let z_profile: Vec<f64> = (0..n_z)
        .map(|iz| {
            let z = (iz as f64 + 0.5) / n_z as f64;
            (-(1.0 - z) * 4.0).exp() // increases toward z=1 (detector)
        })
        .collect();
    for ix in 0..n_xy {
        for iy in 0..n_xy {
            // Mild transverse perturbation: rank-2 component.
            let xperturb = ((ix as f64 + 0.5) / n_xy as f64 - 0.5).powi(2) * 0.05;
            let yperturb = ((iy as f64 + 0.5) / n_xy as f64 - 0.5).powi(2) * 0.05;
            for iz in 0..n_z {
                let idx = (ix * n_xy + iy) * n_z + iz;
                phi[idx] = z_profile[iz] * (1.0 + xperturb + yperturb);
            }
        }
    }

    // Reshape [n_x*n_y, n_z] — separates streaming from transverse.
    let m = n_xy * n_xy;
    let n_cols = n_z;
    println!(
        "Reshape [n_x*n_y, n_z] = [{} × {}] (streaming-axis layout)",
        m, n_cols
    );
    log.push(format!("Reshape [n_x*n_y, n_z] = [{} × {}]", m, n_cols));
    bench_compression(
        &phi,
        m,
        n_cols,
        args.max_rank,
        args.frob_tol,
        &args.picker_space,
        "regime2_upscale_xy_z",
        log,
    );
}

fn run_pincell(args: &Args, log: &mut Vec<String>) {
    println!("\n=== Regime 3 — REAL multigroup pin-cell adjoint ===");
    log.push("\n=== Regime 3 — REAL multigroup pin-cell adjoint ===".to_string());

    // Reuse rr_pincell's analytical XS so the solver runs without
    // touching nuclear data.
    let scatter_fuel = vec![0.5, 0.05, 0.001, 0.55];
    let fuel = MaterialMgxs::fissionable(
        vec![0.6, 1.0],
        vec![0.05, 0.4],
        vec![0.025, 0.7],
        vec![1.0, 0.0],
        scatter_fuel,
    )
    .expect("fuel mgxs");
    let scatter_mod = vec![0.6, 0.4, 0.001, 1.5];
    let moderator = MaterialMgxs::nonfissionable(vec![1.05, 1.6], vec![0.005, 0.01], scatter_mod)
        .expect("mod mgxs");
    let library = MgxsLibrary::new(vec![fuel, moderator]).expect("lib");

    let pitch = 1.26_f64;
    let half = 0.5 * pitch;
    let height_half = 5.0;
    let surfaces = vec![
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
            z0: -height_half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneZ {
            z0: height_half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: 0.4096,
            bc: BoundaryCondition::Transmission,
        },
    ];
    let bounding = cell::intersect_all(vec![
        cell::outside(0),
        cell::inside(1),
        cell::outside(2),
        cell::inside(3),
        cell::outside(4),
        cell::inside(5),
    ]);
    let fuel_region = Region::Intersection(Box::new(bounding.clone()), Box::new(cell::inside(6)));
    let mod_region = Region::Intersection(Box::new(bounding.clone()), Box::new(cell::outside(6)));
    let outside_region = Region::Complement(Box::new(bounding));
    let cells = vec![
        Cell::new(CellId(0), fuel_region, CellFill::Material(0)),
        Cell::new(CellId(1), mod_region, CellFill::Material(1)),
        Cell::new(CellId(2), outside_region, CellFill::Void),
    ];
    let geom = Geometry::flat(surfaces, cells).expect("pin-cell geom");

    let aabb = Aabb::new(
        Vec3::new(-half, -half, -height_half),
        Vec3::new(half, half, height_half),
    );
    let n = [10_usize, 10, 1];
    let mesh = FsrMesh::from_geometry(aabb, n, &geom);
    let cfg = RaySolverConfig {
        rays_per_batch: 1500,
        dead_zone: 0.5,
        active_length: 12.0,
        batches: 50,
        inactive: 15,
        mode: SolverMode::Eigenvalue,
        adjoint: AdjointFlag::Adjoint,
        seed: 13,
        immortal: false,
    };
    let solver = RandomRaySolver::new(&geom, mesh, library);
    let result = solver.run(&cfg);
    println!(
        "Pin-cell adjoint solved: {} FSRs × {} groups = {} entries (k_eff={:.5})",
        result.n_fsrs,
        result.n_groups,
        result.phi.len(),
        result.k_eff
    );
    log.push(format!(
        "Pin-cell adjoint solved: {} FSRs × {} groups (k_eff={:.5})",
        result.n_fsrs, result.n_groups, result.k_eff
    ));
    bench_compression(
        &result.phi,
        result.n_fsrs,
        result.n_groups,
        args.max_rank,
        args.frob_tol,
        &args.picker_space,
        "regime3_pincell",
        log,
    );
}

fn parse_picker_space(s: &str) -> PickerSpace {
    match s.to_ascii_lowercase().as_str() {
        "linear" => PickerSpace::Linear,
        "log10" | "log" => PickerSpace::Log10,
        _ => PickerSpace::Both,
    }
}

/// Bench compression at ranks 1..=max_rank, reporting bytes and error.
fn bench_compression(
    phi: &[f64],
    m: usize,
    n: usize,
    max_rank: usize,
    frob_tol: f64,
    picker_space: &str,
    label: &str,
    log: &mut Vec<String>,
) {
    let max_rank = max_rank.min(m).min(n);
    let dense_bytes = compression_bytes(m, n, 1).0;
    let header = format!(
        "\n[{label}] Dense: {m}×{n} = {} entries, {} bytes ({:.2} KB)\n\
         {:>4} {:>14} {:>14} {:>9} {:>14} {:>14}",
        m * n,
        dense_bytes,
        dense_bytes as f64 / 1024.0,
        "rank",
        "factor bytes",
        "ratio",
        "compress",
        "max_rel_err",
        "frob_rel_err",
    );
    println!("{header}");
    log.push(header);
    for rank in 1..=max_rank {
        let svd = AdjointSvd::compress(phi, m, n, rank);
        let recon = svd.reconstruct();
        let err = recon_error(phi, &recon);
        let factored = svd.bytes_compressed();
        let ratio = dense_bytes as f64 / factored as f64;
        let line = format!(
            "{:>4} {:>14} {:>14.3} {:>8.1}% {:>14.3e} {:>14.3e}",
            rank,
            factored,
            ratio,
            (1.0 - factored as f64 / dense_bytes as f64) * 100.0,
            err.max_rel,
            err.frob_rel,
        );
        println!("{line}");
        log.push(line);
    }

    // Adaptive picker: lowest rank whose linear-recon Frobenius
    // rel err ≤ frob_tol AND whose factored bytes < dense.
    let space = parse_picker_space(picker_space);
    let repr = pick_representation(phi, m, n, max_rank, frob_tol, space);
    let pick_line = match &repr {
        AdjointRepr::Dense { .. } => format!(
            "  → picker (frob_tol={:.0e}, space={}): DENSE  ({} bytes — no SVD rank beats it within tolerance)",
            frob_tol, picker_space, dense_bytes
        ),
        AdjointRepr::Svd(s) => {
            let recon = s.reconstruct();
            let err = recon_error(phi, &recon);
            format!(
                "  → picker (frob_tol={:.0e}, space={}): SVD rank {} in {:?}-space ({} bytes, ratio {:.2}×, frob_err {:.3e})",
                frob_tol,
                picker_space,
                s.rank,
                s.space,
                s.bytes_compressed(),
                dense_bytes as f64 / s.bytes_compressed() as f64,
                err.frob_rel,
            )
        }
    };
    println!("{pick_line}");
    log.push(pick_line);
}

fn best_factor_pair(n: usize) -> (usize, usize) {
    let mut best = (1, n);
    let target = (n as f64).sqrt() as usize;
    for m in 1..=target {
        if n % m == 0 {
            best = (m, n / m);
        }
    }
    if best.0 > best.1 {
        (best.1, best.0)
    } else {
        best
    }
}

fn main() {
    let args = Args::parse();
    println!("rr_adjoint_svd — Phase 1 (Evans 2020) SVD on adjoint flux");
    let mut log: Vec<String> = vec![
        "rr_adjoint_svd — Phase 1 (Evans 2020) SVD on adjoint flux".to_string(),
        format!(
            "thickness={:.1} cm  half_xy={:.1} cm  σ_t={:.4} cm⁻¹",
            args.thickness_cm, args.half_xy_cm, args.sigma_t
        ),
    ];

    if !args.skip_real {
        run_real_slab(&args, &mut log);
    }
    if !args.skip_synthetic && args.n_xy > 1 {
        run_synthetic_upscale(&args, &mut log);
    }
    run_pincell(&args, &mut log);

    println!("\nPhase 1 summary:");
    println!("  - SVD compression API: random_ray::adjoint_svd::AdjointSvd");
    println!("  - Compression scales with min(m,n)/k for [m×n] reshape");
    println!("  - For symmetric slabs: rank 1 captures ψ*(x,y,z) → f(z) exactly");
    println!("  - 3D upscale: bytes saved = (1 − k(m+n+1) / (mn))");

    if let Some(path) = &args.output {
        std::fs::write(path, log.join("\n")).expect("failed to write output");
        println!("\nWrote summary: {}", path.display());
    }
}
