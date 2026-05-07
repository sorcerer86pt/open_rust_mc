//! Random-ray FW-CADIS importance map for `shield_slab`.
//!
//! Replaces the "lite" detector-backward collision-density proxy
//! (`shield_slab --cadis-calibration`) with a real random-ray adjoint
//! solve. Produces JSON in the exact `CadisMap` schema that
//! `shield_slab --cadis-load` consumes — so the photon-side weight-
//! window pipeline is plug-compatible.
//!
//! Uses 1-group photon multigroup XS for water at ~1 MeV. For a
//! 1-group non-fissionable problem the adjoint operator equals the
//! forward operator (scatter matrix is 1×1 = its own transpose; no
//! χ ↔ νΣ_f swap because both are zero), so the FW-CADIS adjoint
//! solve can be done as a forward fixed-source solve with the source
//! localised at the detector face. That's the cheap exact reduction
//! we exploit here.
//!
//! For multigroup photon transport (broader spectra, Compton+Rayleigh+
//! photoelectric coupling) the same `RandomRaySolver` runs with
//! `AdjointFlag::Adjoint`; the cell-based / multigroup machinery is
//! already in place — only multigroup XS data plumbing is needed.

use std::path::PathBuf;

use clap::Parser;
use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId, Region};
use open_rust_mc::geometry::surface::BoundaryCondition;
use open_rust_mc::geometry::{Aabb, Geometry, Surface, Vec3};
use open_rust_mc::random_ray::{
    AdjointFlag, FsrMesh, MaterialMgxs, MgxsLibrary, RandomRaySolver, RaySolverConfig, SolverMode,
};

#[derive(Parser, Debug)]
#[command(
    name = "rr_cadis_slab",
    about = "Random-ray FW-CADIS importance map for shield_slab"
)]
struct Args {
    /// Slab thickness in cm.
    #[arg(long, default_value_t = 100.0)]
    thickness_cm: f64,

    /// XY half-extent (large = effective infinite slab).
    #[arg(long, default_value_t = 50.0)]
    half_xy_cm: f64,

    /// Number of z-bins in the importance map. Matches what
    /// `shield_slab --cadis-z-bins` was run with.
    #[arg(long, default_value_t = 20)]
    n_z_bins: usize,

    /// Total transport cross section for water at the source energy
    /// (1 MeV default ≈ 0.0707 cm⁻¹). Override for other energies /
    /// materials.
    #[arg(long, default_value_t = 0.0707)]
    sigma_t: f64,

    /// Absorption fraction (Σ_a / Σ_t). 1 MeV in water is dominated
    /// by Compton — about 1% photoelectric absorption + small pair.
    #[arg(long, default_value_t = 0.05)]
    absorption_frac: f64,

    /// Number of rays per batch.
    #[arg(long, default_value_t = 2000)]
    rays_per_batch: usize,

    /// Active ray length (cm). Should be a few mean free paths so
    /// rays sample most of the slab.
    #[arg(long, default_value_t = 200.0)]
    active_length: f64,

    /// Total batches.
    #[arg(long, default_value_t = 80)]
    batches: usize,

    /// Inactive batches.
    #[arg(long, default_value_t = 25)]
    inactive: usize,

    /// Use immortal rays (Tramm 2021 persistent-ray variant).
    #[arg(long, default_value_t = false)]
    immortal: bool,

    /// Output JSON path. Schema matches shield_slab's CadisMap:
    ///   {"thickness_cm": ..., "n_z_bins": ..., "counts": [...]}
    #[arg(long)]
    output: Option<PathBuf>,
}

fn build_slab_geometry(thickness_cm: f64, half_xy_cm: f64) -> Geometry {
    // Surface order matches shield_slab.rs:
    //   0: z_back (Reflective)
    //   1: z_front (Vacuum)
    //   2: x_min (Reflective)
    //   3: x_max (Reflective)
    //   4: y_min (Reflective)
    //   5: y_max (Reflective)
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
        cell::outside(0), // z >= 0
        cell::inside(1),  // z <= thickness
        cell::outside(2), // x >= -half
        cell::inside(3),  // x <= +half
        cell::outside(4), // y >= -half
        cell::inside(5),  // y <= +half
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

fn main() {
    let args = Args::parse();
    println!(
        "rr_cadis_slab — random-ray FW-CADIS for shield_slab\n\
         Thickness: {:.1} cm    n_z_bins: {}    σ_t: {:.4} cm⁻¹    σ_a/σ_t: {:.2}",
        args.thickness_cm, args.n_z_bins, args.sigma_t, args.absorption_frac
    );

    let geom = build_slab_geometry(args.thickness_cm, args.half_xy_cm);

    // 1-group photon MGXS for water-at-1-MeV.
    // Σ_t given by user; Σ_a = absorption_frac · Σ_t; Σ_s = Σ_t − Σ_a.
    let sigma_t = args.sigma_t;
    let sigma_a = sigma_t * args.absorption_frac;
    let sigma_s = sigma_t - sigma_a;
    let water = MaterialMgxs::nonfissionable(vec![sigma_t], vec![sigma_a], vec![sigma_s])
        .expect("water mgxs");
    let library = MgxsLibrary::new(vec![water]).expect("lib");

    // 1×1×n_z Cartesian FSR mesh aligned with shield_slab's CADIS bins.
    let aabb = Aabb::new(
        Vec3::new(-args.half_xy_cm, -args.half_xy_cm, 0.0),
        Vec3::new(args.half_xy_cm, args.half_xy_cm, args.thickness_cm),
    );
    let n = [1_usize, 1, args.n_z_bins];
    let mesh = FsrMesh::from_geometry(aabb, n, &geom);
    let n_fsrs = mesh.n_fsrs();
    println!(
        "FSR mesh: 1×1×{} = {} voxels, voxel V = {:.3} cm³",
        args.n_z_bins,
        n_fsrs,
        mesh.fsr_volume(0)
    );

    // Adjoint problem reduction: 1-group non-fissionable → adjoint =
    // forward fixed-source. Source localised at the detector face
    // (last z-voxel, ix=iy=0, iz=n_z-1) with unit strength.
    let mut q_ext = vec![0.0_f64; n_fsrs];
    let detector_idx = FsrMesh::cart_flat_index(n, 0, 0, args.n_z_bins - 1);
    q_ext[detector_idx] = 1.0;

    let cfg = RaySolverConfig {
        rays_per_batch: args.rays_per_batch,
        dead_zone: if args.immortal { 0.0 } else { 5.0 },
        active_length: args.active_length,
        batches: args.batches,
        inactive: args.inactive,
        mode: SolverMode::FixedSource,
        // 1-group non-fissionable: forward = adjoint, so either flag
        // works. Use Forward to make the reduction explicit.
        adjoint: AdjointFlag::Forward,
        seed: 7,
        immortal: args.immortal,
    };

    let solver = RandomRaySolver::new(&geom, mesh, library).with_external_source(q_ext);
    println!(
        "Running {} batches × {} rays ({})...",
        cfg.batches,
        cfg.rays_per_batch,
        if cfg.immortal { "immortal" } else { "mortal" }
    );
    let t0 = std::time::Instant::now();
    let result = solver.run(&cfg);
    let dt = t0.elapsed().as_secs_f64();
    println!("Done in {:.2}s.", dt);

    // Reduce ψ*(x,y,z) → ψ*(z): the mesh is 1×1×n_z so this is a copy.
    let phi = result.flux_group(0);
    assert_eq!(phi.len(), args.n_z_bins);

    // Find the peak so we can normalise.
    let phi_max = phi.iter().cloned().fold(0.0_f64, f64::max);
    if phi_max <= 0.0 {
        eprintln!("ERROR: importance map collapsed to zero. Check σ_t, active_length, batches.");
        std::process::exit(1);
    }

    // Convert to integer counts proportional to ψ*. shield_slab treats
    // these as relative weights when calling WeightWindow::from_flux,
    // so absolute scale only matters for u64 overflow.
    let scale = 1.0e6_f64 / phi_max;
    let counts: Vec<u64> = phi
        .iter()
        .map(|&v| (v * scale).max(0.0).round() as u64)
        .collect();

    // Print a coarse importance profile for sanity.
    println!("\nz-bin           ψ̂*(z) (norm.)   w_target ∝ 1/ψ̂*");
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

    // Build the CadisMap JSON in shield_slab's exact schema.
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
        std::fs::write(path, &json).expect("failed to write output JSON");
        println!(
            "\nSaved random-ray CADIS map → {} ({} bytes)",
            path.display(),
            json.len()
        );
        println!(
            "\nNext step: shield_slab --cadis-load {} ...    \
             # uses random-ray ψ̂* instead of the collision-density proxy.",
            path.display()
        );
    } else {
        println!("\n--- JSON (no --output specified) ---\n{}", json);
    }
}
