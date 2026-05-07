//! Mesh-size sweep for the adjoint-flux SVD compression picker.
//!
//! For each `(n_x, n_y, n_z)` configuration, runs a real random-ray
//! adjoint solve on the slab geometry, hands the resulting ψ*(r,g)
//! to `pick_representation`, and records:
//!
//!   - `n_voxels = n_x · n_y · n_z`
//!   - `dense_bytes`
//!   - whether the picker chose dense or SVD
//!   - if SVD: rank, space (linear/log10), factored bytes, ratio,
//!     frob-rel error
//!   - wall time of the random-ray solve
//!
//! Output is CSV to stdout (and optionally a file). The companion
//! `scripts/plot_adjoint_sweep.py` reads the CSV and emits the plots
//! used in `paper/sections/adjoint_compression.tex`.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId, Region};
use open_rust_mc::geometry::surface::BoundaryCondition;
use open_rust_mc::geometry::{Aabb, Geometry, Surface, Vec3};
use open_rust_mc::random_ray::adjoint_svd::{
    AdjointRepr, PickerSpace, pick_representation, recon_error,
};
use open_rust_mc::random_ray::{
    AdjointFlag, FsrMesh, MaterialMgxs, MgxsLibrary, RandomRaySolver, RaySolverConfig, SolverMode,
};

#[derive(Parser, Debug)]
#[command(name = "rr_adjoint_sweep", about = "Mesh-size sweep for the adjoint-SVD picker")]
struct Args {
    /// Slab thickness (cm).
    #[arg(long, default_value_t = 100.0)]
    thickness_cm: f64,

    /// Slab xy half-extent (cm).
    #[arg(long, default_value_t = 50.0)]
    half_xy_cm: f64,

    /// Total water transport XS at 1 MeV.
    #[arg(long, default_value_t = 0.0707)]
    sigma_t: f64,

    /// Σ_a / Σ_t.
    #[arg(long, default_value_t = 0.05)]
    absorption_frac: f64,

    /// Mesh configurations as "nx,ny,nz" triples.
    #[arg(long, value_delimiter = ';', num_args = 1..,
          default_values_t = vec![
              "1,1,25".to_string(),
              "5,5,25".to_string(),
              "10,10,25".to_string(),
              "10,10,50".to_string(),
              "20,20,25".to_string(),
              "20,20,50".to_string(),
              "30,30,50".to_string(),
              "40,40,50".to_string(),
          ])]
    meshes: Vec<String>,

    /// Frobenius rel-err tolerance for the picker.
    #[arg(long, default_value_t = 0.025)]
    frob_tol: f64,

    /// Maximum rank to consider in the picker.
    #[arg(long, default_value_t = 10)]
    max_rank: usize,

    /// Picker space: "linear", "log10", "both".
    #[arg(long, default_value = "both")]
    picker_space: String,

    /// Random-ray batches.
    #[arg(long, default_value_t = 200)]
    batches: usize,

    /// Random-ray inactive batches.
    #[arg(long, default_value_t = 50)]
    inactive: usize,

    /// Rays per batch.
    #[arg(long, default_value_t = 8000)]
    rays_per_batch: usize,

    /// Active ray length (cm).
    #[arg(long, default_value_t = 200.0)]
    active_length: f64,

    /// CSV output file.
    #[arg(long)]
    csv: Option<PathBuf>,
}

fn build_slab_geometry(thickness_cm: f64, half_xy_cm: f64) -> Geometry {
    let surfaces = vec![
        Surface::PlaneZ { z0: 0.0, bc: BoundaryCondition::Reflective },
        Surface::PlaneZ { z0: thickness_cm, bc: BoundaryCondition::Vacuum },
        Surface::PlaneX { x0: -half_xy_cm, bc: BoundaryCondition::Reflective },
        Surface::PlaneX { x0: half_xy_cm, bc: BoundaryCondition::Reflective },
        Surface::PlaneY { y0: -half_xy_cm, bc: BoundaryCondition::Reflective },
        Surface::PlaneY { y0: half_xy_cm, bc: BoundaryCondition::Reflective },
    ];
    let inside = cell::intersect_all(vec![
        cell::outside(0), cell::inside(1),
        cell::outside(2), cell::inside(3),
        cell::outside(4), cell::inside(5),
    ]);
    let outside = Region::Complement(Box::new(cell::intersect_all(vec![
        cell::outside(0), cell::inside(1),
        cell::outside(2), cell::inside(3),
        cell::outside(4), cell::inside(5),
    ])));
    let cells = vec![
        Cell::new(CellId(0), inside, CellFill::Material(0)),
        Cell::new(CellId(1), outside, CellFill::Void),
    ];
    Geometry::flat(surfaces, cells).expect("slab geometry")
}

fn parse_picker_space(s: &str) -> PickerSpace {
    match s.to_ascii_lowercase().as_str() {
        "linear" => PickerSpace::Linear,
        "log10" | "log" => PickerSpace::Log10,
        _ => PickerSpace::Both,
    }
}

fn parse_mesh(s: &str) -> [usize; 3] {
    let parts: Vec<usize> = s
        .split(',')
        .map(|p| p.trim().parse::<usize>().expect("mesh part"))
        .collect();
    assert_eq!(parts.len(), 3, "mesh string must be nx,ny,nz");
    [parts[0], parts[1], parts[2]]
}

/// Reject meshes whose smallest voxel edge is finer than the
/// physics resolution of the underlying random-ray problem.
///
/// For the slab benchmark (water at 1 MeV, mfp ≈ 14.14 cm) the
/// floor is mfp / 8 ≈ 1.77 cm. Below that the geometric subdivision
/// is finer than the physics supports — there is no signal at
/// sub-mfp scale, only random-ray noise. Calling the picker on
/// such data produces a meaningless result that wastes CPU and
/// pollutes the sweep CSV. Refusing at the source makes that
/// kind of mistake unrepresentable.
fn assert_mesh_is_physical(args: &Args, n: [usize; 3]) {
    let slab_x = 2.0 * args.half_xy_cm;
    let slab_y = 2.0 * args.half_xy_cm;
    let slab_z = args.thickness_cm;
    let edge_x = slab_x / n[0].max(1) as f64;
    let edge_y = slab_y / n[1].max(1) as f64;
    let edge_z = slab_z / n[2].max(1) as f64;
    let min_edge = edge_x.min(edge_y).min(edge_z);
    // mfp_water_1MeV ≈ 1 / sigma_t = 1 / 0.0707 ≈ 14.14 cm.
    let mfp = 1.0 / args.sigma_t;
    let floor = mfp / 8.0;
    assert!(
        min_edge >= floor,
        "mesh {:?} has voxel edge {:.3} cm < mfp/8 = {:.3} cm. \
         Sub-mfp voxels carry no signal — refusing to run a \
         meaningless sweep entry. (mfp = 1/sigma_t = {:.2} cm)",
        n, min_edge, floor, mfp
    );
}

fn run_one(args: &Args, n: [usize; 3]) -> String {
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
    let mesh = FsrMesh::from_geometry(aabb, n, &geom);
    let n_fsrs = mesh.n_fsrs();
    let mut q_ext = vec![0.0_f64; n_fsrs];
    let n_xy = n[0] * n[1];
    for ix in 0..n[0] {
        for iy in 0..n[1] {
            q_ext[FsrMesh::cart_flat_index(n, ix, iy, n[2] - 1)] = 1.0 / n_xy as f64;
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
    let t0 = Instant::now();
    let result = solver.run(&cfg);
    let solve_secs = t0.elapsed().as_secs_f64();
    let phi = result.flux_group(0);

    // Reshape decision: if 3D mesh use [n_x*n_y, n_z], else factor n_z.
    let (m, n_cols, reshape) = if n[0] * n[1] > 1 {
        (n[0] * n[1], n[2], format!("xy_z {}x{}", n[0] * n[1], n[2]))
    } else {
        let (a, b) = best_factor_pair(n[2]);
        (a, b, format!("nz_factored {}x{}", a, b))
    };
    let dense_bytes = m * n_cols * std::mem::size_of::<f64>();
    let space = parse_picker_space(&args.picker_space);
    let repr = pick_representation(&phi, m, n_cols, args.max_rank, args.frob_tol, space);
    let (kind, rank, repr_space, comp_bytes, ratio, frob_err) = match &repr {
        AdjointRepr::Dense { .. } => (
            "dense", 0, "—".to_string(), dense_bytes, 1.0_f64, 0.0_f64,
        ),
        AdjointRepr::Svd(s) => {
            let recon = s.reconstruct();
            let err = recon_error(&phi, &recon);
            (
                "svd",
                s.rank,
                format!("{:?}", s.space),
                s.bytes_compressed(),
                dense_bytes as f64 / s.bytes_compressed() as f64,
                err.frob_rel,
            )
        }
    };
    format!(
        "{},{},{},{},{},{},{},{},{:.4},{},{},{:.4},{:.6e},{:.3}",
        n[0], n[1], n[2],
        n[0] * n[1] * n[2],
        reshape,
        m, n_cols,
        dense_bytes,
        args.frob_tol,
        kind,
        rank,
        repr_space,
        frob_err,
        solve_secs,
    ) + &format!(",{},{:.4}", comp_bytes, ratio)
}

fn best_factor_pair(n: usize) -> (usize, usize) {
    let mut best = (1, n);
    let target = (n as f64).sqrt() as usize;
    for m in 1..=target {
        if n % m == 0 {
            best = (m, n / m);
        }
    }
    if best.0 > best.1 { (best.1, best.0) } else { best }
}

fn main() {
    let args = Args::parse();
    let header = "n_x,n_y,n_z,n_voxels,reshape,m,n_cols,dense_bytes,frob_tol,picker_kind,rank,repr_space,frob_err,solve_secs,comp_bytes,ratio";
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    {
        let mut out = stdout.lock();
        let _ = writeln!(out, "{header}");
        let _ = out.flush();
    }
    let mut lines = vec![header.to_string()];
    let total = args.meshes.len();
    for (idx, spec) in args.meshes.iter().enumerate() {
        let n = parse_mesh(spec);
        assert_mesh_is_physical(&args, n);
        let n_voxels = n[0] * n[1] * n[2];
        {
            let mut err = stderr.lock();
            let _ = writeln!(
                err,
                "# [{}/{}] running mesh {:?} = {} voxels ...",
                idx + 1, total, n, n_voxels,
            );
            let _ = err.flush();
        }
        let line = run_one(&args, n);
        {
            let mut out = stdout.lock();
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }
        lines.push(line.clone());
        // Persist each completed row immediately so a later crash
        // doesn't lose the data we already have.
        if let Some(p) = &args.csv {
            let _ = std::fs::write(p, lines.join("\n"));
        }
    }
    if let Some(p) = &args.csv {
        std::fs::write(p, lines.join("\n")).expect("csv write");
        let mut err = stderr.lock();
        let _ = writeln!(err, "# wrote {}", p.display());
        let _ = err.flush();
    }
}
