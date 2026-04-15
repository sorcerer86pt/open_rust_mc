//! PWR pin cell eigenvalue benchmark — multi-material, multi-nuclide.
//!
//! Standard PWR fuel pin: UO2 fuel + Zircaloy clad + light water moderator.
//! This is the real test for SVD compression — 8 nuclides across 3 materials
//! with heterogeneous geometry (cylinder + reflective box).
//!
//! Usage:
//!   pwr_pincell <data_dir> [--rank K] [--batches N] [--particles N] [--mode MODE]

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3};
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::{self, BatchResult, SimConfig, XsProvider};
use open_rust_mc::transport::xs_provider;

#[derive(clap::ValueEnum, Clone, Debug)]
enum XsMode { Svd, Table, Both }

#[derive(Parser)]
#[command(name = "pwr_pincell", about = "PWR pin cell benchmark (multi-material, multi-nuclide)")]
struct Args {
    /// Directory containing nuclide HDF5 files.
    data_dir: PathBuf,

    #[arg(short, long, default_value_t = 5)]
    rank: usize,

    #[arg(short, long, default_value_t = 100)]
    batches: u32,

    #[arg(short, long, default_value_t = 20)]
    inactive: u32,

    #[arg(short, long, default_value_t = 10000)]
    particles: u32,

    #[arg(short, long, default_value_t = 1)]
    temp_idx: usize,

    #[arg(short, long, value_enum, default_value_t = XsMode::Svd)]
    mode: XsMode,
}

/// Nuclide specs: (filename, AWR, fallback nu-bar).
/// Index in this array = xs_kernel_idx used by materials.
const NUCLIDE_SPECS: &[(&str, f64, f64)] = &[
    ("U235.h5", 233.025, 2.43),  // 0
    ("U238.h5", 236.006, 2.49),  // 1
    ("O16.h5",  15.858,  0.0),   // 2
    ("H1.h5",    0.999,  0.0),   // 3
    ("Zr90.h5", 89.132,  0.0),   // 4
    ("Zr91.h5", 90.130,  0.0),   // 5
    ("Zr92.h5", 91.126,  0.0),   // 6
    ("Zr94.h5", 93.120,  0.0),   // 7
];

struct RunResult {
    k_mean: f64,
    k_std: f64,
    load_ms: f64,
    sim_ms: f64,
    xs_memory_bytes: usize,
}

fn run_simulation<XS: XsProvider>(
    config: &SimConfig,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
    inactive: u32,
) -> (Vec<BatchResult>, f64, f64, f64) {
    let t1 = Instant::now();
    let (results, _) = simulate::run_eigenvalue(config, surfaces, cells, materials, xs_provider);
    let sim_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let active: Vec<f64> = results.iter()
        .filter(|r| r.batch > inactive)
        .map(|r| r.k_eff)
        .collect();
    let n = active.len() as f64;
    let k_mean = active.iter().sum::<f64>() / n;
    let k_var = active.iter().map(|&k| (k - k_mean).powi(2)).sum::<f64>() / (n * (n - 1.0));
    let k_std = k_var.sqrt();

    (results, k_mean, k_std, sim_ms)
}

fn setup_geometry() -> (Vec<Surface>, Vec<Cell>) {
    // Standard PWR pin cell dimensions
    let fuel_or = 0.4096;  // cm, fuel outer radius
    let clad_ir = 0.4180;  // cm, clad inner radius
    let clad_or = 0.4750;  // cm, clad outer radius
    let pitch = 1.2600;    // cm, pin pitch
    let half = pitch / 2.0;

    let z_half = half; // Use same extent in Z for reflective bounding

    let surfaces = vec![
        // 0: fuel outer cylinder
        Surface::CylinderZ {
            center_x: 0.0, center_y: 0.0, radius: fuel_or,
            bc: BoundaryCondition::Transmission,
        },
        // 1: clad inner cylinder
        Surface::CylinderZ {
            center_x: 0.0, center_y: 0.0, radius: clad_ir,
            bc: BoundaryCondition::Transmission,
        },
        // 2: clad outer cylinder
        Surface::CylinderZ {
            center_x: 0.0, center_y: 0.0, radius: clad_or,
            bc: BoundaryCondition::Transmission,
        },
        // 3: -X plane
        Surface::PlaneX { x0: -half, bc: BoundaryCondition::Reflective },
        // 4: +X plane
        Surface::PlaneX { x0: half, bc: BoundaryCondition::Reflective },
        // 5: -Y plane
        Surface::PlaneY { y0: -half, bc: BoundaryCondition::Reflective },
        // 6: +Y plane
        Surface::PlaneY { y0: half, bc: BoundaryCondition::Reflective },
        // 7: -Z plane (reflective — infinite lattice in Z)
        Surface::PlaneZ { z0: -z_half, bc: BoundaryCondition::Reflective },
        // 8: +Z plane
        Surface::PlaneZ { z0: z_half, bc: BoundaryCondition::Reflective },
    ];

    let box_aabb = Aabb::new(Vec3::new(-half, -half, -z_half), Vec3::new(half, half, z_half));

    let cells = vec![
        // 0: Fuel (inside fuel cylinder, inside Z box)
        Cell::new(
            CellId(0),
            cell::intersect_all(vec![
                cell::inside(0),
                cell::outside(7), cell::inside(8),
            ]),
            CellFill::Material(0),
        )
        .with_aabb(Aabb::new(
            Vec3::new(-fuel_or, -fuel_or, -z_half),
            Vec3::new(fuel_or, fuel_or, z_half),
        ))
        .with_temperature(900.0),
        // 1: Gap (between fuel and clad) — void (He fill, negligible XS)
        Cell::new(
            CellId(1),
            cell::intersect_all(vec![
                cell::outside(0), cell::inside(1),
                cell::outside(7), cell::inside(8),
            ]),
            CellFill::Void,
        )
        .with_aabb(Aabb::new(
            Vec3::new(-clad_ir, -clad_ir, -z_half),
            Vec3::new(clad_ir, clad_ir, z_half),
        )),
        // 2: Clad (between clad_ir and clad_or)
        Cell::new(
            CellId(2),
            cell::intersect_all(vec![
                cell::outside(1), cell::inside(2),
                cell::outside(7), cell::inside(8),
            ]),
            CellFill::Material(1),
        )
        .with_aabb(Aabb::new(
            Vec3::new(-clad_or, -clad_or, -z_half),
            Vec3::new(clad_or, clad_or, z_half),
        ))
        .with_temperature(600.0),
        // 3: Water (outside clad, inside reflective box)
        Cell::new(
            CellId(3),
            cell::intersect_all(vec![
                cell::outside(2),  // outside clad
                cell::outside(3),  // x > -half
                cell::inside(4),   // x < +half
                cell::outside(5),  // y > -half
                cell::inside(6),   // y < +half
                cell::outside(7),  // z > -z_half
                cell::inside(8),   // z < +z_half
            ]),
            CellFill::Material(2),
        )
        .with_aabb(box_aabb)
        .with_temperature(600.0),
    ];

    (surfaces, cells)
}

fn setup_materials() -> Vec<Material> {
    // Material 0: UO2 fuel (3.1% enriched, 10.4 g/cm³)
    // Atom densities from OpenMC model (atoms/barn-cm)
    let mut fuel = Material::new("UO2", 900.0);
    fuel.add_nuclide(0.00072, 0);   // U-235  (xs_kernel_idx=0)
    fuel.add_nuclide(0.02219, 1);   // U-238  (xs_kernel_idx=1)
    fuel.add_nuclide(0.04582, 2);   // O-16   (xs_kernel_idx=2)

    // Material 1: Zircaloy-4 cladding (6.55 g/cm³, simplified)
    let mut clad = Material::new("Zircaloy", 600.0);
    clad.add_nuclide(0.02189, 4);   // Zr-90  (xs_kernel_idx=4)
    clad.add_nuclide(0.00477, 5);   // Zr-91  (xs_kernel_idx=5)
    clad.add_nuclide(0.00729, 6);   // Zr-92  (xs_kernel_idx=6)
    clad.add_nuclide(0.00739, 7);   // Zr-94  (xs_kernel_idx=7)

    // Material 2: Light water (0.74 g/cm³, 600K)
    let mut water = Material::new("H2O", 600.0);
    water.add_nuclide(0.04937, 3);  // H-1    (xs_kernel_idx=3)
    water.add_nuclide(0.02469, 2);  // O-16   (xs_kernel_idx=2, shared with fuel)

    vec![fuel, clad, water]
}

fn run_svd(args: &Args, surfaces: &[Surface], cells: &[Cell], materials: &[Material], config: &SimConfig) -> RunResult {
    println!("\n── Loading nuclear data (SVD, rank={}) ──", args.rank);
    let t0 = Instant::now();

    let mut kernels = Vec::new();
    for &(filename, awr, nu_bar) in NUCLIDE_SPECS {
        let path = args.data_dir.join(filename);
        if !path.exists() {
            eprintln!("  WARNING: {} not found", path.display());
            kernels.push(xs_provider::NuclideKernels {
                elastic: None, inelastic: None, n2n: None, n3n: None,
                fission: None, capture: None, awr, nu_bar_const: nu_bar,
                nu_bar_table: None, discrete_levels: vec![],
                has_continuum_inelastic: false, elastic_angle: None,
                fission_energy_dist: None, urr_tables: None,
            });
        } else {
            kernels.push(xs_provider::load_nuclide(&path, args.rank, args.temp_idx, awr, nu_bar));
        }
    }

    let xs_mem: usize = kernels.iter().map(|k| k.svd_memory_bytes()).sum();
    let provider = xs_provider::SvdXsProvider { nuclides: kernels };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB ({} nuclides)", xs_mem as f64 / 1024.0, NUCLIDE_SPECS.len());

    println!("\n── Running eigenvalue (SVD) ──\n");
    let (_, k_mean, k_std, sim_ms) = run_simulation(config, surfaces, cells, materials, &provider, args.inactive);

    RunResult { k_mean, k_std, load_ms, sim_ms, xs_memory_bytes: xs_mem }
}

fn run_table(args: &Args, surfaces: &[Surface], cells: &[Cell], materials: &[Material], config: &SimConfig) -> RunResult {
    println!("\n── Loading nuclear data (pointwise table) ──");
    let t0 = Instant::now();

    let mut tables = Vec::new();
    for &(filename, awr, nu_bar) in NUCLIDE_SPECS {
        let path = args.data_dir.join(filename);
        if !path.exists() {
            eprintln!("  WARNING: {} not found", path.display());
            tables.push(xs_provider::NuclideTableData {
                elastic: None, inelastic: None, n2n: None, n3n: None,
                fission: None, capture: None, awr, nu_bar_const: nu_bar,
                nu_bar_table: None, discrete_levels: vec![],
                has_continuum_inelastic: false, elastic_angle: None,
                fission_energy_dist: None, urr_tables: None,
            });
        } else {
            tables.push(xs_provider::load_nuclide_table(&path, args.temp_idx, awr, nu_bar));
        }
    }

    let xs_mem: usize = tables.iter().map(|t| t.table_memory_bytes()).sum();
    let provider = xs_provider::TableXsProvider { nuclides: tables };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB ({} nuclides)", xs_mem as f64 / 1024.0, NUCLIDE_SPECS.len());

    println!("\n── Running eigenvalue (pointwise table) ──\n");
    let (_, k_mean, k_std, sim_ms) = run_simulation(config, surfaces, cells, materials, &provider, args.inactive);

    RunResult { k_mean, k_std, load_ms, sim_ms, xs_memory_bytes: xs_mem }
}

fn print_result(label: &str, r: &RunResult, particles: u32, batches: u32) {
    let mc_pcm = r.k_std / r.k_mean * 1e5;
    println!("  {label}:");
    println!("    k_inf       = {:.5} +/- {:.5}", r.k_mean, r.k_std);
    println!("    MC uncert   = {mc_pcm:.0} pcm");
    println!("    Load time   = {:.0} ms", r.load_ms);
    println!("    Sim time    = {:.0} ms  ({} batches x {} particles)", r.sim_ms, batches, particles);
    println!("    XS memory   = {:.1} KB", r.xs_memory_bytes as f64 / 1024.0);
}

fn main() {
    let args = Args::parse();

    println!("=== open_rust_mc — PWR Pin Cell Benchmark ===\n");
    println!("Data dir:     {}", args.data_dir.display());
    println!("Mode:         {:?}", args.mode);
    if matches!(args.mode, XsMode::Svd | XsMode::Both) {
        println!("SVD rank:     {}", args.rank);
    }
    println!("Batches:      {} ({} inactive + {} active)", args.batches, args.inactive, args.batches - args.inactive);
    println!("Particles:    {}", args.particles);
    println!("Nuclides:     {} (U235, U238, O16, H1, Zr90-94)", NUCLIDE_SPECS.len());
    println!("Materials:    3 (UO2 fuel, Zircaloy clad, H2O moderator)");

    let (surfaces, cells) = setup_geometry();
    let materials = setup_materials();

    let config = SimConfig {
        batches: args.batches,
        inactive: args.inactive,
        particles_per_batch: args.particles,
        seed: 0,
    };

    match args.mode {
        XsMode::Svd => {
            let r = run_svd(&args, &surfaces, &cells, &materials, &config);
            println!("\n{}", "=".repeat(60));
            println!("RESULTS — PWR Pin Cell");
            println!("{}", "=".repeat(60));
            print_result("SVD", &r, args.particles, args.batches);
        }
        XsMode::Table => {
            let r = run_table(&args, &surfaces, &cells, &materials, &config);
            println!("\n{}", "=".repeat(60));
            println!("RESULTS — PWR Pin Cell");
            println!("{}", "=".repeat(60));
            print_result("Pointwise Table", &r, args.particles, args.batches);
        }
        XsMode::Both => {
            let svd = run_svd(&args, &surfaces, &cells, &materials, &config);
            let tbl = run_table(&args, &surfaces, &cells, &materials, &config);

            println!("\n{}", "=".repeat(60));
            println!("PWR PIN CELL — HEAD-TO-HEAD COMPARISON");
            println!("{}", "=".repeat(60));

            print_result(&format!("SVD (rank={})", args.rank), &svd, args.particles, args.batches);
            println!();
            print_result("Pointwise Table", &tbl, args.particles, args.batches);

            let dk_pcm = (svd.k_mean - tbl.k_mean).abs() / tbl.k_mean * 1e5;
            let mem_ratio = if svd.xs_memory_bytes > 0 {
                tbl.xs_memory_bytes as f64 / svd.xs_memory_bytes as f64
            } else { 0.0 };
            let speed_ratio = if svd.sim_ms > 0.0 { tbl.sim_ms / svd.sim_ms } else { 0.0 };

            println!("\n  {}", "-".repeat(50));
            println!("  COMPARISON:");
            println!("    k_inf gap (SVD vs table) = {dk_pcm:.0} pcm");
            println!("    Memory ratio (tbl/svd)   = {mem_ratio:.2}x ({:.1} KB vs {:.1} KB)",
                     svd.xs_memory_bytes as f64 / 1024.0, tbl.xs_memory_bytes as f64 / 1024.0);
            println!("    Speed ratio (tbl/svd)    = {speed_ratio:.2}x ({:.0} ms vs {:.0} ms)",
                     svd.sim_ms, tbl.sim_ms);
        }
    }
}
