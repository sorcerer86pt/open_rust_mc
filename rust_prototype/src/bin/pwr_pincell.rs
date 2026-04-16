//! PWR pin cell eigenvalue benchmark — multi-material, multi-nuclide.
//!
//! Standard PWR fuel pin: UO2 fuel + Zircaloy clad + light water moderator.
//! This is the real test for SVD compression — 8 nuclides across 3 materials
//! with heterogeneous geometry (cylinder + reflective box).
//!
//! Supports multi-seed statistical benchmarking for rigorous time/particle
//! measurements with confidence intervals.
//!
//! Usage:
//!   pwr_pincell <data_dir> [--rank K] [--batches N] [--particles N] [--mode MODE] [--seeds S]

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3};
use open_rust_mc::hdf5_reader;
use open_rust_mc::thermal::ThermalScatteringData;
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::{self, SimConfig, XsProvider};
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

    /// Number of independent seeds for statistical benchmarking.
    #[arg(short, long, default_value_t = 1)]
    seeds: u32,
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

/// Results from one seeded run.
struct SeedResult {
    #[allow(dead_code)]
    seed: u32,
    k_mean: f64,
    k_std: f64,
    sim_ms: f64,
    total_histories: u64,
}

impl SeedResult {
    fn ns_per_particle(&self) -> f64 {
        self.sim_ms * 1e6 / self.total_histories as f64
    }
}

/// Aggregate results across multiple seeds.
struct BenchmarkResult {
    label: String,
    load_ms: f64,
    xs_memory_bytes: usize,
    seed_results: Vec<SeedResult>,
}

impl BenchmarkResult {
    fn k_eff_mean(&self) -> f64 {
        let n = self.seed_results.len() as f64;
        self.seed_results.iter().map(|r| r.k_mean).sum::<f64>() / n
    }

    fn k_eff_std(&self) -> f64 {
        if self.seed_results.len() < 2 { return self.seed_results[0].k_std; }
        let mean = self.k_eff_mean();
        let n = self.seed_results.len() as f64;
        let var = self.seed_results.iter()
            .map(|r| (r.k_mean - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        var.sqrt()
    }

    fn ns_per_particle_mean(&self) -> f64 {
        let n = self.seed_results.len() as f64;
        self.seed_results.iter().map(|r| r.ns_per_particle()).sum::<f64>() / n
    }

    fn ns_per_particle_std(&self) -> f64 {
        if self.seed_results.len() < 2 { return 0.0; }
        let mean = self.ns_per_particle_mean();
        let n = self.seed_results.len() as f64;
        let var = self.seed_results.iter()
            .map(|r| (r.ns_per_particle() - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        var.sqrt()
    }

    fn total_sim_ms(&self) -> f64 {
        self.seed_results.iter().map(|r| r.sim_ms).sum()
    }
}

/// Run multiple seeds with a given XS provider, return aggregate results.
fn run_multi_seed<XS: XsProvider>(
    label: &str,
    args: &Args,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    xs_provider: &XS,
    xs_memory_bytes: usize,
    load_ms: f64,
) -> BenchmarkResult {
    let inactive = args.inactive.min(args.batches.saturating_sub(1));
    let total_histories = (args.batches - inactive) as u64 * args.particles as u64;
    let mut seed_results = Vec::with_capacity(args.seeds as usize);

    for seed in 0..args.seeds {
        let config = SimConfig {
            batches: args.batches,
            inactive,
            particles_per_batch: args.particles,
            seed: seed as u64,
        };

        if args.seeds > 1 {
            print!("  Seed {seed}: ");
            let _ = std::io::stdout().flush();
        } else {
            println!();
        }

        let t1 = Instant::now();
        let (results, _) = simulate::run_eigenvalue(&config, surfaces, cells, materials, xs_provider);
        let sim_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let active: Vec<f64> = results.iter()
            .filter(|r| r.batch > inactive)
            .map(|r| r.k_eff)
            .collect();
        let n = active.len() as f64;
        let k_mean = active.iter().sum::<f64>() / n;
        let k_var = active.iter().map(|&k| (k - k_mean).powi(2)).sum::<f64>() / (n * (n - 1.0));
        let k_std = k_var.sqrt();

        let sr = SeedResult { seed, k_mean, k_std, sim_ms, total_histories };

        if args.seeds > 1 {
            println!("k={k_mean:.5} +/- {k_std:.5}  {sim_ms:.0}ms  ({:.1} ns/particle)",
                     sr.ns_per_particle());
        }

        seed_results.push(sr);
    }

    BenchmarkResult { label: label.to_string(), load_ms, xs_memory_bytes, seed_results }
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

/// Load thermal scattering data and build the per-nuclide thermal vector.
///
/// H1 (xs_kernel_idx=3) gets c_H_in_H2O thermal data if available.
fn load_thermal(data_dir: &PathBuf) -> Vec<Option<Arc<ThermalScatteringData>>> {
    let h2o_path = data_dir.join("c_H_in_H2O.h5");
    let h2o_thermal: Option<Arc<ThermalScatteringData>> = if h2o_path.exists() {
        match hdf5_reader::load_thermal_scattering(&h2o_path) {
            Ok(tsl) => {
                println!("  S(a,b): loaded {} ({} temperatures)", tsl.name, tsl.temp_labels.len());
                Some(Arc::new(tsl))
            }
            Err(e) => {
                eprintln!("  WARNING: failed to load S(a,b) from {}: {e}", h2o_path.display());
                None
            }
        }
    } else {
        println!("  S(a,b): c_H_in_H2O.h5 not found — using free gas model for H");
        None
    };

    // Build thermal vector indexed by xs_kernel_idx (same order as NUCLIDE_SPECS)
    // Index 3 = H1 → gets H₂O thermal data
    let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; NUCLIDE_SPECS.len()];
    if let Some(ref tsl) = h2o_thermal {
        thermal[3] = Some(Arc::clone(tsl)); // H1 = index 3
    }
    thermal
}

fn load_svd(args: &Args) -> (xs_provider::SvdXsProvider, usize, f64) {
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

    let thermal = load_thermal(&args.data_dir);
    let xs_mem: usize = kernels.iter().map(|k| k.svd_memory_bytes()).sum();
    let provider = xs_provider::SvdXsProvider { nuclides: kernels, thermal };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB ({} nuclides)", xs_mem as f64 / 1024.0, NUCLIDE_SPECS.len());

    (provider, xs_mem, load_ms)
}

fn load_table(args: &Args) -> (xs_provider::TableXsProvider, usize, f64) {
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

    let thermal = load_thermal(&args.data_dir);
    let xs_mem: usize = tables.iter().map(|t| t.table_memory_bytes()).sum();
    let provider = xs_provider::TableXsProvider { nuclides: tables, thermal };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB ({} nuclides)", xs_mem as f64 / 1024.0, NUCLIDE_SPECS.len());

    (provider, xs_mem, load_ms)
}

fn print_benchmark(r: &BenchmarkResult) {
    let n_seeds = r.seed_results.len();

    println!("  {}:", r.label);
    println!("    k_inf            = {:.5} +/- {:.5}", r.k_eff_mean(), r.k_eff_std());
    if n_seeds > 1 {
        println!("    ns/particle      = {:.2} +/- {:.2} ({n_seeds} seeds)",
                 r.ns_per_particle_mean(), r.ns_per_particle_std());
    } else {
        println!("    ns/particle      = {:.2}", r.ns_per_particle_mean());
    }
    println!("    Total sim time   = {:.0} ms ({n_seeds} run{})", r.total_sim_ms(), if n_seeds > 1 { "s" } else { "" });
    println!("    Load time        = {:.0} ms", r.load_ms);
    println!("    XS memory        = {:.1} KB", r.xs_memory_bytes as f64 / 1024.0);
}

fn main() {
    let args = Args::parse();

    let inactive = if args.inactive >= args.batches {
        println!("  [Warning] Inactive batches ({}) >= total batches ({}). Capping inactive to {}.",
                 args.inactive, args.batches, args.batches - 1);
        args.batches - 1
    } else {
        args.inactive
    };
    let active_batches = args.batches - inactive;
    let histories_per_run = active_batches as u64 * args.particles as u64;

    println!("=== open_rust_mc — PWR Pin Cell Benchmark ===\n");
    println!("Data dir:     {}", args.data_dir.display());
    println!("Mode:         {:?}", args.mode);
    if matches!(args.mode, XsMode::Svd | XsMode::Both) {
        println!("SVD rank:     {}", args.rank);
    }
    println!("Batches:      {} ({} inactive + {} active)",
             args.batches, inactive, active_batches);
    println!("Particles:    {}/batch", args.particles);
    println!("Histories:    {} per run ({} active batches x {} particles)",
             histories_per_run, active_batches, args.particles);
    println!("Nuclides:     {} (U235, U238, O16, H1, Zr90-94)", NUCLIDE_SPECS.len());
    println!("Materials:    3 (UO2 fuel, Zircaloy clad, H2O moderator)");
    if args.seeds > 1 {
        println!("Seeds:        {} (independent runs for statistical confidence)", args.seeds);
    }
    println!("S(a,b):       c_H_in_H2O (if available in data_dir)");

    let (surfaces, cells) = setup_geometry();
    let materials = setup_materials();

    match args.mode {
        XsMode::Svd => {
            let (provider, xs_mem, load_ms) = load_svd(&args);
            let r = run_multi_seed(
                &format!("SVD (rank={})", args.rank),
                &args, &surfaces, &cells, &materials, &provider, xs_mem, load_ms,
            );
            println!("\n{}", "=".repeat(60));
            println!("RESULTS — PWR Pin Cell");
            println!("{}", "=".repeat(60));
            print_benchmark(&r);
        }
        XsMode::Table => {
            let (provider, xs_mem, load_ms) = load_table(&args);
            let r = run_multi_seed(
                "Pointwise Table",
                &args, &surfaces, &cells, &materials, &provider, xs_mem, load_ms,
            );
            println!("\n{}", "=".repeat(60));
            println!("RESULTS — PWR Pin Cell");
            println!("{}", "=".repeat(60));
            print_benchmark(&r);
        }
        XsMode::Both => {
            let (svd_prov, svd_mem, svd_load) = load_svd(&args);
            let svd = run_multi_seed(
                &format!("SVD (rank={})", args.rank),
                &args, &surfaces, &cells, &materials, &svd_prov, svd_mem, svd_load,
            );
            drop(svd_prov); // free before loading table

            let (tbl_prov, tbl_mem, tbl_load) = load_table(&args);
            let tbl = run_multi_seed(
                "Pointwise Table",
                &args, &surfaces, &cells, &materials, &tbl_prov, tbl_mem, tbl_load,
            );

            println!("\n{}", "=".repeat(60));
            println!("PWR PIN CELL — STATISTICAL BENCHMARK");
            println!("{}", "=".repeat(60));

            print_benchmark(&svd);
            println!();
            print_benchmark(&tbl);

            let dk = (svd.k_eff_mean() - tbl.k_eff_mean()).abs() / tbl.k_eff_mean() * 1e5;
            let speedup = tbl.ns_per_particle_mean() / svd.ns_per_particle_mean();

            println!("\n  {}", "-".repeat(50));
            println!("  COMPARISON:");
            println!("    k_inf gap (SVD - table)  = {dk:.0} pcm");
            println!("    SVD speedup              = {speedup:.2}x ({:.2} vs {:.2} ns/particle)",
                     svd.ns_per_particle_mean(), tbl.ns_per_particle_mean());
            if args.seeds > 1 {
                let s1 = svd.ns_per_particle_std();
                let s2 = tbl.ns_per_particle_std();
                let m1 = svd.ns_per_particle_mean();
                let m2 = tbl.ns_per_particle_mean();
                let ratio_std = speedup * ((s1/m1).powi(2) + (s2/m2).powi(2)).sqrt();
                println!("    Speedup uncertainty      = +/- {ratio_std:.2}x ({} seeds)", args.seeds);
            }
            println!("    Memory ratio (tbl/svd)   = {:.2}x ({:.1} KB vs {:.1} KB)",
                     tbl.xs_memory_bytes as f64 / svd.xs_memory_bytes as f64,
                     svd.xs_memory_bytes as f64 / 1024.0, tbl.xs_memory_bytes as f64 / 1024.0);
        }
    }
}
