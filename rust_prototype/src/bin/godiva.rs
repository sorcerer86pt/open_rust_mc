//! End-to-end Godiva eigenvalue simulation in pure Rust.
//!
//! Reads U-234, U-235, U-238 HDF5 files, builds SVD kernels or pointwise
//! tables, sets up the Godiva geometry, and runs k-eigenvalue power iteration.
//!
//! Supports multi-seed statistical benchmarking for rigorous time/particle
//! measurements with confidence intervals.
//!
//! Usage:
//!   godiva <data_dir> [--rank K] [--batches N] [--particles N] [--mode MODE] [--seeds S]
//!
//! Examples:
//!   godiva data/ --mode both --seeds 5 --particles 20000 --batches 150

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3};
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::{self, SimConfig, XsProvider};
use open_rust_mc::transport::xs_provider;

#[derive(clap::ValueEnum, Clone, Debug)]
enum XsMode { Svd, Table, Both }

#[derive(Parser)]
#[command(name = "godiva", about = "Godiva k-eigenvalue benchmark (pure Rust + SVD)")]
struct Args {
    /// Directory containing nuclide HDF5 files (U234.h5, U235.h5, U238.h5).
    data_dir: PathBuf,

    /// SVD truncation rank (only used in svd/both modes).
    #[arg(short, long, default_value_t = 5)]
    rank: usize,

    /// Total number of batches.
    #[arg(short, long, default_value_t = 50)]
    batches: u32,

    /// Inactive batches (discarded for k_eff averaging).
    #[arg(short, long, default_value_t = 10)]
    inactive: u32,

    /// Particles per batch.
    #[arg(short, long, default_value_t = 5000)]
    particles: u32,

    /// Temperature index to use (0-based, from sorted temperatures).
    #[arg(short, long, default_value_t = 1)]
    temp_idx: usize,

    /// Cross-section lookup mode: svd, table, or both (honesty test).
    #[arg(short, long, value_enum, default_value_t = XsMode::Svd)]
    mode: XsMode,

    /// Number of independent seeds for statistical benchmarking.
    /// Each seed produces a fully independent run. Reports mean ± stddev
    /// of time/particle and k_eff across all seeds.
    #[arg(short, long, default_value_t = 1)]
    seeds: u32,
}

const NUCLIDE_SPECS: &[(&str, f64, f64)] = &[
    ("U234.h5", 232.029, 2.49),
    ("U235.h5", 233.025, 2.43),
    ("U238.h5", 236.006, 2.49),
];

/// Results from one seeded run.
struct SeedResult {
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
    let xs_mem: usize = kernels.iter().map(|k| k.svd_memory_bytes()).sum();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB", xs_mem as f64 / 1024.0);
    (xs_provider::SvdXsProvider { nuclides: kernels, thermal: vec![] }, xs_mem, load_ms)
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
    let xs_mem: usize = tables.iter().map(|t| t.table_memory_bytes()).sum();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB", xs_mem as f64 / 1024.0);
    (xs_provider::TableXsProvider { nuclides: tables, thermal: vec![] }, xs_mem, load_ms)
}

fn print_benchmark(r: &BenchmarkResult, _particles: u32) {
    let k_exp = 1.0000;
    let delta_pcm = (r.k_eff_mean() - k_exp).abs() / k_exp * 1e5;
    let n_seeds = r.seed_results.len();

    println!("  {}:", r.label);
    println!("    k_eff            = {:.5} +/- {:.5}", r.k_eff_mean(), r.k_eff_std());
    if n_seeds > 1 {
        for sr in &r.seed_results {
            println!("      seed {}: k={:.5} +/- {:.5}  ({:.1} ns/p)",
                     sr.seed, sr.k_mean, sr.k_std, sr.ns_per_particle());
        }
    }
    println!("    delta(exp)       = {delta_pcm:.0} pcm");
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

    println!("=== open_rust_mc — Godiva Eigenvalue Benchmark ===\n");
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
    if args.seeds > 1 {
        println!("Seeds:        {} (independent runs for statistical confidence)", args.seeds);
    }

    // ── Geometry: Godiva (ICSBEP HEU-MET-FAST-001) ─────────────────
    let radius = 8.7407;
    let surfaces = vec![
        Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius,
            bc: BoundaryCondition::Vacuum,
        },
    ];
    let cells = vec![
        Cell::new(CellId(0), cell::inside(0), CellFill::Material(0))
            .with_aabb(Aabb::new(
                Vec3::new(-radius, -radius, -radius),
                Vec3::new(radius, radius, radius),
            ))
            .with_temperature(294.0),
        Cell::new(CellId(1), cell::outside(0), CellFill::Void),
    ];
    let mut heu = Material::new("HEU", 294.0);
    heu.add_nuclide(0.000483, 0);
    heu.add_nuclide(0.04509, 1);
    heu.add_nuclide(0.00265, 2);
    let materials = vec![heu];

    // ── Run ─────────────────────────────────────────────────────────
    match args.mode {
        XsMode::Svd => {
            let (provider, xs_mem, load_ms) = load_svd(&args);
            let r = run_multi_seed("SVD", &args, &surfaces, &cells, &materials, &provider, xs_mem, load_ms);
            println!("\n{}", "=".repeat(60));
            println!("RESULTS");
            println!("{}", "=".repeat(60));
            print_benchmark(&r, args.particles);
        }
        XsMode::Table => {
            let (provider, xs_mem, load_ms) = load_table(&args);
            let r = run_multi_seed("Table", &args, &surfaces, &cells, &materials, &provider, xs_mem, load_ms);
            println!("\n{}", "=".repeat(60));
            println!("RESULTS");
            println!("{}", "=".repeat(60));
            print_benchmark(&r, args.particles);
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
            println!("STATISTICAL BENCHMARK — HEAD-TO-HEAD");
            println!("{}", "=".repeat(60));
            print_benchmark(&svd, args.particles);
            println!();
            print_benchmark(&tbl, args.particles);

            let dk = (svd.k_eff_mean() - tbl.k_eff_mean()).abs() / tbl.k_eff_mean() * 1e5;
            let speedup = tbl.ns_per_particle_mean() / svd.ns_per_particle_mean();

            println!("\n  {}", "-".repeat(50));
            println!("  COMPARISON:");
            println!("    k_eff gap (SVD - table)  = {dk:.0} pcm");
            println!("    SVD speedup              = {speedup:.2}x ({:.2} vs {:.2} ns/particle)",
                     svd.ns_per_particle_mean(), tbl.ns_per_particle_mean());
            if args.seeds > 1 {
                // Uncertainty on the speedup ratio via error propagation
                let s1 = svd.ns_per_particle_std();
                let s2 = tbl.ns_per_particle_std();
                let m1 = svd.ns_per_particle_mean();
                let m2 = tbl.ns_per_particle_mean();
                let ratio_std = speedup * ((s1/m1).powi(2) + (s2/m2).powi(2)).sqrt();
                println!("    Speedup uncertainty      = +/- {ratio_std:.2}x ({} seeds)", args.seeds);
            }
            println!("\n  Experimental k_eff = 1.00000");
            println!("  SVD delta(exp)     = {:.0} pcm", (svd.k_eff_mean() - 1.0).abs() * 1e5);
            println!("  Table delta(exp)   = {:.0} pcm", (tbl.k_eff_mean() - 1.0).abs() * 1e5);
        }
    }
}
