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
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3};
use open_rust_mc::transport::hybrid_xs::HybridTableWmpXsProvider;
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::{self, SimConfig, XsProvider};
use open_rust_mc::transport::xs_provider;
use open_rust_mc::wmp::WindowedMultipole;

#[derive(clap::ValueEnum, Clone, Debug)]
enum XsMode {
    Svd,
    Table,
    Both,
    /// ACE pointwise table + Windowed Multipole inside the resolved
    /// resonance window. The industry-standard low-memory baseline.
    Wmp,
    /// Three-way honesty test: SVD vs Table vs ACE+WMP back-to-back.
    All,
}

#[derive(Parser)]
#[command(
    name = "godiva",
    about = "Godiva k-eigenvalue benchmark (pure Rust + SVD)"
)]
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
    /// Ignored when `--target-temp` is specified.
    #[arg(short, long, default_value_t = 1)]
    temp_idx: usize,

    /// Operating temperature in Kelvin. When this lies between two
    /// library endpoints (e.g., 600 K between 294 K and 900 K), the
    /// table provider loads both bracketing temperatures and draws one
    /// per lookup (OpenMC-style pseudo-interpolation), forcing random
    /// cache loads into two XS arrays per collision. The SVD provider
    /// uses Ducru kernel reconstruction for continuous T interpolation.
    /// If unset, falls back to `--temp-idx` (single-library behaviour).
    #[arg(long)]
    target_temp: Option<f64>,

    /// Override the SVD rank used for discrete inelastic levels
    /// (MT=51-91). Reviewer note: levels have almost no temperature
    /// dependence so rank=1 should suffice. Default: same as --rank.
    /// Only applied when routing through the at-temp loader (either
    /// `--target-temp` or `--discrete-rank` is specified).
    #[arg(long)]
    discrete_rank: Option<usize>,

    /// Cross-section lookup mode: svd, table, or both (honesty test).
    #[arg(short, long, value_enum, default_value_t = XsMode::Svd)]
    mode: XsMode,

    /// Number of independent seeds for statistical benchmarking.
    /// Each seed produces a fully independent run. Reports mean ± stddev
    /// of time/particle and k_eff across all seeds.
    #[arg(short, long, default_value_t = 1)]
    seeds: u32,

    /// Replace the fixed `--inactive` count with runtime Shannon-entropy
    /// plateau detection. The simulator discards settle batches until
    /// the fission-site entropy's sliding-window CV drops below 1e-3
    /// (bounded by the policy's [20, 200] min/max inactive).
    #[arg(long, default_value_t = false)]
    auto_inactive: bool,
}

const NUCLIDE_SPECS: &[(&str, f64, f64)] = &[
    ("U234.h5", 232.029, 2.49),
    ("U235.h5", 233.025, 2.43),
    ("U238.h5", 236.006, 2.49),
];

/// WMP file and evaluation temperature for each nuclide in `NUCLIDE_SPECS`.
/// Must match the NUCLIDE_SPECS order. 294 K matches Godiva's room-temp
/// operating condition; the WMP evaluator broadens on the fly from 0 K
/// poles, so a single library covers all temperatures exactly.
const WMP_SPECS: &[(&str, f64)] = &[
    ("092234.h5", 294.0),
    ("092235.h5", 294.0),
    ("092238.h5", 294.0),
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
        if self.seed_results.len() < 2 {
            return self.seed_results[0].k_std;
        }
        let mean = self.k_eff_mean();
        let n = self.seed_results.len() as f64;
        let var = self
            .seed_results
            .iter()
            .map(|r| (r.k_mean - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        var.sqrt()
    }

    fn ns_per_particle_mean(&self) -> f64 {
        let n = self.seed_results.len() as f64;
        self.seed_results
            .iter()
            .map(|r| r.ns_per_particle())
            .sum::<f64>()
            / n
    }

    fn ns_per_particle_std(&self) -> f64 {
        if self.seed_results.len() < 2 {
            return 0.0;
        }
        let mean = self.ns_per_particle_mean();
        let n = self.seed_results.len() as f64;
        let var = self
            .seed_results
            .iter()
            .map(|r| (r.ns_per_particle() - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        var.sqrt()
    }

    fn total_sim_ms(&self) -> f64 {
        self.seed_results.iter().map(|r| r.sim_ms).sum()
    }
}

/// Run multiple seeds with a given XS provider, return aggregate results.
#[allow(clippy::too_many_arguments)]
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
            auto_inactive: if args.auto_inactive {
                Some(open_rust_mc::transport::simulate::EntropyConvergence::default())
            } else {
                None
            },
            verbose: true,
            parallel: true,
        };

        if args.seeds > 1 {
            print!("  Seed {seed}: ");
            let _ = std::io::stdout().flush();
        } else {
            println!();
        }

        let t1 = Instant::now();
        let (results, _) =
            simulate::run_eigenvalue(&config, surfaces, cells, materials, xs_provider);
        let sim_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let active: Vec<f64> = results
            .iter()
            .filter(|r| r.active)
            .map(|r| r.k_eff)
            .collect();
        let n = active.len() as f64;
        let k_mean = active.iter().sum::<f64>() / n;
        let k_var = active.iter().map(|&k| (k - k_mean).powi(2)).sum::<f64>() / (n * (n - 1.0));
        let k_std = k_var.sqrt();

        let sr = SeedResult {
            seed,
            k_mean,
            k_std,
            sim_ms,
            total_histories,
        };

        if args.seeds > 1 {
            println!(
                "k={k_mean:.5} +/- {k_std:.5}  {sim_ms:.0}ms  ({:.1} ns/particle)",
                sr.ns_per_particle()
            );
        }

        seed_results.push(sr);
    }

    BenchmarkResult {
        label: label.to_string(),
        load_ms,
        xs_memory_bytes,
        seed_results,
    }
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
                elastic: None,
                total_table: None,
                total_xs_raw: None,
                missing_xs: None,
                pointwise_xs: None,
                inelastic: None,
                n2n: None,
                n3n: None,
                fission: None,
                capture: None,
                awr,
                nu_bar_const: nu_bar,
                nu_bar_table: None,
                discrete_levels: vec![],
                inelastic_cdf: None,
                discrete_level_angles: vec![],
                has_continuum_inelastic: false,
                elastic_angle: None,
                fission_energy_dist: None,
                inelastic_continuum_edist: None,
                n2n_edist: None,
                n3n_edist: None,
                urr_tables: None,
                photon_products: Vec::new(),
            });
        } else {
            // Route through the at-temp loader whenever target_temp or
            // discrete_rank is set. Otherwise fall back to the single-T
            // legacy loader for bit-for-bit backwards compatibility.
            match (args.target_temp, args.discrete_rank) {
                (Some(t), _) => kernels.push(xs_provider::load_nuclide_at_temp(
                    &path,
                    args.rank,
                    t,
                    awr,
                    nu_bar,
                    args.discrete_rank,
                )),
                (None, Some(_)) => {
                    // Need a target temperature; resolve temp_idx to the
                    // corresponding library value by opening the reader.
                    match open_rust_mc::hdf5_reader::NuclideFileReader::open(&path) {
                        Ok(r) => {
                            let t = r.temperatures.get(args.temp_idx).copied().unwrap_or(294.0);
                            kernels.push(xs_provider::load_nuclide_at_temp(
                                &path,
                                args.rank,
                                t,
                                awr,
                                nu_bar,
                                args.discrete_rank,
                            ));
                        }
                        Err(e) => {
                            eprintln!("  WARNING: cannot open {}: {e}", path.display());
                        }
                    }
                }
                (None, None) => kernels.push(xs_provider::load_nuclide(
                    &path,
                    args.rank,
                    args.temp_idx,
                    awr,
                    nu_bar,
                )),
            }
        }
    }
    let xs_mem: usize = kernels.iter().map(|k| k.svd_memory_bytes()).sum();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB",
        xs_mem as f64 / 1024.0
    );
    (
        xs_provider::SvdXsProvider {
            nuclides: kernels,
            thermal: vec![],
        },
        xs_mem,
        load_ms,
    )
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
                elastic: None,
                total_table: None,
                inelastic: None,
                n2n: None,
                n3n: None,
                fission: None,
                capture: None,
                awr,
                nu_bar_const: nu_bar,
                nu_bar_table: None,
                discrete_levels: vec![],
                discrete_level_angles: vec![],
                has_continuum_inelastic: false,
                elastic_angle: None,
                fission_energy_dist: None,
                inelastic_continuum_edist: None,
                n2n_edist: None,
                n3n_edist: None,
                urr_tables: None,
                photon_products: Vec::new(),
            });
        } else {
            match args.target_temp {
                Some(t) => tables.push(xs_provider::load_nuclide_table_at_temp(
                    &path, t, awr, nu_bar,
                )),
                None => tables.push(xs_provider::load_nuclide_table(
                    &path,
                    args.temp_idx,
                    awr,
                    nu_bar,
                )),
            }
        }
    }
    let xs_mem: usize = tables.iter().map(|t| t.table_memory_bytes()).sum();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB",
        xs_mem as f64 / 1024.0
    );
    (
        xs_provider::TableXsProvider {
            nuclides: tables,
            thermal: vec![],
        },
        xs_mem,
        load_ms,
    )
}

/// Load the ACE+WMP hybrid provider: pointwise tables everywhere (the
/// inner path used by `--mode table`), overridden by WMP evaluation inside
/// each nuclide's resolved-resonance window. Mirrors the industry
/// standard low-memory Monte Carlo lookup used by OpenMC when WMP data
/// is available.
fn load_wmp_hybrid(args: &Args) -> (HybridTableWmpXsProvider, usize, f64) {
    println!("\n── Loading nuclear data (ACE pointwise + WMP in RRR) ──");
    let t0 = Instant::now();
    let (table_provider, table_mem, _) = load_table(args);

    let wmp_dir = args.data_dir.join("..").join("wmp");
    let mut wmps: Vec<Option<(Arc<WindowedMultipole>, f64)>> = Vec::with_capacity(WMP_SPECS.len());
    let mut covered = 0usize;
    for &(wmp_file, t_kelvin) in WMP_SPECS {
        let path = wmp_dir.join(wmp_file);
        if !path.exists() {
            println!("  WMP not found: {}", path.display());
            wmps.push(None);
            continue;
        }
        match WindowedMultipole::from_hdf5(&path) {
            Ok(wmp) => {
                println!(
                    "  Loaded WMP {wmp_file} at T={t_kelvin} K  ({:.3e}-{:.3e} eV, {} poles, {} windows)",
                    wmp.e_min, wmp.e_max, wmp.n_poles, wmp.n_windows
                );
                covered += 1;
                wmps.push(Some((Arc::new(wmp), t_kelvin)));
            }
            Err(e) => {
                eprintln!("  Failed to load {wmp_file}: {e}");
                wmps.push(None);
            }
        }
    }

    let provider = HybridTableWmpXsProvider::new(table_provider, wmps);
    let report = provider.memory_report();
    // Honest accounting: report the *smooth-only* memory, which is what
    // a production implementation would actually carry (pointwise
    // tables scrubbed of resonance energies + WMP payload). That matches
    // the industry baseline the reviewer asked about.
    let xs_mem = report.smooth_only_total();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  WMP covers {covered}/{} nuclides", WMP_SPECS.len());
    println!(
        "  Loaded in {load_ms:.0} ms  |  XS memory (smooth-only): {:.1} KB  [current in-solver: {:.1} KB]",
        xs_mem as f64 / 1024.0,
        (report.current_total()) as f64 / 1024.0,
    );
    println!(
        "    breakdown: pointwise {:.1} KB + WMP payload {:.1} KB (in-solver table total was {:.1} KB)",
        report.smooth_only_svd_bytes as f64 / 1024.0,
        report.wmp_payload_bytes as f64 / 1024.0,
        table_mem as f64 / 1024.0,
    );
    (provider, xs_mem, load_ms)
}

fn print_benchmark(r: &BenchmarkResult, _particles: u32) {
    let k_exp = 1.0000;
    let delta_pcm = (r.k_eff_mean() - k_exp).abs() / k_exp * 1e5;
    let n_seeds = r.seed_results.len();

    println!("  {}:", r.label);
    println!(
        "    k_eff            = {:.5} +/- {:.5}",
        r.k_eff_mean(),
        r.k_eff_std()
    );
    if n_seeds > 1 {
        for sr in &r.seed_results {
            println!(
                "      seed {}: k={:.5} +/- {:.5}  ({:.1} ns/p)",
                sr.seed,
                sr.k_mean,
                sr.k_std,
                sr.ns_per_particle()
            );
        }
    }
    println!("    delta(exp)       = {delta_pcm:.0} pcm");
    if n_seeds > 1 {
        println!(
            "    ns/particle      = {:.2} +/- {:.2} ({n_seeds} seeds)",
            r.ns_per_particle_mean(),
            r.ns_per_particle_std()
        );
    } else {
        println!("    ns/particle      = {:.2}", r.ns_per_particle_mean());
    }
    println!(
        "    Total sim time   = {:.0} ms ({n_seeds} run{})",
        r.total_sim_ms(),
        if n_seeds > 1 { "s" } else { "" }
    );
    println!("    Load time        = {:.0} ms", r.load_ms);
    println!(
        "    XS memory        = {:.1} KB",
        r.xs_memory_bytes as f64 / 1024.0
    );
}

fn main() {
    let args = Args::parse();

    let inactive = if args.inactive >= args.batches {
        println!(
            "  [Warning] Inactive batches ({}) >= total batches ({}). Capping inactive to {}.",
            args.inactive,
            args.batches,
            args.batches - 1
        );
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
    match args.target_temp {
        Some(t) => println!("Target T:     {t:.1} K (stochastic T-interp on table)"),
        None => println!("Target T:     library index {}", args.temp_idx),
    }
    println!(
        "Batches:      {} ({} inactive + {} active)",
        args.batches, inactive, active_batches
    );
    println!("Particles:    {}/batch", args.particles);
    println!(
        "Histories:    {} per run ({} active batches x {} particles)",
        histories_per_run, active_batches, args.particles
    );
    if args.seeds > 1 {
        println!(
            "Seeds:        {} (independent runs for statistical confidence)",
            args.seeds
        );
    }

    // ── Geometry: Godiva (ICSBEP HEU-MET-FAST-001) ─────────────────
    let radius = 8.7407;
    let surfaces = vec![Surface::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius,
        bc: BoundaryCondition::Vacuum,
    }];
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
            let r = run_multi_seed(
                "SVD", &args, &surfaces, &cells, &materials, &provider, xs_mem, load_ms,
            );
            println!("\n{}", "=".repeat(60));
            println!("RESULTS");
            println!("{}", "=".repeat(60));
            print_benchmark(&r, args.particles);
        }
        XsMode::Table => {
            let (provider, xs_mem, load_ms) = load_table(&args);
            let r = run_multi_seed(
                "Table", &args, &surfaces, &cells, &materials, &provider, xs_mem, load_ms,
            );
            println!("\n{}", "=".repeat(60));
            println!("RESULTS");
            println!("{}", "=".repeat(60));
            print_benchmark(&r, args.particles);
        }
        XsMode::Both => {
            let (svd_prov, svd_mem, svd_load) = load_svd(&args);
            let svd = run_multi_seed(
                &format!("SVD (rank={})", args.rank),
                &args,
                &surfaces,
                &cells,
                &materials,
                &svd_prov,
                svd_mem,
                svd_load,
            );
            drop(svd_prov); // free before loading table

            let (tbl_prov, tbl_mem, tbl_load) = load_table(&args);
            let tbl = run_multi_seed(
                "Pointwise Table",
                &args,
                &surfaces,
                &cells,
                &materials,
                &tbl_prov,
                tbl_mem,
                tbl_load,
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
            println!(
                "    SVD speedup              = {speedup:.2}x ({:.2} vs {:.2} ns/particle)",
                svd.ns_per_particle_mean(),
                tbl.ns_per_particle_mean()
            );
            if args.seeds > 1 {
                // Uncertainty on the speedup ratio via error propagation
                let s1 = svd.ns_per_particle_std();
                let s2 = tbl.ns_per_particle_std();
                let m1 = svd.ns_per_particle_mean();
                let m2 = tbl.ns_per_particle_mean();
                let ratio_std = speedup * ((s1 / m1).powi(2) + (s2 / m2).powi(2)).sqrt();
                println!(
                    "    Speedup uncertainty      = +/- {ratio_std:.2}x ({} seeds)",
                    args.seeds
                );
            }
            println!("\n  Experimental k_eff = 1.00000");
            println!(
                "  SVD delta(exp)     = {:.0} pcm",
                (svd.k_eff_mean() - 1.0).abs() * 1e5
            );
            println!(
                "  Table delta(exp)   = {:.0} pcm",
                (tbl.k_eff_mean() - 1.0).abs() * 1e5
            );
        }
        XsMode::Wmp => {
            let (provider, xs_mem, load_ms) = load_wmp_hybrid(&args);
            let r = run_multi_seed(
                "ACE+WMP", &args, &surfaces, &cells, &materials, &provider, xs_mem, load_ms,
            );
            println!("\n{}", "=".repeat(60));
            println!("RESULTS");
            println!("{}", "=".repeat(60));
            print_benchmark(&r, args.particles);
        }
        XsMode::All => {
            let (svd_prov, svd_mem, svd_load) = load_svd(&args);
            let svd = run_multi_seed(
                &format!("SVD (rank={})", args.rank),
                &args,
                &surfaces,
                &cells,
                &materials,
                &svd_prov,
                svd_mem,
                svd_load,
            );
            drop(svd_prov);

            let (tbl_prov, tbl_mem, tbl_load) = load_table(&args);
            let tbl = run_multi_seed(
                "Pointwise Table",
                &args,
                &surfaces,
                &cells,
                &materials,
                &tbl_prov,
                tbl_mem,
                tbl_load,
            );
            drop(tbl_prov);

            let (wmp_prov, wmp_mem, wmp_load) = load_wmp_hybrid(&args);
            let wmp = run_multi_seed(
                "ACE+WMP", &args, &surfaces, &cells, &materials, &wmp_prov, wmp_mem, wmp_load,
            );

            println!("\n{}", "=".repeat(60));
            println!("STATISTICAL BENCHMARK — THREE-WAY HONESTY TEST");
            println!("{}", "=".repeat(60));
            print_benchmark(&svd, args.particles);
            println!();
            print_benchmark(&tbl, args.particles);
            println!();
            print_benchmark(&wmp, args.particles);

            println!("\n  {}", "-".repeat(50));
            println!("  COMPARISON (reference = ACE+WMP, industry baseline):");
            let svd_gap = (svd.k_eff_mean() - wmp.k_eff_mean()).abs() * 1e5;
            let tbl_gap = (tbl.k_eff_mean() - wmp.k_eff_mean()).abs() * 1e5;
            println!("    k_eff gap SVD vs WMP    = {svd_gap:.0} pcm");
            println!("    k_eff gap Table vs WMP  = {tbl_gap:.0} pcm");
            println!(
                "    ns/p  SVD / Table / WMP  = {:.1} / {:.1} / {:.1}",
                svd.ns_per_particle_mean(),
                tbl.ns_per_particle_mean(),
                wmp.ns_per_particle_mean()
            );
            println!(
                "    mem KB SVD / Table / WMP = {:.0} / {:.0} / {:.0}",
                svd_mem as f64 / 1024.0,
                tbl_mem as f64 / 1024.0,
                wmp_mem as f64 / 1024.0,
            );
        }
    }
}
