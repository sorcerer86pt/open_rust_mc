//! End-to-end Godiva eigenvalue simulation in pure Rust.
//!
//! Reads U-234, U-235, U-238 HDF5 files, builds SVD kernels,
//! sets up the Godiva geometry, and runs k-eigenvalue power iteration.
//!
//! Usage:
//!   godiva <data_dir> [--rank K] [--batches N] [--particles N]
//!
//! Example:
//!   godiva path/to/endfb-vii.1-hdf5/neutron --rank 5 --batches 50

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3};
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::{self, SimConfig};
use open_rust_mc::transport::xs_provider::{self, SvdXsProvider};

#[derive(Parser)]
#[command(name = "godiva", about = "Godiva k-eigenvalue benchmark (pure Rust + SVD)")]
struct Args {
    /// Directory containing nuclide HDF5 files (U234.h5, U235.h5, U238.h5).
    data_dir: PathBuf,

    /// SVD truncation rank.
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
}

fn main() {
    let args = Args::parse();

    println!("=== open_rust_mc — Godiva Eigenvalue Simulation ===\n");
    println!("Data dir: {}", args.data_dir.display());
    println!("SVD rank: {}", args.rank);
    println!("Batches: {} ({} inactive + {} active)",
             args.batches, args.inactive, args.batches - args.inactive);
    println!("Particles/batch: {}", args.particles);

    // ── Load nuclear data ───────────────────────────────────────────
    let t0 = Instant::now();

    println!("\nLoading nuclear data...");

    // Nuclide   AWR        nu_bar
    // U-234     232.029    2.49 (fast fission)
    // U-235     233.025    2.43 (thermal/fast)
    // U-238     236.006    2.49 (fast fission, threshold ~1 MeV)
    let nuclide_specs: Vec<(&str, f64, f64)> = vec![
        ("U234.h5", 232.029, 2.49),
        ("U235.h5", 233.025, 2.43),
        ("U238.h5", 236.006, 2.49),
    ];

    let mut nuclide_kernels = Vec::new();
    let mut nuclide_names = Vec::new();

    for (filename, awr, nu_bar) in &nuclide_specs {
        let path = args.data_dir.join(filename);
        if !path.exists() {
            eprintln!("  WARNING: {} not found, using zero cross-sections", path.display());
            nuclide_kernels.push(xs_provider::NuclideKernels {
                elastic: None,
                inelastic: None,
                n2n: None,
                n3n: None,
                fission: None,
                capture: None,
                awr: *awr,
                nu_bar_const: *nu_bar,
                nu_bar_table: None,
                discrete_levels: vec![],
                has_continuum_inelastic: false,
            });
        } else {
            nuclide_kernels.push(xs_provider::load_nuclide(
                &path, args.rank, args.temp_idx, *awr, *nu_bar,
            ));
        }
        nuclide_names.push(filename.replace(".h5", ""));
    }

    let xs_provider = SvdXsProvider { nuclides: nuclide_kernels };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Data loaded in {load_ms:.0} ms");

    // ── Geometry: Godiva (ICSBEP HEU-MET-FAST-001) ─────────────────
    println!("\nSetting up Godiva geometry...");

    let radius = 8.7407; // cm, critical radius

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

    // HEU material: 93.5% U-235, 5.5% U-238, 1% U-234
    // Atom densities for 18.74 g/cm³ HEU
    let mut heu = Material::new("HEU", 294.0);
    heu.add_nuclide(0.000483, 0);  // U-234: N = 4.83e-4 atoms/barn-cm
    heu.add_nuclide(0.04509, 1);   // U-235: N = 4.509e-2 atoms/barn-cm
    heu.add_nuclide(0.00265, 2);   // U-238: N = 2.65e-3 atoms/barn-cm

    let materials = vec![heu];

    println!("  Sphere: R = {radius} cm");
    println!("  Material: HEU (93.5% U-235, 5.5% U-238, 1% U-234)");
    println!("  Density: 18.74 g/cm³");

    // ── Run eigenvalue simulation ───────────────────────────────────
    let config = SimConfig {
        batches: args.batches,
        inactive: args.inactive,
        particles_per_batch: args.particles,
    };

    println!("\nRunning eigenvalue simulation...\n");
    let t1 = Instant::now();

    let (results, _k_final) = simulate::run_eigenvalue(
        &config, &surfaces, &cells, &materials, &xs_provider,
    );

    let sim_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // ── Results ─────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(60));
    println!("RESULTS");
    println!("{}", "=".repeat(60));

    // Compute standard deviation from active batches
    let active_results: Vec<f64> = results.iter()
        .filter(|r| r.batch > args.inactive)
        .map(|r| r.k_eff)
        .collect();

    let n_active = active_results.len() as f64;
    let k_mean = active_results.iter().sum::<f64>() / n_active;
    let k_var = active_results.iter()
        .map(|&k| (k - k_mean).powi(2))
        .sum::<f64>() / (n_active * (n_active - 1.0));
    let k_std = k_var.sqrt();

    println!("\n  k_eff = {k_mean:.5} +/- {k_std:.5}");
    println!("  ({} active batches, {} particles/batch)", active_results.len(), args.particles);
    println!("  Simulation time: {sim_ms:.0} ms");
    println!("  Total time: {:.0} ms (load + sim)", t0.elapsed().as_secs_f64() * 1000.0);

    // Compare with experimental value
    let k_exp = 1.0000;
    let delta_pcm = (k_mean - k_exp).abs() / k_exp * 1e5;
    println!("\n  Experimental k_eff: {k_exp:.5}");
    println!("  Delta: {delta_pcm:.0} pcm");
    println!("  MC uncertainty: {:.0} pcm", k_std / k_mean * 1e5);
}
