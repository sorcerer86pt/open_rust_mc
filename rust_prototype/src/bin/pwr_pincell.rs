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
use open_rust_mc::transport::hybrid_xs::{HybridSvdWmpXsProvider, HybridTableWmpXsProvider};
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::{self, SimConfig, XsProvider};
use open_rust_mc::transport::xs_provider;
use open_rust_mc::wmp::WindowedMultipole;

#[derive(clap::ValueEnum, Clone, Debug)]
enum XsMode {
    Svd,
    Table,
    Both,
    /// SVD + Windowed Multipole hybrid (kept for regression comparison).
    Hybrid,
    /// ACE pointwise table + Windowed Multipole — industry baseline.
    Wmp,
    /// Four-way honesty test: SVD vs Table vs SVD+WMP vs ACE+WMP.
    All,
}

/// WMP filename (ZZAAA.h5) and target temperature (K) per nuclide,
/// parallel to NUCLIDE_SPECS. None = no WMP coverage for this nuclide.
/// Temperatures match the `temp_idx` in NUCLIDE_SPECS:
///   0=250K, 1=294K, 2=600K, 3=900K, 4=1200K, 5=2500K.
const WMP_SPECS: &[(&str, f64)] = &[
    ("092235.h5", 900.0), // 0  U235 fuel
    ("092238.h5", 900.0), // 1  U238 fuel
    ("008016.h5", 900.0), // 2  O16 fuel
    ("001001.h5", 600.0), // 3  H1 water
    ("040090.h5", 600.0), // 4  Zr90 clad
    ("040091.h5", 600.0), // 5  Zr91 clad
    ("040092.h5", 600.0), // 6  Zr92 clad
    ("040094.h5", 600.0), // 7  Zr94 clad
    ("008016.h5", 600.0), // 8  O16 water
];

#[derive(Parser)]
#[command(
    name = "pwr_pincell",
    about = "PWR pin cell benchmark (multi-material, multi-nuclide)"
)]
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

    /// Replace the fixed `--inactive` count with runtime Shannon-entropy
    /// plateau detection. See EntropyConvergence::default for thresholds.
    #[arg(long, default_value_t = false)]
    auto_inactive: bool,

    /// Operating-temperature offset added to every nuclide's library
    /// temperature (K). Shortcut for --fuel-offset and --mod-offset.
    #[arg(long)]
    target_temp_offset: Option<f64>,

    /// Offset (K) applied only to fuel nuclides (U-235, U-238, O-16 in
    /// fuel; NUCLIDE_SPECS indices 0, 1, 2). Isolates fuel-side Doppler
    /// from moderator-side thermal-scattering effects for PWR off-
    /// library diagnostics.
    #[arg(long)]
    fuel_offset: Option<f64>,

    /// Offset (K) applied only to moderator and clad nuclides
    /// (H-1, Zr-90..94, O-16 in water; NUCLIDE_SPECS indices 3..=8).
    #[arg(long)]
    mod_offset: Option<f64>,

    /// Override the SVD rank used for discrete inelastic levels
    /// (MT=51-91). rank=1 captures them (weak T-dependence).
    #[arg(long)]
    discrete_rank: Option<usize>,
}

/// Nuclide specs: (filename, AWR, fallback nu-bar, temp_idx).
/// Index in this array = xs_kernel_idx used by materials.
/// temp_idx after numeric sort: 0=250K, 1=294K, 2=600K, 3=900K, 4=1200K, 5=2500K
/// O16 is duplicated: idx 2 at 900K for fuel, idx 8 at 600K for water.
const NUCLIDE_SPECS: &[(&str, f64, f64, usize)] = &[
    ("U235.h5", 233.025, 2.43, 3), // 0  fuel: 900K
    ("U238.h5", 236.006, 2.49, 3), // 1  fuel: 900K
    ("O16.h5", 15.858, 0.0, 3),    // 2  fuel O16: 900K
    ("H1.h5", 0.999, 0.0, 2),      // 3  water: 600K
    ("Zr90.h5", 89.132, 0.0, 2),   // 4  clad: 600K
    ("Zr91.h5", 90.130, 0.0, 2),   // 5  clad: 600K
    ("Zr92.h5", 91.126, 0.0, 2),   // 6  clad: 600K
    ("Zr94.h5", 93.120, 0.0, 2),   // 7  clad: 600K
    ("O16.h5", 15.858, 0.0, 2),    // 8  water O16: 600K
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

fn setup_geometry() -> (Vec<Surface>, Vec<Cell>) {
    // Standard PWR pin cell dimensions
    let fuel_or = 0.4096; // cm, fuel outer radius
    let clad_ir = 0.4180; // cm, clad inner radius
    let clad_or = 0.4750; // cm, clad outer radius
    let pitch = 1.2600; // cm, pin pitch
    let half = pitch / 2.0;

    let z_half = half; // Use same extent in Z for reflective bounding

    let surfaces = vec![
        // 0: fuel outer cylinder
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: fuel_or,
            bc: BoundaryCondition::Transmission,
        },
        // 1: clad inner cylinder
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: clad_ir,
            bc: BoundaryCondition::Transmission,
        },
        // 2: clad outer cylinder
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: clad_or,
            bc: BoundaryCondition::Transmission,
        },
        // 3: -X plane
        Surface::PlaneX {
            x0: -half,
            bc: BoundaryCondition::Reflective,
        },
        // 4: +X plane
        Surface::PlaneX {
            x0: half,
            bc: BoundaryCondition::Reflective,
        },
        // 5: -Y plane
        Surface::PlaneY {
            y0: -half,
            bc: BoundaryCondition::Reflective,
        },
        // 6: +Y plane
        Surface::PlaneY {
            y0: half,
            bc: BoundaryCondition::Reflective,
        },
        // 7: -Z plane (reflective — infinite lattice in Z)
        Surface::PlaneZ {
            z0: -z_half,
            bc: BoundaryCondition::Reflective,
        },
        // 8: +Z plane
        Surface::PlaneZ {
            z0: z_half,
            bc: BoundaryCondition::Reflective,
        },
    ];

    let box_aabb = Aabb::new(
        Vec3::new(-half, -half, -z_half),
        Vec3::new(half, half, z_half),
    );

    let cells = vec![
        // 0: Fuel (inside fuel cylinder, inside Z box)
        Cell::new(
            CellId(0),
            cell::intersect_all(vec![cell::inside(0), cell::outside(7), cell::inside(8)]),
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
                cell::outside(0),
                cell::inside(1),
                cell::outside(7),
                cell::inside(8),
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
                cell::outside(1),
                cell::inside(2),
                cell::outside(7),
                cell::inside(8),
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
                cell::outside(2), // outside clad
                cell::outside(3), // x > -half
                cell::inside(4),  // x < +half
                cell::outside(5), // y > -half
                cell::inside(6),  // y < +half
                cell::outside(7), // z > -z_half
                cell::inside(8),  // z < +z_half
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
    // Atom densities from OpenMC (atoms/barn-cm)
    let mut fuel = Material::new("UO2", 900.0);
    fuel.add_nuclide(0.000719, 0); // U-235  (xs_kernel_idx=0)
    fuel.add_nuclide(0.022482, 1); // U-238  (xs_kernel_idx=1)
    fuel.add_nuclide(0.046402, 2); // O-16   (xs_kernel_idx=2)

    // Material 1: Zircaloy-4 cladding (6.55 g/cm³)
    let mut clad = Material::new("Zircaloy", 600.0);
    clad.add_nuclide(0.022932, 4); // Zr-90  (xs_kernel_idx=4)
    clad.add_nuclide(0.004996, 5); // Zr-91  (xs_kernel_idx=5)
    clad.add_nuclide(0.007636, 6); // Zr-92  (xs_kernel_idx=6)
    clad.add_nuclide(0.007740, 7); // Zr-94  (xs_kernel_idx=7)

    // Material 2: Light water (0.74 g/cm³, 600K)
    let mut water = Material::new("H2O", 600.0);
    water.add_nuclide(0.049486, 3); // H-1    (xs_kernel_idx=3)
    water.add_nuclide(0.024743, 8); // O-16   (xs_kernel_idx=8, at 600K)

    vec![fuel, clad, water]
}

/// Load thermal scattering data and build the per-nuclide thermal vector.
///
/// H1 (xs_kernel_idx=3) gets c_H_in_H2O thermal data if available.
fn load_thermal(data_dir: &std::path::Path) -> Vec<Option<Arc<ThermalScatteringData>>> {
    let h2o_path = data_dir.join("c_H_in_H2O.h5");
    let h2o_thermal: Option<Arc<ThermalScatteringData>> = if h2o_path.exists() {
        match hdf5_reader::load_thermal_scattering(&h2o_path) {
            Ok(tsl) => {
                println!(
                    "  S(a,b): loaded {} ({} temperatures)",
                    tsl.name,
                    tsl.temp_labels.len()
                );
                Some(Arc::new(tsl))
            }
            Err(e) => {
                eprintln!(
                    "  WARNING: failed to load S(a,b) from {}: {e}",
                    h2o_path.display()
                );
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

/// Resolve the temperature offset (K) to apply for NUCLIDE_SPECS index
/// `nuc_idx`. Fuel (0, 1, 2) vs moderator/clad (3..=8) split. If no
/// per-group knob is set, falls back to the global `--target-temp-offset`.
/// Returns `None` when no offset applies (use legacy single-T loader).
fn resolve_offset_for(nuc_idx: usize, args: &Args) -> Option<f64> {
    let is_fuel = matches!(nuc_idx, 0..=2);
    let specific = if is_fuel {
        args.fuel_offset
    } else {
        args.mod_offset
    };
    specific.or(args.target_temp_offset)
}

fn load_svd(args: &Args) -> (xs_provider::SvdXsProvider, usize, f64) {
    println!("\n── Loading nuclear data (SVD, rank={}) ──", args.rank);
    let t0 = Instant::now();

    let mut kernels = Vec::new();
    for (nuc_idx, &(filename, awr, nu_bar, nuc_temp_idx)) in NUCLIDE_SPECS.iter().enumerate() {
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
                discrete_level_angles: vec![],
                has_continuum_inelastic: false,
                elastic_angle: None,
                fission_energy_dist: None,
                urr_tables: None,
            });
        } else {
            let offset_here = resolve_offset_for(nuc_idx, args);
            // Route through the at-temp loader when any offset or
            // --discrete-rank is set; otherwise keep the legacy
            // single-T fast path for bit-for-bit backwards compat.
            match (offset_here, args.discrete_rank) {
                (None, None) => kernels.push(xs_provider::load_nuclide(
                    &path,
                    args.rank,
                    nuc_temp_idx,
                    awr,
                    nu_bar,
                )),
                (offset, drank) => {
                    let base_t = open_rust_mc::hdf5_reader::NuclideFileReader::open(&path)
                        .ok()
                        .and_then(|r| r.temperatures.get(nuc_temp_idx).copied())
                        .unwrap_or(294.0);
                    let target = base_t + offset.unwrap_or(0.0);
                    kernels.push(xs_provider::load_nuclide_at_temp(
                        &path, args.rank, target, awr, nu_bar, drank,
                    ));
                }
            }
        }
    }

    let thermal = load_thermal(&args.data_dir);
    let xs_mem: usize = kernels.iter().map(|k| k.svd_memory_bytes()).sum();
    let provider = xs_provider::SvdXsProvider {
        nuclides: kernels,
        thermal,
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB ({} nuclides)",
        xs_mem as f64 / 1024.0,
        NUCLIDE_SPECS.len()
    );

    (provider, xs_mem, load_ms)
}

fn load_hybrid(args: &Args) -> (HybridSvdWmpXsProvider, usize, f64) {
    println!(
        "\n── Loading nuclear data (Hybrid SVD rank={} + WMP) ──",
        args.rank
    );
    let t0 = Instant::now();

    // First build the SVD provider exactly as `load_svd` does.
    let (svd_provider, _svd_mem, _) = load_svd(args);

    // Now load per-nuclide WMP data.
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
                    "  Loaded WMP {wmp_file} at T={t_kelvin} K   \
                    (E {:.2e}..{:.2e} eV, {} poles, {} windows)",
                    wmp.e_min, wmp.e_max, wmp.n_poles, wmp.n_windows
                );
                covered += 1;
                wmps.push(Some((Arc::new(wmp), t_kelvin)));
            }
            Err(e) => {
                println!("  WMP load failed for {wmp_file}: {e:?}");
                wmps.push(None);
            }
        }
    }

    let provider = HybridSvdWmpXsProvider::new(svd_provider, wmps);
    let report = provider.memory_report();
    let total_mem = report.current_total();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Hybrid ready in {load_ms:.0} ms  ({} / {} nuclides with WMP)",
        covered,
        WMP_SPECS.len()
    );
    println!("  Memory (current scaffolding):");
    println!(
        "    full SVD basis     = {:.1} KB",
        report.current_svd_bytes as f64 / 1024.0
    );
    println!(
        "    WMP payload        = {:.1} KB",
        report.wmp_payload_bytes as f64 / 1024.0
    );
    println!(
        "    TOTAL (current)    = {:.1} KB",
        report.current_total() as f64 / 1024.0
    );
    println!("  Memory (smooth-only projection, measured from loaded data):");
    println!(
        "    smooth-only SVD    = {:.1} KB",
        report.smooth_only_svd_bytes as f64 / 1024.0
    );
    println!(
        "    WMP payload        = {:.1} KB",
        report.wmp_payload_bytes as f64 / 1024.0
    );
    println!(
        "    TOTAL (projected)  = {:.1} KB",
        report.smooth_only_total() as f64 / 1024.0
    );
    println!(
        "    reduction vs full  = {:.1}x",
        report.current_total() as f64 / report.smooth_only_total() as f64
    );

    (provider, total_mem, load_ms)
}

/// Load ACE pointwise table + WMP in the resolved-resonance window.
/// Industry baseline — matches godiva's `--mode wmp`.
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
                println!("  WMP load failed for {wmp_file}: {e:?}");
                wmps.push(None);
            }
        }
    }

    let provider = HybridTableWmpXsProvider::new(table_provider, wmps);
    let report = provider.memory_report();
    let xs_mem = report.smooth_only_total();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  WMP covers {covered}/{} nuclides", WMP_SPECS.len());
    println!(
        "  Loaded in {load_ms:.0} ms  |  XS memory (smooth-only): {:.1} KB  [in-solver table total: {:.1} KB]",
        xs_mem as f64 / 1024.0,
        table_mem as f64 / 1024.0,
    );
    (provider, xs_mem, load_ms)
}

fn load_table(args: &Args) -> (xs_provider::TableXsProvider, usize, f64) {
    println!("\n── Loading nuclear data (pointwise table) ──");
    let t0 = Instant::now();

    let mut tables = Vec::new();
    for (nuc_idx, &(filename, awr, nu_bar, nuc_temp_idx)) in NUCLIDE_SPECS.iter().enumerate() {
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
                urr_tables: None,
            });
        } else {
            match resolve_offset_for(nuc_idx, args) {
                None => tables.push(xs_provider::load_nuclide_table(
                    &path,
                    nuc_temp_idx,
                    awr,
                    nu_bar,
                )),
                Some(offset) => {
                    let base_t = open_rust_mc::hdf5_reader::NuclideFileReader::open(&path)
                        .ok()
                        .and_then(|r| r.temperatures.get(nuc_temp_idx).copied())
                        .unwrap_or(294.0);
                    let target = base_t + offset;
                    tables.push(xs_provider::load_nuclide_table_at_temp(
                        &path, target, awr, nu_bar,
                    ));
                }
            }
        }
    }

    let thermal = load_thermal(&args.data_dir);
    let xs_mem: usize = tables.iter().map(|t| t.table_memory_bytes()).sum();
    let provider = xs_provider::TableXsProvider {
        nuclides: tables,
        thermal,
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB ({} nuclides)",
        xs_mem as f64 / 1024.0,
        NUCLIDE_SPECS.len()
    );

    (provider, xs_mem, load_ms)
}

fn print_benchmark(r: &BenchmarkResult) {
    let n_seeds = r.seed_results.len();

    println!("  {}:", r.label);
    println!(
        "    k_inf            = {:.5} +/- {:.5}",
        r.k_eff_mean(),
        r.k_eff_std()
    );
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

    println!("=== open_rust_mc — PWR Pin Cell Benchmark ===\n");
    println!("Data dir:     {}", args.data_dir.display());
    println!("Mode:         {:?}", args.mode);
    if matches!(args.mode, XsMode::Svd | XsMode::Both) {
        println!("SVD rank:     {}", args.rank);
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
    println!(
        "Nuclides:     {} (U235, U238, O16, H1, Zr90-94)",
        NUCLIDE_SPECS.len()
    );
    println!("Materials:    3 (UO2 fuel, Zircaloy clad, H2O moderator)");
    if args.seeds > 1 {
        println!(
            "Seeds:        {} (independent runs for statistical confidence)",
            args.seeds
        );
    }
    println!("S(a,b):       c_H_in_H2O (if available in data_dir)");

    let (surfaces, cells) = setup_geometry();
    let materials = setup_materials();

    match args.mode {
        XsMode::Svd => {
            let (provider, xs_mem, load_ms) = load_svd(&args);
            let r = run_multi_seed(
                &format!("SVD (rank={})", args.rank),
                &args,
                &surfaces,
                &cells,
                &materials,
                &provider,
                xs_mem,
                load_ms,
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
                &args,
                &surfaces,
                &cells,
                &materials,
                &provider,
                xs_mem,
                load_ms,
            );
            println!("\n{}", "=".repeat(60));
            println!("RESULTS — PWR Pin Cell");
            println!("{}", "=".repeat(60));
            print_benchmark(&r);
        }
        XsMode::Hybrid => {
            let (provider, xs_mem, load_ms) = load_hybrid(&args);
            let r = run_multi_seed(
                &format!("Hybrid SVD(rank={})+WMP", args.rank),
                &args,
                &surfaces,
                &cells,
                &materials,
                &provider,
                xs_mem,
                load_ms,
            );
            println!("\n{}", "=".repeat(60));
            println!("RESULTS — PWR Pin Cell (Hybrid SVD + WMP)");
            println!("{}", "=".repeat(60));
            print_benchmark(&r);
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
            println!(
                "    SVD speedup              = {speedup:.2}x ({:.2} vs {:.2} ns/particle)",
                svd.ns_per_particle_mean(),
                tbl.ns_per_particle_mean()
            );
            if args.seeds > 1 {
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
            println!(
                "    Memory ratio (tbl/svd)   = {:.2}x ({:.1} KB vs {:.1} KB)",
                tbl.xs_memory_bytes as f64 / svd.xs_memory_bytes as f64,
                svd.xs_memory_bytes as f64 / 1024.0,
                tbl.xs_memory_bytes as f64 / 1024.0
            );
        }
        XsMode::Wmp => {
            let (provider, xs_mem, load_ms) = load_wmp_hybrid(&args);
            let r = run_multi_seed(
                "ACE+WMP", &args, &surfaces, &cells, &materials, &provider, xs_mem, load_ms,
            );
            println!("\n{}", "=".repeat(60));
            println!("RESULTS — PWR Pin Cell (ACE+WMP baseline)");
            println!("{}", "=".repeat(60));
            print_benchmark(&r);
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
            println!("PWR PIN CELL — THREE-WAY HONESTY TEST");
            println!("{}", "=".repeat(60));
            print_benchmark(&svd);
            println!();
            print_benchmark(&tbl);
            println!();
            print_benchmark(&wmp);

            println!("\n  {}", "-".repeat(50));
            println!("  COMPARISON (reference = ACE+WMP, industry baseline):");
            let svd_gap = (svd.k_eff_mean() - wmp.k_eff_mean()).abs() * 1e5;
            let tbl_gap = (tbl.k_eff_mean() - wmp.k_eff_mean()).abs() * 1e5;
            println!("    k_inf gap SVD vs WMP    = {svd_gap:.0} pcm");
            println!("    k_inf gap Table vs WMP  = {tbl_gap:.0} pcm");
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
