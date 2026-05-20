//! Heavy-water-moderated pin-cell demo.
//!
//! Same fuel + clad geometry as `pwr_pincell` (3.1 % UO₂, Zircaloy-4
//! clad), but the moderator is D₂O (heavy water) instead of H₂O.
//! Demonstrates the new ZAID-driven loader: every nuclide is resolved
//! by `(zaid, target_temperature)` against `NuclideLibrary`, and the
//! `D in D₂O` thermal-scattering data is loaded via `ThermalLibrary`
//! rather than a hand-built `data_dir.join("c_D_in_D2O.h5")` call.
//!
//! Heavy-water moderation gives a different physics regime from light
//! water: D has a much smaller absorption cross section than H (σ_a ≈
//! 0.5 mb at thermal vs ~330 mb for H), so the moderator-to-fuel ratio
//! at which the system goes critical is much smaller, and natural-
//! uranium fuel is fissile on D₂O moderation. With our 3.1 %
//! enrichment the pin runs heavily over-moderated, k_inf well above
//! the standard PWR value.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --bin pwr_d2o_pincell -- data/endfb-vii.1-hdf5/neutron \
//!     --rank 5 --batches 80 --inactive 20 --particles 5000 --seeds 3
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Geometry, Vec3};
use open_rust_mc::thermal::ThermalScatteringData;
use open_rust_mc::transport::dispatch::{CpuRunner, EigenvalueRunner};
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::nuclides::{NuclideLibrary, ResolvedNuclide};
use open_rust_mc::transport::simulate::SimConfig;
use open_rust_mc::transport::thermal_library::{ThermalBinding, ThermalLibrary};
use open_rust_mc::transport::xs_provider;

#[derive(Parser, Debug)]
#[command(
    name = "pwr_d2o_pincell",
    about = "Heavy-water-moderated UO₂ pin cell — NuclideLibrary + D-in-D₂O demo"
)]
struct Args {
    /// Directory containing the ENDF/B HDF5 files (neutron + thermal).
    data_dir: PathBuf,
    #[arg(short, long, default_value_t = 5)]
    rank: usize,
    #[arg(short, long, default_value_t = 80)]
    batches: u32,
    #[arg(short, long, default_value_t = 20)]
    inactive: u32,
    #[arg(short, long, default_value_t = 5000)]
    particles: u32,
    #[arg(short, long, default_value_t = 1)]
    seeds: u32,
    /// UO₂ enrichment (mass fraction U-235). 3.1 % matches the PWR
    /// reference; natural uranium is 0.72 %.
    #[arg(long, default_value_t = 0.031)]
    enrichment: f64,
    /// Lattice pitch (cm). 1.26 is the PWR / H₂O reference; CANDU
    /// uses ≈ 28.6 cm with natural-U fuel; ≥ ~3 cm is enough D₂O
    /// volume per pin to show correctly-moderated behaviour with
    /// 3.1 % enriched fuel.
    #[arg(long, default_value_t = 1.260)]
    pitch: f64,
}

// ── Geometry constants (cylinders identical to pwr_pincell; pitch
//    becomes a CLI knob since D₂O needs much more moderator per pin
//    than H₂O for adequate thermalisation). ───────────────────────────
const FUEL_OR: f64 = 0.4096;
const CLAD_IR: f64 = 0.4180;
const CLAD_OR: f64 = 0.4750;

// ── Material temperatures ────────────────────────────────────────────
const T_FUEL_K: f64 = 900.0;
const T_CLAD_K: f64 = 600.0;
const T_MOD_K: f64 = 600.0;

// ── ZAIDs (single source of truth — comes through the catalog) ───────
const ZAID_U235: u32 = 92235;
const ZAID_U238: u32 = 92238;
const ZAID_O16: u32 = 8016;
const ZAID_D: u32 = 1002; // deuterium (H-2)
const ZAID_ZR90: u32 = 40090;
const ZAID_ZR91: u32 = 40091;
const ZAID_ZR92: u32 = 40092;
const ZAID_ZR94: u32 = 40094;

fn build_uo2_atom_densities(enrichment_mass_fraction: f64) -> (f64, f64, f64) {
    // Density 10.4 g/cm³, M(UO₂) ≈ 270 g/mol, N_A = 6.022e23.
    // Number density of UO₂ molecules: ρ N_A / M = 10.4 × 6.022e23 / 270
    // ≈ 2.32e22 atoms/cm³ (per molecule). Per-isotope densities follow
    // from stoichiometry + the U-235/U-238 mass split.
    let n_uo2 = 10.4 * 6.022_140_76e23 / 270.0; // atoms/cm³
    let mass_to_atom = |m_frac: f64, awr_a: f64, awr_b: f64| {
        m_frac / awr_a / (m_frac / awr_a + (1.0 - m_frac) / awr_b)
    };
    let frac_u235 = mass_to_atom(enrichment_mass_fraction, 235.0, 238.0);
    let n_u = n_uo2;
    let n_u235 = n_u * frac_u235 / 1e24;
    let n_u238 = n_u * (1.0 - frac_u235) / 1e24;
    let n_o16 = 2.0 * n_u / 1e24;
    (n_u235, n_u238, n_o16)
}

fn setup_materials(enrichment: f64) -> Vec<Material> {
    let (n_u235, n_u238, n_o16_fuel) = build_uo2_atom_densities(enrichment);

    let mut fuel = Material::new("UO2", T_FUEL_K);
    fuel.add_nuclide(n_u235, 0); // xs_kernel_idx 0
    fuel.add_nuclide(n_u238, 1);
    fuel.add_nuclide(n_o16_fuel, 2);

    let mut clad = Material::new("Zircaloy", T_CLAD_K);
    // Same densities as the H₂O pin cell.
    clad.add_nuclide(2.2932e-2, 4); // Zr-90
    clad.add_nuclide(4.996e-3, 5); // Zr-91
    clad.add_nuclide(7.636e-3, 6); // Zr-92
    clad.add_nuclide(7.740e-3, 7); // Zr-94

    // D₂O at 600 K. Density ~0.83 g/cm³, M(D₂O) = 20.027 g/mol.
    // n_D2O = 0.83 × 6.022e23 / 20.027 ≈ 2.50e22 / cm³ (per molecule)
    // n_D = 2 × n_D2O ≈ 5.00e22 / cm³
    // n_O = 1 × n_D2O ≈ 2.50e22 / cm³
    let n_d2o = 0.83 * 6.022_140_76e23 / 20.027;
    let n_d = 2.0 * n_d2o / 1e24;
    let n_o16_water = n_d2o / 1e24;
    let mut d2o = Material::new("D2O", T_MOD_K);
    d2o.add_nuclide(n_d, 3); // D, xs_kernel_idx 3
    d2o.add_nuclide(n_o16_water, 8); // O-16 in moderator

    vec![fuel, clad, d2o]
}

fn setup_geometry(pitch_cm: f64) -> (Vec<Surface>, Vec<Cell>) {
    let half = pitch_cm / 2.0;
    let surfaces = vec![
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: FUEL_OR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_IR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_OR,
            bc: BoundaryCondition::Transmission,
        },
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
            z0: -half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneZ {
            z0: half,
            bc: BoundaryCondition::Reflective,
        },
    ];

    let aabb = Aabb::new(Vec3::new(-half, -half, -half), Vec3::new(half, half, half));
    // Inside-the-box region: x ∈ [-half, +half], y ∈ [-half, +half],
    // z ∈ [-half, +half]. Used as the axial bound for every material
    // cell so the reflective BC actually triggers when particles
    // approach the box faces (otherwise the radial cylinders extend
    // axially through ±∞ and particles never see the z-planes).
    let inside_box = || {
        cell::intersect_all(vec![
            cell::Region::HalfSpace {
                surface_idx: 3,
                positive: true,
            },
            cell::Region::HalfSpace {
                surface_idx: 4,
                positive: false,
            },
            cell::Region::HalfSpace {
                surface_idx: 5,
                positive: true,
            },
            cell::Region::HalfSpace {
                surface_idx: 6,
                positive: false,
            },
            cell::Region::HalfSpace {
                surface_idx: 7,
                positive: true,
            },
            cell::Region::HalfSpace {
                surface_idx: 8,
                positive: false,
            },
        ])
    };

    let cells = vec![
        Cell::new(
            CellId(0),
            cell::Region::Intersection(Box::new(cell::inside(0)), Box::new(inside_box())),
            CellFill::Material(0),
        )
        .with_aabb(aabb)
        .with_temperature(T_FUEL_K),
        Cell::new(
            CellId(1),
            cell::Region::Intersection(Box::new(cell::between(0, 1)), Box::new(inside_box())),
            CellFill::Void,
        ),
        Cell::new(
            CellId(2),
            cell::Region::Intersection(Box::new(cell::between(1, 2)), Box::new(inside_box())),
            CellFill::Material(1),
        )
        .with_temperature(T_CLAD_K),
        Cell::new(
            CellId(3),
            cell::Region::Intersection(Box::new(cell::outside(2)), Box::new(inside_box())),
            CellFill::Material(2),
        )
        .with_temperature(T_MOD_K),
    ];
    (surfaces, cells)
}

fn main() {
    let args = Args::parse();

    println!("============================================================");
    println!("  Heavy-water (D₂O) moderated UO₂ pin cell");
    println!("============================================================");
    println!("  Enrichment      : {:.2}% U-235", args.enrichment * 100.0);
    println!("  Fuel temperature: {T_FUEL_K} K");
    println!("  Mod  temperature: {T_MOD_K} K  (D₂O)");
    println!(
        "  Geometry        : pitch={:.3} cm, fuel OR={FUEL_OR} cm, clad OR={CLAD_OR} cm",
        args.pitch
    );

    // ── Resolve all nuclides through the library ──────────────────────
    let lib = NuclideLibrary::from_data_dir(&args.data_dir);
    let zaids: &[(u32, f64)] = &[
        (ZAID_U235, T_FUEL_K), // xs_idx 0
        (ZAID_U238, T_FUEL_K), // xs_idx 1
        (ZAID_O16, T_FUEL_K),  // xs_idx 2
        (ZAID_D, T_MOD_K),     // xs_idx 3
        (ZAID_ZR90, T_CLAD_K), // xs_idx 4
        (ZAID_ZR91, T_CLAD_K), // xs_idx 5
        (ZAID_ZR92, T_CLAD_K), // xs_idx 6
        (ZAID_ZR94, T_CLAD_K), // xs_idx 7
        (ZAID_O16, T_MOD_K),   // xs_idx 8 — second O-16 column at 600 K
    ];
    let resolved: Vec<ResolvedNuclide> =
        lib.resolve_many(zaids).expect("nuclide resolution failed");

    println!("\n── NuclideLibrary resolution ──");
    for (i, r) in resolved.iter().enumerate() {
        println!(
            "  [{i:>2}] zaid={:>5} {:<10} AWR={:.4}  T={:.0} K  (idx {} of {})",
            r.zaid,
            r.symbol,
            r.awr,
            r.temperature_k,
            r.temp_idx,
            r.temperatures_k.len()
        );
    }

    // ── Load nuclear data ─────────────────────────────────────────────
    let t0 = Instant::now();
    let kernels: Vec<_> = resolved
        .iter()
        .map(|r| xs_provider::load_nuclide(&r.path, args.rank, r.temp_idx, r.awr, r.nu_bar_const))
        .collect();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  loaded {} nuclides in {load_ms:.0} ms", kernels.len());

    // ── Heavy-water thermal scattering ────────────────────────────────
    let thermal_lib = ThermalLibrary::from_data_dir(&args.data_dir);
    let d2o_thermal: Option<Arc<ThermalScatteringData>> = thermal_lib
        .try_load(ThermalBinding::DInD2O)
        .expect("c_D_in_D2O.h5 read failed")
        .map(Arc::new);
    const KB_EV_PER_K: f64 = 8.617_333_262e-5;
    match &d2o_thermal {
        Some(t) => {
            let temps_k: Vec<f64> = t.kts.iter().map(|kt| kt / KB_EV_PER_K).collect();
            let t_min = temps_k.iter().cloned().fold(f64::INFINITY, f64::min);
            let t_max = temps_k.iter().cloned().fold(0.0_f64, f64::max);
            println!(
                "  Thermal: D in D₂O loaded ({} columns, {:.0}–{:.0} K, E_max = {:.3} eV)",
                temps_k.len(),
                t_min,
                t_max,
                t.energy_max,
            );
        }
        None => {
            println!("  Thermal: c_D_in_D2O.h5 not found — falling back to free-atom elastic");
        }
    }

    let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; resolved.len()];
    if let Some(t) = d2o_thermal {
        // Bind D-in-D₂O to the deuterium nuclide slot (xs_idx 3).
        thermal[3] = Some(t);
    }

    let xs_provider = xs_provider::SvdXsProvider {
        nuclides: kernels.into_iter().map(std::sync::Arc::new).collect(),
        thermal,
    };

    let materials = setup_materials(args.enrichment);
    let (surfaces, cells) = setup_geometry(args.pitch);
    let geometry =
        Geometry::from_slices(&surfaces, &cells).expect("d2o pin cell geometry must validate");

    // ── Source iteration ─────────────────────────────────────────────
    let inactive = args.inactive.min(args.batches.saturating_sub(1));
    let total_histories = (args.batches - inactive) as u64 * args.particles as u64;

    let mut k_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);
    let mut t_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);

    println!("\n── Source iteration ──");

    for seed in 0..args.seeds {
        let config = SimConfig {
            batches: args.batches,
            inactive,
            particles_per_batch: args.particles,
            seed: seed as u64,
            auto_inactive: None,
            verbose: false,
            parallel: true,
            tallies: Default::default(),
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
            gpu_refill_pool_factor: None,
            gpu_auto_refill: false,
        };
        let runner = CpuRunner {
            geometry: &geometry,
            materials: &materials,
            xs_provider: &xs_provider,
        };
        let t1 = Instant::now();
        let outcome = runner.run(&config);
        let sim_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let active: Vec<f64> = outcome
            .batches
            .iter()
            .filter(|r| r.active)
            .map(|r| r.k_eff)
            .collect();
        let n_active = active.len() as f64;
        let k_mean = active.iter().sum::<f64>() / n_active;
        k_per_seed.push(k_mean);
        t_per_seed.push(sim_ms);

        println!("  Seed {seed}: k_inf={k_mean:.5}   {sim_ms:.0} ms");
    }

    let n = k_per_seed.len() as f64;
    let k_mean = k_per_seed.iter().sum::<f64>() / n;
    let k_var = k_per_seed.iter().map(|k| (k - k_mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let k_std = k_var.sqrt();
    let total_t_ms: f64 = t_per_seed.iter().sum();
    let ns_per_p = total_t_ms * 1e6 / (total_histories * args.seeds as u64) as f64;

    println!("\n============================================================");
    println!(
        "  RESULT — D₂O pin cell ({:.2}% U-235)",
        args.enrichment * 100.0
    );
    println!("============================================================");
    println!(
        "  k_inf       = {k_mean:.5} +/- {k_std:.5}  ({} seeds)",
        args.seeds
    );
    println!(
        "  Histories   = {} per seed × {} seeds",
        total_histories, args.seeds
    );
    println!("  Sim time    = {total_t_ms:.0} ms");
    println!("  ns/particle = {ns_per_p:.1}");
}
