#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::manual_is_multiple_of,
    clippy::needless_borrow
)]
//! Integrated PWR pin-cell burnup driver — eigenvalue → flux → CRAM
//! → composition update → repeat.
//!
//! Demonstrates the depletion module wired to the live transport
//! engine. Each burnup step:
//!
//!   1. Run a short eigenvalue solve on the current composition
//!      (with mesh flux tally over the fuel region).
//!   2. Extract the mean fuel-cell flux per source neutron.
//!   3. Compute the source rate `Q` from a target linear power.
//!   4. Build the depletion matrix at the resulting physical flux,
//!      using one-group thermal-XS values pulled from the same
//!      `XsProvider` that drives transport.
//!   5. CRAM-16 step → updated chain composition.
//!   6. Push Xe-135 atom density back into the fuel `Material`.
//!   7. Print `t, k_eff, N_Xe / N_U235`.
//!
//! Scope: this is the smallest end-to-end depletion demo that
//! exercises the full feedback loop. The chain tracks U-235, I-135,
//! Xe-135, Cs-135 only — enough to see the textbook k_eff drop as
//! Xe-135 reaches equilibrium. Multi-nuclide PWR depletion (Pu-239
//! buildup, full fission-product chain, multi-group XS collapse)
//! is the natural follow-on.
//!
//! Usage:
//!   deplete_pwr <data_dir> [--steps N] [--hours-per-step H]
//!                          [--power-w-per-cm P] [--rank K]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::depletion::{
    BurnupMapping, E_PER_FISSION_J,
    chain::{DecayBranch, DepletionChain, NuclideEntry, u235_thermal_iodine_xenon_yields},
    chain_io::ChainSpec,
    cram::CramOrder,
    deplete_ce_li, mean_fissions_per_source, mean_flux_per_source, power_normalized_source,
};
use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3};
use open_rust_mc::hdf5_reader;
use open_rust_mc::thermal::ThermalScatteringData;
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::{self, SimConfig, XsProvider};
use open_rust_mc::transport::tally::{MeshFluxTally, Tallies};
use open_rust_mc::transport::xs_provider;

#[derive(Parser, Debug)]
#[command(name = "deplete_pwr", about = "PWR pin cell burnup with CRAM-16")]
struct Args {
    /// Directory containing nuclide HDF5 files. Must include
    /// U235, U238, O16, H1, Zr90-94, plus Xe135.
    data_dir: PathBuf,

    /// Number of burnup steps.
    #[arg(long, default_value_t = 8)]
    steps: u32,

    /// Wall-clock burnup hours per step. 5 h × 8 = 40 h is enough
    /// to see Xe-135 reach equilibrium.
    #[arg(long, default_value_t = 5.0)]
    hours_per_step: f64,

    /// Target linear power in W/cm of axial pin length. The pin
    /// cell volume is `pitch² × Δz` ≈ `1.26² × 1.26` ≈ 2.0 cm³ for
    /// the default geometry; per-cm power 200 W is typical PWR.
    #[arg(long, default_value_t = 200.0)]
    power_w_per_cm: f64,

    /// SVD rank for the cross-section provider.
    #[arg(long, default_value_t = 5)]
    rank: usize,

    /// Active particles per batch in the eigenvalue solve.
    #[arg(long, default_value_t = 5_000)]
    particles: u32,

    /// Total batches (active + inactive) per eigenvalue step.
    #[arg(long, default_value_t = 50)]
    batches: u32,

    #[arg(long, default_value_t = 15)]
    inactive: u32,

    /// CRAM approximation order. 16 is the default (sufficient for
    /// PWR-typical Δt at < 1e-10 precision); 48 is for stiff
    /// activation chains, shutdown decay heat, or geologic-scale Δt.
    #[arg(long, value_enum, default_value_t = CramOrderArg::Order16)]
    cram_order: CramOrderArg,

    /// Path to a JSON chain file (see `chains/partial_xe.json` for
    /// the schema). When omitted, the embedded partial Xe-only chain
    /// is used. Pass a custom JSON file to swap in any chain library
    /// — actinide buildup, full fission-product set, decay heat,
    /// activation. Schema documented on `depletion::chain_io::ChainSpec`.
    #[arg(long)]
    chain: Option<PathBuf>,

    /// (Reserved.) Switches to a "full" chain library when one is
    /// shipped in-tree. Currently only the embedded partial chain or
    /// a user-supplied `--chain <path>` are supported.
    #[arg(long, default_value_t = false)]
    full_chain: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum CramOrderArg {
    #[value(name = "16")]
    Order16,
    #[value(name = "48")]
    Order48,
}

impl From<CramOrderArg> for CramOrder {
    fn from(a: CramOrderArg) -> Self {
        match a {
            CramOrderArg::Order16 => CramOrder::Cram16,
            CramOrderArg::Order48 => CramOrder::Cram48,
        }
    }
}

// xs_kernel_idx layout (parallel to NUCLIDE_SPECS).
const IDX_U235: usize = 0;
const IDX_U238: usize = 1;
const IDX_O16_FUEL: usize = 2;
const IDX_H1: usize = 3;
const IDX_ZR90: usize = 4;
const IDX_ZR91: usize = 5;
const IDX_ZR92: usize = 6;
const IDX_ZR94: usize = 7;
const IDX_O16_WATER: usize = 8;
const IDX_XE135: usize = 9;
// Actinide chain + Sm-149 chain — populated when the user passes
// `--chain pwr_actinides.json`. Loaded for every depletion run; the
// initial atom densities are zero on the transport side and grow via
// the BurnupMapping push from CRAM.
const IDX_U236: usize = 10;
const IDX_U237: usize = 11;
const IDX_U239: usize = 12;
const IDX_NP237: usize = 13;
const IDX_NP239: usize = 14;
const IDX_PU239: usize = 15;
const IDX_PU240: usize = 16;
const IDX_PU241: usize = 17;
const IDX_PU242: usize = 18;
const IDX_AM241: usize = 19;
const IDX_I135: usize = 20;
const IDX_CS135: usize = 21;
const IDX_PM149: usize = 22;
const IDX_SM149: usize = 23;

const NUCLIDE_SPECS: &[(&str, f64, f64, usize)] = &[
    ("U235.h5", 233.025, 2.43, 3),
    ("U238.h5", 236.006, 2.49, 3),
    ("O16.h5", 15.858, 0.0, 3),
    ("H1.h5", 0.999, 0.0, 2),
    ("Zr90.h5", 89.132, 0.0, 2),
    ("Zr91.h5", 90.130, 0.0, 2),
    ("Zr92.h5", 91.126, 0.0, 2),
    ("Zr94.h5", 93.120, 0.0, 2),
    ("O16.h5", 15.858, 0.0, 2),
    ("Xe135.h5", 133.748, 0.0, 2),
    // pwr_actinides chain — initial density 0, fed by CRAM each step.
    ("U236.h5", 234.018, 0.0, 3),
    ("U237.h5", 235.012, 0.0, 3),
    ("U239.h5", 237.001, 0.0, 3),
    ("Np237.h5", 235.012, 0.0, 3),
    ("Np239.h5", 236.999, 0.0, 3),
    ("Pu239.h5", 236.999, 2.88, 3),
    ("Pu240.h5", 237.992, 2.79, 3),
    ("Pu241.h5", 238.978, 2.95, 3),
    ("Pu242.h5", 239.979, 2.81, 3),
    ("Am241.h5", 238.986, 0.0, 3),
    ("I135.h5", 133.750, 0.0, 3),
    ("Cs135.h5", 133.747, 0.0, 3),
    ("Pm149.h5", 147.639, 0.0, 3),
    ("Sm149.h5", 147.638, 0.0, 3),
];

/// Per-`xs_kernel_idx` (zaid, material_idx) parallel to
/// `NUCLIDE_SPECS`. The depletion driver uses this table to build the
/// `BurnupMapping` automatically — every chain ZAID that has a row
/// here gets its CRAM-evolved density pushed back into transport.
/// ZAIDs in chain-only (e.g. I-135, Cs-135 in `partial_xe.json`,
/// or Pu-239/240/241 in `pwr_actinides.json` until those HDF5 files
/// are loaded too) stay decoupled from transport but still evolve
/// in the chain composition vector.
///
/// To extend the burnup feedback set: add the HDF5 file to
/// `NUCLIDE_SPECS`, add the matching `(zaid, material_idx)` row
/// here, and add the corresponding `mat.add_nuclide(...)` call in
/// `setup_materials`. The chain JSON does not need to change —
/// `BurnupMapping::from_zaid_table` validates each row against the
/// chain at startup and silently drops any that don't apply.
const NUCLIDE_INFO: &[(u32, usize)] = &[
    (92235, 0), // U-235  in fuel
    (92238, 0), // U-238  in fuel
    (8016, 0),  // O-16   in fuel
    (1001, 2),  // H-1    in water
    (40090, 1), // Zr-90  in clad
    (40091, 1), // Zr-91  in clad
    (40092, 1), // Zr-92  in clad
    (40094, 1), // Zr-94  in clad
    (8016, 2),  // O-16   in water (different xs_kernel_idx)
    (54135, 0), // Xe-135 in fuel
    // pwr_actinides chain — actinide buildup + Sm-149 / Pm-149 / I-135 / Cs-135.
    (92236, 0), // U-236  in fuel
    (92237, 0), // U-237  in fuel
    (92239, 0), // U-239  in fuel
    (93237, 0), // Np-237 in fuel
    (93239, 0), // Np-239 in fuel
    (94239, 0), // Pu-239 in fuel
    (94240, 0), // Pu-240 in fuel
    (94241, 0), // Pu-241 in fuel
    (94242, 0), // Pu-242 in fuel
    (95241, 0), // Am-241 in fuel
    (53135, 0), // I-135  in fuel
    (55135, 0), // Cs-135 in fuel
    (61149, 0), // Pm-149 in fuel
    (62149, 0), // Sm-149 in fuel
];

const ZAID_U235: u32 = 92235;
const ZAID_I135: u32 = 53135;
const ZAID_XE135: u32 = 54135;
const ZAID_CS135: u32 = 55135;

const LAMBDA_I135: f64 = 2.926_400e-5;
const LAMBDA_XE135: f64 = 2.106_530e-5;

const E_THERMAL_EV: f64 = 0.0253;

fn setup_materials(initial_xe_density: f64) -> Vec<Material> {
    let mut fuel = Material::new("UO2", 900.0);
    fuel.add_nuclide(7.19e-4, IDX_U235);
    fuel.add_nuclide(2.2482e-2, IDX_U238);
    fuel.add_nuclide(4.6402e-2, IDX_O16_FUEL);
    fuel.add_nuclide(initial_xe_density, IDX_XE135);
    // Actinide / FP chain — initial atom density 0; CRAM grows them.
    fuel.add_nuclide(0.0, IDX_U236);
    fuel.add_nuclide(0.0, IDX_U237);
    fuel.add_nuclide(0.0, IDX_U239);
    fuel.add_nuclide(0.0, IDX_NP237);
    fuel.add_nuclide(0.0, IDX_NP239);
    fuel.add_nuclide(0.0, IDX_PU239);
    fuel.add_nuclide(0.0, IDX_PU240);
    fuel.add_nuclide(0.0, IDX_PU241);
    fuel.add_nuclide(0.0, IDX_PU242);
    fuel.add_nuclide(0.0, IDX_AM241);
    fuel.add_nuclide(0.0, IDX_I135);
    fuel.add_nuclide(0.0, IDX_CS135);
    fuel.add_nuclide(0.0, IDX_PM149);
    fuel.add_nuclide(0.0, IDX_SM149);

    let mut clad = Material::new("Zircaloy", 600.0);
    clad.add_nuclide(2.2932e-2, IDX_ZR90);
    clad.add_nuclide(4.996e-3, IDX_ZR91);
    clad.add_nuclide(7.636e-3, IDX_ZR92);
    clad.add_nuclide(7.740e-3, IDX_ZR94);

    let mut water = Material::new("H2O", 600.0);
    water.add_nuclide(4.9486e-2, IDX_H1);
    water.add_nuclide(2.4743e-2, IDX_O16_WATER);

    vec![fuel, clad, water]
}

fn setup_geometry() -> (Vec<Surface>, Vec<Cell>, Aabb) {
    let fuel_or = 0.4096;
    let clad_ir = 0.4180;
    let clad_or = 0.4750;
    let pitch = 1.2600;
    let half = pitch / 2.0;
    let z_half = half;

    let mut surfaces =
        open_rust_mc::geometry::shapes::pin_cylinders(0.0, 0.0, &[fuel_or, clad_ir, clad_or]);
    let outer_box = open_rust_mc::geometry::shapes::rect_box(
        [half, half, z_half],
        BoundaryCondition::Reflective,
        surfaces.len(),
    );
    surfaces.extend(outer_box.surfaces);

    let fuel_aabb = Aabb::new(
        Vec3::new(-fuel_or, -fuel_or, -z_half),
        Vec3::new(fuel_or, fuel_or, z_half),
    );
    let pin_aabb = Aabb::new(
        Vec3::new(-half, -half, -z_half),
        Vec3::new(half, half, z_half),
    );

    let cells = vec![
        Cell::new(
            CellId(0),
            cell::intersect_all(vec![cell::inside(0), cell::outside(7), cell::inside(8)]),
            CellFill::Material(0),
        )
        .with_aabb(fuel_aabb)
        .with_temperature(900.0),
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
        Cell::new(
            CellId(3),
            cell::Region::Intersection(Box::new(cell::outside(2)), Box::new(outer_box.inside)),
            CellFill::Material(2),
        )
        .with_aabb(pin_aabb)
        .with_temperature(600.0),
    ];
    (surfaces, cells, fuel_aabb)
}

fn load_xs(args: &Args) -> xs_provider::SvdXsProvider {
    let mut kernels = Vec::with_capacity(NUCLIDE_SPECS.len());
    for &(filename, awr, nu_bar, nuc_temp_idx) in NUCLIDE_SPECS.iter() {
        let path = args.data_dir.join(filename);
        if path.exists() {
            kernels.push(xs_provider::load_nuclide(
                &path,
                args.rank,
                nuc_temp_idx,
                awr,
                nu_bar,
            ));
        } else {
            eprintln!("WARN: {} not found — using zero kernel", path.display());
            kernels.push(xs_provider::NuclideKernels::empty(awr, nu_bar));
        }
    }
    let h2o_path = args.data_dir.join("c_H_in_H2O.h5");
    let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; NUCLIDE_SPECS.len()];
    if h2o_path.exists()
        && let Ok(tsl) = hdf5_reader::load_thermal_scattering(&h2o_path)
    {
        thermal[IDX_H1] = Some(Arc::new(tsl));
    }
    xs_provider::SvdXsProvider {
        nuclides: kernels.into_iter().map(Arc::new).collect(),
        thermal,
    }
}

fn build_chain<XS: XsProvider>(provider: &XS) -> DepletionChain {
    let mut chain = DepletionChain::new();
    chain.add_nuclide(NuclideEntry {
        name: "U-235".into(),
        zaid: ZAID_U235,
        decay_constant: 0.0,
        decay_branches: vec![],
    });
    chain.add_nuclide(NuclideEntry {
        name: "I-135".into(),
        zaid: ZAID_I135,
        decay_constant: LAMBDA_I135,
        decay_branches: vec![DecayBranch {
            daughter_zaid: ZAID_XE135,
            branch_ratio: 1.0,
        }],
    });
    chain.add_nuclide(NuclideEntry {
        name: "Xe-135".into(),
        zaid: ZAID_XE135,
        decay_constant: LAMBDA_XE135,
        decay_branches: vec![DecayBranch {
            daughter_zaid: ZAID_CS135,
            branch_ratio: 1.0,
        }],
    });
    chain.add_nuclide(NuclideEntry {
        name: "Cs-135".into(),
        zaid: ZAID_CS135,
        decay_constant: 0.0,
        decay_branches: vec![],
    });

    // One-group thermal XS pulled directly from the live provider —
    // same data the transport solver evaluates. Keeps the chain
    // self-consistent with the engine.
    let micro_u235 = provider.lookup(IDX_U235, E_THERMAL_EV);
    let micro_xe135 = provider.lookup(IDX_XE135, E_THERMAL_EV);

    chain.add_reaction(
        ZAID_U235,
        18,
        micro_u235.fission,
        Some(u235_thermal_iodine_xenon_yields()),
    );
    chain.add_reaction(
        ZAID_XE135,
        102,
        micro_xe135.capture,
        Some(std::collections::HashMap::new()),
    );

    chain
}

fn run_eigenvalue(
    args: &Args,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Material],
    provider: &xs_provider::SvdXsProvider,
    fuel_aabb: &Aabb,
    seed: u64,
) -> (
    Vec<open_rust_mc::transport::simulate::BatchResult>,
    f64,
    MeshFluxTally,
    open_rust_mc::transport::tally::ReactionRateTally,
) {
    let mut tallies = Tallies::default();
    let mesh = MeshFluxTally::from_aabb(fuel_aabb, [4, 4, 1]);
    tallies.mesh_flux = Some(mesh.clone());
    // Reaction-rate tally for chain-XS spectrum collapse. Mirrors
    // the geometry's cells × all NUCLIDE_SPECS slots × the four
    // depletable MTs we track in the chain (fission, capture,
    // (n,2n), (n,3n)).
    // MT list: fission, capture, (n,2n), (n,3n) drive the chain;
    // (n,p) and (n,α) flow through `partial_xs` for granular reporting
    // — they don't drive depletion (already folded into capture for
    // the absorption sampler) but `collapsed_reaction_xs` will collapse
    // them per-cell so the chain JSON can carry them if any future
    // chain wants explicit (n,p) / (n,α) yields.
    let rr = open_rust_mc::transport::tally::ReactionRateTally::new(
        cells.len(),
        NUCLIDE_SPECS.len(),
        vec![18, 102, 16, 17, 103, 107],
    );
    tallies.reaction_rate = Some(rr.clone());

    let cfg = SimConfig {
        batches: args.batches,
        inactive: args.inactive,
        particles_per_batch: args.particles,
        seed,
        auto_inactive: None,
        verbose: false,
        parallel: true,
        tallies,
        statepoint_path: None,
        survival_biasing: None,
        initial_source_bank: None,
        weight_window: None,
        disable_delayed_neutrons: false,
        urr_equivalence: None,
        gpu_refill_pool_factor: None,
    };

    let (results, k_eff) = simulate::run_eigenvalue(&cfg, surfaces, cells, materials, provider);
    (results, k_eff, mesh, rr)
}

fn main() {
    let args = Args::parse();
    println!("==========================================================");
    println!("  PWR pin-cell burnup driver — CRAM-16 + transport feedback");
    println!("==========================================================");
    println!("  data_dir       : {}", args.data_dir.display());
    println!(
        "  steps          : {}  ({:.1} h each)",
        args.steps, args.hours_per_step
    );
    println!(
        "  power          : {:.1} W/cm pin length",
        args.power_w_per_cm
    );
    println!(
        "  particles      : {} × {} batches",
        args.particles, args.batches
    );
    println!();

    let provider = load_xs(&args);
    let mut chain = match &args.chain {
        Some(path) => {
            println!("  Loading chain from {}", path.display());
            let spec = ChainSpec::from_file(path).expect("failed to load chain JSON");
            println!(
                "  Chain '{}': {} nuclides, {} reactions",
                spec.name,
                spec.nuclides.len(),
                spec.reactions.len()
            );
            spec.build()
        }
        None => {
            println!("  Using embedded partial Xe chain (4 nuclides, 2 reactions)");
            build_chain(&provider)
        }
    };

    // Auto-derive the burnup mapping from `NUCLIDE_INFO` (the
    // per-`xs_kernel_idx` ZAID + material_idx table). Every chain
    // ZAID with a transport slot gets wired; chain-only ZAIDs (e.g.
    // I-135, Cs-135 in the partial chain; Pu-239 etc. in the
    // actinides chain when their HDF5 files aren't loaded) stay
    // decoupled but still evolve in the chain composition vector.
    // Transport-only nuclides (Zr clad, etc.) are silently skipped.
    //
    // The intersection is the single source of truth: extend the
    // chain JSON, the burnup loop walks more nuclides through CRAM;
    // extend `NUCLIDE_INFO`+`NUCLIDE_SPECS`+`setup_materials`,
    // transport carries more nuclides; intersect both → the
    // `BurnupMapping` grows accordingly. No hand-edits per chain.
    let zaid_table: Vec<(u32, usize, usize)> = NUCLIDE_INFO
        .iter()
        .enumerate()
        .map(|(xs_idx, &(zaid, mat_idx))| (zaid, mat_idx, xs_idx))
        .collect();

    let (surfaces, cells, fuel_aabb) = setup_geometry();
    let mut materials = setup_materials(0.0);
    let mapping = BurnupMapping::from_zaid_table(&chain, &materials, &zaid_table);
    println!(
        "  BurnupMapping wired {}/{} chain ZAIDs into transport ({} chain-only, {} transport-only)",
        mapping.len(),
        chain.len(),
        chain.len() - mapping.len(),
        zaid_table.len() - mapping.len(),
    );

    // Pull initial composition straight from the live materials —
    // single source of truth, no risk of drift.
    let mut chain_composition = mapping.pull(&chain, &materials);

    // Pin volume (1.26 × 1.26 × 1.26 cm) for power → source-rate scaling.
    let pin_volume = 1.26_f64 * 1.26 * 1.26;
    let target_power = args.power_w_per_cm * 1.26;

    let zaid_pu239 = chain.index_of_zaid(94239);
    let zaid_pu240 = chain.index_of_zaid(94240);
    let zaid_sm149 = chain.index_of_zaid(62149);
    let actinides_in_chain = zaid_pu239.is_some() && zaid_pu240.is_some() && zaid_sm149.is_some();
    if actinides_in_chain {
        println!(
            "  step    t [h]   k_eff   φ_fuel [n/cm²/s]   N_Xe/U235   N_Pu239/U235   N_Pu240/U235   N_Sm149/U235   ΔU235/U235"
        );
        println!(
            "  ----  -------  ------  ----------------  ----------  -------------  -------------  -------------  ----------"
        );
    } else {
        println!(
            "  step       t [h]    k_eff       φ_fuel [n/cm²/s]    N_Xe/N_U235     ΔN_U235/N_U235"
        );
        println!(
            "  -----  --------  --------  -----------------  --------------  ----------------"
        );
    }

    let dt = args.hours_per_step * 3_600.0;
    let order: CramOrder = args.cram_order.into();
    let n_u235_initial = chain_composition[chain.index_of_zaid(ZAID_U235).unwrap()];

    if args.full_chain {
        eprintln!("WARN: --full-chain is reserved for a future chain library; running partial.");
    }

    for step in 0..=args.steps {
        let t_run = Instant::now();
        let (batches, k_eff, mesh, rr_template) = run_eigenvalue(
            &args,
            &surfaces,
            &cells,
            &materials,
            &provider,
            &fuel_aabb,
            42 + step as u64,
        );
        let phi_per_source = mean_flux_per_source(&batches, &mesh);
        let f_per_source = mean_fissions_per_source(&batches);
        // Power normalisation: Q [n/s] from target power and energy
        // per fission. φ_physical = φ_per_source × Q / pin_volume —
        // since the per-source flux is already cm⁻² (track length /
        // volume / N_source), multiplying by Q recovers n/cm²/s.
        let q = power_normalized_source(target_power, f_per_source, E_PER_FISSION_J);
        let phi_physical = phi_per_source * q;

        // **Spectrum-collapsed one-group XS** — overrides the
        // thermal-spectrum values shipped in `chains/pwr_actinides.json`
        // with the actual cell-flux-spectrum-averaged σ collapsed
        // from the eigenvalue's reaction-rate tally. Cell 0 = fuel.
        // For every (xs_idx, MT) the tally saw, look up the
        // corresponding ZAID via NUCLIDE_INFO and update the chain's
        // reaction entry in-place.
        let collapsed = open_rust_mc::depletion::flux::collapsed_reaction_xs(
            &batches,
            &rr_template,
            /* fuel cell */ 0,
        );
        for ((xs_idx, mt), sigma_barns) in &collapsed {
            // Map xs_idx → ZAID via NUCLIDE_INFO. Use the FIRST
            // material_idx == 0 (fuel) entry — water-side O-16 is a
            // separate xs_idx with its own collapsed value but lives
            // outside the depletion chain (fuel-O is the only one we
            // chain).
            let Some(&(zaid, _mat)) = NUCLIDE_INFO.get(*xs_idx) else {
                continue;
            };
            // Skip non-fuel materials' entries.
            if NUCLIDE_INFO[*xs_idx].1 != 0 {
                continue;
            }
            if let Some(rxn) = chain.reactions.get_mut(&(zaid, *mt)) {
                rxn.xs_barns = *sigma_barns;
            }
        }
        if step == 0 {
            // Diagnostic at the first step: print the collapsed
            // values for the dominant chain reactions so the operator
            // can spot pathologies (e.g. σ_f(U-235) = 583 b would
            // indicate the spectrum collapse isn't taking effect).
            let probe = |zaid: u32, mt: u32, label: &str| {
                if let Some(rxn) = chain.reactions.get(&(zaid, mt)) {
                    eprintln!(
                        "    {label:<24} ⟨σ⟩ = {:>8.3} b  (chain JSON: see file)",
                        rxn.xs_barns
                    );
                }
            };
            eprintln!("  Spectrum-collapsed one-group XS at fuel cell (step 0):");
            probe(92235, 18, "U-235  fission");
            probe(92235, 102, "U-235  capture");
            probe(92238, 18, "U-238  fission");
            probe(92238, 102, "U-238  capture");
            probe(94239, 18, "Pu-239 fission");
            probe(94239, 102, "Pu-239 capture");
            probe(54135, 102, "Xe-135 capture");
            probe(62149, 102, "Sm-149 capture");
        }

        let n_u235 = chain_composition[chain.index_of_zaid(ZAID_U235).unwrap()];
        let n_xe135 = chain_composition[chain.index_of_zaid(ZAID_XE135).unwrap()];
        let t_hours = step as f64 * args.hours_per_step;

        if actinides_in_chain {
            let n_pu239 = chain_composition[zaid_pu239.unwrap()];
            let n_pu240 = chain_composition[zaid_pu240.unwrap()];
            let n_sm149 = chain_composition[zaid_sm149.unwrap()];
            let n_u235_safe = n_u235.max(1e-30);
            println!(
                "  {:>4}  {:>7.2}  {:>6.4}  {:>16.3e}  {:>10.3e}  {:>13.3e}  {:>13.3e}  {:>13.3e}  {:>+9.2}%  ({:.0} ms)",
                step,
                t_hours,
                k_eff,
                phi_physical,
                n_xe135 / n_u235_safe,
                n_pu239 / n_u235_safe,
                n_pu240 / n_u235_safe,
                n_sm149 / n_u235_safe,
                100.0 * (n_u235 - n_u235_initial) / n_u235_initial,
                t_run.elapsed().as_secs_f64() * 1000.0,
            );
        } else {
            println!(
                "  {:>5}  {:>8.2}  {:>8.5}  {:>17.3e}  {:>14.4e}  {:>+15.2}%  ({:.0} ms)",
                step,
                t_hours,
                k_eff,
                phi_physical,
                n_xe135 / n_u235.max(1e-30),
                100.0 * (n_u235 - n_u235_initial) / n_u235_initial,
                t_run.elapsed().as_secs_f64() * 1000.0,
            );
        }

        if step == args.steps {
            break;
        }

        // Fresh-corrector closure: when CRAM finishes the predictor
        // step, this runs an independent eigenvalue solve on the
        // PREDICTED composition to get the EOC flux. Materials are
        // cloned so the in-flight `materials` slice stays at BOC for
        // the post-corrector push back.
        //
        // Cost: one extra full-physics eigenvalue solve per burnup
        // step. Standard for any honest predictor-corrector scheme
        // (OpenMC's CE/LI does the same thing).
        let flux_at = |predicted: &[f64]| {
            let mut clone = materials.clone();
            mapping.push(predicted, &mut clone);
            let (eoc_batches, _eoc_k, eoc_mesh, _eoc_rr) = run_eigenvalue(
                &args,
                &surfaces,
                &cells,
                &clone,
                &provider,
                &fuel_aabb,
                42 + step as u64 + 100_000,
            );
            let phi_eoc_per_source = mean_flux_per_source(&eoc_batches, &eoc_mesh);
            let f_eoc_per_source = mean_fissions_per_source(&eoc_batches);
            let q_eoc = power_normalized_source(target_power, f_eoc_per_source, E_PER_FISSION_J);
            phi_eoc_per_source * q_eoc
        };
        let result = deplete_ce_li(&chain, &chain_composition, phi_physical, dt, order, flux_at);
        chain_composition = result.corrected;

        // Push every mapped (chain → material) entry. ZAIDs that
        // don't have a transport slot (I-135, Cs-135) evolve only
        // in `chain_composition` — they don't feed back into macro
        // XS but their decay rates and yields still drive the chain.
        mapping.push(&chain_composition, &mut materials);
    }

    println!();
    println!(
        "  Source rate Q (final step) — physical flux scaled to {:.1} W power",
        target_power
    );
    println!("  Pin cell volume {:.3} cm³", pin_volume);
}
