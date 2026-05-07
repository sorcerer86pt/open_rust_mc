//! Depletion demo — Xe-135 equilibrium poisoning under PWR thermal flux.
//!
//! Classical reactor-physics benchmark for any depletion solver:
//! given a constant thermal flux and a fixed U-235 fissile inventory,
//! the I-135 → Xe-135 → Cs-135 chain reaches an equilibrium where
//! Xe-135 production from fission and from I-135 β-decay balances
//! Xe-135 removal by β-decay and (n,γ) capture. The equilibrium
//! ratio `N_Xe^eq / N_U235` has a closed-form expression that any
//! correct Bateman + CRAM implementation must reproduce.
//!
//! Reaction chain modelled here:
//!   - U-235 fission produces I-135 (cumulative thermal yield 6.309 %)
//!     and direct Xe-135 (independent yield 0.256 %).
//!   - I-135 β-decays to Xe-135 (T₁/₂ = 6.57 h, 100 % branch).
//!   - Xe-135 β-decays to Cs-135 (T₁/₂ = 9.14 h) and is destroyed by
//!     (n,γ) (σ_a ≈ 2.65 × 10⁶ b at 0.025 eV — the largest thermal
//!     capture cross-section of any nuclide).
//!   - Cs-135 is treated as effectively stable on the depletion
//!     timescale (T₁/₂ ≈ 2.3 × 10⁶ y → λ ≈ 0).
//!
//! Reports the analytical equilibrium concentration alongside the
//! CRAM-16 result after the chain has converged. Runs in < 100 ms.
//!
//! Usage:
//!   deplete_demo [--flux <n/cm²/s>] [--total-hours H] [--steps N]

use std::collections::HashMap;

use clap::Parser;

use open_rust_mc::depletion::{
    DepletionStep,
    chain::{DecayBranch, DepletionChain, NuclideEntry, u235_thermal_iodine_xenon_yields},
    cram::CramOrder,
    deplete_ce_li,
};

#[derive(Parser, Debug)]
#[command(
    name = "deplete_demo",
    about = "Xe-135 equilibrium poisoning via CRAM-16"
)]
struct Args {
    /// Thermal-region flux (n / cm² / s). PWR core average ≈ 3e14.
    #[arg(long, default_value_t = 3.0e14)]
    flux: f64,

    /// Total burn time in hours. ~80 h is enough for Xe-135 to reach
    /// equilibrium (a few half-lives of the slower component).
    #[arg(long, default_value_t = 80.0)]
    total_hours: f64,

    /// Number of CE/LI steps. With constant flux they're equivalent
    /// to substepping the Bateman solve.
    #[arg(long, default_value_t = 16)]
    steps: u32,

    /// Initial U-235 number density (atoms / barn cm). Default
    /// matches PWR fuel at 3.1 % enrichment (10.4 g/cm³ UO₂).
    #[arg(long, default_value_t = 7.19e-4)]
    n_u235_0: f64,

    /// CRAM approximation order. 16 is sufficient for PWR-typical Δt;
    /// 48 is for very stiff chains or geologic-scale Δt.
    #[arg(long, value_enum, default_value_t = CramOrderArg::Order16)]
    cram_order: CramOrderArg,
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

const U235: u32 = 92235;
const I135: u32 = 53135;
const XE135: u32 = 54135;
const CS135: u32 = 55135;

const LAMBDA_I135: f64 = 2.926_400e-5; // s⁻¹  (T½ = 6.57 h)
const LAMBDA_XE135: f64 = 2.106_530e-5; // s⁻¹  (T½ = 9.14 h)

const U235_SIGMA_F_BARN: f64 = 583.5; // thermal one-group fission XS
const XE135_SIGMA_C_BARN: f64 = 2.65e6; // thermal (n,γ) on Xe-135

fn build_chain() -> DepletionChain {
    let mut chain = DepletionChain::new();
    chain.add_nuclide(NuclideEntry {
        name: "U-235".into(),
        zaid: U235,
        decay_constant: 0.0, // stable on depletion timescale
        decay_branches: vec![],
    });
    chain.add_nuclide(NuclideEntry {
        name: "I-135".into(),
        zaid: I135,
        decay_constant: LAMBDA_I135,
        decay_branches: vec![DecayBranch {
            daughter_zaid: XE135,
            branch_ratio: 1.0,
        }],
    });
    chain.add_nuclide(NuclideEntry {
        name: "Xe-135".into(),
        zaid: XE135,
        decay_constant: LAMBDA_XE135,
        decay_branches: vec![DecayBranch {
            daughter_zaid: CS135,
            branch_ratio: 1.0,
        }],
    });
    chain.add_nuclide(NuclideEntry {
        name: "Cs-135".into(),
        zaid: CS135,
        decay_constant: 0.0,
        decay_branches: vec![],
    });

    // U-235 thermal fission yields → I-135 (cumulative) + Xe-135 (independent).
    chain.add_reaction(
        U235,
        18,
        U235_SIGMA_F_BARN,
        Some(u235_thermal_iodine_xenon_yields()),
    );
    // Xe-135 (n,γ) → Xe-136 (untracked — out of chain).
    chain.add_reaction(XE135, 102, XE135_SIGMA_C_BARN, Some(HashMap::new()));

    chain
}

/// Analytical equilibrium I-135 / Xe-135 concentrations at constant
/// flux. Derived from `dN_I/dt = dN_Xe/dt = 0`:
///   N_I^eq = γ_I · Σ_f · φ / λ_I
///   N_Xe^eq = (γ_I + γ_Xe) · Σ_f · φ / (λ_Xe + σ_a^Xe · φ)
fn analytical_equilibrium(n_u235: f64, flux: f64) -> (f64, f64) {
    const BARN_CM2: f64 = 1.0e-24;
    let yields = u235_thermal_iodine_xenon_yields();
    let gamma_i = yields[&I135];
    let gamma_xe = yields[&XE135];

    // Σ_f from atom density (atoms / b·cm) is `N · σ_f` in cm⁻¹ when
    // σ_f is in barns. Production rate per cm³ per second is
    // `N · σ_f · φ` after the BARN_CM2 conversion (since
    // [atoms / b·cm] · [cm² / b] · [barn] = atoms / cm³, and · [1/s]).
    // Easier: keep N in atoms/cm³ — the `n_u235` flag is in atoms/(b·cm)
    // which is `1e24 × atoms/cm³`. Same factor cancels on both sides.
    let sigma_f_macro = n_u235 * U235_SIGMA_F_BARN; // [atoms / b·cm × b] = atoms/cm
    let production_i = gamma_i * sigma_f_macro * flux * BARN_CM2;
    let production_xe = (gamma_i + gamma_xe) * sigma_f_macro * flux * BARN_CM2;
    let n_i_eq = production_i / LAMBDA_I135;
    let n_xe_eq = production_xe / (LAMBDA_XE135 + XE135_SIGMA_C_BARN * BARN_CM2 * flux);
    (n_i_eq, n_xe_eq)
}

fn main() {
    let args = Args::parse();
    let chain = build_chain();
    let n_u235_0 = args.n_u235_0;

    // Initial composition: only U-235 present.
    let mut composition = vec![0.0_f64; chain.len()];
    composition[chain.index_of_zaid(U235).expect("U-235 in chain")] = n_u235_0;

    let total_seconds = args.total_hours * 3_600.0;
    let dt = total_seconds / args.steps as f64;
    println!(
        "Xe-135 equilibrium poisoning ({} steps × {:.2} h, φ = {:.2e} n/cm²/s)",
        args.steps,
        dt / 3_600.0,
        args.flux,
    );
    println!();
    println!("  step    t [h]      N_U235        N_I-135       N_Xe-135      N_Cs-135");
    println!("  ----  --------  ------------  ------------  ------------  ------------");
    let order: CramOrder = args.cram_order.into();
    for step in 0..args.steps {
        let result: DepletionStep =
            deplete_ce_li(&chain, &composition, args.flux, dt, order, |_| args.flux);
        composition = result.corrected;
        let t_hr = (step + 1) as f64 * dt / 3_600.0;
        println!(
            "  {:>4}  {:>8.2}  {:>12.4e}  {:>12.4e}  {:>12.4e}  {:>12.4e}",
            step + 1,
            t_hr,
            composition[chain.index_of_zaid(U235).unwrap()],
            composition[chain.index_of_zaid(I135).unwrap()],
            composition[chain.index_of_zaid(XE135).unwrap()],
            composition[chain.index_of_zaid(CS135).unwrap()],
        );
    }

    // Compare to analytical equilibrium.
    let (n_i_eq, n_xe_eq) = analytical_equilibrium(n_u235_0, args.flux);
    let n_i_cram = composition[chain.index_of_zaid(I135).unwrap()];
    let n_xe_cram = composition[chain.index_of_zaid(XE135).unwrap()];

    println!();
    println!("  ── Equilibrium comparison ──────────────────────────────────");
    println!(
        "  N_I-135   analytical = {:>12.4e}    CRAM = {:>12.4e}    Δ = {:>+5.2}%",
        n_i_eq,
        n_i_cram,
        100.0 * (n_i_cram - n_i_eq) / n_i_eq,
    );
    println!(
        "  N_Xe-135  analytical = {:>12.4e}    CRAM = {:>12.4e}    Δ = {:>+5.2}%",
        n_xe_eq,
        n_xe_cram,
        100.0 * (n_xe_cram - n_xe_eq) / n_xe_eq,
    );
    println!();
    println!(
        "  N_Xe / N_U235 (atomic ratio) = {:.4e}",
        n_xe_cram / n_u235_0
    );
}
