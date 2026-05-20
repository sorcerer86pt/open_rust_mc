#![allow(
// SPDX-License-Identifier: MIT
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::manual_is_multiple_of,
    clippy::needless_borrow
)]
//! Point-kinetics transient demo — exercises the new
//! `transport::kinetics` module on canonical reactivity profiles
//! that anchor against textbook closed-form values.
//!
//! Three modes:
//!
//! - `step` : instantaneous reactivity insertion at t=0; integrates
//!   the prompt jump + delayed evolution; prints n(t) and compares
//!   the early-time behaviour to the prompt-jump analytic.
//! - `ramp` : linear reactivity ramp `ρ(t) = ramp_rate · t` until a
//!   stop time; surrogate for control-rod withdrawal.
//! - `scram`: equilibrium → step ρ = −β → traces the prompt drop +
//!   delayed asymptote. Late-time decay rate matches the slowest
//!   delayed group `λ_1` (Keepin: 0.0124 s⁻¹).
//!
//! Output: CSV `(t, n, c1, c2, c3, c4, c5, c6, rho)` to stdout (or
//! `--out FILE`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use open_rust_mc::transport::kinetics::{
    PU239_THERMAL_KEEPIN, PkParams, PkState, U235_THERMAL_KEEPIN, U238_FAST_KEEPIN, blend,
    inhour_period, prompt_jump_ratio, step_crank_nicolson,
};

#[derive(Parser, Debug)]
#[command(name = "point_kinetics_demo")]
struct Args {
    /// Reactivity profile to drive.
    #[arg(long, value_enum, default_value_t = Mode::Step)]
    mode: Mode,
    /// Step / target reactivity in dollars (1 $ = β). For ramps, the
    /// terminal value at `t_stop`.
    #[arg(long, default_value_t = 0.5)]
    rho_dollars: f64,
    /// For `ramp`: rate of reactivity insertion in $/s.
    #[arg(long, default_value_t = 1.0)]
    ramp_rate_per_s: f64,
    /// Mean prompt-neutron generation time Λ (s). LWR ~10 µs, fast
    /// reactor ~0.1 µs.
    #[arg(long, default_value_t = 1e-5)]
    gen_time: f64,
    /// Time horizon to integrate (s).
    #[arg(long, default_value_t = 60.0)]
    t_end: f64,
    /// Step size in seconds. CN is A-stable; smaller dt only buys
    /// 2nd-order accuracy gain.
    #[arg(long, default_value_t = 1e-4)]
    dt: f64,
    /// Sample stride for output rows. dt=1e-4 with stride 100 → row
    /// every 10 ms — 6 000 rows over 60 s.
    #[arg(long, default_value_t = 100)]
    print_stride: usize,
    /// External neutron source rate (n/s). Drives sub-critical
    /// problems where the equations would otherwise zero out.
    #[arg(long, default_value_t = 0.0)]
    source: f64,
    /// Fissile mix as `<nuclide>:<weight>[,<nuclide>:<weight>...]`.
    /// Supported: `u235`, `u238_fast`, `pu239`. Weights are
    /// renormalised. Default = pure U-235 thermal.
    #[arg(long, default_value = "u235:1.0")]
    mix: String,
    /// Optional output file. Defaults to stdout.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode {
    Step,
    Ramp,
    Scram,
}

fn parse_mix(spec: &str) -> Vec<(open_rust_mc::transport::kinetics::DelayedGroups, f64)> {
    let mut out = Vec::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let mut parts = entry.split(':');
        let name = parts.next().unwrap_or("").trim().to_lowercase();
        let weight = parts
            .next()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(1.0);
        let g = match name.as_str() {
            "u235" => U235_THERMAL_KEEPIN,
            "u238_fast" => U238_FAST_KEEPIN,
            "pu239" => PU239_THERMAL_KEEPIN,
            other => panic!("unknown mix nuclide '{other}' (try u235 / u238_fast / pu239)"),
        };
        out.push((g, weight));
    }
    out
}

fn rho_at(args: &Args, beta: f64, t: f64) -> f64 {
    let target = args.rho_dollars * beta;
    match args.mode {
        Mode::Step => target,
        Mode::Scram => -beta, // ρ = -1 $ — full negative reactivity insertion
        Mode::Ramp => {
            let rate = args.ramp_rate_per_s * beta;
            (rate * t).clamp(-10.0 * beta, 10.0 * beta).min(target)
        }
    }
}

fn main() {
    let args = Args::parse();

    let mix = parse_mix(&args.mix);
    let groups = blend(&mix);
    let params = PkParams {
        groups,
        gen_time: args.gen_time,
    };

    let initial = match args.mode {
        Mode::Scram | Mode::Ramp => PkState::equilibrium(1.0, &params),
        Mode::Step => PkState::equilibrium(1.0, &params),
    };

    eprintln!(
        "# point_kinetics_demo  mode={:?}  Λ={:.3e} s",
        args.mode, params.gen_time
    );
    eprintln!(
        "# β = {:.5}    β_i = {:.4?}    λ_i = {:.4?} s⁻¹",
        groups.beta_total,
        groups.beta_i(),
        groups.decay_per_s,
    );
    let target_rho = args.rho_dollars * groups.beta_total;
    if matches!(args.mode, Mode::Step) {
        eprintln!(
            "# step ρ = {:.2} $ = {:.3e}    prompt-jump analytic n_jump = {:.4}",
            args.rho_dollars,
            target_rho,
            prompt_jump_ratio(target_rho, &params),
        );
        let tau = inhour_period(target_rho, &params);
        eprintln!("# inhour stable period τ = {tau:.3} s");
    } else if matches!(args.mode, Mode::Scram) {
        eprintln!(
            "# scram  ρ = -1 $ = {:.3e}    prompt drop ratio = {:.4}",
            -groups.beta_total,
            prompt_jump_ratio(-groups.beta_total, &params),
        );
    }

    // CSV output sink — file or stdout.
    let stdout = std::io::stdout();
    let mut sink: Box<dyn Write> = match &args.out {
        Some(path) => Box::new(BufWriter::new(File::create(path).expect("create out"))),
        None => Box::new(BufWriter::new(stdout.lock())),
    };
    writeln!(sink, "t_s,n,c1,c2,c3,c4,c5,c6,rho").unwrap();
    let mut t = 0.0_f64;
    let mut state = initial;
    let mut step_idx = 0usize;
    while t < args.t_end {
        let rho = rho_at(&args, groups.beta_total, t);
        if step_idx % args.print_stride == 0 {
            writeln!(
                sink,
                "{t:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{rho:.6e}",
                state.n, state.c[0], state.c[1], state.c[2], state.c[3], state.c[4], state.c[5],
            )
            .unwrap();
        }
        // Mid-step reactivity for trapezoidal accuracy on linear
        // ramps; constant for step / scram.
        let rho_mid = rho_at(&args, groups.beta_total, t + 0.5 * args.dt);
        state = step_crank_nicolson(state, &params, rho_mid, args.source, args.dt);
        t += args.dt;
        step_idx += 1;
    }
    // Always emit the final row.
    let rho_end = rho_at(&args, groups.beta_total, t);
    writeln!(
        sink,
        "{t:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{rho_end:.6e}",
        state.n, state.c[0], state.c[1], state.c[2], state.c[3], state.c[4], state.c[5],
    )
    .unwrap();

    eprintln!(
        "# final  t={t:.3} s  n={:.4e}  Σc_i={:.4e}",
        state.n,
        state.c.iter().sum::<f64>(),
    );
}
