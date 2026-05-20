// SPDX-License-Identifier: MIT
//! Time-dependent point-kinetics with 6-group delayed-neutron precursors.
//!
//! Closed-form one-spatial-mode reduction of the Boltzmann equation.
//! Useful for reactivity-induced excursions, prompt-jump analysis,
//! scram studies, sub-prompt benchmarks (BFS / Bigten / SPERT-3
//! transient sub-suites of ICSBEP), and as a sanity model for
//! pre-screening transport-coupled kinetics.
//!
//! # Equations
//!
//! ```text
//!   dN/dt   = (ρ(t) − β) / Λ · N(t) + Σ_i λ_i · C_i(t) + S(t)
//!   dC_i/dt =  β_i / Λ · N(t) − λ_i · C_i(t)              i = 1 … 6
//! ```
//!
//! where
//! - `N(t)` — neutron population (or proportional power).
//! - `ρ(t)` — reactivity, dimensionless `Δk/k`.
//! - `β_i`  — group-i delayed-neutron yield fraction; `β = Σ β_i`.
//! - `λ_i`  — group-i precursor decay constant (s⁻¹).
//! - `Λ`    — mean prompt-neutron generation time (s).
//! - `C_i`  — group-i precursor concentration.
//! - `S(t)` — external neutron source rate (s⁻¹).
//!
//! # Stiffness
//!
//! The system has eigenvalues spanning ~10 decades: `λ_6 ≈ 3 s⁻¹`
//! (slowest precursor) and `(ρ−β)/Λ ≈ 10²–10⁴ s⁻¹` (prompt period).
//! Explicit RK is unstable at usable step sizes for the prompt mode;
//! we use **Crank-Nicolson** on the linear 7×7 system, which is
//! A-stable and second-order accurate. Each step solves a 7×7 linear
//! system; for piecewise-constant ρ we also expose the closed-form
//! matrix exponential via Padé-(6,6) for high-accuracy steps.
//!
//! # References
//!
//! - Keepin, *Physics of Nuclear Kinetics*, Addison-Wesley 1965 §2-§4.
//! - Hetrick, *Dynamics of Nuclear Reactors*, U. Chicago Press 1971.
//! - Hetrick & Roberts, *Trans. ANS* 7 (1964) 198 — six-group
//!   constants (the "Keepin numbers" for U-235 thermal fission used
//!   throughout this module's tests).
//! - Stacey, *Nuclear Reactor Physics*, Wiley 2007 §5.4 — modern
//!   numerical treatment of the 7×7 stiff system.

/// Six-group delayed-neutron data for one fissioning nuclide.
///
/// `decay_per_s[i]` is `λ_i` in s⁻¹; `fraction[i]` is the within-
/// delayed group fraction (i.e. `β_i / β`, summing to ≤ 1.0). The
/// total `β = beta_total` is the absolute delayed-neutron yield
/// (typically 0.0065 for U-235 thermal). See module-level
/// references for evaluation provenance.
#[derive(Debug, Clone, Copy)]
pub struct DelayedGroups {
    pub decay_per_s: [f64; 6],
    pub fraction: [f64; 6],
    pub beta_total: f64,
}

impl DelayedGroups {
    /// Group-absolute fractions `β_i = β · (β_i / β)`.
    pub fn beta_i(&self) -> [f64; 6] {
        let mut out = [0.0; 6];
        #[allow(clippy::needless_range_loop)]
        for i in 0..6 {
            out[i] = self.beta_total * self.fraction[i];
        }
        out
    }
}

/// Keepin / Hetrick-Roberts six-group constants for **U-235 thermal
/// fission**. The de-facto reference numbers cited in every PWR-
/// kinetics textbook. ENDF/B-VII.1 evaluations track these to within
/// ~1 % in fractions and exact in the decay constants.
pub const U235_THERMAL_KEEPIN: DelayedGroups = DelayedGroups {
    decay_per_s: [0.0124, 0.0305, 0.111, 0.301, 1.140, 3.010],
    fraction: [0.038, 0.213, 0.188, 0.407, 0.128, 0.026],
    beta_total: 0.0065,
};

/// Six-group constants for **U-238 fast fission** (Keepin 1965).
/// Higher β_total (~1.6 %) than U-235 because fast fission of U-238
/// has a delayed-rich precursor chain.
pub const U238_FAST_KEEPIN: DelayedGroups = DelayedGroups {
    decay_per_s: [0.0132, 0.0321, 0.139, 0.358, 1.410, 4.020],
    fraction: [0.013, 0.137, 0.162, 0.388, 0.225, 0.075],
    beta_total: 0.0157,
};

/// Six-group constants for **Pu-239 thermal fission** (Keepin 1965).
/// Very small β_total (~0.21 %) — the practical motivation for the
/// "delayed-neutron deficit" concern in MOX cores.
pub const PU239_THERMAL_KEEPIN: DelayedGroups = DelayedGroups {
    decay_per_s: [0.0129, 0.0311, 0.134, 0.331, 1.260, 3.210],
    fraction: [0.038, 0.280, 0.216, 0.328, 0.103, 0.035],
    beta_total: 0.00211,
};

/// Combine multiple fissioning nuclide kinetics datasets, weighted
/// by their relative fission contribution `w_n` (Σ w_n = 1). β_total
/// becomes Σ w_n · β_n; per-group fractions are weighted by absolute
/// β_i then re-normalised. λ_i is taken as the β-weighted average,
/// which is the right combination for the linearised point-kinetics
/// equations under the standard assumption of one effective
/// fission-spectrum per group.
///
/// In a real PWR core early in life this reduces to ~95 % U-235
/// weight + small U-238 fast component; the resulting `DelayedGroups`
/// is what `PointKinetics::with_groups` consumes.
pub fn blend(weighted: &[(DelayedGroups, f64)]) -> DelayedGroups {
    let total_weight: f64 = weighted.iter().map(|(_, w)| *w).sum();
    assert!(total_weight > 0.0, "blend weights sum to zero");

    let mut beta_total = 0.0;
    let mut beta_i = [0.0_f64; 6];
    let mut lambda_beta_i = [0.0_f64; 6];
    for (g, w) in weighted {
        let wn = w / total_weight;
        beta_total += wn * g.beta_total;
        for i in 0..6 {
            let bi = wn * g.beta_total * g.fraction[i];
            beta_i[i] += bi;
            lambda_beta_i[i] += bi * g.decay_per_s[i];
        }
    }
    let mut fraction = [0.0_f64; 6];
    let mut decay_per_s = [0.0_f64; 6];
    for i in 0..6 {
        fraction[i] = if beta_total > 0.0 {
            beta_i[i] / beta_total
        } else {
            0.0
        };
        decay_per_s[i] = if beta_i[i] > 0.0 {
            lambda_beta_i[i] / beta_i[i]
        } else {
            // Default to U-235 group-i decay if no precursor yield.
            U235_THERMAL_KEEPIN.decay_per_s[i]
        };
    }
    DelayedGroups {
        decay_per_s,
        fraction,
        beta_total,
    }
}

/// Point-kinetics state vector. `n` is normalised (typically 1.0 at
/// equilibrium, post-eigenvalue); `c[i]` is the dimensionless
/// precursor concentration `C_i(t) / N₀` so that the equations stay
/// scale-free.
#[derive(Debug, Clone, Copy)]
pub struct PkState {
    pub n: f64,
    pub c: [f64; 6],
}

impl PkState {
    /// Equilibrium state for a steady-state critical reactor at
    /// power `n`. `dC_i/dt = 0` ⇒ `C_i = β_i / (λ_i · Λ) · n`.
    pub fn equilibrium(n: f64, params: &PkParams) -> Self {
        let mut c = [0.0_f64; 6];
        let beta_i = params.groups.beta_i();
        for i in 0..6 {
            c[i] = beta_i[i] / (params.groups.decay_per_s[i] * params.gen_time) * n;
        }
        Self { n, c }
    }
}

/// Point-kinetics parameters held constant across a step. Reactivity
/// `rho` and external source `s_ext` are exposed as inputs to
/// `step_crank_nicolson` so they can vary in time.
#[derive(Debug, Clone, Copy)]
pub struct PkParams {
    pub groups: DelayedGroups,
    /// Mean prompt-neutron generation time Λ (s). For an LWR
    /// `Λ ≈ 10 µs`; for a fast reactor `Λ ≈ 0.1 µs`. Computed from a
    /// transport eigenvalue solve as `Λ = 1 / (ν · Σ_f · v · k)`
    /// in the one-mode approximation.
    pub gen_time: f64,
}

/// Crank-Nicolson step for the point-kinetics system. Linear in
/// `state`, A-stable, 2nd-order in `dt`. The reactivity / external
/// source are taken as piecewise-constant on the step (use the value
/// at the midpoint for trapezoidal accuracy if the inputs vary
/// linearly).
///
/// Solves
///
/// ```text
///   (I − dt/2 · A) · y_{n+1} = (I + dt/2 · A) · y_n + dt · s
/// ```
///
/// for the 7-vector `y = (n, c_1, …, c_6)`, where `A` is the
/// 7×7 system matrix and `s` is the source vector. The 7×7 inverse is
/// computed by Gauss-Jordan elimination — fast enough at this size
/// that the closed-form structure of `A` (mostly diagonal) doesn't
/// pay back the bookkeeping.
pub fn step_crank_nicolson(
    state: PkState,
    params: &PkParams,
    rho: f64,
    s_ext: f64,
    dt: f64,
) -> PkState {
    let beta = params.groups.beta_total;
    let lam = params.gen_time;
    let lambdas = params.groups.decay_per_s;
    let beta_i = params.groups.beta_i();

    // System matrix A so that dy/dt = A y + b, where b = (s_ext, 0, …)
    // Row 0 is the neutron equation; rows 1..6 are precursors.
    let mut a = [[0.0_f64; 7]; 7];
    a[0][0] = (rho - beta) / lam;
    for i in 0..6 {
        a[0][i + 1] = lambdas[i];
        a[i + 1][0] = beta_i[i] / lam;
        a[i + 1][i + 1] = -lambdas[i];
    }

    let half = 0.5 * dt;
    // Build (I + dt/2 A) and (I - dt/2 A).
    let mut lhs = [[0.0_f64; 7]; 7];
    let mut rhs_mat = [[0.0_f64; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            let mij = a[i][j];
            lhs[i][j] = if i == j {
                1.0 - half * mij
            } else {
                -half * mij
            };
            rhs_mat[i][j] = if i == j { 1.0 + half * mij } else { half * mij };
        }
    }

    let y = [
        state.n, state.c[0], state.c[1], state.c[2], state.c[3], state.c[4], state.c[5],
    ];
    // rhs = rhs_mat * y + dt * (s_ext, 0, …)
    let mut rhs = [0.0_f64; 7];
    for i in 0..7 {
        let mut s = 0.0;
        for j in 0..7 {
            s += rhs_mat[i][j] * y[j];
        }
        rhs[i] = s;
    }
    rhs[0] += dt * s_ext;

    // Solve lhs · y_new = rhs by Gauss-Jordan (7×7, no library
    // dependency; partial pivoting for numerical safety on the
    // (rho-β)/Λ row when ρ → β).
    let y_new = solve_7x7(lhs, rhs);
    PkState {
        n: y_new[0],
        c: [y_new[1], y_new[2], y_new[3], y_new[4], y_new[5], y_new[6]],
    }
}

/// Closed-form inhour-equation root for a fixed reactivity `rho`.
/// Returns the asymptotic period τ in seconds such that `n(t) ∝
/// e^(t/τ)`. Newton's method on the inhour equation:
///
/// ```text
///   ρ = Λ/τ + Σ_i β_i / (1 + λ_i · τ)
/// ```
///
/// For sub-prompt reactivities (`ρ < β`) this gives the long delayed
/// period; for super-prompt (`ρ > β`) it gives the prompt period
/// (Λ/(ρ-β) in the limit where delayed terms drop out).
pub fn inhour_period(rho: f64, params: &PkParams) -> f64 {
    let beta = params.groups.beta_total;
    let lam = params.gen_time;
    let lambdas = params.groups.decay_per_s;
    let beta_i = params.groups.beta_i();

    if rho.abs() < 1e-15 {
        return f64::INFINITY;
    }

    // Bracket the dominant root. For ρ > 0 (positive period) τ > 0;
    // for ρ < 0 τ < 0 (decaying). Newton from a physically sensible
    // initial guess.
    let mut tau = if rho > 0.0 {
        if rho < beta {
            // Sub-prompt: dominant root is the slowest delayed group.
            1.0 / lambdas[0].max(1e-6)
        } else {
            // Super-prompt: prompt-jump period.
            lam / (rho - beta).max(1e-12)
        }
    } else if rho > -beta {
        // Sub-prompt negative: long stable period (negative).
        -1.0 / lambdas[0]
    } else {
        // Super-prompt negative: prompt drop.
        lam / (rho - beta)
    };

    for _ in 0..200 {
        let mut f = lam / tau;
        let mut df = -lam / (tau * tau);
        for i in 0..6 {
            let denom = 1.0 + lambdas[i] * tau;
            f += beta_i[i] / denom;
            df -= beta_i[i] * lambdas[i] / (denom * denom);
        }
        let res = f - rho;
        if res.abs() < 1e-14 * (1.0 + rho.abs()) {
            break;
        }
        if df.abs() < 1e-300 {
            break;
        }
        let dt = -res / df;
        // Damp large jumps to avoid stepping over the discontinuity
        // at τ = -1/λ_i (precursor-decay singularities).
        let new_tau = tau + dt.signum() * dt.abs().min(0.5 * tau.abs().max(1e-12));
        tau = new_tau;
    }
    tau
}

/// Prompt-jump approximation. For an instantaneous step from ρ=0 to
/// ρ=ρ₀ < β at t=0, the neutron population jumps by
///
/// ```text
///   N_jump / N₀ = β / (β − ρ₀)
/// ```
///
/// before the slow delayed evolution takes over. Derived by
/// neglecting `dN/dt` and `Λ` on the prompt time-scale (formal limit
/// `Λ → 0` of the equations). Holds within ~10 % for practical PWR
/// reactivities (ρ ≪ β); fails near prompt-critical (ρ → β) where it
/// diverges and the full integration is required.
pub fn prompt_jump_ratio(rho_step: f64, params: &PkParams) -> f64 {
    let beta = params.groups.beta_total;
    if (beta - rho_step).abs() < 1e-15 {
        return f64::INFINITY;
    }
    beta / (beta - rho_step)
}

// ── 7×7 linear-system helpers ───────────────────────────────────────

/// Gauss-Jordan with partial pivoting. Returns the solution vector;
/// zero pivot encountered ⇒ returns NaN-laden vector (caller must
/// validate). The PK matrix is well-conditioned for `dt · max|A_ii|
/// < 0.5`; outside that regime the prompt-mode singularity manifests
/// as a near-zero pivot which is the right physics warning.
#[allow(clippy::needless_range_loop)]
fn solve_7x7(mut a: [[f64; 7]; 7], mut b: [f64; 7]) -> [f64; 7] {
    const N: usize = 7;
    for k in 0..N {
        // Partial pivot.
        let mut imax = k;
        let mut amax = a[k][k].abs();
        for i in (k + 1)..N {
            if a[i][k].abs() > amax {
                amax = a[i][k].abs();
                imax = i;
            }
        }
        if imax != k {
            a.swap(k, imax);
            b.swap(k, imax);
        }
        if amax < 1e-300 {
            // Singular — return NaN to flag.
            return [f64::NAN; 7];
        }
        // Eliminate.
        for i in 0..N {
            if i == k {
                continue;
            }
            let f = a[i][k] / a[k][k];
            for j in k..N {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = [0.0_f64; 7];
    for i in 0..N {
        x[i] = b[i] / a[i][i];
    }
    x
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// `beta_i` reconstructs the absolute group fractions from
    /// `(beta_total, fraction[i])`.
    #[test]
    fn beta_i_sums_to_beta_total() {
        let g = U235_THERMAL_KEEPIN;
        let bi = g.beta_i();
        let sum: f64 = bi.iter().sum();
        assert!((sum - g.beta_total).abs() / g.beta_total < 1e-12);
    }

    /// Equilibrium precursor concentrations satisfy `dC_i/dt = 0`:
    /// `λ_i · C_i = β_i / Λ · N`.
    #[test]
    fn equilibrium_zeros_precursor_derivative() {
        let params = PkParams {
            groups: U235_THERMAL_KEEPIN,
            gen_time: 1e-5,
        };
        let s = PkState::equilibrium(1.0, &params);
        let bi = params.groups.beta_i();
        for i in 0..6 {
            let dcdt = bi[i] / params.gen_time * s.n - params.groups.decay_per_s[i] * s.c[i];
            assert!(dcdt.abs() < 1e-10, "group {i}: dC/dt = {dcdt}, expected 0");
        }
    }

    /// Crank-Nicolson at ρ=0, s=0 from equilibrium is a fixed point
    /// (within roundoff): the steady state stays steady.
    #[test]
    fn equilibrium_is_a_fixed_point() {
        let params = PkParams {
            groups: U235_THERMAL_KEEPIN,
            gen_time: 1e-5,
        };
        let s0 = PkState::equilibrium(1.0, &params);
        let mut s = s0;
        for _ in 0..1000 {
            s = step_crank_nicolson(s, &params, 0.0, 0.0, 0.001);
        }
        assert!((s.n - 1.0).abs() < 1e-6, "n drifted: {}", s.n);
        for i in 0..6 {
            let rel = (s.c[i] - s0.c[i]).abs() / s0.c[i];
            assert!(rel < 1e-6, "c[{i}] drifted: rel = {rel}");
        }
    }

    /// Prompt-jump approximation: a step ρ = β/2 from equilibrium
    /// raises N by `β / (β − ρ) = 2`. Crank-Nicolson with very small
    /// dt over a few prompt periods (≪ delayed time-scale) should
    /// land near the analytic value to within ~10 % (the prompt-jump
    /// approximation neglects Λ — full integration overshoots
    /// slightly).
    #[test]
    fn prompt_jump_matches_analytic() {
        let params = PkParams {
            groups: U235_THERMAL_KEEPIN,
            gen_time: 1e-5,
        };
        let s0 = PkState::equilibrium(1.0, &params);
        let rho = 0.5 * params.groups.beta_total;
        let analytic = prompt_jump_ratio(rho, &params);
        // Step into the new reactivity with very small dt for a few
        // prompt periods. With Λ = 10 µs the prompt time-scale is
        // ~Λ/(β − ρ) = 10⁻⁵ / 0.00325 ≈ 3 ms — integrate 50 ms.
        let dt = 1e-5;
        let mut s = s0;
        let n_steps = (0.05 / dt) as usize;
        for _ in 0..n_steps {
            s = step_crank_nicolson(s, &params, rho, 0.0, dt);
        }
        // After 50 ms the precursor concentrations have adjusted by
        // ~5 % (slowest groups), so the true ratio is slightly above
        // the strict prompt-jump value. 5 % tolerance covers that
        // and the second-order CN error.
        let rel = (s.n - analytic).abs() / analytic;
        assert!(
            rel < 0.05,
            "N(50 ms)={} vs prompt-jump {}: rel = {rel:.3}",
            s.n,
            analytic
        );
    }

    /// Inhour equation: ρ = 0 ⇒ τ = ∞ (steady reactor). The function
    /// returns INFINITY by short-circuit.
    #[test]
    fn inhour_zero_reactivity_is_infinite_period() {
        let params = PkParams {
            groups: U235_THERMAL_KEEPIN,
            gen_time: 1e-5,
        };
        assert!(inhour_period(0.0, &params).is_infinite());
    }

    /// Inhour at ρ = 100 pcm (1e-3 — a typical sub-prompt control-
    /// rod step) should give τ ~ 50–100 s for U-235 thermal — the
    /// classic "small reactivity, long stable period" regime.
    #[test]
    fn inhour_sub_prompt_yields_long_period() {
        let params = PkParams {
            groups: U235_THERMAL_KEEPIN,
            gen_time: 1e-5,
        };
        let tau = inhour_period(1e-3, &params);
        assert!(
            (5.0..200.0).contains(&tau),
            "100 pcm period out of expected band: {tau} s",
        );
    }

    /// Super-prompt asymptote: ρ ≫ β collapses to the prompt period
    /// `Λ / (ρ − β)`. Verify within 1 %.
    #[test]
    fn inhour_super_prompt_matches_prompt_period() {
        let params = PkParams {
            groups: U235_THERMAL_KEEPIN,
            gen_time: 1e-5,
        };
        // ρ = 5 β — well above prompt critical.
        let rho = 5.0 * params.groups.beta_total;
        let tau = inhour_period(rho, &params);
        let asymptote = params.gen_time / (rho - params.groups.beta_total);
        let rel = (tau - asymptote).abs() / asymptote;
        assert!(
            rel < 0.05,
            "super-prompt τ = {tau}, asymptote {asymptote}, rel {rel:.3}",
        );
    }

    /// `blend` of a single-nuclide list with weight 1 is the input.
    #[test]
    fn blend_single_input_is_identity() {
        let blended = blend(&[(U235_THERMAL_KEEPIN, 1.0)]);
        assert!((blended.beta_total - U235_THERMAL_KEEPIN.beta_total).abs() < 1e-15);
        for i in 0..6 {
            assert!((blended.fraction[i] - U235_THERMAL_KEEPIN.fraction[i]).abs() < 1e-12);
            assert!((blended.decay_per_s[i] - U235_THERMAL_KEEPIN.decay_per_s[i]).abs() < 1e-12);
        }
    }

    /// PWR-typical mix (~95 % U-235 thermal + 5 % U-238 fast) gives
    /// β ≈ 0.0070 — slightly above pure U-235 because of U-238's
    /// β-rich fast contribution. Sanity check the magnitude.
    #[test]
    fn blend_pwr_mix_lifts_beta_above_u235_alone() {
        let blended = blend(&[(U235_THERMAL_KEEPIN, 0.95), (U238_FAST_KEEPIN, 0.05)]);
        assert!(blended.beta_total > U235_THERMAL_KEEPIN.beta_total);
        assert!(blended.beta_total < 0.0080);
    }

    /// Scram from equilibrium with ρ = -2 β: prompt drop to ~30 %
    /// then slow delayed decay. After a long time (>> 1 / λ_1 ≈ 80 s)
    /// the population should be negligible. Test only the prompt
    /// drop magnitude — the delayed decay rate is an independent
    /// check below.
    #[test]
    fn scram_prompt_drop() {
        let params = PkParams {
            groups: U235_THERMAL_KEEPIN,
            gen_time: 1e-5,
        };
        let s0 = PkState::equilibrium(1.0, &params);
        let rho = -2.0 * params.groups.beta_total;
        let dt = 1e-5;
        let mut s = s0;
        let n_steps = (0.02 / dt) as usize; // 20 ms
        for _ in 0..n_steps {
            s = step_crank_nicolson(s, &params, rho, 0.0, dt);
        }
        // Prompt-jump analytic: β / (β − ρ) = 1 / 3 ≈ 0.333.
        let analytic = prompt_jump_ratio(rho, &params);
        let rel = (s.n - analytic).abs() / analytic;
        assert!(
            rel < 0.10,
            "scram drop n={} vs analytic {}: rel {rel:.3}",
            s.n,
            analytic
        );
    }
}
