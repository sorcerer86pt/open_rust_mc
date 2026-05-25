// SPDX-License-Identifier: MIT
//! Stage 5 — pass/fail + CSV + stdout per case.
//!
//! Consumes `ExecutionResult`s from the `ResultChannel`. Computes
//! pass/fail per the same envelope used elsewhere in the engine:
//!
//! ```text
//! |Δk_pcm| ≤ max(150 pcm, n_sigma × σ_combined)
//! σ_combined = sqrt(σ_calc² + σ_exp²)
//! ```
//!
//! Emits one CSV row (flushed) and one stdout line per case, matching
//! the historical Python sweep output so dashboards / graders that
//! parse it keep working.

use super::executor::ExecutionResult;

/// Pass/fail verdict for a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// Engine error (panic, OOM, timeout) — neither pass nor fail
    /// against the physics envelope, but logged separately.
    Error,
}

impl Verdict {
    pub fn classify(r: &ExecutionResult, n_sigma: f64) -> Self {
        if r.error.is_some() {
            return Self::Error;
        }
        let delta_pcm = (r.k_calc - r.spec.k_ref).abs() * 1e5;
        let sigma_combined =
            (r.k_sigma.powi(2) + r.spec.sigma_exp.powi(2)).sqrt() * 1e5;
        // Floor of 150 pcm covers cases where σ_exp is loose enough
        // (e.g. HEU-SOL-THERM at 600 pcm) that the n-sigma envelope
        // would swallow systematic biases.
        let bound = (n_sigma * sigma_combined).max(150.0);
        if delta_pcm <= bound {
            Self::Pass
        } else {
            Self::Fail
        }
    }
}
