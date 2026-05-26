// SPDX-License-Identifier: MIT
//! Running pass/fail / error tally for the sweep.
//!
//! Owned by the ResultProcessor thread; read by the Finalizer for
//! the end-of-sweep summary.

use super::executor::ExecutionResult;
use super::result_processor::Verdict;

#[derive(Default)]
pub struct RunState {
    pub passes: usize,
    pub fails: usize,
    pub errors: usize,
    pub results: Vec<ExecutionResult>,
    /// Per-case `(case_id, delta_pcm)` for the scatter plot. Kept
    /// separately so the plotting hook doesn't have to walk
    /// `results` to recompute Δ_pcm per refresh.
    pub deltas: Vec<(String, f64)>,
}

impl RunState {
    pub fn record(&mut self, r: ExecutionResult, n_sigma: f64) {
        let v = Verdict::classify(&r, n_sigma);
        match v {
            Verdict::Pass => self.passes += 1,
            Verdict::Fail => self.fails += 1,
            Verdict::Error => self.errors += 1,
        }
        if r.error.is_none() {
            let delta_pcm = (r.k_calc - r.summary.k_ref) * 1e5;
            self.deltas.push((r.summary.case_id.clone(), delta_pcm));
        }
        self.results.push(r);
    }

    /// Exit code per §5 of the spec: 0 all-pass, 1 any-fail, 2 any-error.
    pub fn exit_code(&self) -> i32 {
        if self.errors > 0 {
            2
        } else if self.fails > 0 {
            1
        } else {
            0
        }
    }

    pub fn total(&self) -> usize {
        self.passes + self.fails + self.errors
    }
}
