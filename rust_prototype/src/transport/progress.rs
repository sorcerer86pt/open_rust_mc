// SPDX-License-Identifier: MIT
//! Live progress bar for the eigenvalue power-iteration loop.
//!
//! One `EigenProgress` per `EigenvalueRunner::run` call. The bar
//! auto-detects whether `stderr` is a TTY:
//!   - Interactive terminal → animated bar with running k_eff ± σ.
//!   - Piped / redirected   → `ProgressBar::hidden()`; no output.
//!
//! `verbose=true` mode also suppresses the bar so it doesn't fight
//! with `println!` debug output.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;

enum Sink {
    /// Animated in-place bar — only when stderr is a real TTY.
    Bar(ProgressBar),
    /// Plain `eprintln!` per tick — used when the user forces
    /// progress on but stderr is being piped (cargo test --nocapture,
    /// CI logs, `tee`). Indicatif's bar collapses to nothing under a
    /// pipe, so we fall back to one log line per batch.
    Lines {
        backend: String,
        total: u32,
    },
    Off,
}

pub struct EigenProgress {
    sink: Sink,
    inactive: u32,
    n_active: u32,
    k_sum: f64,
    k_sq_sum: f64,
}

impl EigenProgress {
    pub fn new(batches: u32, inactive: u32, backend: &str, verbose: bool) -> Self {
        // `verbose=true` is for line-buffered debug printing — never
        // show the bar there (would fight with println!). Otherwise:
        //   stderr is a TTY                    → animated bar
        //   OPEN_RUST_MC_PROGRESS=1 + non-TTY  → one log line per batch
        //   else                                → silent
        // The env-var override is the path for `cargo test --
        // --nocapture` (cargo wraps stderr → no TTY) and CI sessions
        // where the user still wants per-batch feedback.
        let forced = matches!(
            std::env::var("OPEN_RUST_MC_PROGRESS")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1" | "force" | "always" | "true" | "yes")
        );
        let is_tty = std::io::stderr().is_terminal();
        let sink = if verbose {
            Sink::Off
        } else if is_tty {
            let pb = ProgressBar::with_draw_target(
                Some(batches as u64),
                ProgressDrawTarget::stderr_with_hz(4),
            );
            let style = ProgressStyle::with_template(
                "{prefix:>4} [{bar:30.cyan/blue}] {pos:>4}/{len} | {msg} | ETA {eta}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▉▊▋▌▍▎▏  ");
            pb.set_style(style);
            pb.set_prefix(backend.to_string());
            Sink::Bar(pb)
        } else if forced {
            Sink::Lines {
                backend: backend.to_string(),
                total: batches,
            }
        } else {
            Sink::Off
        };
        Self {
            sink,
            inactive,
            n_active: 0,
            k_sum: 0.0,
            k_sq_sum: 0.0,
        }
    }

    /// Advance the bar by one batch and refresh the running-mean
    /// message. `k_batch` is this batch's k_eff estimate.
    pub fn tick(&mut self, batch: u32, k_batch: f64) {
        let active = batch > self.inactive;
        if active {
            self.n_active += 1;
            self.k_sum += k_batch;
            self.k_sq_sum += k_batch * k_batch;
        }
        let msg = if self.n_active >= 2 {
            let n = self.n_active as f64;
            let mean = self.k_sum / n;
            // Stderr of the mean: sample variance / N.
            let var = ((self.k_sq_sum - n * mean * mean) / (n - 1.0)).max(0.0);
            let sigma_pcm = (var / n).sqrt() * 1.0e5;
            format!("k={k_batch:.5}  <k>={mean:.5} +/- {sigma_pcm:.0} pcm")
        } else if active {
            format!("k={k_batch:.5}  (1st active)")
        } else {
            format!("k={k_batch:.5}  (inactive)")
        };
        match &self.sink {
            Sink::Bar(pb) => {
                pb.set_message(msg);
                pb.inc(1);
            }
            Sink::Lines { backend, total } => {
                eprintln!("  [{backend}] batch {batch:>4}/{total} | {msg}");
            }
            Sink::Off => {}
        }
    }

    pub fn finish(&self) {
        if let Sink::Bar(pb) = &self.sink {
            pb.finish_and_clear();
        }
    }
}
