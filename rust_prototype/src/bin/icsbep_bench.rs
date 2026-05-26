// SPDX-License-Identifier: MIT
//! **DEPRECATED** — thin shim around `benchmark_runner`.
//!
//! The original `icsbep_bench` (subprocess-per-case loop) is retired
//! per `docs/benchmark-pipeline-spec.md` Phase 5. The in-process,
//! heterogeneous CPU/GPU pipeline lives at `bin/benchmark_runner.rs`
//! and accepts a superset of the historical flags.
//!
//! This shim:
//!   1. Translates the legacy flag surface (`<bench_dir>` positional,
//!      `--data-dir`, `--filter`, `--output`, `--n-sigma`) into the
//!      new `benchmark_runner` flags.
//!   2. Spawns `benchmark_runner` with the translated argv and waits
//!      for it to exit, propagating the exit code.
//!   3. Prints a one-line deprecation notice on stderr the first time
//!      it's invoked in a sweep so CI / dashboards notice the
//!      migration.
//!
//! Once downstream pipelines are confirmed migrated, delete this
//! shim. Until then, every caller of `icsbep_bench` transparently
//! gets the new pipeline.

use std::path::PathBuf;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    eprintln!(
        "[DEPRECATED] icsbep_bench is a shim around `benchmark_runner`. \
         See docs/benchmark-pipeline-spec.md Phase 5. Update CI / sweep \
         scripts to invoke `benchmark_runner` directly."
    );

    // Locate the new binary. It must live next to this one in the
    // same `target/<profile>/` directory.
    let me = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("benchmark_runner"));
    let bin_dir = me.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let new_bin = if cfg!(windows) {
        bin_dir.join("benchmark_runner.exe")
    } else {
        bin_dir.join("benchmark_runner")
    };
    if !new_bin.exists() {
        eprintln!(
            "icsbep_bench: cannot find benchmark_runner at {} — \
             rebuild the workspace (`cargo build --bin benchmark_runner`).",
            new_bin.display(),
        );
        return ExitCode::from(127);
    }

    // Translate legacy argv → new argv. The legacy form was:
    //   icsbep_bench <bench_dir> --data-dir <X> [--filter <Y>]
    //                 [--output <CSV>] [--n-sigma <S>]
    // All other args (--seeds, --particles, --batches, --inactive) are
    // already named identically by both binaries, so they pass
    // through unchanged.
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut translated: Vec<String> = Vec::with_capacity(args.len() + 4);
    // First positional arg = bench dir (if not preceded by a flag).
    if let Some(first) = args.first() {
        if !first.starts_with("--") {
            translated.push("--bench-dir".to_string());
            translated.push(first.clone());
            args.remove(0);
        }
    }
    // Remap `--output` → `--csv`. Pass everything else as-is.
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--output" {
            translated.push("--csv".to_string());
            i += 1;
            if i < args.len() {
                translated.push(args[i].clone());
                i += 1;
            }
        } else {
            translated.push(args[i].clone());
            i += 1;
        }
    }

    match Command::new(&new_bin).args(&translated).status() {
        Ok(status) => match status.code() {
            Some(code) => ExitCode::from(code as u8),
            None => ExitCode::from(130), // killed by signal
        },
        Err(e) => {
            eprintln!("icsbep_bench: failed to spawn {}: {e}", new_bin.display());
            ExitCode::from(126)
        }
    }
}
