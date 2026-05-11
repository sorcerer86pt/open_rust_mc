//! ICSBEP benchmark runner.
//!
//! Walks a directory of benchmark case files (`*.json`), invokes the
//! configured engine binary for each, parses k_eff / σ from stdout, and
//! emits a CSV of `Δ_pcm` and pass/fail against the published reference
//! k_eff. The same harness handles ICSBEP cases, internal regression
//! suites, and any future benchmark family — the case file picks the
//! binary and labels the suite.
//!
//! Each case file is forward-compatible with the `.nmc` manifest format:
//! the `benchmark` block matches §3.1 of `specs/nmc/NMC_SPEC.md`
//! verbatim. The `runner` block is the bench-harness extension that
//! describes which binary to invoke with which args — once
//! `Geometry::from_json` is implemented, `runner` will be dropped and
//! the bench runner will instead load `scene.json` and call
//! `EigenvalueRunner` directly.
//!
//! # Case file format
//!
//! ```json
//! {
//!   "benchmark": {
//!     "suite":           "ICSBEP",
//!     "case_id":         "HEU-MET-FAST-001",
//!     "case_name":       "Godiva",
//!     "category":        "HEU-MET-FAST",
//!     "k_eff_reference": 1.0000,
//!     "k_eff_sigma":     0.0010,
//!     "source":          "ICSBEP Handbook 2022, HMF-001"
//!   },
//!   "runner": {
//!     "binary": "godiva",
//!     "args":   ["--mode", "table", "--batches", "150"],
//!     "data_arg": "positional",
//!     "k_label":  "k_eff (collision)"
//!   }
//! }
//! ```
//!
//! # CLI
//!
//! ```text
//! icsbep_bench <bench_dir> --data-dir <ENDF_dir>
//!     [--bin-dir <target/release>] [--output <file.csv>]
//!     [--filter <substr>] [--n-sigma <2.0>]
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(
    name = "icsbep_bench",
    about = "Run a directory of ICSBEP / regression benchmark cases and report against reference k_eff"
)]
struct Args {
    /// Directory containing benchmark case files (`*.json`).
    bench_dir: PathBuf,

    /// Directory containing ENDF/B-VII.1 HDF5 nuclear data. Passed to
    /// each runner binary as the positional `data_dir` argument.
    #[arg(long)]
    data_dir: PathBuf,

    /// Directory containing pre-built binaries. Defaults to
    /// `target/release/` relative to the workspace root if found,
    /// otherwise to `target/debug/`. Each case's
    /// `runner.binary` is resolved as `<bin_dir>/<binary>(.exe on Windows)`.
    #[arg(long)]
    bin_dir: Option<PathBuf>,

    /// Output CSV path. Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Only run cases whose `case_id` contains this substring (case-
    /// sensitive). Useful for re-running a single case while iterating.
    #[arg(long)]
    filter: Option<String>,

    /// Pass criterion: `|Δk| < n_sigma · √(σ_calc² + σ_exp²)`. Default
    /// 2.0 — i.e. 95 %-confidence two-sided pass.
    #[arg(long, default_value_t = 2.0)]
    n_sigma: f64,

    /// Print full subprocess stdout/stderr for cases that fail to
    /// parse. Otherwise only a short error is printed.
    #[arg(long, default_value_t = false)]
    verbose_on_error: bool,
}

#[derive(Deserialize, Debug)]
struct CaseFile {
    benchmark: BenchmarkBlock,
    /// Optional. When present, the runner invokes the named binary
    /// with these CLI args and parses k_eff from its stdout. When
    /// absent, the case is `BLOCKED` until `Geometry::from_json`
    /// lands — at that point the bench runner will instead deserialize
    /// `scene` and run the case in-process via `EigenvalueRunner`.
    #[serde(default)]
    runner: Option<RunnerBlock>,
    /// Optional. Full geometry+materials in NMC scene-schema form, used
    /// by the in-process runner once `Geometry::from_json` exists.
    /// Present on cases imported from `mit-crpg/benchmarks`; absent on
    /// hand-authored cases that exercise an existing binary directly.
    #[serde(default)]
    #[allow(dead_code)]
    scene: Option<serde_json::Value>,
    /// When set to `true`, the case is treated as a placeholder — the
    /// runner emits a `SKIP` row instead of invoking the binary. Used
    /// for cases where the geometry/materials are known but the
    /// official ICSBEP-handbook composition values have not yet been
    /// keyed in. Author-side gate to prevent shipping cases with
    /// approximate compositions.
    #[serde(default)]
    pending_composition: bool,
}

#[derive(Deserialize, Debug)]
struct BenchmarkBlock {
    suite: String,
    case_id: String,
    #[serde(default)]
    case_name: Option<String>,
    #[serde(default)]
    category: Option<String>,
    k_eff_reference: f64,
    k_eff_sigma: f64,
    #[serde(default)]
    source: Option<String>,
    /// Free-text caveats from the benchmark spec. Carried through to
    /// the CSV `source` column when present.
    #[serde(default)]
    #[allow(dead_code)]
    notes: Option<String>,
}

#[derive(Deserialize, Debug)]
struct RunnerBlock {
    /// Name of the binary to invoke (without path or `.exe` suffix).
    binary: String,
    /// CLI args to pass before / after `data_arg` placement.
    #[serde(default)]
    args: Vec<String>,
    /// Where to place the data_dir. `"positional"` appends it as the
    /// last argument (the convention `godiva`, `pwr_pincell`, … all
    /// use). `"flag:--data-dir"` would pass `--data-dir <path>` (not
    /// yet used by any binary in tree but supported for forward
    /// compatibility).
    #[serde(default = "default_data_arg")]
    data_arg: String,
    /// Substring of the stdout line containing the k_eff. Default is
    /// `"k_eff (collision)"` which matches `godiva.rs` output. Other
    /// binaries print `"k_inf"` and similar — set per case.
    #[serde(default = "default_k_label")]
    k_label: String,
}

fn default_data_arg() -> String {
    "positional".to_string()
}

fn default_k_label() -> String {
    "k_eff (collision)".to_string()
}

#[derive(Debug)]
struct CaseResult {
    case_id: String,
    suite: String,
    category: String,
    case_name: String,
    k_ref: f64,
    sigma_exp: f64,
    k_calc: Option<f64>,
    sigma_calc: Option<f64>,
    delta_pcm: Option<f64>,
    n_sigma_actual: Option<f64>,
    pass: Option<bool>,
    skipped: bool,
    /// True when the case has a `scene` block but no `runner` — it
    /// can't be executed yet (waiting on `Geometry::from_json`).
    /// Distinct from `error`: blocked cases are expected, errors are
    /// not.
    blocked: bool,
    wall_s: f64,
    error: Option<String>,
    source: String,
}

fn parse_k_eff_line(stdout: &str, label_substring: &str) -> Option<(f64, f64)> {
    // Looks for a line containing `label_substring` and pulls two
    // floating-point numbers separated by "+/-". Tolerates leading
    // whitespace, label variants ("k_eff (collision)", "k_inf",
    // "k_eff (track-len)"), and engine print drift like `=` vs `:`.
    for line in stdout.lines() {
        if !line.contains(label_substring) {
            continue;
        }
        // Find "+/-" or "±" and split.
        let separator_idx = line.find("+/-").or_else(|| line.find('±'))?;
        let left = &line[..separator_idx];
        let right = &line[separator_idx + 3..];
        let k = extract_float(left)?;
        let sigma = extract_float(right)?;
        return Some((k, sigma));
    }
    None
}

/// Find the *last* float in `s`. Floats are runs of digits + optional
/// '.' + optional exponent. Returns `None` if no float is present.
fn extract_float(s: &str) -> Option<f64> {
    let bytes = s.as_bytes();
    let mut end = bytes.len();
    while end > 0 && !is_float_char(bytes[end - 1]) {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && is_float_char(bytes[start - 1]) {
        start -= 1;
    }
    if start == end {
        return None;
    }
    s.get(start..end)?.parse().ok()
}

fn is_float_char(b: u8) -> bool {
    b.is_ascii_digit() || b == b'.' || b == b'-' || b == b'+' || b == b'e' || b == b'E'
}

fn resolve_bin_dir(arg: Option<PathBuf>) -> PathBuf {
    if let Some(p) = arg {
        return p;
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let release = cwd.join("target").join("release");
    if release.exists() {
        return release;
    }
    cwd.join("target").join("debug")
}

fn binary_path(bin_dir: &Path, name: &str) -> PathBuf {
    let mut p = bin_dir.join(name);
    if cfg!(windows) {
        p.set_extension("exe");
    }
    p
}

/// Build a default `CaseResult` carrying only the benchmark-block
/// fields. Callers then mutate the result fields (`error`, `pass`,
/// `k_calc`, `blocked`, …) before returning. Keeps the function from
/// repeating the full 15-field struct literal at every early return.
fn empty_result(case: &CaseFile) -> CaseResult {
    CaseResult {
        case_id:        case.benchmark.case_id.clone(),
        suite:          case.benchmark.suite.clone(),
        category:       case.benchmark.category.clone().unwrap_or_default(),
        case_name:      case.benchmark.case_name.clone().unwrap_or_default(),
        k_ref:          case.benchmark.k_eff_reference,
        sigma_exp:      case.benchmark.k_eff_sigma,
        k_calc:         None,
        sigma_calc:     None,
        delta_pcm:      None,
        n_sigma_actual: None,
        pass:           None,
        skipped:        false,
        blocked:        false,
        wall_s:         0.0,
        error:          None,
        source:         case.benchmark.source.clone().unwrap_or_default(),
    }
}

fn run_case(case: &CaseFile, bin_dir: &Path, data_dir: &Path, n_sigma: f64) -> CaseResult {
    // Cases marked `pending_composition = true` are author-side gates
    // (composition not yet keyed in from the handbook). Always SKIP.
    if case.pending_composition {
        let mut r = empty_result(case);
        r.skipped = true;
        r.error = Some("pending_composition gate — composition not yet entered from official handbook".into());
        return r;
    }

    let runner = match &case.runner {
        Some(r) => r,
        None => {
            // No runner block. If there's a `scene` block we know the
            // case will run via `Geometry::from_json` once that
            // deserializer lands; until then it's BLOCKED, not an
            // error. If neither is present that's a malformed case.
            let mut r = empty_result(case);
            if case.scene.is_some() {
                r.blocked = true;
                r.error = Some("BLOCKED on Geometry::from_json — scene-only case, no runner block".into());
            } else {
                r.error = Some("malformed case: neither `runner` nor `scene` present".into());
            }
            return r;
        }
    };

    let bin_path = binary_path(bin_dir, &runner.binary);
    if !bin_path.exists() {
        let mut r = empty_result(case);
        r.error = Some(format!("binary not found: {}", bin_path.display()));
        return r;
    }

    let mut cmd = Command::new(&bin_path);
    // Order: <args from manifest>, then the data_dir argument.
    for a in &runner.args {
        cmd.arg(a);
    }
    match runner.data_arg.as_str() {
        "positional" => {
            cmd.arg(data_dir);
        }
        flag if flag.starts_with("flag:") => {
            cmd.arg(&flag[5..]).arg(data_dir);
        }
        other => {
            let mut r = empty_result(case);
            r.error = Some(format!("unknown data_arg mode: {other}"));
            return r;
        }
    }

    let start = Instant::now();
    let output = cmd.output();
    let wall_s = start.elapsed().as_secs_f64();
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            let mut r = empty_result(case);
            r.wall_s = wall_s;
            r.error = Some(format!("subprocess error: {e}"));
            return r;
        }
    };
    if !output.status.success() {
        let mut r = empty_result(case);
        r.wall_s = wall_s;
        r.error = Some(format!("non-zero exit: {}", output.status));
        return r;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = parse_k_eff_line(&stdout, &runner.k_label);
    let (k_calc, sigma_calc) = match parsed {
        Some(pair) => pair,
        None => {
            let mut r = empty_result(case);
            r.wall_s = wall_s;
            r.error = Some(format!(
                "could not parse k_eff line containing '{}'",
                runner.k_label
            ));
            return r;
        }
    };

    let delta_pcm = (k_calc - case.benchmark.k_eff_reference) * 1.0e5;
    let combined_sigma = (sigma_calc * sigma_calc
        + case.benchmark.k_eff_sigma * case.benchmark.k_eff_sigma)
        .sqrt();
    let n_sigma_actual = if combined_sigma > 0.0 {
        (k_calc - case.benchmark.k_eff_reference).abs() / combined_sigma
    } else {
        f64::INFINITY
    };
    let pass = n_sigma_actual <= n_sigma;
    let mut r = empty_result(case);
    r.k_calc = Some(k_calc);
    r.sigma_calc = Some(sigma_calc);
    r.delta_pcm = Some(delta_pcm);
    r.n_sigma_actual = Some(n_sigma_actual);
    r.pass = Some(pass);
    r.wall_s = wall_s;
    r
}

fn write_csv(results: &[CaseResult], out: &mut dyn std::io::Write) -> std::io::Result<()> {
    writeln!(
        out,
        "suite,case_id,case_name,category,k_calc,sigma_calc,k_ref,sigma_exp,delta_pcm,n_sigma,pass,wall_s,error,source"
    )?;
    for r in results {
        let k_calc = r.k_calc.map(|v| format!("{v:.6}")).unwrap_or_default();
        let sigma_calc = r.sigma_calc.map(|v| format!("{v:.6}")).unwrap_or_default();
        let delta = r.delta_pcm.map(|v| format!("{v:.1}")).unwrap_or_default();
        let n_sigma = r.n_sigma_actual.map(|v| format!("{v:.2}")).unwrap_or_default();
        let pass = match r.pass {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "ERROR",
        };
        let error = r.error.clone().unwrap_or_default().replace(',', ";");
        let source = r.source.replace(',', ";");
        writeln!(
            out,
            "{},{},{},{},{},{},{:.6},{:.6},{},{},{},{:.1},{},{}",
            r.suite,
            r.case_id,
            r.case_name,
            r.category,
            k_calc,
            sigma_calc,
            r.k_ref,
            r.sigma_exp,
            delta,
            n_sigma,
            pass,
            r.wall_s,
            error,
            source,
        )?;
    }
    Ok(())
}

/// Five terminal labels: PASS / FAIL / SKIP / BLOCK / ERR.
///
/// Note: this orchestrator is transitional. Once `Geometry::from_json`
/// lands, the canonical ICSBEP regression run becomes
/// `cargo test --release --ignored` with a libtest-mimic harness that
/// loads the same `bench/icsbep/*.json` files — at which point the
/// PASS/FAIL/IGNORED reporting is `cargo test`'s built-in. This bench
/// binary only exists to (a) drive the small set of cases that
/// currently invoke a separate compiled binary (`godiva`, `pwr_pincell`)
/// as a subprocess — something libtest doesn't orchestrate cleanly —
/// and (b) make the 367 scene-only cases visible as `BLOCKED` rather
/// than silently absent.
fn status_label(r: &CaseResult) -> &'static str {
    if r.blocked { "BLOCK" }
    else if r.skipped { "SKIP" }
    else if r.error.is_some() { "ERR" }
    else if r.pass == Some(true) { "PASS" }
    else { "FAIL" }
}

fn human_summary(results: &[CaseResult]) {
    println!("\n=== ICSBEP regression summary ===");
    println!(
        "{:<5} {:<32} {:<14} {:>10} {:>10} {:>10} {:>8} {:>6}",
        "STAT", "case", "suite", "k_calc", "k_ref", "Δ pcm", "n_σ", "wall"
    );
    // Per-suite tallies: (pass, fail, skip, block, err).
    let mut by_suite: BTreeMap<String, [usize; 5]> = BTreeMap::new();
    for r in results {
        let lbl = status_label(r);
        let counts = by_suite.entry(r.suite.clone()).or_default();
        match lbl {
            "PASS"  => counts[0] += 1,
            "FAIL"  => counts[1] += 1,
            "SKIP"  => counts[2] += 1,
            "BLOCK" => counts[3] += 1,
            _       => counts[4] += 1,
        }
        let k_calc_str = r.k_calc.map(|v| format!("{v:.5}")).unwrap_or_else(|| "—".into());
        let delta_str  = r.delta_pcm.map(|v| format!("{v:+.0}")).unwrap_or_else(|| "—".into());
        let n_sig_str  = r.n_sigma_actual.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into());
        println!(
            "{:<5} {:<32} {:<14} {:>10} {:>10.5} {:>10} {:>8} {:>5.1}s",
            lbl, r.case_id, r.suite, k_calc_str, r.k_ref, delta_str, n_sig_str, r.wall_s,
        );
        if let Some(err) = &r.error
            && !r.blocked
            && !r.skipped
        {
            println!("       error: {err}");
        }
    }
    println!("\nPer-suite totals:");
    for (suite, c) in &by_suite {
        println!(
            "  {:<14}  {:>3} PASS  {:>3} FAIL  {:>3} SKIP  {:>4} BLOCK  {:>3} ERR  ({} total)",
            suite, c[0], c[1], c[2], c[3], c[4], c.iter().sum::<usize>(),
        );
    }
}

fn main() {
    let args = Args::parse();
    let bin_dir = resolve_bin_dir(args.bin_dir);

    // Discover case files.
    let mut case_paths: Vec<PathBuf> = match fs::read_dir(&args.bench_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect(),
        Err(e) => {
            eprintln!("cannot read bench dir {}: {e}", args.bench_dir.display());
            std::process::exit(2);
        }
    };
    case_paths.sort();
    if case_paths.is_empty() {
        eprintln!("no *.json cases found under {}", args.bench_dir.display());
        std::process::exit(2);
    }

    println!("=== open_rust_mc — ICSBEP regression run ===");
    println!("Bench dir:  {}", args.bench_dir.display());
    println!("Data dir:   {}", args.data_dir.display());
    println!("Binaries:   {}", bin_dir.display());
    println!("Pass crit:  |Δk| < {:.1} σ_combined", args.n_sigma);
    println!("Cases:      {}", case_paths.len());

    let mut results = Vec::with_capacity(case_paths.len());
    for path in &case_paths {
        let case: CaseFile = match fs::read_to_string(path).and_then(|s| {
            serde_json::from_str::<CaseFile>(&s)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  [parse error] {}: {e}", path.display());
                continue;
            }
        };

        if let Some(f) = &args.filter
            && !case.benchmark.case_id.contains(f.as_str())
        {
            continue;
        }
        print!("  running {} ... ", case.benchmark.case_id);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let result = run_case(&case, &bin_dir, &args.data_dir, args.n_sigma);
        match status_label(&result) {
            "PASS" => println!("PASS ({:.1}s)", result.wall_s),
            "FAIL" => println!(
                "FAIL ({:.1}s, Δ = {:+.0} pcm, {:.2}σ)",
                result.wall_s,
                result.delta_pcm.unwrap_or(0.0),
                result.n_sigma_actual.unwrap_or(0.0),
            ),
            "BLOCK" => println!("BLOCKED (Geometry::from_json not yet implemented)"),
            "SKIP" => println!(
                "SKIP ({})",
                result.error.as_deref().unwrap_or("pending_composition")
            ),
            _ => {
                println!(
                    "ERROR ({:.1}s): {}",
                    result.wall_s,
                    result.error.as_deref().unwrap_or("(no error message)")
                );
                if args.verbose_on_error
                    && let Some(err) = &result.error
                {
                    eprintln!("    {err}");
                }
            }
        }
        results.push(result);
    }

    human_summary(&results);

    if let Some(out_path) = &args.output {
        match fs::File::create(out_path) {
            Ok(mut f) => {
                if let Err(e) = write_csv(&results, &mut f) {
                    eprintln!("CSV write error: {e}");
                    std::process::exit(2);
                }
                println!("\nCSV written: {}", out_path.display());
            }
            Err(e) => {
                eprintln!("cannot create {}: {e}", out_path.display());
                std::process::exit(2);
            }
        }
    }

    // Exit code reflects worst case: 0 if all PASS, 1 if any FAIL, 2 if any ERROR.
    let any_err = results.iter().any(|r| r.error.is_some());
    let any_fail = results.iter().any(|r| matches!(r.pass, Some(false)));
    let code = if any_err {
        2
    } else if any_fail {
        1
    } else {
        0
    };
    std::process::exit(code);
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn extract_float_picks_last_number() {
        assert_eq!(extract_float("k_eff = 1.00079"), Some(1.00079));
        assert_eq!(extract_float("    1.23e-4"), Some(1.23e-4));
        assert_eq!(extract_float("    -1.5"), Some(-1.5));
        assert_eq!(extract_float("no numbers here"), None);
        assert_eq!(extract_float(""), None);
    }

    #[test]
    fn parse_godiva_collision_line() {
        let stdout = r#"
=== open_rust_mc — Godiva Eigenvalue Benchmark ===

  Table run:
    k_eff (collision) = 1.00079 +/- 0.00038
    k_eff (track-len) = 0.99950 +/- 0.00020
    delta(exp)       = 79 pcm
"#;
        let (k, sigma) = parse_k_eff_line(stdout, "k_eff (collision)").unwrap();
        assert!((k - 1.00079).abs() < 1e-9);
        assert!((sigma - 0.00038).abs() < 1e-9);
    }

    #[test]
    fn parse_pwr_pincell_kinf_line() {
        let stdout = r#"
=== PWR pin cell ===
    k_inf            = 1.32775 +/- 0.00183
    delta(exp)       = ... pcm
"#;
        let (k, sigma) = parse_k_eff_line(stdout, "k_inf").unwrap();
        assert!((k - 1.32775).abs() < 1e-9);
        assert!((sigma - 0.00183).abs() < 1e-9);
    }

    #[test]
    fn parse_handles_unicode_sigma() {
        // Some binaries print "± 0.00018" with the unicode glyph.
        let stdout = "    k_eff (collision) = 1.00342 ± 0.00018\n";
        let (k, sigma) = parse_k_eff_line(stdout, "k_eff (collision)").unwrap();
        assert!((k - 1.00342).abs() < 1e-9);
        assert!((sigma - 0.00018).abs() < 1e-9);
    }

    #[test]
    fn parse_returns_none_when_label_missing() {
        let stdout = "    k_eff (track-len) = 1.0 +/- 0.001\n";
        assert!(parse_k_eff_line(stdout, "k_eff (collision)").is_none());
    }

    #[test]
    fn case_file_round_trips_through_serde() {
        let s = r#"{
            "benchmark": {
              "suite": "ICSBEP",
              "case_id": "HEU-MET-FAST-001",
              "case_name": "Godiva",
              "k_eff_reference": 1.0,
              "k_eff_sigma": 0.001,
              "source": "ICSBEP Handbook 2022, HMF-001",
              "category": "HEU-MET-FAST"
            },
            "runner": {
              "binary": "godiva",
              "args": ["--mode", "table"],
              "k_label": "k_eff (collision)"
            }
        }"#;
        let case: CaseFile = serde_json::from_str(s).unwrap();
        assert_eq!(case.benchmark.case_id, "HEU-MET-FAST-001");
        let runner = case.runner.as_ref().expect("runner block present");
        assert_eq!(runner.binary, "godiva");
        assert_eq!(runner.data_arg, "positional");
        assert_eq!(runner.k_label, "k_eff (collision)");
    }

    #[test]
    fn scene_only_case_resolves_to_blocked() {
        let s = r#"{
            "benchmark": {
              "suite": "ICSBEP",
              "case_id": "PU-MET-FAST-001",
              "k_eff_reference": 1.0,
              "k_eff_sigma": 0.002
            },
            "scene": {
              "surfaces": [],
              "cells": [],
              "universes": [],
              "materials": [],
              "root_universe_id": 0
            }
        }"#;
        let case: CaseFile = serde_json::from_str(s).unwrap();
        assert!(case.runner.is_none());
        assert!(case.scene.is_some());
        // run_case() with absent runner should return blocked, not error.
        let r = run_case(&case, Path::new("/nonexistent"), Path::new("/nonexistent"), 2.0);
        assert!(r.blocked, "scene-only case should be blocked");
        assert!(r.error.as_deref().unwrap_or("").contains("BLOCKED"));
        assert_eq!(status_label(&r), "BLOCK");
    }

    #[test]
    fn pending_composition_resolves_to_skip() {
        let s = r#"{
            "benchmark": {
              "suite": "ICSBEP",
              "case_id": "PMF-001-pending",
              "k_eff_reference": 1.0,
              "k_eff_sigma": 0.002
            },
            "pending_composition": true
        }"#;
        let case: CaseFile = serde_json::from_str(s).unwrap();
        let r = run_case(&case, Path::new("/nonexistent"), Path::new("/nonexistent"), 2.0);
        assert!(r.skipped);
        assert_eq!(status_label(&r), "SKIP");
    }

    #[test]
    fn malformed_case_no_runner_no_scene_is_error() {
        let s = r#"{
            "benchmark": {
              "suite": "ICSBEP",
              "case_id": "malformed",
              "k_eff_reference": 1.0,
              "k_eff_sigma": 0.001
            }
        }"#;
        let case: CaseFile = serde_json::from_str(s).unwrap();
        let r = run_case(&case, Path::new("/nonexistent"), Path::new("/nonexistent"), 2.0);
        assert!(!r.blocked);
        assert!(!r.skipped);
        assert!(r.error.is_some());
        assert_eq!(status_label(&r), "ERR");
    }
}
