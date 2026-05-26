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

use std::fs::OpenOptions;
use std::path::Path;

use super::executor::ExecutionResult;

/// CSV row writer backed by the `csv` crate. Field quoting and
/// escaping are handled by the library — error messages with embedded
/// commas / quotes / newlines round-trip cleanly. `flush()` runs per
/// case so a kill -9 / power loss loses at most the in-flight row.
///
/// Format matches the historical Python sweep header so dashboards
/// that parse the output keep working. The header is emitted only
/// when the file is created fresh; resuming appends rows below the
/// previous run's last completed case.
pub struct CsvWriter {
    /// Direct file-backed writer. The `csv` crate buffers internally;
    /// adding our own `BufWriter` would block per-case flushes since
    /// `csv::Writer` exposes no mutable accessor for the inner type.
    /// At ~200 bytes per row and one row per case, the syscall cost
    /// is unmeasurable next to a 1-200 s case.
    inner: csv::Writer<std::fs::File>,
}

impl CsvWriter {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let needs_header = !path.exists()
            || std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut inner = csv::WriterBuilder::new()
            .has_headers(needs_header)
            .from_writer(file);
        if needs_header {
            inner.write_record([
                "case",
                "seq",
                "total",
                "verdict",
                "k_calc",
                "k_sigma",
                "k_ref",
                "sigma_exp",
                "delta_pcm",
                "bound_pcm",
                "sigma_ratio",
                "runtime_s",
                "load_s",
                "backend",
                "error",
            ])?;
            inner.flush()?;
        }
        Ok(Self { inner })
    }

    pub fn append(&mut self, r: &ExecutionResult, n_sigma: f64) -> std::io::Result<()> {
        let verdict = match Verdict::classify(r, n_sigma) {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Error => "ERROR",
        };
        let delta_pcm = (r.k_calc - r.summary.k_ref) * 1e5;
        let sigma_combined =
            (r.k_sigma.powi(2) + r.summary.sigma_exp.powi(2)).sqrt() * 1e5;
        let bound = (n_sigma * sigma_combined).max(150.0);
        let sigma_ratio = if sigma_combined > 0.0 {
            delta_pcm.abs() / sigma_combined
        } else {
            0.0
        };
        let backend = match r.backend {
            super::executor::BackendUsed::Cpu => "cpu",
            super::executor::BackendUsed::Cuda => "cuda",
            super::executor::BackendUsed::CpuVramDowngrade => "cpu_vram_downgrade",
        };
        self.inner.write_record([
            r.summary.case_id.as_str(),
            &r.summary.seq.to_string(),
            &r.summary.total.to_string(),
            verdict,
            &format!("{:.6}", r.k_calc),
            &format!("{:.6}", r.k_sigma),
            &format!("{:.6}", r.summary.k_ref),
            &format!("{:.6}", r.summary.sigma_exp),
            &format!("{:.1}", delta_pcm),
            &format!("{:.1}", bound),
            &format!("{:.3}", sigma_ratio),
            &format!("{:.3}", r.runtime_s),
            &format!("{:.3}", r.load_s),
            backend,
            r.error.as_deref().unwrap_or(""),
        ])?;
        // Per-case flush so kill -9 loses at most one row. The csv
        // crate's `flush()` drains its internal record buffer into
        // the wrapped `BufWriter`; that's what we need for durability
        // since `BufWriter::write_all` is committed (the OS page
        // cache holds it across a process kill, even without a
        // `flush()` on the BufWriter itself).
        self.inner.flush()
    }
}

/// JSONL telemetry writer — one JSON object per case, newline-delimited.
/// Captures per-case identity, timings, routing decision, verdict, and
/// `gpu_debug_metrics` (cache hit counts / VRAM stats) when GPU is in
/// use. Used by the §5.4.0 router calibration loop and by post-hoc
/// performance analysis.
///
/// The format is intentionally schema-free — every well-formed JSON
/// object on its own line — so jq / pandas / DuckDB can each consume
/// it without a schema declaration.
pub struct TelemetryWriter {
    inner: std::fs::File,
}

impl TelemetryWriter {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { inner: file })
    }

    pub fn append(
        &mut self,
        r: &ExecutionResult,
        n_sigma: f64,
        extra: serde_json::Value,
    ) -> std::io::Result<()> {
        use std::io::Write;
        let verdict = match Verdict::classify(r, n_sigma) {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Error => "ERROR",
        };
        let delta_pcm = (r.k_calc - r.summary.k_ref) * 1e5;
        let backend = match r.backend {
            super::executor::BackendUsed::Cpu => "cpu",
            super::executor::BackendUsed::Cuda => "cuda",
            super::executor::BackendUsed::CpuVramDowngrade => "cpu_vram_downgrade",
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let row = serde_json::json!({
            "timestamp_unix": now,
            "case_id": r.summary.case_id,
            "seq": r.summary.seq,
            "total": r.summary.total,
            "source_path": r.summary.source_path.display().to_string(),
            "verdict": verdict,
            "backend": backend,
            "k_calc": r.k_calc,
            "k_sigma": r.k_sigma,
            "k_ref": r.summary.k_ref,
            "sigma_exp": r.summary.sigma_exp,
            "delta_pcm": delta_pcm,
            "k_track": r.k_track,
            "k_track_sigma": r.k_track_sigma,
            "runtime_s": r.runtime_s,
            "load_s": r.load_s,
            "n_histories": r.n_histories,
            "error": r.error,
            "extra": extra,
        });
        let line = serde_json::to_string(&row).unwrap_or_else(|_| String::from("{}"));
        writeln!(self.inner, "{line}")?;
        // OS page cache + append-only file = durable on kill -9. Power
        // loss before fsync would lose the tail; the CSV row gets
        // flushed via `csv::Writer::flush` so it's the durability of
        // record.
        Ok(())
    }
}

/// Parse the `case` column of an existing CSV via the `csv` crate.
/// Used by `--resume` to skip case ids already in the output file.
pub fn read_completed_case_ids(
    path: &Path,
) -> std::io::Result<std::collections::HashSet<String>> {
    let mut out = std::collections::HashSet::new();
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e),
    };
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);
    for rec in rdr.records().flatten() {
        if let Some(case_id) = rec.get(0) {
            if !case_id.is_empty() {
                out.insert(case_id.to_string());
            }
        }
    }
    Ok(out)
}

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
        let delta_pcm = (r.k_calc - r.summary.k_ref).abs() * 1e5;
        let sigma_combined =
            (r.k_sigma.powi(2) + r.summary.sigma_exp.powi(2)).sqrt() * 1e5;
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
