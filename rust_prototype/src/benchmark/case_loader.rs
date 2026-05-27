// SPDX-License-Identifier: MIT
//! Stage 2 — case file parsing.
//!
//! Walks `bench_dir` for `*.json`, parses each into a `CaseSpec`,
//! and pushes onto the `LoadQueue` channel that Stage 3 reads from.
//! Does **no** HDF5 I/O — that belongs to Stage 3.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::geometry::scene_io::{self, LoadedScene, SceneDto};
use crate::geometry::Geometry;
use crate::transport::simulate::SimConfig;

/// Routing hint for the `RunnerHint::Auto` router (§5.4.0). Stage 4
/// dispatches per case based on this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerHint {
    /// Force CPU regardless of complexity.
    Cpu,
    /// Force GPU. Stage 3 may still downgrade to CPU if the VRAM
    /// gate refuses.
    Gpu,
    /// Apply the routing rules in `auto_pick` (§5.4.0).
    Auto,
}

impl Default for RunnerHint {
    fn default() -> Self {
        Self::Auto
    }
}

/// Reference k_eff source — distinguishes a handbook value from a
/// locally-validated OpenMC k_eff on the same scene JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceSource {
    /// ICSBEP handbook value.
    Handbook,
    /// `local_validation.openmc_k_eff` from the case JSON. Used when
    /// available to grade the engine against an OpenMC run on the
    /// exact same scene transcription.
    OpenMc,
}

/// Parsed case, ready for Stage 3 (data resolution).
pub struct CaseSpec {
    // Identity
    pub case_id: String,
    pub seq: usize,
    pub total: usize,

    // Reference values
    pub k_ref: f64,
    pub sigma_exp: f64,
    pub source: ReferenceSource,

    // Geometry (parsed — `Arc` so the loader and any later stage
    // can share without re-parsing).
    pub geometry: Arc<Geometry>,
    pub scene: Arc<SceneDto>,

    // Sim settings (merged: CLI > JSON `recommended_settings` > default).
    pub config: SimConfig,

    // Routing
    pub runner: RunnerHint,

    // Source file (kept for error messages + telemetry).
    pub source_path: PathBuf,
}

impl CaseSpec {
    /// True if any material in the scene references a thermal-
    /// scattering file. Read by the §5.4.0 router (rule 1).
    pub fn has_thermal_scattering(&self) -> bool {
        self.scene
            .materials
            .iter()
            .any(|m| !m.thermal_files.is_empty())
    }

    /// Number of materials in the scene. Read by the §5.4.0 router
    /// (rule 3).
    pub fn material_count(&self) -> usize {
        self.scene.materials.len()
    }

    /// True if the scene declares any lattice geometry. Read by the
    /// §5.4.0 router (rule 3).
    pub fn has_lattice(&self) -> bool {
        !self.scene.rect_lattices.is_empty() || !self.scene.hex_lattices.is_empty()
    }

    /// Total work proxy: `particles_per_batch × batches`. Read by
    /// the §5.4.0 router (rule 2).
    pub fn work_proxy(&self) -> u64 {
        (self.config.particles_per_batch as u64)
            .saturating_mul(self.config.batches as u64)
    }
}

/// Errors surfaced while parsing a case JSON. Keep variants narrow so
/// callers (Stage 2 thread, single-case CLI driver) can distinguish
/// "skip silently" vs "halt the sweep".
#[derive(Debug, thiserror::Error)]
pub enum CaseParseError {
    #[error("read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse JSON {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "case JSON {path} has no `scene` block — likely a CLI-runner \
         manifest (`runner.binary` references a built-in like `godiva` \
         / `pwr_pincell`). The benchmark pipeline only consumes \
         scene-based cases."
    )]
    NoSceneBlock { path: PathBuf },
    #[error("benchmark.k_eff_reference missing in {path}")]
    MissingKRef { path: PathBuf },
    #[error("benchmark.k_eff_sigma missing in {path}")]
    MissingSigma { path: PathBuf },
    #[error("scene_io::load_scene_from_json({path}): {source}")]
    SceneLoad {
        path: PathBuf,
        #[source]
        source: crate::geometry::scene_io::SceneLoadError,
    },
}

/// Settings carried over from CLI / `RunArgs` that influence the
/// merged [`SimConfig`] of every case. Mirrors the precedence the
/// historical Python sweep applies — CLI overrides JSON
/// `recommended_settings`, which overrides the engine's built-in
/// defaults.
#[derive(Debug, Clone, Copy)]
pub struct CaseDefaults {
    pub particles_per_batch: Option<u32>,
    pub batches: Option<u32>,
    pub inactive_batches: Option<u32>,
    pub base_seed: u64,
    /// When true, every loaded case gets
    /// `SimConfig::survival_biasing = Some(SurvivalBiasing::default())`
    /// (OpenMC defaults `w_min=0.25, w_survive=1.0`). Drives implicit
    /// capture + Bernoulli-banked fission + Russian roulette on both
    /// CPU and GPU paths. Necessary for high-particle-count runs
    /// (≥200k) on small-VRAM GPUs where the analog tail otherwise
    /// hits `SimLimits::max_events_per_history = 1_000_000`.
    pub survival_biasing: bool,
}

impl CaseDefaults {
    /// `(particles, batches, inactive)` resolved against per-case JSON
    /// `recommended_settings` and the engine's built-in defaults. The
    /// CLI override (if any) wins; otherwise the JSON value; otherwise
    /// the engine default.
    fn resolve(&self, json_recommended: &serde_json::Value) -> (u32, u32, u32) {
        let particles = self.particles_per_batch.unwrap_or_else(|| {
            json_recommended["particles_per_batch"]
                .as_u64()
                .map(|v| v as u32)
                .unwrap_or(100_000)
        });
        let batches = self.batches.unwrap_or_else(|| {
            json_recommended["batches"]
                .as_u64()
                .map(|v| v as u32)
                .unwrap_or(150)
        });
        let inactive = self.inactive_batches.unwrap_or_else(|| {
            json_recommended["inactive_batches"]
                .as_u64()
                .map(|v| v as u32)
                .unwrap_or(30)
        });
        (particles, batches, inactive)
    }
}

/// Parse one case JSON into a [`CaseSpec`].
///
/// Mirrors the parsing logic in
/// `bindings/python/src/lib.rs::run_icsbep_case` so the in-process
/// pipeline reads the same files the historical Python sweep
/// consumes. Used both by Stage 2 (CaseLoader thread) and by the
/// single-case CLI driver.
///
/// `seq`/`total` are passed in rather than computed here so the
/// caller controls iteration order (the Stage 2 thread walks
/// `bench_dir` and assigns sequential ids).
pub fn parse_case_json(
    path: &Path,
    seq: usize,
    total: usize,
    defaults: &CaseDefaults,
    hint: RunnerHint,
) -> Result<(CaseSpec, LoadedScene), CaseParseError> {
    let text = std::fs::read_to_string(path).map_err(|source| CaseParseError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|source| CaseParseError::Json {
            path: path.to_path_buf(),
            source,
        })?;

    let benchmark = &value["benchmark"];
    let scene = &value["scene"];
    if scene.is_null() {
        return Err(CaseParseError::NoSceneBlock {
            path: path.to_path_buf(),
        });
    }

    let handbook_k = benchmark["k_eff_reference"]
        .as_f64()
        .ok_or_else(|| CaseParseError::MissingKRef {
            path: path.to_path_buf(),
        })?;
    let handbook_sigma = benchmark["k_eff_sigma"]
        .as_f64()
        .ok_or_else(|| CaseParseError::MissingSigma {
            path: path.to_path_buf(),
        })?;

    // Prefer the local OpenMC validation k_eff when present (same
    // logic as `run_icsbep_case`). Keeps the engine graded against a
    // value measured on the exact same scene transcription.
    let (k_ref, sigma_exp, source) = match benchmark.get("local_validation") {
        Some(lv) if lv.get("openmc_k_eff").and_then(|v| v.as_f64()).is_some() => {
            let k = lv["openmc_k_eff"].as_f64().unwrap();
            let s_omc = lv["openmc_k_sigma_seeds"].as_f64().unwrap_or(0.001);
            (k, s_omc.max(handbook_sigma), ReferenceSource::OpenMc)
        }
        _ => (handbook_k, handbook_sigma, ReferenceSource::Handbook),
    };

    let loaded = scene_io::load_scene_from_json(&scene.to_string()).map_err(|source| {
        CaseParseError::SceneLoad {
            path: path.to_path_buf(),
            source,
        }
    })?;

    let recommended = benchmark
        .get("recommended_settings")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let (particles, batches, inactive) = defaults.resolve(&recommended);

    let case_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();

    // The `Arc<Geometry>` is shared between CaseSpec and any later
    // bundle that owns the case. Cloning the inner LoadedScene out
    // here keeps the borrow checker happy on the materials path.
    let dto: SceneDto =
        serde_json::from_str(&scene.to_string()).map_err(|source| CaseParseError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let geometry = Arc::new(loaded.geometry.clone());
    let scene_arc = Arc::new(dto);

    let config = SimConfig {
        batches,
        inactive,
        particles_per_batch: particles,
        seed: defaults.base_seed,
        auto_inactive: None,
        verbose: false,
        parallel: true,
        tallies: Default::default(),
        statepoint_path: None,
        survival_biasing: if defaults.survival_biasing {
            Some(crate::transport::simulate::SurvivalBiasing::default())
        } else {
            None
        },
        initial_source_bank: None,
        weight_window: None,
        disable_delayed_neutrons: false,
        urr_equivalence: None,
        gpu_refill_pool_factor: None,
        gpu_auto_refill: true,
    };

    Ok((
        CaseSpec {
            case_id,
            seq,
            total,
            k_ref,
            sigma_exp,
            source,
            geometry,
            scene: scene_arc,
            config,
            runner: hint,
            source_path: path.to_path_buf(),
        },
        loaded,
    ))
}
