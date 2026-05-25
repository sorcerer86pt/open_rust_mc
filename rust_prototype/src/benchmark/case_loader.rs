// SPDX-License-Identifier: MIT
//! Stage 2 — case file parsing.
//!
//! Walks `bench_dir` for `*.json`, parses each into a `CaseSpec`,
//! and pushes onto the `LoadQueue` channel that Stage 3 reads from.
//! Does **no** HDF5 I/O — that belongs to Stage 3.

use std::path::PathBuf;
use std::sync::Arc;

use crate::geometry::scene_io::SceneDto;
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
