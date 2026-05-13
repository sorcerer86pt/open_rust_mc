//! Engine-policy bounds and tolerances — knobs that aren't user
//! intent per run (those live in [`SimConfig`]) but instead bound the
//! engine's behaviour so it stays responsive on pathological inputs.
//!
//! All fields have sensible defaults that reproduce the engine's
//! historical behaviour bit-for-bit. Long-shielding / large-lattice /
//! degraded-source problems can override via TOML; the file is
//! optional, and any unspecified key falls back to the default.
//!
//! Layout of the TOML file:
//!
//! ```toml
//! # config/sim_limits.toml
//! max_events_per_history = 5000          # GPU per-history step budget
//! fis_capacity_factor = 4                # GPU fission-bank preallocation × particles
//! sab_temperature_tolerance = 0.5        # tsl.select_temperature tolerance
//! initial_source_max_attempts_factor = 10000  # rejection sampler budget × particles
//! ```
//!
//! Callers use [`SimLimits::default()`] for the historical values, or
//! [`SimLimits::from_toml_file`] when a config file is supplied via
//! CLI / env. The struct is `Clone`, so threading it through a
//! per-run setup is cheap.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SimLimits {
    /// Per-history step budget on the GPU. Each particle terminates
    /// after this many collisions / surface crossings even if not
    /// absorbed / leaked, preventing runaway histories on degenerate
    /// geometries. CPU transport carries its own death conditions and
    /// does not consult this. Historical value: 5_000.
    pub max_events_per_history: u32,

    /// Multiplier on `SimConfig::particles_per_batch` for the GPU
    /// fission-bank preallocation. Each batch reserves `n × factor`
    /// slots for the next-generation bank; if the realised bank
    /// exceeds this, the excess is dropped (the bank is resampled
    /// down to `n` regardless, so overflow just biases the normalise
    /// step). Historical value: 4×.
    pub fis_capacity_factor: usize,

    /// Tolerance passed to `ThermalScatteringData::select_temperature`
    /// when picking which TSL temperature index to bind on the GPU.
    /// Smaller values prefer exact-match temperatures; larger ones
    /// interpolate more aggressively. Historical value: 0.5.
    pub sab_temperature_tolerance: f64,

    /// Multiplier on `SimConfig::particles_per_batch` for the
    /// rejection-sampler attempt budget inside
    /// `simulate::try_initial_source`. The sampler returns an error
    /// after `n × factor` attempts rather than spinning forever on a
    /// geometry whose fissile region the AABB heuristic missed.
    /// Historical value: 10_000.
    pub initial_source_max_attempts_factor: u64,
}

impl Default for SimLimits {
    fn default() -> Self {
        Self {
            max_events_per_history: 5_000,
            fis_capacity_factor: 4,
            sab_temperature_tolerance: 0.5,
            initial_source_max_attempts_factor: 10_000,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimLimitsError {
    #[error("failed to read sim_limits TOML at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse sim_limits TOML: {0}")]
    Parse(#[from] toml::de::Error),
}

impl SimLimits {
    /// Parse [`SimLimits`] from an in-memory TOML string. Unknown
    /// keys are rejected so typos surface loudly instead of silently
    /// defaulting.
    pub fn from_toml_str(text: &str) -> Result<Self, SimLimitsError> {
        toml::from_str(text).map_err(SimLimitsError::Parse)
    }

    /// Load [`SimLimits`] from a TOML file on disk. Returns the
    /// historical defaults when `path` doesn't exist (so callers can
    /// pass an optional config path without branching).
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self, SimLimitsError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path).map_err(|source| SimLimitsError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml_str(&text)
    }

    /// `fis_capacity_factor × particles_per_batch` saturating to at
    /// least 1 slot so device allocators never see a zero-size
    /// request.
    #[inline]
    pub fn fis_capacity(&self, particles_per_batch: usize) -> usize {
        particles_per_batch
            .saturating_mul(self.fis_capacity_factor)
            .max(1)
    }

    /// `initial_source_max_attempts_factor × n` with a 1_000_000-attempt
    /// floor — matches the historical `simulate::try_initial_source`
    /// behaviour for tiny `n`.
    #[inline]
    pub fn initial_source_max_attempts(&self, n: usize) -> u64 {
        (n as u64)
            .saturating_mul(self.initial_source_max_attempts_factor)
            .max(1_000_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_historical_constants() {
        let l = SimLimits::default();
        assert_eq!(l.max_events_per_history, 5_000);
        assert_eq!(l.fis_capacity_factor, 4);
        assert!((l.sab_temperature_tolerance - 0.5).abs() < 1e-12);
        assert_eq!(l.initial_source_max_attempts_factor, 10_000);
    }

    #[test]
    fn fis_capacity_floors_at_one() {
        let l = SimLimits::default();
        assert_eq!(l.fis_capacity(0), 1);
        assert_eq!(l.fis_capacity(10), 40);
    }

    #[test]
    fn initial_source_attempts_floors_at_million() {
        let l = SimLimits::default();
        assert_eq!(l.initial_source_max_attempts(10), 1_000_000);
        assert_eq!(l.initial_source_max_attempts(10_000), 100_000_000);
    }

    #[test]
    fn parses_toml() {
        let text = r#"
            max_events_per_history = 20000
            fis_capacity_factor = 8
            sab_temperature_tolerance = 0.1
            initial_source_max_attempts_factor = 50000
        "#;
        let l = SimLimits::from_toml_str(text).unwrap();
        assert_eq!(l.max_events_per_history, 20_000);
        assert_eq!(l.fis_capacity_factor, 8);
        assert!((l.sab_temperature_tolerance - 0.1).abs() < 1e-12);
        assert_eq!(l.initial_source_max_attempts_factor, 50_000);
    }

    #[test]
    fn partial_toml_merges_with_defaults() {
        let l = SimLimits::from_toml_str("max_events_per_history = 999").unwrap();
        assert_eq!(l.max_events_per_history, 999);
        // Others fall back to defaults.
        assert_eq!(l.fis_capacity_factor, 4);
    }

    #[test]
    fn unknown_key_rejected() {
        let err = SimLimits::from_toml_str("unknown_key = 42").unwrap_err();
        assert!(matches!(err, SimLimitsError::Parse(_)));
    }
}
