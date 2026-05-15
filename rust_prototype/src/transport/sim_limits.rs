//! Engine-policy bounds — separates "engine should stay responsive"
//! from per-run user intent (which lives in `SimConfig`). Defaults
//! reproduce the engine's historical behaviour bit-for-bit.
//!
//! TOML example:
//! ```toml
//! max_events_per_history = 5000
//! fis_capacity_factor = 4
//! sab_temperature_tolerance = 0.5
//! initial_source_max_attempts_factor = 10000
//! ```

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SimLimits {
    /// GPU per-history step budget. CPU transport ignores this.
    pub max_events_per_history: u32,
    /// `n × factor` slot reserve for the GPU fission bank; overflow
    /// dropped (bank resampled to `n` regardless).
    pub fis_capacity_factor: usize,
    /// Tolerance for `ThermalScatteringData::select_temperature` on
    /// the GPU TSL binding.
    pub sab_temperature_tolerance: f64,
    /// `n × factor` attempt cap on `try_initial_source` rejection
    /// sampling; returns an error rather than spinning.
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
    /// `deny_unknown_fields` — typos surface loudly.
    pub fn from_toml_str(text: &str) -> Result<Self, SimLimitsError> {
        toml::from_str(text).map_err(SimLimitsError::Parse)
    }

    /// Returns defaults when `path` doesn't exist — callers pass an
    /// optional path without branching.
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

    /// Floors at 1 so device allocators never see a zero-size request.
    #[inline]
    pub fn fis_capacity(&self, particles_per_batch: usize) -> usize {
        particles_per_batch
            .saturating_mul(self.fis_capacity_factor)
            .max(1)
    }

    /// 1M-attempt floor — historical `try_initial_source` behaviour
    /// on tiny `n`.
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
