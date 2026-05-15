//! Centralised constants. Canonical definitions live in owning
//! modules (`photon::compton::M_E_C2_EV` etc.); this module
//! re-exports them so analysis code has one consistent path.

pub use crate::photon::compton::{HC_EV_ANGSTROM, M_E_C2_EV, R_E_SQ_CM2};

/// CODATA-2018 exact.
pub const AVOGADRO: f64 = 6.022_140_76e23;
pub const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;
pub const TWO_PI: f64 = std::f64::consts::TAU;

/// μ = 1 (+ẑ); default `mu_in` for `shield_slab` source photons.
pub const MU_AXIAL_FORWARD: f64 = 1.0;
/// μ = -1 (-ẑ); detector-backward pseudo-source.
pub const MU_AXIAL_BACKWARD: f64 = -1.0;

pub const NEE_NO_EXCLUSION: f64 = 0.0;
/// MCNP / OpenMC convention.
pub const WW_DEFAULT_RATIO: f64 = 5.0;
pub const WW_DEFAULT_FLOOR: f64 = 1.0e-3;

/// Taylor switch for `(1 − exp(−τ))/τ` to avoid cancellation.
pub const SMALL_TAU_SERIES_THRESHOLD: f64 = 1.0e-4;
pub const NUMERICAL_ZERO: f64 = 1.0e-12;
