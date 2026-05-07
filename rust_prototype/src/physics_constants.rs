//! Centralized physics and numerical constants.
//!
//! Convention: when a non-trivial number appears in more than one
//! place, or has a physical meaning beyond "the value 1.0", it gets
//! a name here. Magic numbers in formula implementations are fine
//! when they're truly local (e.g. `0.5` for a midpoint average); but
//! anything a future contributor would have to grep the literature to
//! understand should be named. The math is no less correct with a
//! magic constant, but the diff history and review experience are
//! noticeably better with a named one.
//!
//! Constants are grouped by domain:
//! - **Physics**: fundamental SI values (electron mass-energy,
//!   classical electron radius, Avogadro), used in cross-section and
//!   kinematics formulas.
//! - **Geometry / direction**: unit-direction sentinel values used as
//!   default arguments to estimator routines (e.g. `MU_AXIAL_FORWARD`
//!   for "photon traveling +ẑ").
//! - **Variance reduction**: defaults / sentinels for next-event,
//!   weight-window, and CADIS routines (e.g. `NEE_NO_EXCLUSION`).
//! - **Numerical / quadrature**: tolerances and stable-evaluation
//!   thresholds that show up in series expansions.
//!
//! Many of these are also defined in their owning physics modules
//! (e.g. `photon::compton::M_E_C2_EV`); this module re-exports them
//! so a caller writing analysis or test code can reach for one
//! consistent path. We keep the canonical definitions in their owning
//! modules so domain-specific code doesn't acquire a cross-module
//! dependency for a single constant.

// ── Physics — re-exports of the canonical owning-module values ────

pub use crate::photon::compton::{HC_EV_ANGSTROM, M_E_C2_EV, R_E_SQ_CM2};

/// Avogadro's number (CODATA-2018 exact: 6.022 140 76 × 10²³).
/// Used for atomic-density conversions in material setup.
pub const AVOGADRO: f64 = 6.022_140_76e23;

/// `4π` — solid angle of a sphere; appears throughout MoC, NEE,
/// and isotropic-source normalisations.
pub const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

/// `2π` — full azimuthal range. Same value `std::f64::consts::TAU`,
/// repeated here so callers don't have to remember which name we
/// chose.
pub const TWO_PI: f64 = std::f64::consts::TAU;

// ── Direction / geometry sentinels ────────────────────────────────

/// `μ = 1`: photon (or particle) direction along `+ẑ`. Used as the
/// default `mu_in` for source-photon NEE calls in the `shield_slab`
/// geometry where the source is born going `+ẑ`.
pub const MU_AXIAL_FORWARD: f64 = 1.0;

/// `μ = -1`: photon direction along `-ẑ` (e.g. detector-backward
/// pseudo-source for lite calibration prior to the random-ray
/// adjoint replacement).
pub const MU_AXIAL_BACKWARD: f64 = -1.0;

// ── Variance reduction defaults ───────────────────────────────────

/// Sentinel for "no exclusion zone" in NEE calls — the v0 numeric
/// `0.0` argument is replaced by this name at call sites.
pub const NEE_NO_EXCLUSION: f64 = 0.0;

/// Default WW band ratio (textbook value, MCNP / OpenMC convention).
pub const WW_DEFAULT_RATIO: f64 = 5.0;

/// Default WW floor as a fraction of `φ_max` — voxels with importance
/// below this are flagged inactive.
pub const WW_DEFAULT_FLOOR: f64 = 1.0e-3;

// ── Numerical tolerances / thresholds ─────────────────────────────

/// Below this `τ`, use a Taylor series for `(1 − exp(−τ))/τ` to avoid
/// catastrophic cancellation. Matches the threshold used in
/// `random_ray::integrator::exp_m1_over` and the NEE exclusion-zone
/// regulariser. Typical scale where the direct-form division loses
/// 4–5 significant digits.
pub const SMALL_TAU_SERIES_THRESHOLD: f64 = 1.0e-4;

/// Tolerance for "essentially zero" comparisons of dimensionless
/// quantities. Looser than `f64::EPSILON` (≈ 2.2e-16) so reasonable
/// numerical drift doesn't trip checks.
pub const NUMERICAL_ZERO: f64 = 1.0e-12;
