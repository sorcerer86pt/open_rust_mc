//! Traditional pointwise table lookup — the baseline competitor.
//!
//! This simulates what OpenMC does today: binary search the energy grid,
//! then linearly interpolate the cross-section between the two bracketing
//! energy points. Each lookup is a random-access read into a large array.

use std::path::Path;
use std::sync::Arc;

use ndarray_npy::ReadNpyExt;

use crate::error::{Result, SvdError};
use crate::kernel::EnergyHashTable;

/// Pointwise cross-section table for one nuclide, one reaction, one temperature.
pub struct PointwiseTable {
    /// Energy grid (sorted ascending), eV — shared across reactions in the same nuclide.
    energies: Arc<[f64]>,
    /// Cross-section values (barns), same length as `energies`.
    xs: Vec<f64>,
    /// Hash table for O(1) energy lookup.
    hash_table: Option<EnergyHashTable>,
}

impl PointwiseTable {
    /// Load from numpy arrays.
    pub fn from_npy(energy_path: &Path, xs_path: &Path) -> Result<Self> {
        let e_file = std::fs::File::open(energy_path)?;
        let energies: Vec<f64> = ndarray::Array1::<f64>::read_npy(e_file)
            .map_err(|e| SvdError::NpyLoad {
                path: energy_path.display().to_string(),
                source: e,
            })?
            .to_vec();

        let xs_file = std::fs::File::open(xs_path)?;
        let xs = ndarray::Array1::<f64>::read_npy(xs_file)
            .map_err(|e| SvdError::NpyLoad {
                path: xs_path.display().to_string(),
                source: e,
            })?
            .to_vec();

        let e: Arc<[f64]> = energies.into();
        // Hash table disabled for PointwiseTable — the log-log interpolation
        // is sensitive to bracket accuracy. SVD uses hash (index-only, no interp).
        Ok(Self {
            energies: e,
            xs,
            hash_table: None,
        })
    }

    /// Build from raw vectors (for benchmarking without file I/O).
    pub fn from_vecs(energies: Vec<f64>, xs: Vec<f64>) -> Self {
        debug_assert_eq!(energies.len(), xs.len());
        let e: Arc<[f64]> = energies.into();
        Self {
            energies: e,
            xs,
            hash_table: None,
        }
    }

    /// Build from a shared energy grid and owned XS values.
    pub fn from_shared_grid(energies: Arc<[f64]>, xs: Vec<f64>) -> Self {
        debug_assert_eq!(energies.len(), xs.len());
        Self {
            energies,
            xs,
            hash_table: None,
        }
    }

    /// Memory footprint of XS data only (bytes), excluding shared energy grid.
    pub fn memory_bytes(&self) -> usize {
        self.xs.len() * std::mem::size_of::<f64>()
    }

    /// Lookup via binary search + linear interpolation.
    ///
    /// This is the hot path for traditional Monte Carlo: called once per
    /// collision for each nuclide in the material.
    #[inline]
    pub fn lookup(&self, energy: f64) -> f64 {
        let n = self.energies.len();

        // Use hash table for O(1) lookup when available, else binary search.
        // Both return the lower bracket index for interpolation.
        let idx = if let Some(ref ht) = self.hash_table {
            let i = ht.lookup(energy, &self.energies);
            if energy <= self.energies[0] {
                return self.xs[0];
            }
            if i >= n {
                return self.xs[n - 1];
            }
            // Hash returns upper bracket; we need lower for interpolation
            if i > 0 { i - 1 } else { 0 }
        } else {
            match self
                .energies
                .binary_search_by(|e| e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
            {
                Ok(i) => return self.xs[i],
                Err(0) => return self.xs[0],
                Err(i) if i >= n => return self.xs[n - 1],
                Err(i) => i - 1,
            }
        };

        // Linear interpolation in log-log space (standard in nuclear data)
        let e_lo = self.energies[idx];
        let e_hi = self.energies[idx + 1];
        let xs_lo = self.xs[idx];
        let xs_hi = self.xs[idx + 1];

        if xs_lo <= 0.0 || xs_hi <= 0.0 {
            // Fallback to linear interpolation for near-zero values
            let frac = (energy - e_lo) / (e_hi - e_lo);
            return xs_lo + frac * (xs_hi - xs_lo);
        }

        // Log-log interpolation: log(σ) = log(σ_lo) + f * (log(σ_hi) - log(σ_lo))
        // where f = log(E/E_lo) / log(E_hi/E_lo)
        // Using exp2 identity: a^b = exp2(b * log2(a)) — 3-5x faster than powf
        let f = (energy / e_lo).ln() / (e_hi / e_lo).ln();
        let ratio = xs_hi / xs_lo;
        xs_lo * f64::exp2(f * ratio.log2())
    }

    /// Lower-bracket grid index for `energy`: largest `idx` where
    /// `energies[idx] <= energy`, or `0` / `n-1` for out-of-range.
    ///
    /// Exposes the search so a caller that holds several tables on the
    /// same `Arc<[f64]>` grid can search once and then call
    /// `lookup_at_idx` on each. Matches the search performed internally
    /// by `lookup` exactly.
    #[inline]
    pub fn bracket_idx(&self, energy: f64) -> usize {
        let n = self.energies.len();
        if n == 0 {
            return 0;
        }
        match self
            .energies
            .binary_search_by(|e| e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) if i >= n => n - 1,
            Err(i) => i - 1,
        }
    }

    /// Lookup at a pre-computed grid index — skips the energy search.
    ///
    /// Caller must provide `idx` from the same shared energy grid (lower
    /// bracket: largest `idx` with `energies[idx] <= energy`) and the
    /// energy itself for the interpolation. Produces exactly the same
    /// value as `lookup(energy)` when `idx` is correct; no precision
    /// difference. Use this when the caller has already done a single
    /// search across multiple reactions on the shared grid.
    #[inline]
    pub fn lookup_at_idx(&self, energy: f64, idx: usize) -> f64 {
        let n = self.energies.len();
        if n == 0 {
            return 0.0;
        }
        if energy <= self.energies[0] {
            return self.xs[0];
        }
        if idx + 1 >= n {
            return self.xs[n - 1];
        }

        let e_lo = self.energies[idx];
        let e_hi = self.energies[idx + 1];
        let xs_lo = self.xs[idx];
        let xs_hi = self.xs[idx + 1];

        if xs_lo <= 0.0 || xs_hi <= 0.0 {
            let frac = (energy - e_lo) / (e_hi - e_lo);
            return xs_lo + frac * (xs_hi - xs_lo);
        }

        let f = (energy / e_lo).ln() / (e_hi / e_lo).ln();
        let ratio = xs_hi / xs_lo;
        xs_lo * f64::exp2(f * ratio.log2())
    }

    /// Batch lookup for benchmarking: look up many random energies.
    pub fn batch_lookup(&self, energies: &[f64], out: &mut [f64]) {
        for (e, o) in energies.iter().zip(out.iter_mut()) {
            *o = self.lookup(*e);
        }
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }
}

// ── Stochastic temperature interpolation ───────────────────────────────
//
// Pseudo-interpolation test (PHYSOR-style): when the operating temperature
// T is between two library endpoints T_lo and T_hi, OpenMC samples one of
// the two tables per lookup with probability p_lo = (T_hi-T)/(T_hi-T_lo).
// This forces random memory loads into two XS arrays per collision rather
// than streaming one resident table — a realistic cache-pressure scenario.

thread_local! {
    /// Per-thread splitmix64 state for cheap binary xi draws in XS lookup.
    /// Separate from the particle RNG to avoid perturbing Monte Carlo
    /// reproducibility; this only controls which library temperature is
    /// touched, and the XS values are continuous in T so the statistical
    /// effect averages to the exact linear interpolant.
    static STOCH_STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0x9E37_79B9_7F4A_7C15) };
}

#[inline]
fn draw_xi() -> f64 {
    STOCH_STATE.with(|c| {
        let mut z = c.get().wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        c.set(z);
        (z >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    })
}

/// A pointwise table that may be backed by two library temperatures,
/// with a stochastic per-lookup pick between them.
///
/// For on-library temperatures, use [`StochTempTable::single`]; no picking
/// happens and performance matches `PointwiseTable`. For off-library
/// temperatures, use [`StochTempTable::stochastic`]: each lookup draws a
/// uniform xi and returns `lo` with probability `(T_hi - T)/(T_hi - T_lo)`.
pub struct StochTempTable {
    lo: PointwiseTable,
    hi: Option<PointwiseTable>,
    p_lo: f64,
}

impl StochTempTable {
    /// Wrap a single table (on-library temperature — no stochastic pick).
    pub fn single(table: PointwiseTable) -> Self {
        Self {
            lo: table,
            hi: None,
            p_lo: 1.0,
        }
    }

    /// Build a two-temperature stochastic table. `target_t` should lie in
    /// `[t_lo, t_hi]`; `p_lo` is clamped to [0, 1].
    pub fn stochastic(
        lo: PointwiseTable,
        hi: PointwiseTable,
        target_t: f64,
        t_lo: f64,
        t_hi: f64,
    ) -> Self {
        let p_lo = if (t_hi - t_lo).abs() < 1e-6 {
            1.0
        } else {
            ((t_hi - target_t) / (t_hi - t_lo)).clamp(0.0, 1.0)
        };
        Self {
            lo,
            hi: Some(hi),
            p_lo,
        }
    }

    /// True if this table has two library endpoints (cache-pressure mode).
    pub fn is_stochastic(&self) -> bool {
        self.hi.is_some()
    }

    /// Memory footprint of XS data (both endpoints if stochastic), bytes.
    pub fn memory_bytes(&self) -> usize {
        self.lo.memory_bytes() + self.hi.as_ref().map_or(0, PointwiseTable::memory_bytes)
    }

    /// Lower-bracket grid index (matches `PointwiseTable::bracket_idx`).
    /// Both endpoints share the same energy grid, so `lo` is authoritative.
    #[inline]
    pub fn bracket_idx(&self, energy: f64) -> usize {
        self.lo.bracket_idx(energy)
    }

    /// Stochastic lookup at a pre-computed grid index. Draws its own xi
    /// per call — use [`lookup_at_idx_with_pick`] when a single MicroXs
    /// build touches multiple channels and they must all use the same
    /// library endpoint (consistency of partial channels within a
    /// collision is a correctness requirement for OpenMC-style
    /// pseudo-interpolation).
    #[inline]
    pub fn lookup_at_idx(&self, energy: f64, idx: usize) -> f64 {
        match &self.hi {
            Some(hi) if draw_xi() > self.p_lo => hi.lookup_at_idx(energy, idx),
            _ => self.lo.lookup_at_idx(energy, idx),
        }
    }

    /// Stochastic lookup with an externally-chosen endpoint. `use_hi =
    /// true` picks the upper library temp; `false` picks the lower. The
    /// caller should draw one xi at the start of a collision's MicroXs
    /// build and pass the same `use_hi` to every channel lookup so all
    /// partial channels come from the same library temperature.
    #[inline]
    pub fn lookup_at_idx_with_pick(&self, energy: f64, idx: usize, use_hi: bool) -> f64 {
        match &self.hi {
            Some(hi) if use_hi => hi.lookup_at_idx(energy, idx),
            _ => self.lo.lookup_at_idx(energy, idx),
        }
    }

    /// Draw a consistent pick for this stochastic table. Returns
    /// `(use_hi, p_lo)` — `use_hi = draw_xi() > p_lo`. Returns
    /// `(false, 1.0)` for single-temp tables (lo is authoritative).
    #[inline]
    pub fn draw_pick(&self) -> bool {
        match &self.hi {
            Some(_) => draw_xi() > self.p_lo,
            None => false,
        }
    }

    /// Stochastic lookup with internal bracket search.
    #[inline]
    pub fn lookup(&self, energy: f64) -> f64 {
        let idx = self.bracket_idx(energy);
        self.lookup_at_idx(energy, idx)
    }
}
