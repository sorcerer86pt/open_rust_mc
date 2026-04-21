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
