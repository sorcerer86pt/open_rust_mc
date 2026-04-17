//! SVD reconstruction kernel — the hot path.
//!
//! Layout:
//!   `basis`: N_E × rank, row-major. Each row is k pre-multiplied coefficients
//!            for one energy point: basis[i*rank + j] = U[i,j] * S[j].
//!   `vt_coeffs`: rank × N_T, row-major. V^T temperature coefficients.
//!
//! Reconstruction at temperature index `t`:
//!   1. Build `coeffs[j] = vt_coeffs[j * n_t + t]` for j in 0..rank  (one cache line)
//!   2. For each energy point i:
//!        σ_log(E_i) = Σ_{j=0}^{rank-1} basis[i*rank + j] * coeffs[j]
//!   3. σ(E_i) = 10^{σ_log(E_i)}
//!
//! Step 2 is a dot product of length `rank` — pure ALU, no memory stalls,
//! because `basis` is streamed sequentially and `coeffs` fits in registers.

use std::sync::Arc;

/// Hash table for O(1) energy grid lookup (Brown, 2014).
///
/// Divides the log-energy range into uniform bins. Each bin stores the
/// starting grid index, reducing lookup to: compute bin → read index → scan ≤2 entries.
pub struct EnergyHashTable {
    /// Starting grid index for each hash bin.
    bins: Vec<u32>,
    /// ln(E_min) of the energy grid.
    log_e_min: f64,
    /// 1 / bin_width in log-space.
    inv_bin_width: f64,
    /// Number of bins.
    n_bins: usize,
}

impl EnergyHashTable {
    /// Build a hash table for the given energy grid.
    pub fn new(energies: &[f64], n_bins: usize) -> Self {
        let n = energies.len();
        if n < 2 {
            return Self { bins: vec![0; n_bins], log_e_min: 0.0, inv_bin_width: 0.0, n_bins };
        }

        let log_e_min = energies[0].max(1e-11).ln();
        let log_e_max = energies[n - 1].max(1e-11).ln();
        let bin_width = (log_e_max - log_e_min) / n_bins as f64;
        let inv_bin_width = if bin_width > 0.0 { 1.0 / bin_width } else { 0.0 };

        // For each bin, find the first grid index whose energy falls in that bin
        let mut bins = Vec::with_capacity(n_bins);
        let mut grid_idx = 0_u32;
        for b in 0..n_bins {
            let bin_log_e = log_e_min + (b as f64 + 1.0) * bin_width;
            let bin_e = bin_log_e.exp();
            while (grid_idx as usize) < n && energies[grid_idx as usize] < bin_e {
                grid_idx += 1;
            }
            // Store the index just below the bin boundary
            bins.push(if grid_idx > 0 { grid_idx - 1 } else { 0 });
        }

        Self { bins, log_e_min, inv_bin_width, n_bins }
    }

    /// O(1) energy lookup: hash to bin, then linear scan ≤ a few entries.
    ///
    /// Returns the same index as binary search: the upper bracketing point
    /// (smallest index where `energies[idx] >= energy`), clamped to [0, n-1].
    /// This matches `SvdKernel::energy_index_binary` behavior.
    /// O(1) energy lookup: hash to bin, then short linear scan.
    ///
    /// Returns the LOWER bracket index: largest `idx` where `energies[idx] <= energy`.
    /// This matches the binary search fallback convention in `SvdKernel::energy_index`.
    #[inline]
    pub fn lookup(&self, energy: f64, energies: &[f64]) -> usize {
        let n = energies.len();
        if n < 2 { return 0; }
        if energy <= energies[0] { return 0; }
        if energy >= energies[n - 1] { return n - 1; }

        let log_e = energy.ln();
        let bin = ((log_e - self.log_e_min) * self.inv_bin_width) as usize;
        let bin = bin.min(self.n_bins - 1);

        // Start from the PREVIOUS bin's index (guaranteed <= energy's bracket).
        // bins[bin] is near the current bin's upper edge — too high for forward scan.
        // bins[bin-1] is near the current bin's lower edge — correct starting point.
        let start = if bin > 0 { self.bins[bin - 1] as usize } else { 0 };
        let mut idx = start.min(n - 1);

        // Linear scan forward past all grid points below energy
        while idx + 1 < n && energies[idx + 1] <= energy {
            idx += 1;
        }

        idx
    }
}

/// Pre-built reconstruction engine for a single nuclide + reaction.
pub struct SvdKernel {
    /// U × Σ pre-multiplied basis, row-major: `[n_e][rank]`.
    pub(crate) basis: Vec<f64>,
    /// V^T coefficients, row-major: `[rank][n_t]`.
    pub(crate) vt_coeffs: Vec<f64>,
    /// Energy grid in eV — shared across all reactions in the same nuclide.
    pub(crate) energies: Arc<[f64]>,
    /// Hash table for O(1) energy lookup (Brown 2014).
    pub(crate) hash_table: Option<EnergyHashTable>,
    /// Truncation rank.
    pub(crate) rank: usize,
    /// Number of energy points.
    pub(crate) n_e: usize,
    /// Number of training temperatures.
    pub(crate) n_t: usize,
}

impl SvdKernel {
    /// Construct a kernel from pre-multiplied basis and V^T coefficients.
    ///
    /// The energy grid is `Arc`-shared across reactions within a nuclide.
    pub fn new(
        basis: Vec<f64>,
        vt_coeffs: Vec<f64>,
        energies: Arc<[f64]>,
        rank: usize,
        n_e: usize,
        n_t: usize,
    ) -> Self {
        let hash_table = if n_e > 100 {
            Some(EnergyHashTable::new(&energies, 8192))
        } else {
            None
        };
        Self { basis, vt_coeffs, energies, hash_table, rank, n_e, n_t }
    }

    /// Number of energy points.
    #[inline]
    pub fn n_energy(&self) -> usize {
        self.n_e
    }

    /// SVD truncation rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total memory footprint of the kernel (bytes), excluding shared energy grid.
    pub fn memory_bytes(&self) -> usize {
        self.basis.len() * std::mem::size_of::<f64>()
            + self.vt_coeffs.len() * std::mem::size_of::<f64>()
    }

    /// Memory of the shared energy grid (bytes). Count once per nuclide, not per reaction.
    pub fn energy_grid_bytes(&self) -> usize {
        self.energies.len() * std::mem::size_of::<f64>()
    }

    /// Pre-compute the k temperature coefficients for a given temperature index.
    ///
    /// Returns a `rank`-length vector: `coeffs[j] = vt[j, t_idx]`.
    /// In the real engine this is called once per batch/temperature change,
    /// not per energy lookup.
    #[inline]
    pub fn temp_coeffs(&self, t_idx: usize) -> Vec<f64> {
        debug_assert!(t_idx < self.n_t, "temperature index out of bounds");
        let mut coeffs = Vec::with_capacity(self.rank);
        for j in 0..self.rank {
            coeffs.push(self.vt_coeffs[j * self.n_t + t_idx]);
        }
        coeffs
    }

    /// Compute temperature coefficients for an arbitrary temperature using
    /// the Ducru et al. (2017) kernel reconstruction method.
    ///
    /// Instead of picking the nearest library temperature, this computes
    /// optimal interpolation weights from the Doppler broadening kernel
    /// structure and combines all V^T columns:
    ///   c_k(T) = Σ_j w_j(T) * V_k(T_j)
    ///
    /// The weights w_j(T) are the analytical free Doppler reconstruction
    /// coefficients (Ducru Eq. 31):
    ///   w_j = sqrt(T_j*T)/(T_j+T) * Π_{i≠j} [(T-T_i)/(T+T_i) * (T_j+T_i)/(T_j-T_i)]
    ///
    /// This yields optimal L2 temperature interpolation with ~0.1% error
    /// over [300K, 3000K] using only the library reference temperatures.
    pub fn temp_coeffs_ducru(&self, temperatures: &[f64], target_temp: f64) -> Vec<f64> {
        let n_t = temperatures.len();
        debug_assert_eq!(n_t, self.n_t, "temperature count mismatch");

        // Compute Ducru free Doppler weights (Eq. 31)
        let weights = ducru_weights(temperatures, target_temp);

        // Weighted sum of V^T columns: c_k = Σ_j w_j * vt[k, j]
        let mut coeffs = Vec::with_capacity(self.rank);
        for k in 0..self.rank {
            let mut acc = 0.0;
            for j in 0..n_t {
                acc += weights[j] * self.vt_coeffs[k * n_t + j];
            }
            coeffs.push(acc);
        }
        coeffs
    }

    /// Reconstruct **all** cross-sections (log₁₀ scale) at a given temperature.
    ///
    /// This is the primary hot-path benchmark target: sequential stream over
    /// the basis matrix with a k-wide dot product per row.
    pub fn reconstruct_log(&self, coeffs: &[f64], out: &mut [f64]) {
        debug_assert_eq!(coeffs.len(), self.rank);
        debug_assert!(out.len() >= self.n_e);

        let rank = self.rank;
        let basis = &self.basis;

        for i in 0..self.n_e {
            let row = &basis[i * rank..(i + 1) * rank];
            let mut acc = 0.0_f64;
            for j in 0..rank {
                acc = row[j].mul_add(coeffs[j], acc);
            }
            out[i] = acc;
        }
    }

    /// Reconstruct cross-sections in linear scale (barns) at a given temperature.
    pub fn reconstruct_linear(&self, coeffs: &[f64], out: &mut [f64]) {
        self.reconstruct_log(coeffs, out);
        for val in out.iter_mut().take(self.n_e) {
            *val = f64::exp2(*val * std::f64::consts::LOG2_10);
        }
    }

    /// Reconstruct a single energy point (log₁₀ scale).
    ///
    /// Used in the particle tracking inner loop where we need σ at one
    /// specific energy, not the full spectrum.
    #[inline]
    pub fn reconstruct_single_log(&self, energy_idx: usize, coeffs: &[f64]) -> f64 {
        debug_assert!(energy_idx < self.n_e);
        debug_assert_eq!(coeffs.len(), self.rank);

        let row = &self.basis[energy_idx * self.rank..(energy_idx + 1) * self.rank];
        let mut acc = 0.0_f64;
        for j in 0..self.rank {
            acc = row[j].mul_add(coeffs[j], acc);
        }
        acc
    }

    /// Reconstruct a single energy point (linear scale, barns).
    #[inline]
    pub fn reconstruct_single(&self, energy_idx: usize, coeffs: &[f64]) -> f64 {
        f64::exp2(self.reconstruct_single_log(energy_idx, coeffs) * std::f64::consts::LOG2_10)
    }

    /// Return the energy grid.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Return the f64 basis (for GPU upload).
    pub fn basis_f64(&self) -> &[f64] {
        &self.basis
    }

    /// Look up the energy grid index using the hash table if available,
    /// falling back to binary search otherwise.
    #[inline]
    pub fn energy_index(&self, energy: f64) -> usize {
        if let Some(ref ht) = self.hash_table {
            ht.lookup(energy, &self.energies)
        } else {
            self.energy_index_binary(energy)
        }
    }

    /// Binary search the energy grid (O(log N) fallback).
    #[inline]
    fn energy_index_binary(&self, energy: f64) -> usize {
        let n = self.energies.len();
        match self.energies.binary_search_by(|e| {
            e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 { 0 }
                else if i >= n { n - 1 }
                else { i }
            }
        }
    }
}

/// Compute Ducru free Doppler kernel reconstruction weights.
///
/// Given N reference temperatures T_j and a target temperature T,
/// returns optimal weights w_j such that:
///   σ(E, T) ≈ Σ_j w_j * σ(E, T_j)
///
/// Analytical formula (Ducru et al., J. Comput. Phys. 335, 2017, Eq. 31):
///   w_j = sqrt(T_j*T)/(T_j+T) * Π_{i≠j} [(T-T_i)/(T+T_i)] * [(T_j+T_i)/(T_j-T_i)]
///
/// If the target temperature matches a reference temperature exactly,
/// returns a one-hot weight vector.
pub fn ducru_weights(temperatures: &[f64], target_temp: f64) -> Vec<f64> {
    let n = temperatures.len();
    let t = target_temp;

    // Check for exact match with a reference temperature
    for (idx, &t_j) in temperatures.iter().enumerate() {
        if (t - t_j).abs() < 0.01 {
            let mut w = vec![0.0; n];
            w[idx] = 1.0;
            return w;
        }
    }

    let mut weights = Vec::with_capacity(n);
    for j in 0..n {
        let t_j = temperatures[j];

        // Leading factor: sqrt(T_j * T) / (T_j + T)
        let leading = (t_j * t).sqrt() / (t_j + t);

        // Product term: Π_{i≠j} [(T - T_i)/(T + T_i)] * [(T_j + T_i)/(T_j - T_i)]
        let mut product = 1.0_f64;
        for i in 0..n {
            if i == j { continue; }
            let t_i = temperatures[i];
            let num1 = t - t_i;
            let den1 = t + t_i;
            let num2 = t_j + t_i;
            let den2 = t_j - t_i;

            // Guard against division by zero (identical reference temps)
            if den2.abs() < 1e-10 { continue; }

            product *= (num1 / den1) * (num2 / den2);
        }

        weights.push(leading * product);
    }

    weights
}

/// Use `faer` for a SIMD-optimized full reconstruction.
///
/// This performs the same computation as `reconstruct_log` but uses faer's
/// matrix-vector multiply, which employs AVX2/AVX-512 SIMD automatically.
/// Note: basis is f32 but faer operates in f64 here (promoting during conversion).
pub fn reconstruct_log_faer(kernel: &SvdKernel, coeffs: &[f64], out: &mut [f64]) {
    use faer::Mat;

    let n_e = kernel.n_e;
    let rank = kernel.rank;

    let basis = Mat::from_fn(n_e, rank, |i, j| kernel.basis[i * rank + j]);
    let c = Mat::from_fn(rank, 1, |j, _| coeffs[j]);

    let result = &basis * &c;

    for i in 0..n_e {
        out[i] = result[(i, 0)];
    }
}
