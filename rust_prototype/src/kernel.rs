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

/// Pre-built reconstruction engine for a single nuclide + reaction.
pub struct SvdKernel {
    /// U × Σ pre-multiplied basis, row-major: `[n_e][rank]`.
    pub(crate) basis: Vec<f64>,
    /// V^T coefficients, row-major: `[rank][n_t]`.
    pub(crate) vt_coeffs: Vec<f64>,
    /// Energy grid in eV.
    pub(crate) energies: Vec<f64>,
    /// Truncation rank.
    pub(crate) rank: usize,
    /// Number of energy points.
    pub(crate) n_e: usize,
    /// Number of training temperatures.
    pub(crate) n_t: usize,
}

impl SvdKernel {
    /// Construct a kernel from pre-multiplied basis and V^T coefficients.
    pub fn new(
        basis: Vec<f64>,
        vt_coeffs: Vec<f64>,
        energies: Vec<f64>,
        rank: usize,
        n_e: usize,
        n_t: usize,
    ) -> Self {
        Self { basis, vt_coeffs, energies, rank, n_e, n_t }
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

    /// Total memory footprint of the kernel (bytes).
    pub fn memory_bytes(&self) -> usize {
        (self.basis.len() + self.vt_coeffs.len() + self.energies.len())
            * std::mem::size_of::<f64>()
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
                // This compiles to FMA on x86 with -C target-cpu=native
                acc = row[j].mul_add(coeffs[j], acc);
            }
            out[i] = acc;
        }
    }

    /// Reconstruct cross-sections in linear scale (barns) at a given temperature.
    pub fn reconstruct_linear(&self, coeffs: &[f64], out: &mut [f64]) {
        self.reconstruct_log(coeffs, out);
        for val in out.iter_mut().take(self.n_e) {
            *val = 10.0_f64.powf(*val);
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
        10.0_f64.powf(self.reconstruct_single_log(energy_idx, coeffs))
    }

    /// Return the energy grid.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }
}

/// Use `faer` for a SIMD-optimized full reconstruction.
///
/// This performs the same computation as `reconstruct_log` but uses faer's
/// matrix-vector multiply, which employs AVX2/AVX-512 SIMD automatically.
pub fn reconstruct_log_faer(kernel: &SvdKernel, coeffs: &[f64], out: &mut [f64]) {
    use faer::Mat;

    let n_e = kernel.n_e;
    let rank = kernel.rank;

    // Wrap the pre-allocated basis as a faer matrix view (no copy).
    // faer uses column-major by default, but we stored row-major.
    // Build a faer Mat from our row-major data.
    let basis = Mat::from_fn(n_e, rank, |i, j| kernel.basis[i * rank + j]);
    let c = Mat::from_fn(rank, 1, |j, _| coeffs[j]);

    let result = &basis * &c;

    for i in 0..n_e {
        out[i] = result[(i, 0)];
    }
}
