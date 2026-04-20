//! SVD decomposition using faer — compute U, Σ, V^T from the cross-section matrix.

use faer::Mat;

/// Result of an SVD decomposition.
pub struct SvdResult {
    /// Left singular vectors: N_E × rank, row-major.
    pub u: Vec<f64>,
    /// Singular values (descending), length = rank.
    pub s: Vec<f64>,
    /// Right singular vectors transposed: rank × N_T, row-major.
    pub vt: Vec<f64>,
    /// Matrix dimensions.
    pub n_e: usize,
    pub n_t: usize,
    pub rank: usize,
}

impl SvdResult {
    /// Reconstruct the full matrix (log scale) at truncation rank `k`.
    /// Returns row-major N_E × N_T.
    pub fn reconstruct_log(&self, k: usize) -> Vec<f64> {
        let k = k.min(self.rank);
        let mut mat = vec![0.0_f64; self.n_e * self.n_t];

        for i in 0..self.n_e {
            for t in 0..self.n_t {
                let mut acc = 0.0_f64;
                for j in 0..k {
                    let u_ij = self.u[i * self.rank + j];
                    let s_j = self.s[j];
                    let vt_jt = self.vt[j * self.n_t + t];
                    acc = (u_ij * s_j).mul_add(vt_jt, acc);
                }
                mat[i * self.n_t + t] = acc;
            }
        }
        mat
    }
}

/// Perform thin SVD on a row-major N_E × N_T matrix using faer.
pub fn svd(matrix: &[f64], n_e: usize, n_t: usize) -> SvdResult {
    assert_eq!(matrix.len(), n_e * n_t);

    let a = Mat::from_fn(n_e, n_t, |i, j| matrix[i * n_t + j]);

    #[allow(non_snake_case)]
    let decomp = a.thin_svd().expect("SVD failed");

    #[allow(non_snake_case)]
    let U = decomp.U();
    let s_col = decomp.S().column_vector();
    #[allow(non_snake_case)]
    let V = decomp.V();

    let rank = n_t.min(n_e);

    // Extract U: N_E × rank, row-major
    let mut u = vec![0.0_f64; n_e * rank];
    for i in 0..n_e {
        for j in 0..rank {
            u[i * rank + j] = U[(i, j)];
        }
    }

    // Extract S: rank
    let mut s = vec![0.0_f64; rank];
    for j in 0..rank {
        s[j] = s_col[j];
    }

    // Extract V^T: rank × N_T, row-major. V is N_T × rank, V^T[j,t] = V[t,j]
    let mut vt = vec![0.0_f64; rank * n_t];
    for j in 0..rank {
        for t in 0..n_t {
            vt[j * n_t + t] = V[(t, j)];
        }
    }

    SvdResult {
        u,
        s,
        vt,
        n_e,
        n_t,
        rank,
    }
}
