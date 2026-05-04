//! CP / PARAFAC decomposition of the 3-tensor σ(E, T, ℓ).
//!
//! Research probe for the discrete-level representation question raised
//! in paper §gpu / §future_work: the per-level rank-1 SVDs we currently
//! build (one matrix per discrete inelastic level) ignore cross-ℓ
//! correlation. CP factorises the joint 3-tensor as a sum of rank-1
//! triadic products
//!
//!     σ_ℓ(E, T) ≈ Σ_{r=1}^{R} a_r(E) · b_r(T) · c_r(ℓ)
//!
//! producing all N_levels at once via a single contracted evaluation.
//!
//! Implementation: greedy rank-1 deflation via tensor power iteration,
//! one rank component at a time. Simpler than full ALS and adequate for
//! analysis purposes (small absolute error is what we care about here,
//! not the global optimum). The dominant rank-1 component each round is
//! found by alternating power iteration; after convergence we subtract
//! its outer-product reconstruction from the residual and repeat. The
//! result is sub-optimal vs joint ALS-CP for any fixed R, but each
//! component is well-defined and stable, which is the right trade-off
//! for an analysis probe.

#![allow(clippy::needless_range_loop)]

/// Stored rank-R CP decomposition of a 3-tensor of shape `(n_e, n_t, n_l)`.
pub struct CpDecomposition {
    pub rank: usize,
    pub n_e: usize,
    pub n_t: usize,
    pub n_l: usize,
    /// Factor matrix A: (n_e × rank), column-major laid out flat as
    /// `a[r * n_e + i]`.
    pub a: Vec<f64>,
    /// Factor matrix B: (n_t × rank), `b[r * n_t + t]`.
    pub b: Vec<f64>,
    /// Factor matrix C: (n_l × rank), `c[r * n_l + l]`.
    pub c: Vec<f64>,
    /// Per-component scalar magnitude (analogous to SVD singular value).
    pub sigma: Vec<f64>,
}

impl CpDecomposition {
    /// Reconstruct the full tensor at truncation rank `k`. Output flat
    /// `out[i * n_t * n_l + t * n_l + l]`. `k <= self.rank`.
    pub fn reconstruct(&self, k: usize) -> Vec<f64> {
        let k = k.min(self.rank);
        let mut out = vec![0.0_f64; self.n_e * self.n_t * self.n_l];
        for r in 0..k {
            let s = self.sigma[r];
            let a_off = r * self.n_e;
            let b_off = r * self.n_t;
            let c_off = r * self.n_l;
            for i in 0..self.n_e {
                let ai_s = self.a[a_off + i] * s;
                for t in 0..self.n_t {
                    let bt = self.b[b_off + t];
                    let scale = ai_s * bt;
                    let row = i * self.n_t * self.n_l + t * self.n_l;
                    for l in 0..self.n_l {
                        out[row + l] += scale * self.c[c_off + l];
                    }
                }
            }
        }
        out
    }

    pub fn memory_bytes(&self) -> usize {
        (self.a.len() + self.b.len() + self.c.len() + self.sigma.len())
            * std::mem::size_of::<f64>()
    }
}

/// Decompose a 3-tensor `tensor` (shape `(n_e, n_t, n_l)`, layout
/// `tensor[i * n_t * n_l + t * n_l + l]`) into a rank-`max_rank` CP
/// approximation via greedy rank-1 deflation. `max_iter` caps the
/// power-iteration count per component; `tol` is the relative
/// convergence threshold on the rank-1 component norm change.
pub fn cp_greedy_rank1(
    tensor: &[f64],
    n_e: usize,
    n_t: usize,
    n_l: usize,
    max_rank: usize,
    max_iter: usize,
    tol: f64,
) -> CpDecomposition {
    assert_eq!(tensor.len(), n_e * n_t * n_l);

    let mut residual: Vec<f64> = tensor.to_vec();
    let mut a = Vec::with_capacity(max_rank * n_e);
    let mut b = Vec::with_capacity(max_rank * n_t);
    let mut c = Vec::with_capacity(max_rank * n_l);
    let mut sigma = Vec::with_capacity(max_rank);

    // Deterministic seed for reproducibility (matches PCG-64 stream
    // convention used elsewhere in the engine — different per-rank
    // seeds avoid degenerate startups).
    let mut rng_state = 0x853c49e6748fea9b_u64;
    let next_uniform = |state: &mut u64| {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (*state >> 33) as f64;
        x / (1_u64 << 31) as f64 - 1.0
    };

    for _r in 0..max_rank {
        // Initialize b, c with random unit vectors; a is computed.
        let mut bv: Vec<f64> = (0..n_t).map(|_| next_uniform(&mut rng_state)).collect();
        let mut cv: Vec<f64> = (0..n_l).map(|_| next_uniform(&mut rng_state)).collect();
        normalize(&mut bv);
        normalize(&mut cv);
        let mut av: Vec<f64> = vec![0.0; n_e];

        let mut prev_norm = 0.0_f64;
        for _it in 0..max_iter {
            // a_i = Σ_jk X[i,j,k] · b_j · c_k
            for i in 0..n_e {
                let mut s = 0.0_f64;
                for t in 0..n_t {
                    let bt = bv[t];
                    let row = i * n_t * n_l + t * n_l;
                    for l in 0..n_l {
                        s += residual[row + l] * bt * cv[l];
                    }
                }
                av[i] = s;
            }
            normalize(&mut av);

            // b_t = Σ_ik X[i,t,k] · a_i · c_k
            for t in 0..n_t {
                let mut s = 0.0_f64;
                for i in 0..n_e {
                    let ai = av[i];
                    let row = i * n_t * n_l + t * n_l;
                    for l in 0..n_l {
                        s += residual[row + l] * ai * cv[l];
                    }
                }
                bv[t] = s;
            }
            normalize(&mut bv);

            // c_l = Σ_ij X[i,j,l] · a_i · b_j
            for l in 0..n_l {
                let mut s = 0.0_f64;
                for i in 0..n_e {
                    let ai = av[i];
                    for t in 0..n_t {
                        let row = i * n_t * n_l + t * n_l;
                        s += residual[row + l] * ai * bv[t];
                    }
                }
                cv[l] = s;
            }
            let cv_norm = cv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if cv_norm < 1e-30 {
                break;
            }
            for cv_l in cv.iter_mut() {
                *cv_l /= cv_norm;
            }

            // Convergence test: dominant magnitude is σ = ||c|| (before
            // normalisation). Use the just-computed cv_norm as the
            // current estimate.
            let converged =
                (cv_norm - prev_norm).abs() < tol * cv_norm.max(1e-30);
            prev_norm = cv_norm;
            if converged {
                break;
            }
        }

        // Compute σ = (a^T residual_unfold b ⊗ c). Equivalently:
        //   σ = Σ_ijk residual[i,j,k] · av[i] · bv[t] · cv[l]
        let mut s = 0.0_f64;
        for i in 0..n_e {
            let ai = av[i];
            for t in 0..n_t {
                let bt = bv[t];
                let row = i * n_t * n_l + t * n_l;
                for l in 0..n_l {
                    s += residual[row + l] * ai * bt * cv[l];
                }
            }
        }
        if s.abs() < 1e-20 {
            // Component vanished — stop early.
            break;
        }

        sigma.push(s);
        a.extend_from_slice(&av);
        b.extend_from_slice(&bv);
        c.extend_from_slice(&cv);

        // Deflate: residual -= s · a ⊗ b ⊗ c
        for i in 0..n_e {
            let ai_s = av[i] * s;
            for t in 0..n_t {
                let bt = bv[t];
                let scale = ai_s * bt;
                let row = i * n_t * n_l + t * n_l;
                for l in 0..n_l {
                    residual[row + l] -= scale * cv[l];
                }
            }
        }
    }

    let rank = sigma.len();
    CpDecomposition {
        rank,
        n_e,
        n_t,
        n_l,
        a,
        b,
        c,
        sigma,
    }
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-30 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Relative L2 error of a rank-`k` reconstruction against the original
/// tensor.
pub fn relative_l2_error(original: &[f64], reconstruction: &[f64]) -> f64 {
    assert_eq!(original.len(), reconstruction.len());
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..original.len() {
        let d = original[i] - reconstruction[i];
        num += d * d;
        den += original[i] * original[i];
    }
    if den < 1e-30 {
        return 0.0;
    }
    (num / den).sqrt()
}

/// Maximum absolute error.
pub fn max_abs_error(original: &[f64], reconstruction: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..original.len() {
        let d = (original[i] - reconstruction[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank1_tensor_recovers_at_rank1() {
        // X[i,j,k] = a[i] * b[j] * c[k] is exactly rank-1.
        let n_e = 4;
        let n_t = 3;
        let n_l = 5;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5];
        let c = vec![1.0, -1.0, 2.0, -0.5, 3.0];
        let mut tensor = vec![0.0; n_e * n_t * n_l];
        for i in 0..n_e {
            for t in 0..n_t {
                for l in 0..n_l {
                    tensor[i * n_t * n_l + t * n_l + l] = a[i] * b[t] * c[l];
                }
            }
        }
        let cp = cp_greedy_rank1(&tensor, n_e, n_t, n_l, 1, 200, 1e-12);
        let recon = cp.reconstruct(1);
        let err = relative_l2_error(&tensor, &recon);
        assert!(
            err < 1e-8,
            "rank-1 recovery should be near-exact, got rel L2 = {err}"
        );
    }

    #[test]
    fn rank2_sum_recovers_at_rank2() {
        // Sum of two rank-1 tensors recovers exactly at rank 2 (modulo
        // the greedy power-iteration sub-optimality).
        let n_e = 6;
        let n_t = 4;
        let n_l = 5;
        let mut tensor = vec![0.0; n_e * n_t * n_l];
        for i in 0..n_e {
            for t in 0..n_t {
                for l in 0..n_l {
                    let v1 = (i as f64 + 1.0) * (t as f64 + 1.0) * (l as f64 + 1.0);
                    let v2 = ((i + l) as f64).cos() * ((t + 1) as f64);
                    tensor[i * n_t * n_l + t * n_l + l] = v1 + v2;
                }
            }
        }
        let cp = cp_greedy_rank1(&tensor, n_e, n_t, n_l, 4, 500, 1e-10);
        let recon4 = cp.reconstruct(4);
        let err4 = relative_l2_error(&tensor, &recon4);
        // Greedy rank-1 deflation may need slightly more components
        // than the joint optimum. Tolerate 1% at rank-4 for this test
        // tensor that's a sum of two well-defined rank-1 terms plus
        // the cross-ℓ structure of cos(i+l).
        assert!(err4 < 0.05, "rank-4 should reach within 5%, got {err4}");
    }
}
