//! Multigroup cross-section data for random-ray transport.
//!
//! `MaterialMgxs` holds per-group total / absorption / νΣ_f / χ, plus a
//! group-to-group scattering matrix `Σ_s[g_in][g_out]`. `MgxsLibrary` is
//! a vector of these indexed by the material id used by `Geometry`
//! (`EffectiveFill::Material(u32)`). `n_groups` is fixed across the
//! library — mixing group structures is rejected at construction.
//!
//! Adjoint mode is a *view* of the same data with the scattering matrix
//! transposed and (χ, νΣ_f) swapped. `MaterialMgxs::adjoint_view` lets
//! the integrator sweep with the adjoint operator without copying any
//! XS data — the solver passes a `bool is_adjoint` flag through and the
//! per-segment integrator picks the right look-up.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MgxsError {
    #[error("group count mismatch: expected {expected}, got {got}")]
    GroupMismatch { expected: usize, got: usize },
    #[error("Σ_s row {row} has length {got} but n_groups is {expected}")]
    ScatterRowLength {
        row: usize,
        got: usize,
        expected: usize,
    },
    #[error("Σ_t,{group} = {value} is non-positive — random ray needs strictly positive Σ_t")]
    NonPositiveSigmaT { group: usize, value: f64 },
    #[error("χ has nonzero entries but does not sum to 1 (got {got})")]
    ChiNotNormalised { got: f64 },
}

/// Group-to-group scattering matrix. Row-major: `data[g_in * n + g_out]`.
///
/// `data[g_in * n + g_out]` is Σ_s,g_in→g_out — the macroscopic
/// scattering cross section for a neutron starting in group `g_in` and
/// ending in group `g_out`.
#[derive(Debug, Clone)]
pub struct ScatterMatrix {
    pub n_groups: usize,
    pub data: Vec<f64>,
}

impl ScatterMatrix {
    pub fn new(n_groups: usize, data: Vec<f64>) -> Result<Self, MgxsError> {
        if data.len() != n_groups * n_groups {
            return Err(MgxsError::ScatterRowLength {
                row: 0,
                got: data.len(),
                expected: n_groups * n_groups,
            });
        }
        Ok(Self { n_groups, data })
    }

    /// Σ_s,g_in→g_out (forward).
    #[inline]
    pub fn forward(&self, g_in: usize, g_out: usize) -> f64 {
        self.data[g_in * self.n_groups + g_out]
    }

    /// Σ_s,g_out→g_in (adjoint — transpose of `forward`).
    #[inline]
    pub fn adjoint(&self, g_in: usize, g_out: usize) -> f64 {
        self.data[g_out * self.n_groups + g_in]
    }

    /// Total out-scatter from group `g_in`: `Σ_g_out Σ_s,g_in→g_out`.
    pub fn total_out(&self, g_in: usize) -> f64 {
        let mut acc = 0.0;
        for g_out in 0..self.n_groups {
            acc += self.forward(g_in, g_out);
        }
        acc
    }
}

/// Per-material multigroup cross sections.
#[derive(Debug, Clone)]
pub struct MaterialMgxs {
    pub n_groups: usize,
    /// Σ_t per group (cm⁻¹). Must be strictly positive.
    pub sigma_t: Vec<f64>,
    /// Σ_a per group (cm⁻¹).
    pub sigma_a: Vec<f64>,
    /// νΣ_f per group (cm⁻¹).
    pub nu_sigma_f: Vec<f64>,
    /// χ per group (prompt + delayed fission spectrum, normalised so
    /// `Σ χ_g = 1`). All zeros for non-fissionable materials.
    pub chi: Vec<f64>,
    /// Group-to-group scattering matrix.
    pub scatter: ScatterMatrix,
}

impl MaterialMgxs {
    pub fn new(
        sigma_t: Vec<f64>,
        sigma_a: Vec<f64>,
        nu_sigma_f: Vec<f64>,
        chi: Vec<f64>,
        scatter: ScatterMatrix,
    ) -> Result<Self, MgxsError> {
        let n = sigma_t.len();
        if sigma_a.len() != n || nu_sigma_f.len() != n || chi.len() != n || scatter.n_groups != n {
            return Err(MgxsError::GroupMismatch {
                expected: n,
                got: sigma_a
                    .len()
                    .max(nu_sigma_f.len())
                    .max(chi.len())
                    .max(scatter.n_groups),
            });
        }
        for (g, &v) in sigma_t.iter().enumerate() {
            if v <= 0.0 {
                return Err(MgxsError::NonPositiveSigmaT { group: g, value: v });
            }
        }
        let chi_sum: f64 = chi.iter().sum();
        // χ may be all-zero for non-fissionable materials. If any
        // entry is positive, require it to sum to ~1.
        if chi_sum > 0.0 && (chi_sum - 1.0).abs() > 1e-6 {
            return Err(MgxsError::ChiNotNormalised { got: chi_sum });
        }
        Ok(Self {
            n_groups: n,
            sigma_t,
            sigma_a,
            nu_sigma_f,
            chi,
            scatter,
        })
    }

    /// Build a fissionable material from the standard (Σ_t, Σ_a, νΣ_f,
    /// χ, Σ_s) tuple. Convenience over `new` + `ScatterMatrix::new`.
    pub fn fissionable(
        sigma_t: Vec<f64>,
        sigma_a: Vec<f64>,
        nu_sigma_f: Vec<f64>,
        chi: Vec<f64>,
        scatter: Vec<f64>,
    ) -> Result<Self, MgxsError> {
        let n = sigma_t.len();
        let scatter = ScatterMatrix::new(n, scatter)?;
        Self::new(sigma_t, sigma_a, nu_sigma_f, chi, scatter)
    }

    /// Build a non-fissionable material (νΣ_f = 0, χ = 0).
    pub fn nonfissionable(
        sigma_t: Vec<f64>,
        sigma_a: Vec<f64>,
        scatter: Vec<f64>,
    ) -> Result<Self, MgxsError> {
        let n = sigma_t.len();
        let nu_sigma_f = vec![0.0; n];
        let chi = vec![0.0; n];
        let scatter = ScatterMatrix::new(n, scatter)?;
        Self::new(sigma_t, sigma_a, nu_sigma_f, chi, scatter)
    }
}

/// Library of `MaterialMgxs` indexed by material id.
#[derive(Debug, Clone)]
pub struct MgxsLibrary {
    pub n_groups: usize,
    pub materials: Vec<MaterialMgxs>,
}

impl MgxsLibrary {
    pub fn new(materials: Vec<MaterialMgxs>) -> Result<Self, MgxsError> {
        if materials.is_empty() {
            return Err(MgxsError::GroupMismatch {
                expected: 1,
                got: 0,
            });
        }
        let n_groups = materials[0].n_groups;
        for (i, m) in materials.iter().enumerate().skip(1) {
            if m.n_groups != n_groups {
                return Err(MgxsError::GroupMismatch {
                    expected: n_groups,
                    got: m.n_groups,
                });
            }
            let _ = i;
        }
        Ok(Self {
            n_groups,
            materials,
        })
    }

    #[inline]
    pub fn get(&self, mat: u32) -> Option<&MaterialMgxs> {
        self.materials.get(mat as usize)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn fissionable_constructs_with_normalised_chi() {
        let m = MaterialMgxs::fissionable(
            vec![1.0, 1.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![0.7, 0.3],
            vec![0.8, 0.0, 0.05, 0.85],
        )
        .expect("valid material");
        assert_eq!(m.n_groups, 2);
        assert!((m.chi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_non_normalised_chi() {
        let err = MaterialMgxs::fissionable(
            vec![1.0, 1.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![0.5, 0.3],
            vec![0.8, 0.0, 0.05, 0.85],
        )
        .unwrap_err();
        match err {
            MgxsError::ChiNotNormalised { .. } => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn rejects_non_positive_sigma_t() {
        let err =
            MaterialMgxs::nonfissionable(vec![1.0, 0.0], vec![0.1, 0.1], vec![0.5, 0.0, 0.0, 0.5])
                .unwrap_err();
        match err {
            MgxsError::NonPositiveSigmaT { group: 1, .. } => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn scatter_forward_and_adjoint_are_transposes() {
        let s = ScatterMatrix::new(
            3,
            vec![
                0.1, 0.2, 0.3, // row 0: from g=0
                0.4, 0.5, 0.6, // row 1: from g=1
                0.7, 0.8, 0.9, // row 2: from g=2
            ],
        )
        .expect("valid matrix");
        for g_in in 0..3 {
            for g_out in 0..3 {
                assert_eq!(s.forward(g_in, g_out), s.adjoint(g_out, g_in));
            }
        }
    }

    #[test]
    fn library_rejects_mismatched_group_counts() {
        let m1 = MaterialMgxs::nonfissionable(vec![1.0], vec![0.1], vec![0.5]).expect("1g");
        let m2 =
            MaterialMgxs::nonfissionable(vec![1.0, 1.0], vec![0.1, 0.1], vec![0.5, 0.0, 0.0, 0.5])
                .expect("2g");
        assert!(MgxsLibrary::new(vec![m1, m2]).is_err());
    }
}
