//! SVD-based cross-section provider — connects the SVD reconstruction
//! kernel to the transport loop.
//!
//! For each nuclide, stores SVD kernels for key reactions (elastic,
//! fission, capture). At lookup time, reconstructs σ(E) via a dot
//! product instead of binary-searching a table.

use crate::decompose;
use crate::hdf5_reader::NuclideData;
use crate::kernel::SvdKernel;
use crate::physics::collision::MicroXs;
use crate::transport::simulate::XsProvider;

/// Per-nuclide SVD-compressed cross-section data.
pub struct NuclideKernels {
    /// SVD kernel for elastic scattering (MT=2).
    pub elastic: Option<ReactionKernel>,
    /// SVD kernel for inelastic scattering (MT=4, total inelastic).
    pub inelastic: Option<ReactionKernel>,
    /// SVD kernel for (n,2n) reaction (MT=16).
    pub n2n: Option<ReactionKernel>,
    /// SVD kernel for fission (MT=18). None for non-fissile nuclides.
    pub fission: Option<ReactionKernel>,
    /// SVD kernel for capture (MT=102).
    pub capture: Option<ReactionKernel>,
    /// Atomic weight ratio.
    pub awr: f64,
    /// Average neutrons per fission.
    pub nu_bar: f64,
}

/// SVD kernel + energy grid for a single reaction.
pub struct ReactionKernel {
    pub kernel: SvdKernel,
    /// Pre-computed temperature coefficients (one set per temperature).
    pub coeffs: Vec<f64>,
}

impl ReactionKernel {
    /// Reconstruct cross-section at a given energy (linear scale, barns).
    ///
    /// Uses binary search on the energy grid to find the index, then
    /// reconstructs via the SVD dot product.
    #[inline]
    pub fn lookup(&self, energy: f64) -> f64 {
        let energies = self.kernel.energies();
        let n = energies.len();

        // Binary search for the energy index
        let idx = match energies.binary_search_by(|e| {
            e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 { 0 }
                else if i >= n { n - 1 }
                else { i }
            }
        };

        // Reconstruct at this index via SVD dot product
        let log_val = self.kernel.reconstruct_single_log(idx, &self.coeffs);
        10.0_f64.powf(log_val)
    }
}

/// Cross-section provider backed by SVD-compressed kernels.
pub struct SvdXsProvider {
    pub nuclides: Vec<NuclideKernels>,
}

impl XsProvider for SvdXsProvider {
    fn lookup(&self, nuclide_idx: usize, energy: f64) -> MicroXs {
        let nuc = &self.nuclides[nuclide_idx];

        let elastic = nuc.elastic.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let inelastic = nuc.inelastic.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let n2n = nuc.n2n.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let fission = nuc.fission.as_ref()
            .map_or(0.0, |k| k.lookup(energy));
        let capture = nuc.capture.as_ref()
            .map_or(0.0, |k| k.lookup(energy));

        let total = elastic + inelastic + n2n + fission + capture;

        MicroXs {
            total,
            elastic,
            inelastic,
            n2n,
            fission,
            capture,
            nu_bar: nuc.nu_bar,
            awr: nuc.awr,
        }
    }
}

/// Build an SVD kernel for one reaction of one nuclide from HDF5 data.
///
/// Returns None if the reaction doesn't exist or has no data.
pub fn build_reaction_kernel(
    h5_path: &std::path::Path,
    mt: u32,
    svd_rank: usize,
    temp_idx: usize,
) -> Option<ReactionKernel> {
    let data = NuclideData::from_hdf5(h5_path, mt).ok()?;

    if data.n_energy() == 0 || data.n_temp() == 0 {
        return None;
    }

    let log_matrix = data.to_log_matrix();
    let svd = decompose::svd(&log_matrix, data.n_energy(), data.n_temp());

    let rank = svd_rank.min(svd.rank);
    let n_e = svd.n_e;
    let n_t = svd.n_t;

    // Pre-multiply basis
    let mut basis = vec![0.0_f64; n_e * rank];
    for j in 0..rank {
        let s_j = svd.s[j];
        for i in 0..n_e {
            basis[i * rank + j] = svd.u[i * svd.rank + j] * s_j;
        }
    }

    let mut vt_coeffs = vec![0.0_f64; rank * n_t];
    for j in 0..rank {
        for t in 0..n_t {
            vt_coeffs[j * n_t + t] = svd.vt[j * n_t + t];
        }
    }

    let kernel = SvdKernel::new(basis, vt_coeffs, data.energies.clone(), rank, n_e, n_t);

    // Pre-compute coefficients for the requested temperature
    let t_idx = temp_idx.min(n_t - 1);
    let coeffs = kernel.temp_coeffs(t_idx);

    Some(ReactionKernel { kernel, coeffs })
}

/// Load a complete nuclide (all key reactions) from HDF5 and build SVD kernels.
pub fn load_nuclide(
    h5_path: &std::path::Path,
    svd_rank: usize,
    temp_idx: usize,
    awr: f64,
    nu_bar: f64,
) -> NuclideKernels {
    println!("  Loading {} (rank={svd_rank})...", h5_path.display());

    let elastic = build_reaction_kernel(h5_path, 2, svd_rank, temp_idx);
    if elastic.is_some() { println!("    MT=2 (elastic): loaded"); }

    let inelastic = build_reaction_kernel(h5_path, 4, svd_rank, temp_idx);
    if inelastic.is_some() { println!("    MT=4 (inelastic): loaded"); }

    let n2n = build_reaction_kernel(h5_path, 16, svd_rank, temp_idx);
    if n2n.is_some() { println!("    MT=16 (n,2n): loaded"); }

    let fission = build_reaction_kernel(h5_path, 18, svd_rank, temp_idx);
    if fission.is_some() { println!("    MT=18 (fission): loaded"); }

    let capture = build_reaction_kernel(h5_path, 102, svd_rank, temp_idx);
    if capture.is_some() { println!("    MT=102 (capture): loaded"); }

    NuclideKernels { elastic, inelastic, n2n, fission, capture, awr, nu_bar }
}
