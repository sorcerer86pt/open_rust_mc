//! Per-nuclide GPU XS bundle (Stage C of the GPU cache redesign).
//!
//! Stores a single nuclide's worth of GPU-side data so the bundle
//! cache can de-duplicate at the nuclide level instead of the
//! whole-bundle level. A 376-case ICSBEP sweep has ~50 unique
//! nuclides but ~376 unique bundle compositions; per-nuclide caching
//! cuts redundant H→D traffic by ~75× (~530 GB → ~7 GB).
//!
//! See `docs/stage-c-data-model.md` for the full schema + landing
//! order. This module is added empty-handed: subsequent commits
//! extend `PerNuclideGpu` field coverage and wire `assemble_bundle`
//! into `gpu_transport.rs::upload_nuclide_data_uncached`. The kernel
//! ABI stays unchanged until Stage 4 (separate commit, gated on
//! `metal_stats_diag` 3-way passing).

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

use crate::transport::xs_provider::NuclideKernels;

/// Optional per-nuclide ν̄(E) table. Holds either prompt-total
/// (`nu_bar_table`) or delayed-only (`delayed_nu_bar_table`); both
/// share this shape. `n_points` mirrors `energies.len()` and is
/// pre-stored so the assembly stage doesn't need a `cuMemcpy` to
/// read it back.
pub struct NuBarSlicesGpu {
    pub n_points: i32,
    pub energies: CudaSlice<f64>,
    pub values: CudaSlice<f64>,
}

impl NuBarSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.energies.num_bytes() + self.values.num_bytes()
    }
}

/// Single nuclide's GPU-resident XS data. Sized for one nuclide
/// only — no inter-nuclide concatenation. Cached upstream by
/// `NuclideKey = (file_hash, policy_hash, temp_idx, format_version)`;
/// `rank` is captured in the policy hash so the same nuclide at
/// rank=5 vs rank=7 produces two cache entries.
///
/// Fields land incrementally per `docs/stage-c-data-model.md`. Each
/// commit converts one category and keeps `cargo test --features
/// cuda --lib` green. Fields not yet ported are `None` and the
/// bundle assembly stage falls through to the legacy
/// `upload_nuclide_data_uncached` packing path for that category.
pub struct PerNuclideGpu {
    /// Global SVD rank this nuclide was rank-padded for. Bundle
    /// assembly must verify `bundle_rank == per_nuc.rank` for every
    /// nuclide; mismatch means the cache key was wrong.
    pub rank: i32,

    /// Number of points in this nuclide's union energy grid. Stored
    /// scalar (not derivable from `energy_grid.len()` cheaply on the
    /// device side; needed at assembly stage).
    pub n_energy: i32,

    // ── Category A.1 — energy grid (always populated) ──
    pub energy_grid: CudaSlice<f64>,

    // ── Category A.2 — pointwise tables ──
    /// Sum of every HDF5 reaction at each grid point. `None` when
    /// the nuclide didn't ship a total table (most thermal-scattering-
    /// only entries, some photon products).
    pub total_xs: Option<CudaSlice<f64>>,
    /// 7-channel pointwise [n_e × 7]: el / inel / n2n / n3n / fis /
    /// cap / total. Used by the GPU's pointwise XS path for non-SVD
    /// channels.
    pub pointwise_xs: Option<CudaSlice<f64>>,

    // ── Category A.3 — ν̄ tables ──
    pub nu_bar: Option<NuBarSlicesGpu>,
    pub delayed_nu_bar: Option<NuBarSlicesGpu>,
    // ── Future categories land here per `docs/stage-c-data-model.md` ──
}

impl PerNuclideGpu {
    /// Sum of every owned `CudaSlice`'s `num_bytes()`. Cheap (no
    /// device traffic). Feeds the per-nuclide LFU's byte budget.
    pub fn device_bytes(&self) -> usize {
        let mut total = self.energy_grid.num_bytes();
        if let Some(s) = &self.total_xs {
            total += s.num_bytes();
        }
        if let Some(s) = &self.pointwise_xs {
            total += s.num_bytes();
        }
        if let Some(n) = &self.nu_bar {
            total += n.device_bytes();
        }
        if let Some(n) = &self.delayed_nu_bar {
            total += n.device_bytes();
        }
        total
    }
}

/// Upload a single nuclide's per-nuclide-only fields to the device.
/// Categories that are still bundle-only (per-reaction SVD, discrete
/// levels, fission edist, …) land in subsequent commits.
///
/// `rank` is the *global* rank for the bundle — recorded on the
/// returned `PerNuclideGpu` for later cross-checking and used when
/// rank-padding per-level basis arrays (Stage C category C, not
/// yet implemented).
pub fn upload_one_nuclide(
    stream: &Arc<CudaStream>,
    nuc: &NuclideKernels,
    rank: usize,
) -> Result<PerNuclideGpu, Box<dyn std::error::Error>> {
    // Energy grid — shared across all reactions on this nuclide.
    // Pull from whichever kernel exists; matches the priority order
    // used by `gpu_transport::upload_nuclide_data_uncached`.
    let any_kernel = nuc
        .elastic
        .as_ref()
        .or(nuc.fission.as_ref())
        .or(nuc.capture.as_ref())
        .or(nuc.inelastic.as_ref())
        .or(nuc.n2n.as_ref())
        .or(nuc.n3n.as_ref());
    let (energy_grid_vec, n_energy) = match any_kernel {
        Some(rk) => (rk.energies().to_vec(), rk.n_energy() as i32),
        // Sentinel: device pointers must be non-null even when the
        // nuclide has no kernels (e.g. tracking-only entries). The
        // legacy bundle path uses the same `[0.0]` sentinel.
        None => (vec![0.0_f64], 0_i32),
    };
    let energy_grid = stream.clone_htod(&energy_grid_vec)?;

    // Pointwise total XS — present when the HDF5 carries an explicit
    // total table.
    let total_xs = match &nuc.total_xs_raw {
        Some(xs) if !xs.is_empty() => Some(stream.clone_htod(xs)?),
        _ => None,
    };

    // 7-channel pointwise XS [n_e × 7].
    let pointwise_xs = match &nuc.pointwise_xs {
        Some(xs) if !xs.is_empty() => Some(stream.clone_htod(xs)?),
        _ => None,
    };

    let nu_bar = nuc
        .nu_bar_table
        .as_ref()
        .filter(|t| !t.energies.is_empty())
        .map(|t| -> Result<NuBarSlicesGpu, Box<dyn std::error::Error>> {
            Ok(NuBarSlicesGpu {
                n_points: t.energies.len() as i32,
                energies: stream.clone_htod(&t.energies)?,
                values: stream.clone_htod(&t.values)?,
            })
        })
        .transpose()?;

    let delayed_nu_bar = nuc
        .delayed_nu_bar_table
        .as_ref()
        .filter(|t| !t.energies.is_empty())
        .map(|t| -> Result<NuBarSlicesGpu, Box<dyn std::error::Error>> {
            Ok(NuBarSlicesGpu {
                n_points: t.energies.len() as i32,
                energies: stream.clone_htod(&t.energies)?,
                values: stream.clone_htod(&t.values)?,
            })
        })
        .transpose()?;

    Ok(PerNuclideGpu {
        rank: rank as i32,
        n_energy,
        energy_grid,
        total_xs,
        pointwise_xs,
        nu_bar,
        delayed_nu_bar,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::xs_provider::{NuclideKernels, ReactionKernel};
    use cudarc::driver::CudaContext;

    fn try_cuda_stream() -> Option<Arc<CudaStream>> {
        // Skip the test gracefully when no CUDA device is available
        // (CI containers without GPUs, dev boxes booted without the
        // driver loaded, etc.). Same pattern as other GPU lib tests.
        let ctx = CudaContext::new(0).ok()?;
        Some(ctx.default_stream())
    }

    fn minimal_nuclide() -> NuclideKernels {
        // Hand-built kernel with a Table elastic so `energies()`
        // resolves without pulling HDF5 data. Three grid points keep
        // the test fast and the device buffer non-empty.
        let energies = vec![1.0e-5, 1.0, 2.0e7];
        let xs = vec![10.0, 5.0, 1.0];
        let kernel = ReactionKernel::from_table(energies, xs);
        let mut nuc = NuclideKernels::empty(1.0, 0.0);
        nuc.elastic = Some(kernel);
        nuc.total_xs_raw = Some(vec![20.0, 10.0, 2.0]);
        nuc.pointwise_xs = Some(vec![
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,
        ]);
        nuc
    }

    #[test]
    fn upload_one_nuclide_round_trips_energy_grid() {
        let Some(stream) = try_cuda_stream() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let nuc = minimal_nuclide();
        let per_nuc =
            upload_one_nuclide(&stream, &nuc, 5).expect("upload_one_nuclide failed");

        assert_eq!(per_nuc.rank, 5);
        assert_eq!(per_nuc.n_energy, 3);
        // Energy grid bytes survive the round-trip.
        let mut host_grid = vec![0.0_f64; 3];
        stream
            .memcpy_dtoh(&per_nuc.energy_grid, &mut host_grid)
            .expect("dtoh failed");
        assert_eq!(host_grid, vec![1.0e-5, 1.0, 2.0e7]);
        // total_xs_raw round-trips.
        let total = per_nuc.total_xs.as_ref().expect("total_xs missing");
        let mut host_total = vec![0.0_f64; 3];
        stream.memcpy_dtoh(total, &mut host_total).unwrap();
        assert_eq!(host_total, vec![20.0, 10.0, 2.0]);
        // pointwise present, sized for [3 × 7].
        let pw = per_nuc.pointwise_xs.as_ref().expect("pointwise missing");
        assert_eq!(pw.len(), 21);
        // ν̄ / delayed-ν̄ absent on this synthetic nuclide.
        assert!(per_nuc.nu_bar.is_none());
        assert!(per_nuc.delayed_nu_bar.is_none());
        // device_bytes ≥ what we uploaded.
        assert!(per_nuc.device_bytes() >= (3 + 3 + 21) * 8);
    }
}
