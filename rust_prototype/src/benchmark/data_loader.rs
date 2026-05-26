// SPDX-License-Identifier: MIT
//! Stage 3 — per-case nuclide / material resolution + GPU upload.
//!
//! Consumes a `CaseSpec` + the `LoadedScene` returned alongside it
//! by Stage 2, calls `material_resolve::resolve_materials_with_data_dir`
//! to bind every JSON nuclide to a real HDF5 + SVD payload, computes
//! the lazy-TSL temperature filter, and (when CUDA is enabled and the
//! case is routed to GPU) ships the H→D upload on `stream_transfer`
//! and records the completion event.
//!
//! Phase 2: CPU path implemented end-to-end. GPU upload assembly is
//! stubbed with a clear TODO — once it lands the same Pipeline::run
//! drives heterogeneous CPU + GPU sweeps unchanged.
//!
//! See `docs/benchmark-pipeline-spec.md` §5.3 and
//! `docs/benchmark-pipeline-spec-addendum-lazy-tsl.md` for the design.

use std::sync::Arc;
use std::time::Instant;

use crate::geometry::scene_io::LoadedScene;
use crate::transport::material_resolve::{self, ResolvedMaterials, ResolveError};
use crate::transport::nuclides::NuclideLibrary;

use super::case_bundle::{compute_tsl_temp_filter, CaseBundle};
use super::case_loader::CaseSpec;

/// Errors surfaced by Stage 3. Distinct from `CaseParseError` so the
/// Pipeline can classify failures (parse error = malformed input;
/// resolve error = nuclear-data issue; upload error = device state).
#[derive(Debug, thiserror::Error)]
pub enum DataLoadError {
    #[error("resolve_materials: {0}")]
    Resolve(#[from] ResolveError),

    #[error(
        "material[{material_idx}] {material_name:?} has {n_nuclides} nuclides, but the engine \
         supports at most {max_allowed} per material (MAX_NUCLIDES_PER_MATERIAL). \
         Split the material or raise the limit and rebuild."
    )]
    TooManyNuclides {
        material_idx: usize,
        material_name: String,
        n_nuclides: usize,
        max_allowed: usize,
    },

    #[cfg(feature = "cuda")]
    #[error("GPU upload: {0}")]
    GpuUpload(String),
}

/// Slot list ready to hand to `upload_sab_data_multi_cached_filtered`.
/// Pair of (TSL arc, xs_kernel_idx) in the order the kernel expects.
pub type SabSlots = Vec<(
    Arc<crate::thermal::ThermalScatteringData>,
    usize, /* nuc_idx */
)>;

/// Stage 3 entry point — resolve materials + compute TSL filter. CPU
/// path only (no GPU upload here). The benchmark router selects the
/// backend AFTER stage 3; bundles destined for the GPU executor get
/// `resolve_case_gpu` instead, which adds the H→D upload.
pub fn resolve_case(
    spec: CaseSpec,
    loaded: &LoadedScene,
    lib: &NuclideLibrary,
    svd_rank: usize,
    thermal_dir: &std::path::Path,
) -> Result<CaseBundle, DataLoadError> {
    let load_start = Instant::now();

    let resolved = material_resolve::resolve_materials_with_data_dir(
        &loaded.materials,
        lib,
        svd_rank,
        thermal_dir,
    )?;

    let max_nuc = crate::MAX_NUCLIDES_PER_MATERIAL;
    for (mi, mat) in resolved.materials.iter().enumerate() {
        if mat.nuclides.len() > max_nuc {
            return Err(DataLoadError::TooManyNuclides {
                material_idx: mi,
                material_name: mat.name.clone(),
                n_nuclides: mat.nuclides.len(),
                max_allowed: max_nuc,
            });
        }
    }

    Ok(CaseBundle {
        spec,
        resolved,
        #[cfg(feature = "cuda")]
        gpu_data: None,
        load_start,
        load_end: Instant::now(),
    })
}

/// GPU variant of `resolve_case`: runs the CPU resolution first, then
/// builds a `GpuBundleHandle` on `stream_transfer` so the executor
/// can `stream_compute.wait` on the recorded event before kernel
/// launch.
#[cfg(feature = "cuda")]
pub fn resolve_case_gpu(
    spec: CaseSpec,
    loaded: &LoadedScene,
    lib: &NuclideLibrary,
    svd_rank: usize,
    thermal_dir: &std::path::Path,
    ctx: &crate::gpu_transport::GpuTransportContext,
    stream_transfer: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<CaseBundle, DataLoadError> {
    let mut bundle = resolve_case(spec, loaded, lib, svd_rank, thermal_dir)?;

    let sab_slots = build_sab_slots(&bundle.resolved);
    let temp_filter = build_temp_filter(&bundle.resolved, &sab_slots);

    bundle.gpu_data = gpu::upload_gpu_bundle(
        ctx,
        &bundle.resolved,
        &sab_slots,
        &temp_filter,
        svd_rank,
        stream_transfer,
    )?;
    // Refresh load_end so the recorded timing includes the H→D upload.
    bundle.load_end = Instant::now();
    Ok(bundle)
}

/// Build the `(Arc<TSL>, nuc_idx)` slot list from a resolved provider.
///
/// Only nuclides that actually carry thermal-scattering data
/// contribute. The order matches `provider.thermal`'s iteration order,
/// which is also the iteration order the cache key uses — keep stable
/// to maximise cross-case cache hits.
pub fn build_sab_slots(resolved: &ResolvedMaterials) -> SabSlots {
    resolved
        .provider
        .thermal
        .iter()
        .enumerate()
        .filter_map(|(i, opt)| opt.as_ref().map(|arc| (Arc::clone(arc), i)))
        .collect()
}

/// Compute the lazy-TSL per-slot temperature filter for this case.
///
/// Thin wrapper around `compute_tsl_temp_filter` that decouples the
/// `CaseBundle` consumer from the underlying helper signature. Future
/// pipeline-level overrides (e.g. depletion-aware widened brackets)
/// land here without touching every caller.
pub fn build_temp_filter(
    resolved: &ResolvedMaterials,
    slots: &SabSlots,
) -> Vec<Vec<usize>> {
    compute_tsl_temp_filter(resolved, slots)
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use crate::gpu_transport::GpuTransportContext;

    /// Estimate the GPU bundle's total device footprint before issuing
    /// the upload. Sum of per-nuclide bytes + filtered SAB bytes +
    /// flat-pack overhead + particle-bank reservation. The Stage 3
    /// VRAM gate compares this against `cuMemGetInfo().free` × safety.
    pub fn estimate_bundle_bytes(
        resolved: &ResolvedMaterials,
        sab_slots: &SabSlots,
        temp_filter: &[Vec<usize>],
        particles_per_batch: u32,
    ) -> usize {
        let n_nuc = resolved.provider.nuclides.len().max(1);

        // TODO Phase 3: replace this constant with a real per-nuclide
        // device-bytes estimator that walks the SVD ranks + table sizes
        // (mirrors `GpuTransportContext::upload_nuclide_data`'s
        // accumulators, the way `estimate_sab_device_bytes` mirrors
        // its SAB sibling). Today the only callers are the planner-
        // level VRAM gate, and an ~upper-bound constant is safer
        // than under-estimating: 8 MB / nuclide covers a rank-15
        // U-235 + URR + nu(E) within ~30%.
        let nuclide_bytes: usize = resolved.provider.nuclides.len() * 8 * 1024 * 1024;

        let sab_bytes =
            GpuTransportContext::estimate_sab_device_bytes_filtered(sab_slots, temp_filter, n_nuc);

        // Flat-pack overhead — `GpuNuclideData` ships ~N_PARAMS pointer
        // arrays, each `n_nuc * 8` bytes. Constant per-case.
        let flat_pack = n_nuc * 186 * 8;

        // Particle bank reservation — 2× particles per batch to cover
        // fission emission. Each particle is a ~96-byte SoA row on the
        // device.
        let particles = (particles_per_batch as usize) * 2 * 96;

        nuclide_bytes + sab_bytes + flat_pack + particles
    }

    /// Build the per-case GPU bundle on `stream_transfer`. Stage 3 of
    /// the benchmark pipeline calls this once a case has been
    /// resolved; the resulting `GpuBundleHandle` carries every
    /// device-side buffer plus a `CudaEvent` recorded on
    /// `stream_transfer` after the final `clone_htod`. Stage 4's GPU
    /// executor `stream_compute.wait`s on that event before launching
    /// the transport kernel — gives the driver explicit cross-stream
    /// ordering without serialising the two streams.
    ///
    /// Uses the stream-parameterised upload variants
    /// (`upload_nuclide_data_on_stream`,
    /// `upload_material_data_on_stream`,
    /// `upload_sab_data_multi_cached_filtered`,
    /// `upload_wmp_data_empty_on_stream`) so every H→D copy lands on
    /// `stream_transfer`, not the context default stream. Per the v2
    /// spec §3.2 this is what makes Stage 3 ↔ Stage 4 overlap real.
    pub fn upload_gpu_bundle(
        ctx: &GpuTransportContext,
        resolved: &ResolvedMaterials,
        sab_slots: &SabSlots,
        temp_filter: &[Vec<usize>],
        svd_rank: usize,
        stream_transfer: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Option<crate::benchmark::case_bundle::GpuBundleHandle>, DataLoadError> {
        let n_nuc = resolved.provider.nuclides.len();

        let nuc_data = ctx
            .upload_nuclide_data_on_stream(
                &resolved.provider.nuclides,
                svd_rank,
                stream_transfer,
            )
            .map_err(|e| DataLoadError::GpuUpload(format!("upload_nuclide_data: {e}")))?;

        let awrs: Vec<f64> = resolved
            .provider
            .nuclides
            .iter()
            .map(|n| n.awr)
            .collect();
        let nu_bars: Vec<f64> = resolved
            .provider
            .nuclides
            .iter()
            .map(|n| n.nu_bar_const)
            .collect();
        let q_n2ns: Vec<f64> = resolved
            .provider
            .nuclides
            .iter()
            .map(|n| n.q_n2n)
            .collect();
        let q_n3ns: Vec<f64> = resolved
            .provider
            .nuclides
            .iter()
            .map(|n| n.q_n3n)
            .collect();
        let mat_data = ctx
            .upload_material_data_on_stream(
                &resolved.materials,
                &awrs,
                &nu_bars,
                &q_n2ns,
                &q_n3ns,
                stream_transfer,
            )
            .map_err(|e| DataLoadError::GpuUpload(format!("upload_material_data: {e}")))?;

        let sab_data = ctx
            .upload_sab_data_multi_cached_filtered(sab_slots, temp_filter, n_nuc)
            .map_err(|e| DataLoadError::GpuUpload(format!("upload_sab_data: {e}")))?;

        let wmp_data = ctx
            .upload_wmp_data_empty_on_stream(n_nuc, stream_transfer)
            .map_err(|e| DataLoadError::GpuUpload(format!("upload_wmp_data_empty: {e}")))?;

        let upload_done = ctx
            .new_event()
            .map_err(|e| DataLoadError::GpuUpload(format!("new_event: {e}")))?;
        upload_done
            .record(stream_transfer.as_ref())
            .map_err(|e| DataLoadError::GpuUpload(format!("event record: {e}")))?;

        Ok(Some(crate::benchmark::case_bundle::GpuBundleHandle {
            nuc_data,
            mat_data,
            sab_data,
            wmp_data,
            upload_done: Some(upload_done),
            stream_transfer: std::sync::Arc::clone(stream_transfer),
        }))
    }
}

#[cfg(feature = "cuda")]
pub use gpu::{estimate_bundle_bytes, upload_gpu_bundle};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn build_sab_slots_skips_nuclides_without_tsl() {
        // No need to actually resolve any HDF5 — fabricate a provider
        // shell with mixed Some/None thermal entries and confirm only
        // the Some indices come through.
        use crate::transport::xs_provider::SvdXsProvider;
        let provider = SvdXsProvider {
            nuclides: Vec::new(),
            thermal: vec![None, None, None],
        };
        let resolved = ResolvedMaterials {
            provider,
            materials: Vec::new(),
        };
        let slots = build_sab_slots(&resolved);
        assert!(slots.is_empty());
    }
}
