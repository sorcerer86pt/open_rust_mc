// SPDX-License-Identifier: MIT
//! Slot pool and per-case bundle.
//!
//! Stage 3 produces a `CaseBundle` (resolved nuclides + GPU upload,
//! when applicable) and writes it into one of the `SlotArray` slots.
//! Stage 4 takes it out, runs transport, and marks the slot `Done`.
//! Bounded slot count gives the pipeline natural backpressure: when
//! every slot is full, Stage 3 blocks on `empty_notify` until Stage 4
//! finishes a case.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaEvent, CudaStream};

use crate::transport::material_resolve::ResolvedMaterials;

use super::case_loader::CaseSpec;

/// Routing target after the §5.4.0 rule fires. Carried on the slot
/// so the executor threads can claim only their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
}

/// Per-case GPU bundle handle. Owns the device buffers for the
/// duration of the case; drop frees VRAM (the in-engine LFU caches
/// may keep some `Arc`s alive longer for cross-case reuse).
///
/// `upload_done` is the `CudaEvent` recorded on `stream_transfer` at
/// the end of Stage 3's H→D copies. Stage 4 must `stream_compute.wait`
/// on it before launching kernels that read these buffers.
#[cfg(feature = "cuda")]
pub struct GpuBundleHandle {
    pub nuc_data: Arc<crate::gpu_transport::GpuNuclideData>,
    pub mat_data: crate::gpu_transport::GpuMaterialData,
    pub sab_data: Arc<crate::gpu_transport::GpuSabData>,
    pub wmp_data: crate::gpu_transport::GpuWmpData,
    pub upload_done: Option<CudaEvent>,
    /// The stream the upload went through. Stored so debugging can
    /// re-record on the same stream.
    pub stream_transfer: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuBundleHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBundleHandle")
            .field("upload_done", &self.upload_done.is_some())
            .finish_non_exhaustive()
    }
}

/// Resolved + (optionally) uploaded case, ready for the executor.
pub struct CaseBundle {
    pub spec: CaseSpec,
    pub resolved: ResolvedMaterials,
    #[cfg(feature = "cuda")]
    pub gpu_data: Option<GpuBundleHandle>,
    pub load_start: Instant,
    pub load_end: Instant,
}

/// State machine for one slot in the `SlotArray`. Transitions:
/// `Empty → Loading → Ready → Running → Done → Empty`.
pub enum Slot {
    Empty,
    Loading { case_id: String },
    Ready(Box<CaseBundle>),
    Running { case_id: String },
    /// Result emitted; slot is awaiting reuse. ResultProcessor flips
    /// it back to `Empty` after consuming the `ExecutionResult`.
    Done,
}

/// Bounded pool of slots shared between Stage 3 (writer) and the two
/// Stage 4 executors (readers). Capacity defined by §5.3.1.
pub struct SlotArray {
    pub slots: Vec<Mutex<Slot>>,
    /// Signalled when any slot transitions to `Ready`. Both executors
    /// wait on this; the one whose backend matches the slot's runner
    /// claims it.
    pub ready_notify: Condvar,
    /// Signalled when any slot transitions to `Done` / `Empty`.
    /// Stage 3 waits on this when all slots are full.
    pub empty_notify: Condvar,
}

impl SlotArray {
    pub fn with_capacity(n: usize) -> Self {
        let slots = (0..n).map(|_| Mutex::new(Slot::Empty)).collect();
        Self {
            slots,
            ready_notify: Condvar::new(),
            empty_notify: Condvar::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}

/// Compute the per-slot temperature filter for a case's TSL upload.
///
/// For each `(tsl, nuc_idx)` slot, return the sorted union of bracket
/// pairs returned by `tsl.bracket(material.temperature)` across every
/// material in `resolved.materials` that references `nuc_idx`. The
/// resulting filter is what the benchmark pipeline's Stage 3 hands to
/// `upload_sab_data_multi_cached_filtered` so the device only sees the
/// kT columns each material can actually sample.
///
/// Safety contract: for any material `m` with `m.nuclides[k]
/// .xs_kernel_idx == nuc_idx`, the returned filter MUST contain both
/// indices returned by `tsl.bracket(m.temperature)`. Violating that
/// invariant lets the kernel clamp on an off-bracket boundary and
/// silently shift the effective material temperature.
///
/// Degenerate cases:
///   * No material references `nuc_idx` → return all temps (safe;
///     the kernel only reads slots that materials touch anyway).
///   * `tsl.kts.len() == 1` → return `[0]` (matches `bracket()`).
///
/// See `docs/benchmark-pipeline-spec-addendum-lazy-tsl.md` §2.4.
pub fn compute_tsl_temp_filter(
    resolved: &crate::transport::material_resolve::ResolvedMaterials,
    slots: &[(
        Arc<crate::thermal::ThermalScatteringData>,
        usize, /* nuc_idx */
    )],
) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(slots.len());
    for (tsl, nuc_idx) in slots.iter() {
        let mut indices: Vec<usize> = Vec::with_capacity(4);

        // Walk every material; if any nuclide entry binds to this
        // nuc_idx, include that material's bracket pair.
        let mut referenced = false;
        for mat in resolved.materials.iter() {
            let touches = mat
                .nuclides
                .iter()
                .any(|n| n.xs_kernel_idx == *nuc_idx);
            if !touches {
                continue;
            }
            referenced = true;
            let (lo, hi) = tsl.bracket(mat.temperature);
            indices.push(lo);
            if hi != lo {
                indices.push(hi);
            }
        }

        if !referenced {
            // Nuclide carries a TSL but no material references it.
            // Shouldn't normally happen (slot list is derived from
            // material binding) but defend against caller-side
            // mismatches by uploading the full grid — same payload
            // as the pre-lazy-TSL path.
            indices.extend(0..tsl.kts.len());
        }

        indices.sort_unstable();
        indices.dedup();
        out.push(indices);
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::thermal::{
        ContinuousInelastic, InelasticDist, InelasticThermal, ThermalScatteringData,
    };
    use crate::transport::material::{Material, NuclideEntry};
    use crate::transport::material_resolve::ResolvedMaterials;
    use crate::transport::xs_provider::SvdXsProvider;

    fn make_tsl(kts_ev: Vec<f64>) -> Arc<ThermalScatteringData> {
        let n_temps = kts_ev.len();
        let inelastic: Vec<InelasticThermal> = (0..n_temps)
            .map(|_| InelasticThermal {
                energy: vec![1e-5, 1.0],
                xs: vec![0.0, 0.0],
                dist: InelasticDist::Continuous(ContinuousInelastic {
                    n_inc: 0,
                    offsets: Vec::new(),
                    interp: Vec::new(),
                    e_out: Vec::new(),
                    pdf_e: Vec::new(),
                    cdf_e: Vec::new(),
                    mu_interp: Vec::new(),
                    mu_offsets: Vec::new(),
                    mu: Vec::new(),
                    pdf_mu: Vec::new(),
                    cdf_mu: Vec::new(),
                }),
            })
            .collect();
        Arc::new(ThermalScatteringData {
            name: "test".to_string(),
            nuclides: vec!["X".to_string()],
            energy_max: 4.0,
            awr: 1.0,
            kts: kts_ev,
            temp_labels: (0..n_temps).map(|i| format!("{i}K")).collect(),
            inelastic,
            elastic: None,
        })
    }

    fn make_resolved(
        materials: Vec<(f64 /* T_K */, usize /* nuc_idx_used */)>,
    ) -> ResolvedMaterials {
        let mats = materials
            .into_iter()
            .enumerate()
            .map(|(i, (t, k))| {
                let mut m = Material::new(&format!("m{i}"), t);
                m.nuclides.push(NuclideEntry {
                    atom_density: 1.0,
                    xs_kernel_idx: k,
                });
                m
            })
            .collect();
        ResolvedMaterials {
            provider: SvdXsProvider {
                nuclides: Vec::new(),
                thermal: Vec::new(),
            },
            materials: mats,
        }
    }

    /// Material at 295 K (strictly between two tabulated kT) on a TSL
    /// gridded at {77, 100, 294, 296, 400} K → bracket = (2, 3).
    /// Uses 295 K rather than 294 K to avoid the floating-point edge
    /// where `kt == kts[i]` exactly and `select_temperature` would
    /// land on the lower pair.
    #[test]
    fn single_material_between_two_temps_picks_pair() {
        let kts: Vec<f64> = [77.0, 100.0, 294.0, 296.0, 400.0]
            .iter()
            .map(|t| t * 8.617_333_262e-5)
            .collect();
        let tsl = make_tsl(kts);
        let resolved = make_resolved(vec![(295.0, 0)]);
        let slots = vec![(Arc::clone(&tsl), 0)];
        let filter = compute_tsl_temp_filter(&resolved, &slots);
        assert_eq!(filter.len(), 1);
        assert_eq!(filter[0], vec![2, 3]);
    }

    /// Two materials at different non-edge temperatures → union of
    /// their brackets. 295 K → {2, 3}; 600 K → {4, 5}. Picks the
    /// upper-pair indices because each kt lies strictly above the
    /// lower kT in its bracket.
    #[test]
    fn multi_material_unions_brackets() {
        let kts: Vec<f64> = [77.0, 100.0, 294.0, 296.0, 500.0, 700.0]
            .iter()
            .map(|t| t * 8.617_333_262e-5)
            .collect();
        let tsl = make_tsl(kts);
        let resolved = make_resolved(vec![(295.0, 0), (600.0, 0)]);
        let slots = vec![(Arc::clone(&tsl), 0)];
        let filter = compute_tsl_temp_filter(&resolved, &slots);
        assert_eq!(filter[0], vec![2, 3, 4, 5]);
    }

    /// Material at exactly a tabulated kT → bracket is the LOWER pair
    /// because `select_temperature` advances `i` only while
    /// `kts[i+1] < kt` (strict). Mirroring this in `bracket()` keeps
    /// the filter consistent with what the kernel will actually
    /// sample.
    #[test]
    fn material_on_grid_picks_lower_pair() {
        let kts: Vec<f64> = [77.0, 100.0, 294.0, 296.0, 400.0]
            .iter()
            .map(|t| t * 8.617_333_262e-5)
            .collect();
        let tsl = make_tsl(kts);
        let resolved = make_resolved(vec![(294.0, 0)]);
        let slots = vec![(Arc::clone(&tsl), 0)];
        let filter = compute_tsl_temp_filter(&resolved, &slots);
        // bracket(294 K) → (1, 2) because kts[2] == kt so the loop
        // never advances past i=1.
        assert_eq!(filter[0], vec![1, 2]);
    }

    /// Material temperature above the grid top → clamps to last index;
    /// pair collapses.
    #[test]
    fn above_grid_clamps_to_last() {
        let kts: Vec<f64> = [77.0, 294.0]
            .iter()
            .map(|t| t * 8.617_333_262e-5)
            .collect();
        let tsl = make_tsl(kts);
        let resolved = make_resolved(vec![(600.0, 0)]);
        let slots = vec![(Arc::clone(&tsl), 0)];
        let filter = compute_tsl_temp_filter(&resolved, &slots);
        assert_eq!(filter[0], vec![1]);
    }

    /// No material references the slot's nuclide → fallback to all
    /// temps (defensive — avoids silent under-uploading if the caller
    /// hands a stale slot list).
    #[test]
    fn unreferenced_nuclide_falls_back_to_all_temps() {
        let kts: Vec<f64> = [77.0, 294.0, 600.0]
            .iter()
            .map(|t| t * 8.617_333_262e-5)
            .collect();
        let tsl = make_tsl(kts);
        // Material binds nuc_idx 1; slot is for nuc_idx 0.
        let resolved = make_resolved(vec![(300.0, 1)]);
        let slots = vec![(Arc::clone(&tsl), 0)];
        let filter = compute_tsl_temp_filter(&resolved, &slots);
        assert_eq!(filter[0], vec![0, 1, 2]);
    }
}
