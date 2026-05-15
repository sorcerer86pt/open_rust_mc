//! Event-based GPU neutron transport.
//!
//! CUDA kernel source in `gpu/cuda/transport.cu`, loaded via `include_str!`.
//! Uses packed `TransportParams` struct — all read-only data in one device buffer.
//! Persistent kernel with warp-level reductions and energy-sorted compaction.
//! Full physics parity with CPU: SVD XS, S(α,β), discrete levels, URR, angular dist.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc;

/// Number of u64 fields in the packed TransportParams buffer.
/// Must match N_PARAMS in transport.cu.
const N_PARAMS: usize = 164;

/// NVRTC compile-options builder. Every site that compiles
/// `TRANSPORT_KERNELS` must thread `MAX_NUC_PER_MAT` in from the Rust
/// constant — the CU has no fallback `#define` anymore (intentionally,
/// so host / device cannot silently disagree).
#[allow(dead_code)]
fn transport_kernel_options() -> nvrtc::CompileOptions {
    nvrtc::CompileOptions {
        options: vec![format!(
            "-DMAX_NUC_PER_MAT={}",
            crate::MAX_NUCLIDES_PER_MATERIAL
        )],
        ..Default::default()
    }
}

// ── CUDA kernel source ────────────────────────────────────────────────

/// All CUDA kernels for event-based transport.
///
/// PWR pin cell geometry is hardcoded (9 surfaces, 4 cells, 3 materials).
/// SVD basis data is passed via global memory, coefficients via shared memory.
const TRANSPORT_KERNELS: &str = include_str!("../gpu/cuda/transport.cu");

/// Per-nuclide pointer-identity key. Works because
/// `nuclide_cache::TieredStore` returns the same
/// `Arc<NuclideKernels>` for the same
/// `(file_hash, policy_hash, temp_idx)` tuple — Arc address therefore
/// implies byte-identical content. Two cases sharing U-235 hit the
/// same key and reuse the GPU upload across cases (the cross-case
/// sharing win that's the whole point of Stage C).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PerNuclideCacheKey {
    nuc_ptr: usize,
    rank: usize,
}

/// Per-nuclide cache budget = `BUNDLE_CACHE_DEFAULT_FRACTION × total_mem`.
/// Same env vars as before — `OPEN_RUST_MC_GPU_BUNDLE_CACHE_BYTES`
/// / `_FRACTION` — semantics shift from "one bundle's bytes" to
/// "the union of cached per-nuclide bytes."
const BUNDLE_CACHE_DEFAULT_FRACTION: f64 = 0.75;
const BUNDLE_CACHE_FRACTION_MIN: f64 = 0.05;
const BUNDLE_CACHE_FRACTION_MAX: f64 = 0.95;

/// Compiled CUDA kernels for event-based transport.
pub struct GpuTransportContext {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    k_init_source: CudaFunction,
    #[allow(dead_code)] // retained for future event-based path; loaded from PTX
    k_count_alive: CudaFunction,
    k_compact_alive: CudaFunction,
    k_energy_bin_count: CudaFunction,
    k_energy_bin_scatter: CudaFunction,
    k_transport_persistent: CudaFunction,
    /// Per-nuclide byte-budgeted LFU-with-recency cache (Stage C
    /// step C). Cross-case sharing comes from upstream Arc-dedup in
    /// `nuclide_cache::TieredStore::L1MemoryStore`: same nuclide
    /// across cases → same Arc::as_ptr → same cache key.
    per_nuclide_cache: std::sync::Mutex<PerNuclideCache>,
    cached_bundle_budget: std::sync::OnceLock<usize>,
}

use crate::transport::nuclide_cache::eviction::{
    EvictionStats, LfuEntries, LfuEntriesMut, DEFAULT_AGE_DECAY,
    evict_to_budget,
};

struct PerNuclideCache {
    entries: std::collections::HashMap<
        PerNuclideCacheKey,
        (Arc<crate::gpu_per_nuclide::PerNuclideGpu>, EvictionStats),
    >,
    counter: u64,
    total_bytes: usize,
}

impl PerNuclideCache {
    fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            counter: 0,
            total_bytes: 0,
        }
    }
}

struct PerNuclideCacheAdapter<'a> {
    inner: &'a mut PerNuclideCache,
}

impl LfuEntries for PerNuclideCacheAdapter<'_> {
    type Key = PerNuclideCacheKey;
    fn total_bytes(&self) -> usize {
        self.inner.total_bytes
    }
    fn len(&self) -> usize {
        self.inner.entries.len()
    }
    fn iter_stats(&self) -> Box<dyn Iterator<Item = (&Self::Key, &EvictionStats)> + '_> {
        Box::new(self.inner.entries.iter().map(|(k, (_, s))| (k, s)))
    }
    fn remove(&mut self, key: &Self::Key) {
        if let Some((_, stats)) = self.inner.entries.remove(key) {
            self.inner.total_bytes = self.inner.total_bytes.saturating_sub(stats.bytes);
        }
    }
}

impl LfuEntriesMut for PerNuclideCacheAdapter<'_> {
    fn set_preload_weight(&mut self, key: &Self::Key, weight: u64) {
        if let Some((_, stats)) = self.inner.entries.get_mut(key) {
            stats.preload_weight = weight;
        }
    }
}

/// SVD data + physics tables uploaded to GPU for all nuclides.
pub struct GpuNuclideData {
    // ── Per-nuclide ownership pin (Stage C step D groundwork) ──
    /// Keeps per-nuclide CudaSlices alive for the bundle's lifetime.
    /// The pointer-array fields below (`basis_ptrs` etc.) point into
    /// these per-nuclide allocations; dropping them would invalidate
    /// the device pointers the kernel will eventually load through.
    pub per_nucs: Vec<Arc<crate::gpu_per_nuclide::PerNuclideGpu>>,

    // ── Pointer-array fields (Step D scaffold) ──
    /// `[n_nuc × N_RXN_SLOTS]` flat — each slot is the `CUdeviceptr`
    /// (as `u64`) of the corresponding per-nuclide
    /// `PerNuclideGpu::basis[slot]` CudaSlice, or `0` when
    /// `has_reaction[slot] == 0`. The kernel will eventually read
    /// these via `(double*)basis_ptrs[i]` instead of dereferencing
    /// `all_basis[basis_offsets[i] + ...]` (Stage C step D); this
    /// commit only populates the slice so the wiring is provable
    /// before the kernel ABI is touched.
    pub basis_ptrs: CudaSlice<u64>,
    /// `[n_nuc × N_RXN_SLOTS]` — paired with `basis_ptrs`. Same
    /// semantics; `0` when `has_reaction[slot] == 0`.
    pub coeffs_ptrs: CudaSlice<u64>,
    /// `[n_nuc]` — `CUdeviceptr` of each nuclide's
    /// `PerNuclideGpu::pointwise_xs` (sized `[n_e × 7]`), or `0`
    /// when the nuclide carries no pointwise table. The kernel
    /// gates on `has_pw[ni]` before dereferencing.
    pub pw_xs_ptrs: CudaSlice<u64>,
    /// `[n_nuc]` — `CUdeviceptr` of `PerNuclideGpu::total_xs`, or
    /// `0`. Kernel gates on `has_total_xs[ni]`.
    pub total_xs_ptrs: CudaSlice<u64>,
    /// `[n_nuc]` — `CUdeviceptr` of each nuclide's
    /// `PerNuclideGpu::nu_bar.energies` / `.values`, or `0` when
    /// no ν̄ table. Paired arrays; kernel gates on
    /// `nu_bar_sizes[ni] > 0`.
    pub nb_e_ptrs: CudaSlice<u64>,
    pub nb_v_ptrs: CudaSlice<u64>,
    /// `[n_nuc]` — same shape as `nb_*_ptrs` but for the delayed-
    /// only ν̄ table. Kernel gates on `delayed_nu_bar_sizes[ni] > 0`.
    pub dnb_e_ptrs: CudaSlice<u64>,
    pub dnb_v_ptrs: CudaSlice<u64>,
    /// `[n_nuc]` — `CUdeviceptr` of each nuclide's URR sub-tables.
    /// Six paired arrays: energies + cum_prob + total/elastic/
    /// fission/capture factors. Kernel gates on
    /// `urr_n_energies[ni] > 0`. Absent → `0`.
    pub urr_e_ptrs: CudaSlice<u64>,
    pub urr_cp_ptrs: CudaSlice<u64>,
    pub urr_tf_ptrs: CudaSlice<u64>,
    pub urr_ef_ptrs: CudaSlice<u64>,
    pub urr_ff_ptrs: CudaSlice<u64>,
    pub urr_cf_ptrs: CudaSlice<u64>,
    /// `[n_nuc]` — `CUdeviceptr` of each nuclide's synthesized
    /// inelastic CDF tensor, or `0`. Kernel gates on
    /// `inel_cdf_off[ni] >= 0`.
    pub inel_cdf_ptrs: CudaSlice<u64>,
    /// `[n_nuc]` — `CUdeviceptr` of each nuclide's
    /// `PerNuclideGpu::levels.basis` / `.coeffs`. Per-level basis
    /// rank-padding from `1654c4d` lives inside each per-nuclide
    /// slice (every level pre-padded to `[n_e × global_rank]`).
    pub level_basis_ptrs: CudaSlice<u64>,
    pub level_coeffs_ptrs: CudaSlice<u64>,
    /// `[total_levels]` — within-nuc byte offsets, concat of every
    /// nuclide's `levels.basis_local_off` / `.coeffs_local_off`.
    /// Kernel indexes by global level idx `gl` and uses
    /// `level_basis_ptrs[hit_nuc]` for the base.
    pub level_basis_local_off: CudaSlice<i32>,
    pub level_coeffs_local_off: CudaSlice<i32>,
    /// Step D — per-nuc base pointers for elastic + per-level angular.
    /// `[n_nuc]` each.
    pub ang_e_ptrs: CudaSlice<u64>,
    pub ang_mu_ptrs: CudaSlice<u64>,
    pub ang_cdf_ptrs: CudaSlice<u64>,
    pub lev_ang_e_ptrs: CudaSlice<u64>,
    pub lev_ang_mu_ptrs: CudaSlice<u64>,
    pub lev_ang_cdf_ptrs: CudaSlice<u64>,
    /// `[total_e]` — un-shifted within-nuc ang_mu offsets for the
    /// elastic angular path. Pairs with `ang_mu_ptrs[hit_nuc]`.
    pub ang_dist_local_off: CudaSlice<i32>,
    /// `[total_levels]` — un-shifted within-nuc ang_energy offsets
    /// for the per-level angular path. Pairs with
    /// `lev_ang_e_ptrs[hit_nuc]`.
    pub lev_ang_lev_local_off: CudaSlice<i32>,
    /// `[total_ang_dist]` — un-shifted within-nuc ang_mu offsets
    /// for the per-level angular path. Indexed by global ang_energy
    /// idx (same as `lev_ang_dist_off`); pairs with
    /// `lev_ang_mu_ptrs[hit_nuc]`.
    pub lev_ang_dist_local_off: CudaSlice<i32>,

    // SVD basis data
    pub all_basis: CudaSlice<f64>,
    pub all_coeffs: CudaSlice<f64>,
    pub all_energy_grids: CudaSlice<f64>,
    pub basis_offsets: CudaSlice<i32>,
    pub grid_offsets: CudaSlice<i32>,
    pub n_energies: CudaSlice<i32>,
    pub has_reaction: CudaSlice<i32>,
    pub coeffs_offsets: CudaSlice<i32>,
    pub rank: i32,
    pub total_xs: CudaSlice<f64>,
    pub total_xs_offsets: CudaSlice<i32>,
    pub has_total_xs: CudaSlice<i32>,
    pub pointwise_xs: CudaSlice<f64>,
    pub pw_offsets: CudaSlice<i32>,
    pub has_pw: CudaSlice<i32>,
    // Energy-dependent nu-bar tables (ν_total = ν_prompt + Σ ν_delayed).
    pub nu_bar_energies: CudaSlice<f64>,
    pub nu_bar_values: CudaSlice<f64>,
    pub nu_bar_offsets: CudaSlice<i32>,
    pub nu_bar_sizes: CudaSlice<i32>,
    // Delayed-only ν̄(E) per nuclide. Empty entries (`delayed_nu_bar_sizes[i]==0`)
    // mean the nuclide carries no delayed-neutron data and the device-side
    // fission emitter falls through to the prompt χ path. When non-empty,
    // β(E) = ν_d(E) / ν_t(E) and the emitter draws each banked neutron from
    // the soft Watt delayed spectrum (sample_delayed_energy) with probability
    // β, matching `physics/collision.rs::sample_delayed_energy`.
    pub delayed_nu_bar_energies: CudaSlice<f64>,
    pub delayed_nu_bar_values: CudaSlice<f64>,
    pub delayed_nu_bar_offsets: CudaSlice<i32>,
    pub delayed_nu_bar_sizes: CudaSlice<i32>,
    // Discrete inelastic levels (Q-values + SVD basis for XS-proportional sampling)
    pub level_q_values: CudaSlice<f64>, // flat: all Q-values concatenated
    pub level_thresholds: CudaSlice<f64>, // flat: all thresholds concatenated
    pub level_offsets: CudaSlice<i32>,  // per-nuclide offset into level arrays
    pub level_counts: CudaSlice<i32>,   // per-nuclide number of levels
    pub level_basis: CudaSlice<f64>,    // flat: SVD basis for each level's XS
    pub level_coeffs: CudaSlice<f64>,   // flat: SVD coefficients for each level
    pub level_basis_offsets: CudaSlice<i32>, // per-level offset into level_basis
    pub level_coeffs_offsets: CudaSlice<i32>, // per-level offset into level_coeffs
    pub level_has_kernel: CudaSlice<i32>, // per-level: 1 if kernel exists, 0 if not
    pub level_mt: CudaSlice<i32>,       // per-level: MT number (51-91)
    // Per-discrete-level CM-frame angular distributions (ENDF MT=51-91).
    // Indexed by global level index (same space as level_q_values).
    pub lev_ang_energies: CudaSlice<f64>, // flat: incident-energy grid per level
    pub lev_ang_mu: CudaSlice<f64>,       // flat: cosine values
    pub lev_ang_cdf: CudaSlice<f64>,      // flat: CDF values
    pub lev_ang_dist_off: CudaSlice<i32>, // per (global_level, energy_idx) → offset
    pub lev_ang_dist_sz: CudaSlice<i32>,  // per (global_level, energy_idx) → size
    pub lev_ang_lev_off: CudaSlice<i32>,  // per global level → offset into lev_ang_energies
    pub lev_ang_lev_ne: CudaSlice<i32>,   // per global level → number of incident energies
    // Anisotropic elastic scattering angular distributions
    pub ang_energies: CudaSlice<f64>, // flat: energy grids for angular dist
    pub ang_mu: CudaSlice<f64>,       // flat: cosine values
    pub ang_cdf: CudaSlice<f64>,      // flat: CDF values
    pub ang_dist_offsets: CudaSlice<i32>, // per (nuc, energy) → offset into mu/cdf
    pub ang_dist_sizes: CudaSlice<i32>, // per (nuc, energy) → n_mu
    pub ang_nuc_offsets: CudaSlice<i32>, // per-nuclide → offset into ang_energies
    pub ang_nuc_n_energies: CudaSlice<i32>, // per-nuclide → number of angular energies
    pub ang_is_cm: CudaSlice<i32>,    // per-nuclide → 1 if CM frame
    // Fission energy distributions (tabulated CDF — ENDF Law 4/61).
    pub fis_inc_energies: CudaSlice<f64>,
    pub fis_dist_offsets: CudaSlice<i32>,
    pub fis_dist_sizes: CudaSlice<i32>,
    pub fis_e_out: CudaSlice<f64>,
    pub fis_cdf: CudaSlice<f64>,
    /// PDF samples aligned 1:1 with `fis_e_out` / `fis_cdf`. Enables
    /// the quadratic lin-lin CDF inversion in `sample_eout_bin`
    /// (OpenMC `Tabular::sample`) on the GPU. Pre-fix the GPU used a
    /// linear-CDF / histogram-PDF fallback, which biased the χ
    /// outgoing spectrum hard → less leakage → +500-700 pcm hot on
    /// fast-metal benchmarks (Godiva, PMF). When non-empty, falls
    /// back to linear interpolation only when the PDF slope is
    /// degenerate.
    pub fis_pdf: CudaSlice<f64>,
    pub fis_nuc_offsets: CudaSlice<i32>,
    pub fis_nuc_n_inc: CudaSlice<i32>,
    // MT=91 continuum-inelastic outgoing energy distributions
    // (ENDF Law 4 tabular). Layout mirrors the fission distribution
    // buffers above. Wired to close a +400 keV ⟨E_out⟩ gap vs CPU
    // (the GPU used to fall back unconditionally to a Weisskopf
    // evaporation approximation, the source of the +500-700 pcm
    // fast-metal hot bias on Godiva / Jezebel). When
    // `inel91_nuc_n_inc[i] == 0` the GPU kernel falls back to the
    // evaporation formula — matches the CPU behaviour on nuclides
    // whose evaluation does not ship a tabulated MT=91 distribution.
    pub inel91_inc_energies: CudaSlice<f64>,
    pub inel91_dist_offsets: CudaSlice<i32>,
    pub inel91_dist_sizes: CudaSlice<i32>,
    pub inel91_e_out: CudaSlice<f64>,
    pub inel91_cdf: CudaSlice<f64>,
    pub inel91_pdf: CudaSlice<f64>,
    pub inel91_nuc_offsets: CudaSlice<i32>,
    pub inel91_nuc_n_inc: CudaSlice<i32>,
    // Closed-form fission χ parameters per nuclide (ENDF Law 11 Watt).
    // When `fis_nuc_n_inc[i] == 0` and `watt_nuc_n[i] > 0`, the
    // device-side `sample_fission_energy` interpolates
    // a(E_in) and b(E_in) from `watt_inc_e[off..off+n]` →
    // `watt_a[off..off+n]` / `watt_b[off..off+n]` and samples via
    // the math-correct Watt sampler in transport.cu. Replaces the
    // hardcoded U-235 Cranberg fallback for every nuclide whose
    // evaluation actually carries Watt parameters (U-233, U-234
    // multi-chance fission products, etc.). `watt_u` is the eV
    // cutoff applied as `E_out ≤ E_in − u`.
    pub watt_inc_energies: CudaSlice<f64>,
    pub watt_a: CudaSlice<f64>,
    pub watt_b: CudaSlice<f64>,
    pub watt_u: CudaSlice<f64>,
    pub watt_nuc_offsets: CudaSlice<i32>,
    pub watt_nuc_n: CudaSlice<i32>,
    // Maxwell (Law 7) / Evaporation (Law 9) closed-form fission χ
    // per nuclide. Single shared θ(E_in) table — both laws use the
    // same parameter table; `maxevap_law[i]` selects the sampler at
    // collision time (7 = Maxwell, 9 = Evaporation, 0 = none). When
    // `maxevap_nuc_n[i] == 0` the kernel falls through to Watt (104)
    // and then to the Cranberg fallback (the existing dispatch
    // chain in transport.cu::sample_fission_energy). Closes the
    // wrong-spectrum GPU bias for U-233 (Maxwell), U-234 (Maxwell),
    // and Pu-240/Pu-241 (Evaporation in several evaluations).
    pub maxevap_inc_energies: CudaSlice<f64>,
    pub maxevap_theta: CudaSlice<f64>,
    pub maxevap_u: CudaSlice<f64>,
    pub maxevap_law: CudaSlice<i32>,
    pub maxevap_nuc_offsets: CudaSlice<i32>,
    pub maxevap_nuc_n: CudaSlice<i32>,
    // URR probability tables
    pub urr_energies: CudaSlice<f64>,
    pub urr_cum_prob: CudaSlice<f64>,
    pub urr_total_f: CudaSlice<f64>,
    pub urr_elastic_f: CudaSlice<f64>,
    pub urr_fission_f: CudaSlice<f64>,
    pub urr_capture_f: CudaSlice<f64>,
    pub urr_offsets: CudaSlice<i32>,
    pub urr_n_energies: CudaSlice<i32>,
    pub urr_n_bands: CudaSlice<i32>,
    pub urr_multiply_smooth: CudaSlice<i32>,
    // ── Synthesized MT=4 + per-level CDF for nuclides whose ENDF/B-VII.1
    //    evaluation omits the total-inelastic block (Zr-90/91/92/94, U-238).
    //    Replaces the do_inelastic 13-level walk with a single binary
    //    search in a log-decimated CDF (~200 energy points).
    /// Flat CDF tensor: cdf[e_dec * n_t * n_lev + t * n_lev + l]
    /// concatenated across all nuclides; per-nuclide slice located via
    /// `inel_cdf_offsets`.
    pub inel_cdf_data: CudaSlice<f64>,
    /// Per-nuclide offset into `inel_cdf_data`. -1 means "no CDF, use
    /// the legacy per-level walk in do_inelastic".
    pub inel_cdf_off: CudaSlice<i32>,
    /// Per-nuclide number of decimated energy points.
    pub inel_cdf_n_e: CudaSlice<i32>,
    /// Per-nuclide number of temperature columns.
    pub inel_cdf_n_t: CudaSlice<i32>,
    /// Per-nuclide number of levels in the CDF (parallel to
    /// `level_counts` when both are non-zero).
    pub inel_cdf_n_lev: CudaSlice<i32>,
    /// Per-nuclide log10(E_min) of the decimated grid.
    pub inel_cdf_log_e_min: CudaSlice<f64>,
    /// Per-nuclide log10(E_max) of the decimated grid.
    pub inel_cdf_log_e_max: CudaSlice<f64>,
}

impl GpuNuclideData {
    /// Total on-device byte footprint of every `CudaSlice` this
    /// bundle owns. Used by the byte-budgeted bundle LRU to decide
    /// when to evict before the next upload.
    ///
    /// Sums every field; dominated by `level_basis` (~1.2 GB on a
    /// fast-metal case) + `all_basis` (~180 MB) + `pointwise_xs`
    /// (~40 MB). Cheap (one virtual call per field, no device
    /// traffic).
    pub fn device_bytes(&self) -> usize {
        let s = self;
        // f64 slices.
        let f64_slices: [&CudaSlice<f64>; 41] = [
            &s.all_basis,
            &s.all_coeffs,
            &s.all_energy_grids,
            &s.total_xs,
            &s.pointwise_xs,
            &s.nu_bar_energies,
            &s.nu_bar_values,
            &s.delayed_nu_bar_energies,
            &s.delayed_nu_bar_values,
            &s.level_q_values,
            &s.level_thresholds,
            &s.level_basis,
            &s.level_coeffs,
            &s.lev_ang_energies,
            &s.lev_ang_mu,
            &s.lev_ang_cdf,
            &s.ang_energies,
            &s.ang_mu,
            &s.ang_cdf,
            &s.fis_inc_energies,
            &s.fis_e_out,
            &s.fis_cdf,
            &s.fis_pdf,
            &s.inel91_inc_energies,
            &s.inel91_e_out,
            &s.inel91_cdf,
            &s.inel91_pdf,
            &s.watt_inc_energies,
            &s.watt_a,
            &s.watt_b,
            &s.watt_u,
            &s.maxevap_inc_energies,
            &s.maxevap_theta,
            &s.maxevap_u,
            &s.urr_energies,
            &s.urr_cum_prob,
            &s.urr_total_f,
            &s.urr_elastic_f,
            &s.urr_fission_f,
            &s.urr_capture_f,
            &s.inel_cdf_data,
        ];
        // i32 slices.
        let i32_slices: [&CudaSlice<i32>; 39] = [
            &s.basis_offsets,
            &s.grid_offsets,
            &s.n_energies,
            &s.has_reaction,
            &s.coeffs_offsets,
            &s.total_xs_offsets,
            &s.has_total_xs,
            &s.pw_offsets,
            &s.has_pw,
            &s.nu_bar_offsets,
            &s.nu_bar_sizes,
            &s.delayed_nu_bar_offsets,
            &s.delayed_nu_bar_sizes,
            &s.level_offsets,
            &s.level_counts,
            &s.level_basis_offsets,
            &s.level_coeffs_offsets,
            &s.level_has_kernel,
            &s.level_mt,
            &s.lev_ang_dist_off,
            &s.lev_ang_dist_sz,
            &s.lev_ang_lev_off,
            &s.lev_ang_lev_ne,
            &s.ang_dist_offsets,
            &s.ang_dist_sizes,
            &s.ang_nuc_offsets,
            &s.ang_nuc_n_energies,
            &s.ang_is_cm,
            &s.fis_dist_offsets,
            &s.fis_dist_sizes,
            &s.fis_nuc_offsets,
            &s.fis_nuc_n_inc,
            &s.inel91_dist_offsets,
            &s.inel91_dist_sizes,
            &s.inel91_nuc_offsets,
            &s.inel91_nuc_n_inc,
            &s.watt_nuc_offsets,
            &s.watt_nuc_n,
            &s.maxevap_law,
        ];
        // f64 fields we missed in the array above. Two more i32
        // groups appear after `inel_cdf_data` on the inel-cdf path.
        let i32_extra: [&CudaSlice<i32>; 9] = [
            &s.maxevap_nuc_offsets,
            &s.maxevap_nuc_n,
            &s.urr_offsets,
            &s.urr_n_energies,
            &s.urr_n_bands,
            &s.urr_multiply_smooth,
            &s.inel_cdf_off,
            &s.inel_cdf_n_e,
            &s.inel_cdf_n_t,
        ];
        let f64_extra: [&CudaSlice<f64>; 2] =
            [&s.inel_cdf_log_e_min, &s.inel_cdf_log_e_max];
        let i32_extra2: [&CudaSlice<i32>; 1] = [&s.inel_cdf_n_lev];

        let f64_total: usize = f64_slices
            .iter()
            .chain(f64_extra.iter())
            .map(|x| x.num_bytes())
            .sum();
        let i32_total: usize = i32_slices
            .iter()
            .chain(i32_extra.iter())
            .chain(i32_extra2.iter())
            .map(|x| x.num_bytes())
            .sum();
        // Step-D pointer-array buffers (u64 each). The per-nuclide
        // Arcs themselves are not counted — they live in the per-
        // nuclide cache and are accounted for there.
        let ptr_total = self.basis_ptrs.num_bytes()
            + self.coeffs_ptrs.num_bytes()
            + self.pw_xs_ptrs.num_bytes()
            + self.total_xs_ptrs.num_bytes()
            + self.nb_e_ptrs.num_bytes()
            + self.nb_v_ptrs.num_bytes()
            + self.dnb_e_ptrs.num_bytes()
            + self.dnb_v_ptrs.num_bytes()
            + self.urr_e_ptrs.num_bytes()
            + self.urr_cp_ptrs.num_bytes()
            + self.urr_tf_ptrs.num_bytes()
            + self.urr_ef_ptrs.num_bytes()
            + self.urr_ff_ptrs.num_bytes()
            + self.urr_cf_ptrs.num_bytes()
            + self.inel_cdf_ptrs.num_bytes()
            + self.level_basis_ptrs.num_bytes()
            + self.level_coeffs_ptrs.num_bytes()
            + self.level_basis_local_off.num_bytes()
            + self.level_coeffs_local_off.num_bytes()
            + self.ang_e_ptrs.num_bytes()
            + self.ang_mu_ptrs.num_bytes()
            + self.ang_cdf_ptrs.num_bytes()
            + self.lev_ang_e_ptrs.num_bytes()
            + self.lev_ang_mu_ptrs.num_bytes()
            + self.lev_ang_cdf_ptrs.num_bytes()
            + self.ang_dist_local_off.num_bytes()
            + self.lev_ang_lev_local_off.num_bytes()
            + self.lev_ang_dist_local_off.num_bytes();
        f64_total + i32_total + ptr_total
    }
}

/// S(α,β) thermal scattering data on GPU.
///
/// Multiple TSLs (e.g. H-in-H₂O + D-in-D₂O + C-in-graphite) are packed
/// into the same flat arrays; `slot_per_nuc[nuc_idx]` is `-1` (no SAB)
/// or `slot_idx` (≥ 0) and indexes into the per-slot offset/size arrays
/// that locate this slot's run inside each flat array. The legacy
/// scalars `n_inc` / `energy_max` mirror slot 0 for backward
/// compatibility with kernels that haven't been ported off the
/// single-slot fast path.
pub struct GpuSabData {
    // Flat data arrays (concatenated across all slots).
    pub inc_energies: CudaSlice<f64>,
    pub eout_offsets: CudaSlice<i32>,
    pub eout_sizes: CudaSlice<i32>,
    pub e_out: CudaSlice<f64>,
    pub cdf_e: CudaSlice<f64>,
    pub pdf_e: CudaSlice<f64>,
    pub mu_offsets: CudaSlice<i32>,
    pub mu_sizes: CudaSlice<i32>,
    pub mu: CudaSlice<f64>,
    pub cdf_mu: CudaSlice<f64>,
    pub xs: CudaSlice<f64>,

    // Per-slot indirection.
    /// Number of populated slots. `0` means no SAB.
    pub n_slots: i32,
    /// `[n_nuc]`: nuclide → slot index, or `-1`. Always allocated even
    /// when `n_slots == 0` (filled with `-1`) so the kernel can
    /// indirect unconditionally.
    pub slot_per_nuc: CudaSlice<i32>,
    /// `[n_slots]`: offset into `inc_energies` / `xs` where this slot's
    /// inc-energy grid starts.
    pub slot_inc_e_off: CudaSlice<i32>,
    /// `[n_slots]`: number of inc-energy points in this slot.
    pub slot_n_inc: CudaSlice<i32>,
    /// `[n_slots]`: offset into `eout_offsets` / `eout_sizes` where
    /// this slot's per-inc-energy table starts.
    pub slot_eout_table_off: CudaSlice<i32>,
    /// `[n_slots]`: offset into `mu_offsets` / `mu_sizes` where this
    /// slot's per-eout-bin table starts.
    pub slot_mu_table_off: CudaSlice<i32>,
    /// `[n_slots]`: per-slot `energy_max` (eV).
    pub slot_emax: CudaSlice<f64>,

    // Legacy single-slot mirrors (slot 0). Kept so the original
    // single-slot fast path in transport.cu (`SCALAR_I(p, P_SAB_N_INC)`
    // / `SCALAR_D(p, P_SAB_EMAX)`) continues to work as a fallback
    // until every call site is on the slot-aware path.
    pub n_inc: i32,
    pub energy_max: f64,
}

/// Material composition data on GPU.
pub struct GpuMaterialData {
    pub mat_n_nuclides: CudaSlice<i32>,
    pub mat_nuclide_idx: CudaSlice<i32>,
    pub mat_atom_density: CudaSlice<f64>,
    pub awr_table: CudaSlice<f64>,
    pub nu_bar_const: CudaSlice<f64>,
}

/// Windowed-Multipole data on GPU, keyed by nuclide index. Empty (all
/// `has[i] = 0`) for the SVD-only path; populated for `--mode hybrid`.
/// The kernel reads poles as `double2` (16-byte aligned) via pointer cast
/// over the raw f64 storage; `pole_offsets[i]` is in complex units.
pub struct GpuWmpData {
    pub has: CudaSlice<i32>,              // [n_nuc]
    pub e_min: CudaSlice<f64>,            // [n_nuc]
    pub e_max: CudaSlice<f64>,            // [n_nuc]
    pub spacing: CudaSlice<f64>,          // [n_nuc]
    pub sqrt_awr: CudaSlice<f64>,         // [n_nuc]
    pub t_kelvin: CudaSlice<f64>,         // [n_nuc]
    pub fit_order: CudaSlice<i32>,        // [n_nuc]
    pub n_windows: CudaSlice<i32>,        // [n_nuc]
    pub fissionable: CudaSlice<i32>,      // [n_nuc]
    pub poles: CudaSlice<f64>,            // flat f64 (re/im pairs), read as double2
    pub pole_offsets: CudaSlice<i32>,     // [n_nuc], offsets in complex units
    pub windows: CudaSlice<i32>,          // flat (n_windows * 2) per nuclide
    pub window_offsets: CudaSlice<i32>,   // [n_nuc]
    pub broaden: CudaSlice<i8>,           // flat n_windows per nuclide
    pub broaden_offsets: CudaSlice<i32>,  // [n_nuc]
    pub curvefit: CudaSlice<f64>,         // flat n_windows*(fit_order+1)*3 per nuclide
    pub curvefit_offsets: CudaSlice<i32>, // [n_nuc]
}

/// Result of debug trace on GPU.
pub struct GpuTraceResult {
    pub data: Vec<f64>,        // [n_particles * max_steps * TRACE_COLS]
    pub step_counts: Vec<i32>, // [n_particles]
}

/// Result of one batch on GPU.
pub struct GpuBatchResult {
    pub k_eff: f64,
    pub collisions: u32,
    pub fissions: u32,
    pub leakage: u32,
    pub surface_crossings: u32,
    /// Fission sites for next generation.
    pub fission_bank: Vec<(f64, f64, f64, f64)>, // (x, y, z, energy)
}

const BLOCK_SIZE: u32 = 256;

impl GpuTransportContext {
    /// Compile all CUDA kernels and initialize GPU context.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        // Tune nvrtc/ptxas for our target hardware (Ampere: RTX 3080,
        // RTX A1000 laptop — both sm_86). Verbose ptxas output
        // (`--ptxas-options=-v -warn-spills`) surfaces register usage
        // and spills during JIT for occupancy tuning per NVIDIA BPG §10.2.
        // If cudarc returns a compile error the log is attached; on
        // success the driver may still print to stderr if
        // `CUDA_CACHE_DISABLE=1` + `CUDA_CACHE_LOG=1` are set.
        let opts = nvrtc::CompileOptions {
            arch: Some("sm_86"),
            options: vec![
                // Single source of truth for the per-material nuclide
                // cap — matches `simulate.rs::MAX_NUCLIDES` and the
                // Rust-side upload arrays.
                format!("-DMAX_NUC_PER_MAT={}", crate::MAX_NUCLIDES_PER_MATERIAL),
                "--ptxas-options=-v".to_string(),
                "-Xptxas".to_string(),
                "-warn-spills".to_string(),
            ],
            ..Default::default()
        };
        let ptx = nvrtc::compile_ptx_with_opts(TRANSPORT_KERNELS, opts)?;
        let module = ctx.load_module(ptx)?;

        let k_init_source = module.load_function("init_source")?;
        let k_count_alive = module.load_function("count_alive")?;
        let k_compact_alive = module.load_function("compact_alive")?;
        let k_energy_bin_count = module.load_function("energy_bin_count")?;
        let k_energy_bin_scatter = module.load_function("energy_bin_scatter")?;
        let k_transport_persistent = module.load_function("transport_persistent")?;
        let stream = ctx.default_stream();

        println!("  GPU transport kernels compiled (8 kernels)");

        Ok(Self {
            _ctx: ctx,
            stream,
            k_init_source,
            k_count_alive,
            k_compact_alive,
            k_energy_bin_count,
            k_energy_bin_scatter,
            k_transport_persistent,
            per_nuclide_cache: std::sync::Mutex::new(PerNuclideCache::new()),
            cached_bundle_budget: std::sync::OnceLock::new(),
        })
    }

    /// Process-wide shared context. First caller pays the NVRTC
    /// compile + CUDA init cost (~30-150 ms on RTX A1000); every
    /// subsequent caller — across cases in an ICSBEP sweep, across
    /// PyO3 entry points, across rayon worker threads — gets the
    /// same `Arc<GpuTransportContext>` and therefore shares the
    /// per-context `nuclide_buffer_cache`. This is what makes the
    /// `Arc::as_ptr()`-keyed cache actually fire across cases: a
    /// fresh context per case (the prior pattern) had an empty cache
    /// every time.
    ///
    /// Returns `Err` only on first-call failure (no CUDA device, no
    /// driver, NVRTC compile error). Failures are *not* cached —
    /// retry-on-error works because the error path doesn't write to
    /// the OnceLock; only a successful init seals the slot.
    pub fn shared() -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        static SHARED: std::sync::OnceLock<Arc<GpuTransportContext>> =
            std::sync::OnceLock::new();
        if let Some(arc) = SHARED.get() {
            return Ok(Arc::clone(arc));
        }
        let ctx = Arc::new(Self::new()?);
        // Either we win the race and store our Arc, or someone else
        // beat us — either way the slot now holds a valid Arc.
        let _ = SHARED.set(Arc::clone(&ctx));
        Ok(Arc::clone(SHARED.get().expect("OnceLock just set above")))
    }

    /// Debug: sample angular distributions at given (energy, xi) pairs.
    /// Returns (stairstep_mu, interpolated_mu) for comparison with CPU.
    pub fn debug_angular_sample(
        &self,
        energies: &[f64],
        xis: &[f64],
        nuc_idx: i32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        geom_type: i32,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
        let n = energies.len();
        assert_eq!(n, xis.len());

        let d_energies = self.stream.clone_htod(energies)?;
        let d_xis = self.stream.clone_htod(xis)?;
        let mut d_out_ss: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_out_interp: CudaSlice<f64> = self.stream.alloc_zeros(n)?;

        // Build params buffer (same as run_batch)
        macro_rules! dptr {
            ($slice:expr) => {{
                let (ptr, _guard) = $slice.device_ptr(&self.stream);
                ptr
            }};
        }
        let mut params_vec: Vec<u64> = vec![
            dptr!(&nuc_data.all_basis),
            dptr!(&nuc_data.all_coeffs),
            dptr!(&nuc_data.all_energy_grids),
            dptr!(&nuc_data.basis_offsets),
            dptr!(&nuc_data.grid_offsets),
            dptr!(&nuc_data.n_energies),
            dptr!(&nuc_data.has_reaction),
            dptr!(&nuc_data.coeffs_offsets),
            nuc_data.rank as u64,
            dptr!(&mat_data.mat_n_nuclides),
            dptr!(&mat_data.mat_nuclide_idx),
            dptr!(&mat_data.mat_atom_density),
            dptr!(&mat_data.awr_table),
            dptr!(&mat_data.nu_bar_const),
            dptr!(&nuc_data.nu_bar_energies),
            dptr!(&nuc_data.nu_bar_values),
            dptr!(&nuc_data.nu_bar_offsets),
            dptr!(&nuc_data.nu_bar_sizes),
            dptr!(&nuc_data.fis_inc_energies),
            dptr!(&nuc_data.fis_dist_offsets),
            dptr!(&nuc_data.fis_dist_sizes),
            dptr!(&nuc_data.fis_e_out),
            dptr!(&nuc_data.fis_cdf),
            dptr!(&nuc_data.fis_nuc_offsets),
            dptr!(&nuc_data.fis_nuc_n_inc),
            dptr!(&nuc_data.level_q_values),
            dptr!(&nuc_data.level_thresholds),
            dptr!(&nuc_data.level_offsets),
            dptr!(&nuc_data.level_counts),
            dptr!(&nuc_data.level_basis),
            dptr!(&nuc_data.level_coeffs),
            dptr!(&nuc_data.level_basis_offsets),
            dptr!(&nuc_data.level_coeffs_offsets),
            dptr!(&nuc_data.level_has_kernel),
            dptr!(&nuc_data.level_mt),
            dptr!(&nuc_data.ang_energies),
            dptr!(&nuc_data.ang_mu),
            dptr!(&nuc_data.ang_cdf),
            dptr!(&nuc_data.ang_dist_offsets),
            dptr!(&nuc_data.ang_dist_sizes),
            dptr!(&nuc_data.ang_nuc_offsets),
            dptr!(&nuc_data.ang_nuc_n_energies),
            dptr!(&nuc_data.ang_is_cm),
            dptr!(&sab_data.inc_energies),
            sab_data.n_inc as u64,
            dptr!(&sab_data.eout_offsets),
            dptr!(&sab_data.eout_sizes),
            dptr!(&sab_data.e_out),
            dptr!(&sab_data.cdf_e),
            dptr!(&sab_data.mu_offsets),
            dptr!(&sab_data.mu_sizes),
            dptr!(&sab_data.mu),
            dptr!(&sab_data.cdf_mu),
            dptr!(&sab_data.xs),
            sab_data.energy_max.to_bits(),
            dptr!(&sab_data.pdf_e),
            dptr!(&nuc_data.urr_energies),
            dptr!(&nuc_data.urr_cum_prob),
            dptr!(&nuc_data.urr_total_f),
            dptr!(&nuc_data.urr_elastic_f),
            dptr!(&nuc_data.urr_fission_f),
            dptr!(&nuc_data.urr_capture_f),
            dptr!(&nuc_data.urr_offsets),
            dptr!(&nuc_data.urr_n_energies),
            dptr!(&nuc_data.urr_n_bands),
            dptr!(&nuc_data.urr_multiply_smooth),
            geom_type as u64,
            dptr!(&nuc_data.total_xs),
            dptr!(&nuc_data.total_xs_offsets),
            dptr!(&nuc_data.has_total_xs),
            dptr!(&nuc_data.pointwise_xs),
            dptr!(&nuc_data.pw_offsets),
            dptr!(&nuc_data.has_pw),
        ];
        // Debug helper only reads the elastic-angular slots; pad the
        // remaining slots (WMP + per-level angular) with nulls so the
        // runtime TransportParams layout check passes. These arrays are
        // not referenced from the `debug_angular_sample` kernel.
        while params_vec.len() < N_PARAMS {
            params_vec.push(0_u64);
        }
        assert_eq!(params_vec.len(), N_PARAMS);
        let d_params = self.stream.clone_htod(&params_vec)?;

        // Load debug kernel
        let ptx = nvrtc::compile_ptx_with_opts(TRANSPORT_KERNELS, transport_kernel_options())?;
        let module = self._ctx.load_module(ptx)?;
        let k_debug = module.load_function("debug_angular_sample")?;

        let n_i32 = n as i32;
        let grid = ((n as u32 + 255) / 256, 1, 1);
        let block = (256u32, 1, 1);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&k_debug)
                .arg(&d_params)
                .arg(&d_energies)
                .arg(&d_xis)
                .arg(&n_i32)
                .arg(&nuc_idx)
                .arg(&mut d_out_ss)
                .arg(&mut d_out_interp)
                .launch(cfg)?;
        }

        let ss = self.stream.clone_dtoh(&d_out_ss)?;
        let interp = self.stream.clone_dtoh(&d_out_interp)?;
        Ok((ss, interp))
    }

    /// Debug: reconstruct XS at given energies for a nuclide on GPU.
    /// Returns [n * 6] flat array: elastic, inelastic, n2n, n3n, fission, capture per energy.
    pub fn debug_xs_reconstruct(
        &self,
        energies: &[f64],
        nuc_idx: i32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        geom_type: i32,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = energies.len();
        let d_energies = self.stream.clone_htod(energies)?;
        let mut d_out: CudaSlice<f64> = self.stream.alloc_zeros(n * 6)?;

        macro_rules! dptr {
            ($slice:expr) => {{
                let (ptr, _guard) = $slice.device_ptr(&self.stream);
                ptr
            }};
        }
        let mut params_vec: Vec<u64> = vec![
            dptr!(&nuc_data.all_basis),
            dptr!(&nuc_data.all_coeffs),
            dptr!(&nuc_data.all_energy_grids),
            dptr!(&nuc_data.basis_offsets),
            dptr!(&nuc_data.grid_offsets),
            dptr!(&nuc_data.n_energies),
            dptr!(&nuc_data.has_reaction),
            dptr!(&nuc_data.coeffs_offsets),
            nuc_data.rank as u64,
            dptr!(&mat_data.mat_n_nuclides),
            dptr!(&mat_data.mat_nuclide_idx),
            dptr!(&mat_data.mat_atom_density),
            dptr!(&mat_data.awr_table),
            dptr!(&mat_data.nu_bar_const),
            dptr!(&nuc_data.nu_bar_energies),
            dptr!(&nuc_data.nu_bar_values),
            dptr!(&nuc_data.nu_bar_offsets),
            dptr!(&nuc_data.nu_bar_sizes),
            dptr!(&nuc_data.fis_inc_energies),
            dptr!(&nuc_data.fis_dist_offsets),
            dptr!(&nuc_data.fis_dist_sizes),
            dptr!(&nuc_data.fis_e_out),
            dptr!(&nuc_data.fis_cdf),
            dptr!(&nuc_data.fis_nuc_offsets),
            dptr!(&nuc_data.fis_nuc_n_inc),
            dptr!(&nuc_data.level_q_values),
            dptr!(&nuc_data.level_thresholds),
            dptr!(&nuc_data.level_offsets),
            dptr!(&nuc_data.level_counts),
            dptr!(&nuc_data.level_basis),
            dptr!(&nuc_data.level_coeffs),
            dptr!(&nuc_data.level_basis_offsets),
            dptr!(&nuc_data.level_coeffs_offsets),
            dptr!(&nuc_data.level_has_kernel),
            dptr!(&nuc_data.level_mt),
            dptr!(&nuc_data.ang_energies),
            dptr!(&nuc_data.ang_mu),
            dptr!(&nuc_data.ang_cdf),
            dptr!(&nuc_data.ang_dist_offsets),
            dptr!(&nuc_data.ang_dist_sizes),
            dptr!(&nuc_data.ang_nuc_offsets),
            dptr!(&nuc_data.ang_nuc_n_energies),
            dptr!(&nuc_data.ang_is_cm),
            dptr!(&sab_data.inc_energies),
            sab_data.n_inc as u64,
            dptr!(&sab_data.eout_offsets),
            dptr!(&sab_data.eout_sizes),
            dptr!(&sab_data.e_out),
            dptr!(&sab_data.cdf_e),
            dptr!(&sab_data.mu_offsets),
            dptr!(&sab_data.mu_sizes),
            dptr!(&sab_data.mu),
            dptr!(&sab_data.cdf_mu),
            dptr!(&sab_data.xs),
            sab_data.energy_max.to_bits(),
            dptr!(&sab_data.pdf_e),
            dptr!(&nuc_data.urr_energies),
            dptr!(&nuc_data.urr_cum_prob),
            dptr!(&nuc_data.urr_total_f),
            dptr!(&nuc_data.urr_elastic_f),
            dptr!(&nuc_data.urr_fission_f),
            dptr!(&nuc_data.urr_capture_f),
            dptr!(&nuc_data.urr_offsets),
            dptr!(&nuc_data.urr_n_energies),
            dptr!(&nuc_data.urr_n_bands),
            dptr!(&nuc_data.urr_multiply_smooth),
            geom_type as u64,
            dptr!(&nuc_data.total_xs),
            dptr!(&nuc_data.total_xs_offsets),
            dptr!(&nuc_data.has_total_xs),
            dptr!(&nuc_data.pointwise_xs),
            dptr!(&nuc_data.pw_offsets),
            dptr!(&nuc_data.has_pw),
        ];
        while params_vec.len() < N_PARAMS {
            params_vec.push(0_u64);
        }
        assert_eq!(params_vec.len(), N_PARAMS);
        let d_params = self.stream.clone_htod(&params_vec)?;

        let ptx = nvrtc::compile_ptx_with_opts(TRANSPORT_KERNELS, transport_kernel_options())?;
        let module = self._ctx.load_module(ptx)?;
        let k_debug = module.load_function("debug_xs_reconstruct")?;

        let n_i32 = n as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (((n as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&k_debug)
                .arg(&d_params)
                .arg(&d_energies)
                .arg(&n_i32)
                .arg(&nuc_idx)
                .arg(&mut d_out)
                .launch(cfg)?;
        }
        Ok(self.stream.clone_dtoh(&d_out)?)
    }

    /// Expose the CUDA stream for diagnostic buffer downloads.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Expose the CUDA context so callers (e.g. `GpuRecursiveContext`)
    /// can share the same primary context and reuse already-uploaded
    /// device buffers without re-allocating.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self._ctx
    }

    /// Build the 104-slot packed `TransportParams` buffer that
    /// `transport_persistent` and `transport_recursive_persistent` both
    /// read. Centralised here so the recursive kernel does not duplicate
    /// the slot layout. `geom_type` is written into the P_GEOM_TYPE slot
    /// — irrelevant for the recursive kernel (which uses its own
    /// geometry tables) but kept so the buffer round-trips through the
    /// existing `transport_persistent` debug entry points.
    pub fn build_transport_params_vec(
        &self,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        wmp_data: &GpuWmpData,
        geom_type: i32,
    ) -> Vec<u64> {
        macro_rules! dptr {
            ($slice:expr) => {{
                let (ptr, _guard) = $slice.device_ptr(&self.stream);
                ptr
            }};
        }
        let v: Vec<u64> = vec![
            dptr!(&nuc_data.all_basis),
            dptr!(&nuc_data.all_coeffs),
            dptr!(&nuc_data.all_energy_grids),
            dptr!(&nuc_data.basis_offsets),
            dptr!(&nuc_data.grid_offsets),
            dptr!(&nuc_data.n_energies),
            dptr!(&nuc_data.has_reaction),
            dptr!(&nuc_data.coeffs_offsets),
            nuc_data.rank as u64,
            dptr!(&mat_data.mat_n_nuclides),
            dptr!(&mat_data.mat_nuclide_idx),
            dptr!(&mat_data.mat_atom_density),
            dptr!(&mat_data.awr_table),
            dptr!(&mat_data.nu_bar_const),
            dptr!(&nuc_data.nu_bar_energies),
            dptr!(&nuc_data.nu_bar_values),
            dptr!(&nuc_data.nu_bar_offsets),
            dptr!(&nuc_data.nu_bar_sizes),
            dptr!(&nuc_data.fis_inc_energies),
            dptr!(&nuc_data.fis_dist_offsets),
            dptr!(&nuc_data.fis_dist_sizes),
            dptr!(&nuc_data.fis_e_out),
            dptr!(&nuc_data.fis_cdf),
            dptr!(&nuc_data.fis_nuc_offsets),
            dptr!(&nuc_data.fis_nuc_n_inc),
            dptr!(&nuc_data.level_q_values),
            dptr!(&nuc_data.level_thresholds),
            dptr!(&nuc_data.level_offsets),
            dptr!(&nuc_data.level_counts),
            dptr!(&nuc_data.level_basis),
            dptr!(&nuc_data.level_coeffs),
            dptr!(&nuc_data.level_basis_offsets),
            dptr!(&nuc_data.level_coeffs_offsets),
            dptr!(&nuc_data.level_has_kernel),
            dptr!(&nuc_data.level_mt),
            dptr!(&nuc_data.ang_energies),
            dptr!(&nuc_data.ang_mu),
            dptr!(&nuc_data.ang_cdf),
            dptr!(&nuc_data.ang_dist_offsets),
            dptr!(&nuc_data.ang_dist_sizes),
            dptr!(&nuc_data.ang_nuc_offsets),
            dptr!(&nuc_data.ang_nuc_n_energies),
            dptr!(&nuc_data.ang_is_cm),
            dptr!(&sab_data.inc_energies),
            sab_data.n_inc as u64,
            dptr!(&sab_data.eout_offsets),
            dptr!(&sab_data.eout_sizes),
            dptr!(&sab_data.e_out),
            dptr!(&sab_data.cdf_e),
            dptr!(&sab_data.mu_offsets),
            dptr!(&sab_data.mu_sizes),
            dptr!(&sab_data.mu),
            dptr!(&sab_data.cdf_mu),
            dptr!(&sab_data.xs),
            sab_data.energy_max.to_bits(),
            dptr!(&sab_data.pdf_e),
            dptr!(&nuc_data.urr_energies),
            dptr!(&nuc_data.urr_cum_prob),
            dptr!(&nuc_data.urr_total_f),
            dptr!(&nuc_data.urr_elastic_f),
            dptr!(&nuc_data.urr_fission_f),
            dptr!(&nuc_data.urr_capture_f),
            dptr!(&nuc_data.urr_offsets),
            dptr!(&nuc_data.urr_n_energies),
            dptr!(&nuc_data.urr_n_bands),
            dptr!(&nuc_data.urr_multiply_smooth),
            geom_type as u64,
            dptr!(&nuc_data.total_xs),
            dptr!(&nuc_data.total_xs_offsets),
            dptr!(&nuc_data.has_total_xs),
            dptr!(&nuc_data.pointwise_xs),
            dptr!(&nuc_data.pw_offsets),
            dptr!(&nuc_data.has_pw),
            dptr!(&wmp_data.has),
            dptr!(&wmp_data.e_min),
            dptr!(&wmp_data.e_max),
            dptr!(&wmp_data.spacing),
            dptr!(&wmp_data.sqrt_awr),
            dptr!(&wmp_data.t_kelvin),
            dptr!(&wmp_data.fit_order),
            dptr!(&wmp_data.n_windows),
            dptr!(&wmp_data.fissionable),
            dptr!(&wmp_data.poles),
            dptr!(&wmp_data.pole_offsets),
            dptr!(&wmp_data.windows),
            dptr!(&wmp_data.window_offsets),
            dptr!(&wmp_data.broaden),
            dptr!(&wmp_data.broaden_offsets),
            dptr!(&wmp_data.curvefit),
            dptr!(&wmp_data.curvefit_offsets),
            dptr!(&nuc_data.lev_ang_energies),
            dptr!(&nuc_data.lev_ang_mu),
            dptr!(&nuc_data.lev_ang_cdf),
            dptr!(&nuc_data.lev_ang_dist_off),
            dptr!(&nuc_data.lev_ang_dist_sz),
            dptr!(&nuc_data.lev_ang_lev_off),
            dptr!(&nuc_data.lev_ang_lev_ne),
            dptr!(&nuc_data.inel_cdf_data),
            dptr!(&nuc_data.inel_cdf_off),
            dptr!(&nuc_data.inel_cdf_n_e),
            dptr!(&nuc_data.inel_cdf_n_t),
            dptr!(&nuc_data.inel_cdf_n_lev),
            dptr!(&nuc_data.inel_cdf_log_e_min),
            dptr!(&nuc_data.inel_cdf_log_e_max),
            // Watt closed-form χ (Law 11), slots 104-109. See
            // transport.cu sample_fission_energy for the dispatch.
            dptr!(&nuc_data.watt_inc_energies),
            dptr!(&nuc_data.watt_a),
            dptr!(&nuc_data.watt_b),
            dptr!(&nuc_data.watt_u),
            dptr!(&nuc_data.watt_nuc_offsets),
            dptr!(&nuc_data.watt_nuc_n),
            // Delayed-only ν̄(E) (slots 110-113). Drives β(E) =
            // ν_d / ν_t at the fission emission site so each banked
            // neutron picks between prompt χ and the soft-Watt delayed
            // spectrum — see transport.cu sample_fission_emit_energy.
            dptr!(&nuc_data.delayed_nu_bar_energies),
            dptr!(&nuc_data.delayed_nu_bar_values),
            dptr!(&nuc_data.delayed_nu_bar_offsets),
            dptr!(&nuc_data.delayed_nu_bar_sizes),
            // Fission χ PDF (slot 114). Drives the OpenMC quadratic
            // lin-lin CDF inversion in `sample_eout_bin`. See
            // P_FIS_PDF comment in transport.cu.
            dptr!(&nuc_data.fis_pdf),
            // MT=91 continuum inelastic distributions (slots 115-122).
            // Replaces the Weisskopf evaporation fallback in the GPU's
            // inelastic branch; restores the ENDF tabular path the CPU
            // already used. Fixes the +400 keV ⟨E_out inelastic⟩ gap
            // that drove the +500-700 pcm fast-metal `k_eff` bias.
            dptr!(&nuc_data.inel91_inc_energies),
            dptr!(&nuc_data.inel91_dist_offsets),
            dptr!(&nuc_data.inel91_dist_sizes),
            dptr!(&nuc_data.inel91_e_out),
            dptr!(&nuc_data.inel91_cdf),
            dptr!(&nuc_data.inel91_pdf),
            dptr!(&nuc_data.inel91_nuc_offsets),
            dptr!(&nuc_data.inel91_nuc_n_inc),
            // Multi-slot S(α,β) lookup (slots 123-129). The flat data
            // arrays still live at slots 43-55; these per-slot tables
            // (length n_slots) plus the per-nuclide lookup (length
            // n_nuc) drive `sab_total_xs` / `sab_sample` for problems
            // with more than one TSL-bearing nuclide (e.g. H-in-H2O
            // + D-in-D2O + C-in-graphite).
            sab_data.n_slots as u64,
            dptr!(&sab_data.slot_per_nuc),
            dptr!(&sab_data.slot_inc_e_off),
            dptr!(&sab_data.slot_n_inc),
            dptr!(&sab_data.slot_eout_table_off),
            dptr!(&sab_data.slot_mu_table_off),
            dptr!(&sab_data.slot_emax),
            // Maxwell (Law 7) / Evaporation (Law 9) closed-form χ —
            // slots 130-135. See P_MAXEVAP_* in transport.cu and the
            // dispatch in sample_fission_energy.
            dptr!(&nuc_data.maxevap_inc_energies),
            dptr!(&nuc_data.maxevap_theta),
            dptr!(&nuc_data.maxevap_u),
            dptr!(&nuc_data.maxevap_law),
            dptr!(&nuc_data.maxevap_nuc_offsets),
            dptr!(&nuc_data.maxevap_nuc_n),
            // Stage C step D — per-nuclide pointer arrays. Slot
            // 136-137 carry [n_nuc × N_RXN_SLOTS] basis / coeffs
            // pointers; slot 138 carries the per-nuclide
            // pointwise_xs base address [n_nuc]. Kernel reads
            // `((double*)PTR_U64(p, P_*)[key])[…]` instead of
            // indirecting through all_basis / pointwise_xs slabs.
            dptr!(&nuc_data.basis_ptrs),
            dptr!(&nuc_data.coeffs_ptrs),
            dptr!(&nuc_data.pw_xs_ptrs),
            dptr!(&nuc_data.total_xs_ptrs),
            dptr!(&nuc_data.nb_e_ptrs),
            dptr!(&nuc_data.nb_v_ptrs),
            dptr!(&nuc_data.dnb_e_ptrs),
            dptr!(&nuc_data.dnb_v_ptrs),
            dptr!(&nuc_data.urr_e_ptrs),
            dptr!(&nuc_data.urr_cp_ptrs),
            dptr!(&nuc_data.urr_tf_ptrs),
            dptr!(&nuc_data.urr_ef_ptrs),
            dptr!(&nuc_data.urr_ff_ptrs),
            dptr!(&nuc_data.urr_cf_ptrs),
            dptr!(&nuc_data.inel_cdf_ptrs),
            dptr!(&nuc_data.level_basis_ptrs),
            dptr!(&nuc_data.level_coeffs_ptrs),
            dptr!(&nuc_data.level_basis_local_off),
            dptr!(&nuc_data.level_coeffs_local_off),
            dptr!(&nuc_data.ang_e_ptrs),
            dptr!(&nuc_data.ang_mu_ptrs),
            dptr!(&nuc_data.ang_cdf_ptrs),
            dptr!(&nuc_data.lev_ang_e_ptrs),
            dptr!(&nuc_data.lev_ang_mu_ptrs),
            dptr!(&nuc_data.lev_ang_cdf_ptrs),
            dptr!(&nuc_data.ang_dist_local_off),
            dptr!(&nuc_data.lev_ang_lev_local_off),
            dptr!(&nuc_data.lev_ang_dist_local_off),
        ];
        debug_assert_eq!(v.len(), N_PARAMS);
        v
    }

    /// Upload SVD nuclide data through the per-nuclide LFU cache.
    /// For each nuclide, looks up `(Arc::as_ptr, rank)` in the
    /// per-nuclide cache — hits reuse the existing GPU upload,
    /// misses run `upload_one_nuclide` and insert the result. Pre-
    /// evicts before each miss using the new nuclide's estimated
    /// device-side cost so peak VRAM stays bounded.
    ///
    /// The flat-pack `GpuNuclideData` is assembled fresh on every
    /// call from the (cached + new) per-nuclide pieces via direct
    /// device-to-device copies (no H→D round-trip). Assembly cost
    /// is small (~20 ms per 4 GB bundle) and dominated by DtoD
    /// bandwidth, so caching the assembled bundle for repeat
    /// same-composition uploads is not worth the doubled VRAM.
    pub fn upload_nuclide_data(
        &self,
        nuclides: &[Arc<crate::transport::xs_provider::NuclideKernels>],
        rank: usize,
    ) -> Result<Arc<GpuNuclideData>, Box<dyn std::error::Error>> {
        use crate::gpu_per_nuclide::{
            assemble_a4_cat, assemble_a5_cat, assemble_a6_cat, assemble_a7_cat,
            assemble_a8_cat, assemble_a_cat, assemble_b_cat, assemble_c_cat,
            upload_one_nuclide, PerNuclideGpu,
        };

        let budget = self.bundle_cache_budget_bytes();
        let mut per_nucs: Vec<Arc<PerNuclideGpu>> = Vec::with_capacity(nuclides.len());

        for nuc in nuclides {
            let key = PerNuclideCacheKey {
                nuc_ptr: Arc::as_ptr(nuc) as usize,
                rank,
            };
            // Lookup.
            {
                let mut guard = self
                    .per_nuclide_cache
                    .lock()
                    .expect("per_nuclide_cache poisoned");
                if guard.entries.contains_key(&key) {
                    guard.counter = guard.counter.wrapping_add(1);
                    let now = guard.counter;
                    let (arc, stats) = guard
                        .entries
                        .get_mut(&key)
                        .expect("contains_key just checked");
                    let arc = Arc::clone(arc);
                    stats.hits = stats.hits.wrapping_add(1);
                    stats.last_touch = now;
                    per_nucs.push(arc);
                    continue;
                }
            }

            // Miss path: pre-evict (estimate the new nuclide's bytes
            // from the host-side approx; device-side overhead vs
            // host roughly tracks within an order of magnitude for
            // our payloads), upload, insert.
            let predicted = nuc.approx_host_bytes();
            let evicted_any = {
                let mut guard = self
                    .per_nuclide_cache
                    .lock()
                    .expect("per_nuclide_cache poisoned");
                let now = guard.counter;
                let mut adapter = PerNuclideCacheAdapter { inner: &mut guard };
                let evicted = evict_to_budget(
                    &mut adapter,
                    predicted,
                    budget,
                    now,
                    DEFAULT_AGE_DECAY,
                );
                evicted > 0
            };
            if evicted_any {
                self.trim_async_mempool();
            }

            let fresh = upload_one_nuclide(&self.stream, nuc, rank)?;
            let bytes = fresh.device_bytes();
            let arc = Arc::new(fresh);
            {
                let mut guard = self
                    .per_nuclide_cache
                    .lock()
                    .expect("per_nuclide_cache poisoned");
                // Concurrent-insert race: another thread may have
                // already populated this key while we uploaded.
                if guard.entries.contains_key(&key) {
                    guard.counter = guard.counter.wrapping_add(1);
                    let now = guard.counter;
                    let (existing, stats) = guard
                        .entries
                        .get_mut(&key)
                        .expect("contains_key just checked");
                    let arc = Arc::clone(existing);
                    stats.hits = stats.hits.wrapping_add(1);
                    stats.last_touch = now;
                    per_nucs.push(arc);
                    continue;
                }
                // Post-upload trim with the actual size.
                {
                    let now = guard.counter;
                    let mut adapter = PerNuclideCacheAdapter { inner: &mut guard };
                    let _ = evict_to_budget(
                        &mut adapter,
                        bytes,
                        budget,
                        now,
                        DEFAULT_AGE_DECAY,
                    );
                }
                let counter = guard.counter;
                guard.counter = guard.counter.wrapping_add(1);
                let stats = EvictionStats::new(bytes, counter);
                guard.total_bytes = guard.total_bytes.saturating_add(bytes);
                guard.entries.insert(key, (Arc::clone(&arc), stats));
            }
            per_nucs.push(arc);
        }

        // Assemble the flat-pack bundle. Pure DtoD; no H→D for the
        // already-cached payloads.
        let a = assemble_a_cat(&self.stream, &per_nucs)?;
        let b = assemble_b_cat(&self.stream, &per_nucs)?;
        let c = assemble_c_cat(&self.stream, &per_nucs)?;
        let a4 = assemble_a4_cat(&self.stream, &per_nucs)?;
        let a5 = assemble_a5_cat(&self.stream, &per_nucs)?;
        let a6 = assemble_a6_cat(&self.stream, &per_nucs)?;
        let a7 = assemble_a7_cat(&self.stream, &per_nucs)?;
        let a8 = assemble_a8_cat(&self.stream, &per_nucs)?;

        let (basis_ptrs, coeffs_ptrs) =
            crate::gpu_per_nuclide::build_per_nuclide_ptr_arrays(&self.stream, &per_nucs)?;
        let pw_xs_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.pointwise_xs.as_ref(),
        )?;
        let total_xs_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.total_xs.as_ref(),
        )?;
        let nb_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.nu_bar.as_ref().map(|nb| &nb.energies),
        )?;
        let nb_v_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.nu_bar.as_ref().map(|nb| &nb.values),
        )?;
        let dnb_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.delayed_nu_bar.as_ref().map(|nb| &nb.energies),
        )?;
        let dnb_v_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.delayed_nu_bar.as_ref().map(|nb| &nb.values),
        )?;
        let urr_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.energies),
        )?;
        let urr_cp_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.cum_prob),
        )?;
        let urr_tf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.total_factor),
        )?;
        let urr_ef_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.elastic_factor),
        )?;
        let urr_ff_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.fission_factor),
        )?;
        let urr_cf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.capture_factor),
        )?;
        let inel_cdf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.inel_cdf.as_ref().map(|c| &c.data),
        )?;
        let (
            level_basis_ptrs,
            level_coeffs_ptrs,
            level_basis_local_off,
            level_coeffs_local_off,
        ) = crate::gpu_per_nuclide::build_per_nuc_level_ptr_and_offsets(
            &self.stream,
            &per_nucs,
        )?;
        let ang_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.elastic_angle.as_ref().map(|a| &a.energies),
        )?;
        let ang_mu_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.elastic_angle.as_ref().map(|a| &a.mu),
        )?;
        let ang_cdf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.elastic_angle.as_ref().map(|a| &a.cdf),
        )?;
        let lev_ang_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| {
                if p.levels.n_levels > 0 {
                    Some(&p.levels.ang_energies)
                } else {
                    None
                }
            },
        )?;
        let lev_ang_mu_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| {
                if p.levels.n_levels > 0 {
                    Some(&p.levels.ang_mu)
                } else {
                    None
                }
            },
        )?;
        let lev_ang_cdf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| {
                if p.levels.n_levels > 0 {
                    Some(&p.levels.ang_cdf)
                } else {
                    None
                }
            },
        )?;
        let ang_dist_local_off = self.stream.clone_htod(&a4.ang_dist_local_off_vec)?;
        let lev_ang_lev_local_off =
            self.stream.clone_htod(&c.lev_ang_lev_local_off_vec)?;
        let lev_ang_dist_local_off =
            self.stream.clone_htod(&c.lev_ang_dist_local_off_vec)?;

        Ok(Arc::new(GpuNuclideData {
            per_nucs: per_nucs.clone(),
            basis_ptrs,
            coeffs_ptrs,
            pw_xs_ptrs,
            total_xs_ptrs,
            nb_e_ptrs,
            nb_v_ptrs,
            dnb_e_ptrs,
            dnb_v_ptrs,
            urr_e_ptrs,
            urr_cp_ptrs,
            urr_tf_ptrs,
            urr_ef_ptrs,
            urr_ff_ptrs,
            urr_cf_ptrs,
            inel_cdf_ptrs,
            level_basis_ptrs,
            level_coeffs_ptrs,
            level_basis_local_off,
            level_coeffs_local_off,
            ang_e_ptrs,
            ang_mu_ptrs,
            ang_cdf_ptrs,
            lev_ang_e_ptrs,
            lev_ang_mu_ptrs,
            lev_ang_cdf_ptrs,
            ang_dist_local_off,
            lev_ang_lev_local_off,
            lev_ang_dist_local_off,
            all_basis: b.all_basis,
            all_coeffs: b.all_coeffs,
            all_energy_grids: a.all_energy_grids,
            basis_offsets: self.stream.clone_htod(&b.basis_offsets_vec)?,
            grid_offsets: self.stream.clone_htod(&a.grid_offsets_vec)?,
            n_energies: self.stream.clone_htod(&a.n_energies_vec)?,
            has_reaction: self.stream.clone_htod(&b.has_reaction_vec)?,
            coeffs_offsets: self.stream.clone_htod(&b.coeffs_offsets_vec)?,
            rank: rank as i32,
            total_xs: a.total_xs,
            total_xs_offsets: self.stream.clone_htod(&a.total_xs_off_vec)?,
            has_total_xs: self.stream.clone_htod(&a.has_total_xs_vec)?,
            pointwise_xs: a.pointwise_xs,
            pw_offsets: self.stream.clone_htod(&a.pw_off_vec)?,
            has_pw: self.stream.clone_htod(&a.has_pw_vec)?,
            nu_bar_energies: a.nu_bar_energies,
            nu_bar_values: a.nu_bar_values,
            nu_bar_offsets: self.stream.clone_htod(&a.nu_bar_offsets_vec)?,
            nu_bar_sizes: self.stream.clone_htod(&a.nu_bar_sizes_vec)?,
            delayed_nu_bar_energies: a.delayed_nu_bar_energies,
            delayed_nu_bar_values: a.delayed_nu_bar_values,
            delayed_nu_bar_offsets: self
                .stream
                .clone_htod(&a.delayed_nu_bar_offsets_vec)?,
            delayed_nu_bar_sizes: self.stream.clone_htod(&a.delayed_nu_bar_sizes_vec)?,
            level_q_values: c.level_q_values,
            level_thresholds: c.level_thresholds,
            level_offsets: self.stream.clone_htod(&c.level_offsets_vec)?,
            level_counts: self.stream.clone_htod(&c.level_counts_vec)?,
            level_basis: c.level_basis,
            level_coeffs: c.level_coeffs,
            level_basis_offsets: self.stream.clone_htod(&c.level_basis_offsets_vec)?,
            level_coeffs_offsets: self.stream.clone_htod(&c.level_coeffs_offsets_vec)?,
            level_has_kernel: c.level_has_kernel,
            level_mt: c.level_mt,
            lev_ang_energies: c.lev_ang_energies,
            lev_ang_mu: c.lev_ang_mu,
            lev_ang_cdf: c.lev_ang_cdf,
            lev_ang_dist_off: self.stream.clone_htod(&c.lev_ang_dist_off_vec)?,
            lev_ang_dist_sz: self.stream.clone_htod(&c.lev_ang_dist_sz_vec)?,
            lev_ang_lev_off: self.stream.clone_htod(&c.lev_ang_lev_off_vec)?,
            lev_ang_lev_ne: self.stream.clone_htod(&c.lev_ang_lev_ne_vec)?,
            ang_energies: a4.ang_energies,
            ang_mu: a4.ang_mu,
            ang_cdf: a4.ang_cdf,
            ang_dist_offsets: self.stream.clone_htod(&a4.ang_dist_offsets_vec)?,
            ang_dist_sizes: self.stream.clone_htod(&a4.ang_dist_sizes_vec)?,
            ang_nuc_offsets: self.stream.clone_htod(&a4.ang_nuc_offsets_vec)?,
            ang_nuc_n_energies: self.stream.clone_htod(&a4.ang_nuc_n_energies_vec)?,
            ang_is_cm: self.stream.clone_htod(&a4.ang_is_cm_vec)?,
            fis_inc_energies: a5.fis_inc_energies,
            fis_dist_offsets: self.stream.clone_htod(&a5.fis_dist_offsets_vec)?,
            fis_dist_sizes: self.stream.clone_htod(&a5.fis_dist_sizes_vec)?,
            fis_e_out: a5.fis_e_out,
            fis_cdf: a5.fis_cdf,
            fis_pdf: a5.fis_pdf,
            fis_nuc_offsets: self.stream.clone_htod(&a5.fis_nuc_offsets_vec)?,
            fis_nuc_n_inc: self.stream.clone_htod(&a5.fis_nuc_n_inc_vec)?,
            watt_inc_energies: a5.watt_inc_energies,
            watt_a: a5.watt_a,
            watt_b: a5.watt_b,
            watt_u: self.stream.clone_htod(&a5.watt_u_vec)?,
            watt_nuc_offsets: self.stream.clone_htod(&a5.watt_nuc_offsets_vec)?,
            watt_nuc_n: self.stream.clone_htod(&a5.watt_nuc_n_vec)?,
            maxevap_inc_energies: a5.maxevap_inc_energies,
            maxevap_theta: a5.maxevap_theta,
            maxevap_u: self.stream.clone_htod(&a5.maxevap_u_vec)?,
            maxevap_law: self.stream.clone_htod(&a5.maxevap_law_vec)?,
            maxevap_nuc_offsets: self.stream.clone_htod(&a5.maxevap_nuc_offsets_vec)?,
            maxevap_nuc_n: self.stream.clone_htod(&a5.maxevap_nuc_n_vec)?,
            inel91_inc_energies: a6.inel91_inc_energies,
            inel91_dist_offsets: self.stream.clone_htod(&a6.inel91_dist_offsets_vec)?,
            inel91_dist_sizes: self.stream.clone_htod(&a6.inel91_dist_sizes_vec)?,
            inel91_e_out: a6.inel91_e_out,
            inel91_cdf: a6.inel91_cdf,
            inel91_pdf: a6.inel91_pdf,
            inel91_nuc_offsets: self.stream.clone_htod(&a6.inel91_nuc_offsets_vec)?,
            inel91_nuc_n_inc: self.stream.clone_htod(&a6.inel91_nuc_n_inc_vec)?,
            urr_energies: a7.urr_energies,
            urr_cum_prob: a7.urr_cum_prob,
            urr_total_f: a7.urr_total_f,
            urr_elastic_f: a7.urr_elastic_f,
            urr_fission_f: a7.urr_fission_f,
            urr_capture_f: a7.urr_capture_f,
            urr_offsets: self.stream.clone_htod(&a7.urr_offsets_vec)?,
            urr_n_energies: self.stream.clone_htod(&a7.urr_n_energies_vec)?,
            urr_n_bands: self.stream.clone_htod(&a7.urr_n_bands_vec)?,
            urr_multiply_smooth: self.stream.clone_htod(&a7.urr_multiply_smooth_vec)?,
            inel_cdf_data: a8.inel_cdf_data,
            inel_cdf_off: self.stream.clone_htod(&a8.inel_cdf_off_vec)?,
            inel_cdf_n_e: self.stream.clone_htod(&a8.inel_cdf_n_e_vec)?,
            inel_cdf_n_t: self.stream.clone_htod(&a8.inel_cdf_n_t_vec)?,
            inel_cdf_n_lev: self.stream.clone_htod(&a8.inel_cdf_n_lev_vec)?,
            inel_cdf_log_e_min: self.stream.clone_htod(&a8.inel_cdf_log_e_min_vec)?,
            inel_cdf_log_e_max: self.stream.clone_htod(&a8.inel_cdf_log_e_max_vec)?,
        }))
    }

    /// Drop the per-context GPU per-nuclide cache and trim the async
    /// mempool. Callers that need to free GPU memory between long
    /// sweeps without dropping the whole context should call this.
    pub fn clear_nuclide_buffer_cache(&self) {
        let mut guard = self
            .per_nuclide_cache
            .lock()
            .expect("per_nuclide_cache poisoned");
        guard.entries.clear();
        guard.total_bytes = 0;
        drop(guard);
        self.trim_async_mempool();
    }

    /// Resolution order: `_BYTES` env (explicit) → `_FRACTION` env ×
    /// `total_mem` → `BUNDLE_CACHE_DEFAULT_FRACTION × total_mem`.
    /// Fallback floor 1 GiB when `cuDeviceTotalMem` returns zero.
    pub fn bundle_cache_budget_bytes(&self) -> usize {
        *self.cached_bundle_budget.get_or_init(|| {
            const HARD_FLOOR: usize = crate::hardware_profile::GIB;
            if let Some(v) = std::env::var_os("OPEN_RUST_MC_GPU_BUNDLE_CACHE_BYTES") {
                if let Ok(n) = v.to_string_lossy().parse::<usize>() {
                    return n.max(1);
                }
            }
            let frac = std::env::var("OPEN_RUST_MC_GPU_BUNDLE_CACHE_FRACTION")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(BUNDLE_CACHE_DEFAULT_FRACTION)
                .clamp(BUNDLE_CACHE_FRACTION_MIN, BUNDLE_CACHE_FRACTION_MAX);
            let total = self._ctx.total_mem().unwrap_or(0);
            if total == 0 {
                return HARD_FLOOR;
            }
            ((total as f64) * frac) as usize
        })
    }

    /// `cuMemPoolTrimTo(default_pool, 0)`. `cuMemFreeAsync` parks
    /// freed bytes in the stream pool until trimmed; without this
    /// call nvidia-smi keeps showing the high-water mark and the
    /// next allocation grows VRAM rather than reusing pool bytes.
    /// No-op on pre-CUDA-11.2 / no-async-alloc devices.
    pub fn trim_async_mempool(&self) {
        if !self._ctx.has_async_alloc() {
            return;
        }
        // SAFETY: live Arc<CudaContext> → valid device handle.
        unsafe {
            let dev = self._ctx.cu_device();
            if let Ok(pool) = cudarc::driver::result::device::get_default_mem_pool(dev) {
                let _ = cudarc::driver::result::mem_pool::trim_to(pool, 0);
            }
        }
    }

    /// Uncached upload via the per-nuclide pipeline. Splits the work
    /// into two stages:
    ///   1. `upload_one_nuclide` extracts each nuclide's data into
    ///      its own `PerNuclideGpu` (per-nuclide CudaSlices, no
    ///      inter-nuclide concatenation).
    ///   2. `assemble_*_cat` reconstructs the flat-pack
    ///      `GpuNuclideData` via direct device-to-device copies —
    ///      no round-trip through host memory.
    ///
    /// Byte-identical to the legacy path (preserved as
    /// `upload_nuclide_data_uncached_legacy`); the per-category
    /// `assemble_*_cat_matches_legacy_bundle_byte_for_byte` tests
    /// in `gpu_per_nuclide::tests` prove parity.
    ///
    /// Production path inlines this logic directly into
    /// `upload_nuclide_data` (per-nuclide cache hits, then assembly).
    /// Kept around purely as the byte-equality reference for
    /// `gpu_per_nuclide::tests`.
    #[allow(dead_code)]
    pub(crate) fn upload_nuclide_data_uncached(
        &self,
        nuclides: &[Arc<crate::transport::xs_provider::NuclideKernels>],
        rank: usize,
    ) -> Result<GpuNuclideData, Box<dyn std::error::Error>> {
        use crate::gpu_per_nuclide::*;

        let per_nucs: Vec<Arc<PerNuclideGpu>> = nuclides
            .iter()
            .map(|n| upload_one_nuclide(&self.stream, n, rank).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;

        let a = assemble_a_cat(&self.stream, &per_nucs)?;
        let b = assemble_b_cat(&self.stream, &per_nucs)?;
        let c = assemble_c_cat(&self.stream, &per_nucs)?;
        let a4 = assemble_a4_cat(&self.stream, &per_nucs)?;
        let a5 = assemble_a5_cat(&self.stream, &per_nucs)?;
        let a6 = assemble_a6_cat(&self.stream, &per_nucs)?;
        let a7 = assemble_a7_cat(&self.stream, &per_nucs)?;
        let a8 = assemble_a8_cat(&self.stream, &per_nucs)?;
        let (basis_ptrs, coeffs_ptrs) =
            crate::gpu_per_nuclide::build_per_nuclide_ptr_arrays(&self.stream, &per_nucs)?;
        let pw_xs_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.pointwise_xs.as_ref(),
        )?;
        let total_xs_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.total_xs.as_ref(),
        )?;
        let nb_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.nu_bar.as_ref().map(|nb| &nb.energies),
        )?;
        let nb_v_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.nu_bar.as_ref().map(|nb| &nb.values),
        )?;
        let dnb_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.delayed_nu_bar.as_ref().map(|nb| &nb.energies),
        )?;
        let dnb_v_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.delayed_nu_bar.as_ref().map(|nb| &nb.values),
        )?;
        let urr_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.energies),
        )?;
        let urr_cp_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.cum_prob),
        )?;
        let urr_tf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.total_factor),
        )?;
        let urr_ef_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.elastic_factor),
        )?;
        let urr_ff_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.fission_factor),
        )?;
        let urr_cf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.urr.as_ref().map(|u| &u.capture_factor),
        )?;
        let inel_cdf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.inel_cdf.as_ref().map(|c| &c.data),
        )?;
        let (
            level_basis_ptrs,
            level_coeffs_ptrs,
            level_basis_local_off,
            level_coeffs_local_off,
        ) = crate::gpu_per_nuclide::build_per_nuc_level_ptr_and_offsets(
            &self.stream,
            &per_nucs,
        )?;
        let ang_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.elastic_angle.as_ref().map(|a| &a.energies),
        )?;
        let ang_mu_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.elastic_angle.as_ref().map(|a| &a.mu),
        )?;
        let ang_cdf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| p.elastic_angle.as_ref().map(|a| &a.cdf),
        )?;
        let lev_ang_e_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| {
                if p.levels.n_levels > 0 {
                    Some(&p.levels.ang_energies)
                } else {
                    None
                }
            },
        )?;
        let lev_ang_mu_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| {
                if p.levels.n_levels > 0 {
                    Some(&p.levels.ang_mu)
                } else {
                    None
                }
            },
        )?;
        let lev_ang_cdf_ptrs = crate::gpu_per_nuclide::build_per_nuc_optional_ptr_array(
            &self.stream,
            &per_nucs,
            |p| {
                if p.levels.n_levels > 0 {
                    Some(&p.levels.ang_cdf)
                } else {
                    None
                }
            },
        )?;
        let ang_dist_local_off = self.stream.clone_htod(&a4.ang_dist_local_off_vec)?;
        let lev_ang_lev_local_off =
            self.stream.clone_htod(&c.lev_ang_lev_local_off_vec)?;
        let lev_ang_dist_local_off =
            self.stream.clone_htod(&c.lev_ang_dist_local_off_vec)?;

        Ok(GpuNuclideData {
            per_nucs,
            basis_ptrs,
            coeffs_ptrs,
            pw_xs_ptrs,
            total_xs_ptrs,
            nb_e_ptrs,
            nb_v_ptrs,
            dnb_e_ptrs,
            dnb_v_ptrs,
            urr_e_ptrs,
            urr_cp_ptrs,
            urr_tf_ptrs,
            urr_ef_ptrs,
            urr_ff_ptrs,
            urr_cf_ptrs,
            inel_cdf_ptrs,
            level_basis_ptrs,
            level_coeffs_ptrs,
            level_basis_local_off,
            level_coeffs_local_off,
            ang_e_ptrs,
            ang_mu_ptrs,
            ang_cdf_ptrs,
            lev_ang_e_ptrs,
            lev_ang_mu_ptrs,
            lev_ang_cdf_ptrs,
            ang_dist_local_off,
            lev_ang_lev_local_off,
            lev_ang_dist_local_off,
            // Category A.1 + B
            all_basis: b.all_basis,
            all_coeffs: b.all_coeffs,
            all_energy_grids: a.all_energy_grids,
            basis_offsets: self.stream.clone_htod(&b.basis_offsets_vec)?,
            grid_offsets: self.stream.clone_htod(&a.grid_offsets_vec)?,
            n_energies: self.stream.clone_htod(&a.n_energies_vec)?,
            has_reaction: self.stream.clone_htod(&b.has_reaction_vec)?,
            coeffs_offsets: self.stream.clone_htod(&b.coeffs_offsets_vec)?,
            rank: rank as i32,
            // Category A.2
            total_xs: a.total_xs,
            total_xs_offsets: self.stream.clone_htod(&a.total_xs_off_vec)?,
            has_total_xs: self.stream.clone_htod(&a.has_total_xs_vec)?,
            pointwise_xs: a.pointwise_xs,
            pw_offsets: self.stream.clone_htod(&a.pw_off_vec)?,
            has_pw: self.stream.clone_htod(&a.has_pw_vec)?,
            // Category A.3
            nu_bar_energies: a.nu_bar_energies,
            nu_bar_values: a.nu_bar_values,
            nu_bar_offsets: self.stream.clone_htod(&a.nu_bar_offsets_vec)?,
            nu_bar_sizes: self.stream.clone_htod(&a.nu_bar_sizes_vec)?,
            delayed_nu_bar_energies: a.delayed_nu_bar_energies,
            delayed_nu_bar_values: a.delayed_nu_bar_values,
            delayed_nu_bar_offsets: self
                .stream
                .clone_htod(&a.delayed_nu_bar_offsets_vec)?,
            delayed_nu_bar_sizes: self.stream.clone_htod(&a.delayed_nu_bar_sizes_vec)?,
            // Category C
            level_q_values: c.level_q_values,
            level_thresholds: c.level_thresholds,
            level_offsets: self.stream.clone_htod(&c.level_offsets_vec)?,
            level_counts: self.stream.clone_htod(&c.level_counts_vec)?,
            level_basis: c.level_basis,
            level_coeffs: c.level_coeffs,
            level_basis_offsets: self.stream.clone_htod(&c.level_basis_offsets_vec)?,
            level_coeffs_offsets: self.stream.clone_htod(&c.level_coeffs_offsets_vec)?,
            level_has_kernel: c.level_has_kernel,
            level_mt: c.level_mt,
            lev_ang_energies: c.lev_ang_energies,
            lev_ang_mu: c.lev_ang_mu,
            lev_ang_cdf: c.lev_ang_cdf,
            lev_ang_dist_off: self.stream.clone_htod(&c.lev_ang_dist_off_vec)?,
            lev_ang_dist_sz: self.stream.clone_htod(&c.lev_ang_dist_sz_vec)?,
            lev_ang_lev_off: self.stream.clone_htod(&c.lev_ang_lev_off_vec)?,
            lev_ang_lev_ne: self.stream.clone_htod(&c.lev_ang_lev_ne_vec)?,
            // Category A.4
            ang_energies: a4.ang_energies,
            ang_mu: a4.ang_mu,
            ang_cdf: a4.ang_cdf,
            ang_dist_offsets: self.stream.clone_htod(&a4.ang_dist_offsets_vec)?,
            ang_dist_sizes: self.stream.clone_htod(&a4.ang_dist_sizes_vec)?,
            ang_nuc_offsets: self.stream.clone_htod(&a4.ang_nuc_offsets_vec)?,
            ang_nuc_n_energies: self.stream.clone_htod(&a4.ang_nuc_n_energies_vec)?,
            ang_is_cm: self.stream.clone_htod(&a4.ang_is_cm_vec)?,
            // Category A.5
            fis_inc_energies: a5.fis_inc_energies,
            fis_dist_offsets: self.stream.clone_htod(&a5.fis_dist_offsets_vec)?,
            fis_dist_sizes: self.stream.clone_htod(&a5.fis_dist_sizes_vec)?,
            fis_e_out: a5.fis_e_out,
            fis_cdf: a5.fis_cdf,
            fis_pdf: a5.fis_pdf,
            fis_nuc_offsets: self.stream.clone_htod(&a5.fis_nuc_offsets_vec)?,
            fis_nuc_n_inc: self.stream.clone_htod(&a5.fis_nuc_n_inc_vec)?,
            watt_inc_energies: a5.watt_inc_energies,
            watt_a: a5.watt_a,
            watt_b: a5.watt_b,
            watt_u: self.stream.clone_htod(&a5.watt_u_vec)?,
            watt_nuc_offsets: self.stream.clone_htod(&a5.watt_nuc_offsets_vec)?,
            watt_nuc_n: self.stream.clone_htod(&a5.watt_nuc_n_vec)?,
            maxevap_inc_energies: a5.maxevap_inc_energies,
            maxevap_theta: a5.maxevap_theta,
            maxevap_u: self.stream.clone_htod(&a5.maxevap_u_vec)?,
            maxevap_law: self.stream.clone_htod(&a5.maxevap_law_vec)?,
            maxevap_nuc_offsets: self.stream.clone_htod(&a5.maxevap_nuc_offsets_vec)?,
            maxevap_nuc_n: self.stream.clone_htod(&a5.maxevap_nuc_n_vec)?,
            // Category A.6
            inel91_inc_energies: a6.inel91_inc_energies,
            inel91_dist_offsets: self.stream.clone_htod(&a6.inel91_dist_offsets_vec)?,
            inel91_dist_sizes: self.stream.clone_htod(&a6.inel91_dist_sizes_vec)?,
            inel91_e_out: a6.inel91_e_out,
            inel91_cdf: a6.inel91_cdf,
            inel91_pdf: a6.inel91_pdf,
            inel91_nuc_offsets: self.stream.clone_htod(&a6.inel91_nuc_offsets_vec)?,
            inel91_nuc_n_inc: self.stream.clone_htod(&a6.inel91_nuc_n_inc_vec)?,
            // Category A.7
            urr_energies: a7.urr_energies,
            urr_cum_prob: a7.urr_cum_prob,
            urr_total_f: a7.urr_total_f,
            urr_elastic_f: a7.urr_elastic_f,
            urr_fission_f: a7.urr_fission_f,
            urr_capture_f: a7.urr_capture_f,
            urr_offsets: self.stream.clone_htod(&a7.urr_offsets_vec)?,
            urr_n_energies: self.stream.clone_htod(&a7.urr_n_energies_vec)?,
            urr_n_bands: self.stream.clone_htod(&a7.urr_n_bands_vec)?,
            urr_multiply_smooth: self.stream.clone_htod(&a7.urr_multiply_smooth_vec)?,
            // Category A.8
            inel_cdf_data: a8.inel_cdf_data,
            inel_cdf_off: self.stream.clone_htod(&a8.inel_cdf_off_vec)?,
            inel_cdf_n_e: self.stream.clone_htod(&a8.inel_cdf_n_e_vec)?,
            inel_cdf_n_t: self.stream.clone_htod(&a8.inel_cdf_n_t_vec)?,
            inel_cdf_n_lev: self.stream.clone_htod(&a8.inel_cdf_n_lev_vec)?,
            inel_cdf_log_e_min: self.stream.clone_htod(&a8.inel_cdf_log_e_min_vec)?,
            inel_cdf_log_e_max: self.stream.clone_htod(&a8.inel_cdf_log_e_max_vec)?,
        })
    }

    /// Legacy uncached upload — inline host-side packing of every
    /// per-nuclide field into a single concatenated Vec per slab,
    /// then one `clone_htod` per slab. Kept as the byte-equality
    /// reference for the new per-nuclide path; never called in
    /// production. Removed in a follow-up commit once Step D drops
    /// the assembled-flat-pack `GpuNuclideData` entirely.
    #[allow(dead_code)]
    pub(crate) fn upload_nuclide_data_uncached_legacy(
        &self,
        nuclides: &[Arc<crate::transport::xs_provider::NuclideKernels>],
        rank: usize,
    ) -> Result<GpuNuclideData, Box<dyn std::error::Error>> {
        let n_nuc = nuclides.len();
        let n_rxn = 7; // elastic, inelastic, n2n, n3n, fission, capture, total

        // Concatenate all basis, coefficients, and energy grids
        let mut all_basis_vec: Vec<f64> = Vec::new();
        let mut all_coeffs_vec: Vec<f64> = Vec::new();
        let mut all_grids_vec: Vec<f64> = Vec::new();
        let mut basis_offsets_vec = vec![0_i32; n_nuc * n_rxn];
        let mut coeffs_offsets_vec = vec![0_i32; n_nuc * n_rxn];
        let mut grid_offsets_vec = vec![0_i32; n_nuc];
        let mut n_energies_vec = vec![0_i32; n_nuc];
        let mut has_reaction_vec = vec![0_i32; n_nuc * n_rxn];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            // Energy grid (shared across reactions for this nuclide)
            let grid_offset = all_grids_vec.len();
            grid_offsets_vec[nuc_idx] = grid_offset as i32;

            // Get energy grid from any available reaction
            let any_kernel = nuc
                .elastic
                .as_ref()
                .or(nuc.fission.as_ref())
                .or(nuc.capture.as_ref())
                .or(nuc.inelastic.as_ref())
                .or(nuc.n2n.as_ref())
                .or(nuc.n3n.as_ref());

            if let Some(rk) = any_kernel {
                all_grids_vec.extend_from_slice(rk.energies());
                n_energies_vec[nuc_idx] = rk.n_energy() as i32;
            }

            // Each reaction
            let reactions: [Option<&crate::transport::xs_provider::ReactionKernel>; 7] = [
                nuc.elastic.as_ref(),
                nuc.inelastic.as_ref(),
                nuc.n2n.as_ref(),
                nuc.n3n.as_ref(),
                nuc.fission.as_ref(),
                nuc.capture.as_ref(),
                None, // RXN_TOTAL: not an SVD kernel, handled via pointwise total_xs
            ];

            for (rxn_idx, rxn_opt) in reactions.iter().enumerate() {
                let key = nuc_idx * n_rxn + rxn_idx;
                use crate::transport::xs_provider::ReactionKernel;
                match rxn_opt {
                    Some(ReactionKernel::Svd { kernel, coeffs }) => {
                        has_reaction_vec[key] = 1;
                        basis_offsets_vec[key] = all_basis_vec.len() as i32;
                        all_basis_vec.extend_from_slice(kernel.basis_f64());
                        coeffs_offsets_vec[key] = all_coeffs_vec.len() as i32;
                        all_coeffs_vec.extend_from_slice(coeffs);
                    }
                    Some(ReactionKernel::Table { xs, .. }) => {
                        // Adapt the Table variant into the uniform
                        // rank-`rank` SVD layout the device kernel
                        // expects:
                        //   basis[e * rank + 0] = log10(xs[e])  (clamp to a
                        //                                        large negative
                        //                                        for zero-XS
                        //                                        points)
                        //   basis[e * rank + r] = 0   for r > 0
                        //   coeffs[0]            = 1
                        //   coeffs[r]            = 0   for r > 0
                        // Reconstruction then collapses to
                        //   log_xs = Σ_r basis_r · coeffs_r
                        //          = log10(xs[e])
                        // matching the Table semantics exactly. Slightly
                        // higher device memory (rank× the bytes) than a
                        // dedicated pointwise upload would, but no
                        // device-kernel changes required and the CPU
                        // already keeps SVD-and-Table parity for the
                        // hot path.
                        has_reaction_vec[key] = 1;
                        basis_offsets_vec[key] = all_basis_vec.len() as i32;
                        for &v in xs {
                            let log10_v = if v > 0.0 { v.log10() } else { -300.0 };
                            all_basis_vec.push(log10_v);
                            for _ in 1..rank {
                                all_basis_vec.push(0.0);
                            }
                        }
                        coeffs_offsets_vec[key] = all_coeffs_vec.len() as i32;
                        all_coeffs_vec.push(1.0);
                        for _ in 1..rank {
                            all_coeffs_vec.push(0.0);
                        }
                    }
                    None => {
                        basis_offsets_vec[key] = 0;
                        coeffs_offsets_vec[key] = 0;
                    }
                }
            }
        }

        // Ensure we have data
        if all_basis_vec.is_empty() {
            all_basis_vec.push(0.0);
        }
        if all_coeffs_vec.is_empty() {
            all_coeffs_vec.push(0.0);
        }
        if all_grids_vec.is_empty() {
            all_grids_vec.push(0.0);
        }

        // ── Pack pointwise total XS (sum of all HDF5 reactions) ──
        let mut total_xs_vec: Vec<f64> = Vec::new();
        let mut total_xs_off_vec = vec![0_i32; n_nuc];
        let mut has_total_xs_vec = vec![0_i32; n_nuc];
        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref xs) = nuc.total_xs_raw {
                total_xs_off_vec[nuc_idx] = total_xs_vec.len() as i32;
                has_total_xs_vec[nuc_idx] = 1;
                total_xs_vec.extend_from_slice(xs);
            }
        }
        if total_xs_vec.is_empty() {
            total_xs_vec.push(0.0);
        }

        // ── Pack pointwise XS tables (7 channels per energy point) ──
        let mut pw_xs_vec: Vec<f64> = Vec::new();
        let mut pw_off_vec = vec![0_i32; n_nuc];
        let mut has_pw_vec = vec![0_i32; n_nuc];
        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref pw) = nuc.pointwise_xs {
                pw_off_vec[nuc_idx] = pw_xs_vec.len() as i32;
                has_pw_vec[nuc_idx] = 1;
                pw_xs_vec.extend_from_slice(pw);
            }
        }
        if pw_xs_vec.is_empty() {
            pw_xs_vec.push(0.0);
        }
        println!(
            "  GPU: pointwise XS = {:.1} MB",
            pw_xs_vec.len() as f64 * 8.0 / 1e6
        );

        // ── Pack synthesized MT=4 + per-level CDF (when present) ──
        // Replaces the do_inelastic 13-level walk with a single binary
        // search in a log-decimated CDF (~200 energy points). See
        // xs_provider::InelasticCdf.
        let mut inel_cdf_data_vec: Vec<f64> = Vec::new();
        let mut inel_cdf_off_vec: Vec<i32> = vec![-1; n_nuc];
        let mut inel_cdf_n_e_vec: Vec<i32> = vec![0; n_nuc];
        let mut inel_cdf_n_t_vec: Vec<i32> = vec![0; n_nuc];
        let mut inel_cdf_n_lev_vec: Vec<i32> = vec![0; n_nuc];
        let mut inel_cdf_log_e_min_vec: Vec<f64> = vec![0.0; n_nuc];
        let mut inel_cdf_log_e_max_vec: Vec<f64> = vec![0.0; n_nuc];
        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref cdf) = nuc.inelastic_cdf {
                inel_cdf_off_vec[nuc_idx] = inel_cdf_data_vec.len() as i32;
                inel_cdf_n_e_vec[nuc_idx] = cdf.n_energy as i32;
                inel_cdf_n_t_vec[nuc_idx] = cdf.n_temp as i32;
                inel_cdf_n_lev_vec[nuc_idx] = cdf.n_levels as i32;
                inel_cdf_log_e_min_vec[nuc_idx] = cdf.log_e_min;
                inel_cdf_log_e_max_vec[nuc_idx] = cdf.log_e_max;
                inel_cdf_data_vec.extend_from_slice(&cdf.cdf_flat);
            }
        }
        if inel_cdf_data_vec.is_empty() {
            inel_cdf_data_vec.push(0.0);
        }

        // ── Pack discrete inelastic levels (Q-values + SVD basis) ──
        let mut lev_q_vec: Vec<f64> = Vec::new();
        let mut lev_thr_vec: Vec<f64> = Vec::new();
        let mut lev_off_vec = vec![0_i32; n_nuc];
        let mut lev_cnt_vec = vec![0_i32; n_nuc];
        let mut lev_basis_vec: Vec<f64> = Vec::new();
        let mut lev_coeffs_vec: Vec<f64> = Vec::new();
        let mut lev_basis_off_vec: Vec<i32> = Vec::new();
        let mut lev_coeffs_off_vec: Vec<i32> = Vec::new();
        let mut lev_has_kernel_vec: Vec<i32> = Vec::new();
        let mut lev_mt_vec: Vec<i32> = Vec::new();

        // Per-global-level angular distribution flattening. Indexed by
        // the same global level index as lev_q_vec etc.
        let mut lev_ang_e_vec: Vec<f64> = Vec::new();
        let mut lev_ang_mu_vec: Vec<f64> = Vec::new();
        let mut lev_ang_cdf_vec: Vec<f64> = Vec::new();
        let mut lev_ang_doff_vec: Vec<i32> = Vec::new();
        let mut lev_ang_dsz_vec: Vec<i32> = Vec::new();
        let mut lev_ang_loff_vec: Vec<i32> = Vec::new();
        let mut lev_ang_lne_vec: Vec<i32> = Vec::new();

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            lev_off_vec[nuc_idx] = lev_q_vec.len() as i32;
            lev_cnt_vec[nuc_idx] = nuc.discrete_levels.len() as i32;
            for (li, lev) in nuc.discrete_levels.iter().enumerate() {
                lev_q_vec.push(lev.info.q_value);
                lev_thr_vec.push(lev.info.threshold);
                lev_mt_vec.push(lev.info.mt as i32);
                match lev.kernel.as_ref() {
                    Some(crate::transport::xs_provider::ReactionKernel::Svd { kernel, coeffs }) => {
                        // Per-level SvdKernel may have actual rank
                        // `level_rank < rank` (the global rank uploaded
                        // as `P_RANK`) when the HDF5 grid for the MT is
                        // sparse (high-threshold levels typically have
                        // <15 unique energy points and SVD truncates).
                        //
                        // The device kernel reads basis[e_idx * P_RANK
                        // + j] for j in 0..P_RANK and dots with
                        // coeffs[0..P_RANK]; if we just `extend_from_
                        // slice` the level's raw `n_e * level_rank`
                        // basis we end up storing a NARROWER stride than
                        // the kernel reads — every column j ≥ level_rank
                        // reads past the level's basis into the next
                        // level's data (or into level_coeffs garbage).
                        // The pre-fix consequence on Godiva: levels with
                        // small svd.rank silently returned ~10^0 ≈ 1 b
                        // for their XS (basis row is junk that dot-
                        // products to ~0 in log space), so the GPU's
                        // level-selection sampling was biased toward the
                        // first ~16 low-|Q| levels — ⟨|Q|⟩_GPU = 659 keV
                        // vs CPU 926 keV, the +500-700 pcm fast-metal
                        // hot bias.
                        //
                        // Fix: pad each row to the global stride with
                        // zeros, and pad coeffs to length `rank` with
                        // zeros. The dot product reproduces the level's
                        // true XS (extra coeffs * 0 + extra basis * 0 = 0).
                        let level_rank = kernel.rank();
                        let n_e = kernel.n_energy();
                        let raw_basis = kernel.basis_f64();
                        lev_has_kernel_vec.push(1);
                        lev_basis_off_vec.push(lev_basis_vec.len() as i32);
                        if level_rank == rank {
                            lev_basis_vec.extend_from_slice(raw_basis);
                        } else {
                            for i in 0..n_e {
                                let src = &raw_basis[i * level_rank..(i + 1) * level_rank];
                                lev_basis_vec.extend_from_slice(src);
                                for _ in level_rank..rank {
                                    lev_basis_vec.push(0.0);
                                }
                            }
                        }
                        lev_coeffs_off_vec.push(lev_coeffs_vec.len() as i32);
                        lev_coeffs_vec.extend_from_slice(coeffs);
                        for _ in coeffs.len()..rank {
                            lev_coeffs_vec.push(0.0);
                        }
                    }
                    Some(crate::transport::xs_provider::ReactionKernel::Table { xs, .. }) => {
                        // Discrete-level Table variant — adapt to the
                        // same uniform rank-N SVD layout the device
                        // kernel expects, matching the path used for
                        // the main per-MT channels (basis = log10(xs)
                        // at slot 0, zero elsewhere; coeffs = [1, 0,
                        // ...]).
                        //
                        // Uses the function-level `rank` parameter
                        // directly. The previous code re-derived a
                        // local `rank` by scanning for the first Svd
                        // kernel in elastic/fission, falling back to
                        // 1 when none existed — in production the
                        // first Svd kernel is always present and the
                        // derived rank equals the global rank, so the
                        // shadowing was a no-op outside synthetic
                        // tests that omit SVD channels.
                        lev_has_kernel_vec.push(1);
                        lev_basis_off_vec.push(lev_basis_vec.len() as i32);
                        for &v in xs {
                            let log10_v = if v > 0.0 { v.log10() } else { -300.0 };
                            lev_basis_vec.push(log10_v);
                            for _ in 1..rank {
                                lev_basis_vec.push(0.0);
                            }
                        }
                        lev_coeffs_off_vec.push(lev_coeffs_vec.len() as i32);
                        lev_coeffs_vec.push(1.0);
                        for _ in 1..rank {
                            lev_coeffs_vec.push(0.0);
                        }
                    }
                    None => {
                        lev_has_kernel_vec.push(0);
                        lev_basis_off_vec.push(0);
                        lev_coeffs_off_vec.push(0);
                    }
                }
                // Per-level angular dist: optional slice aligned with
                // discrete_levels. Missing → mark as 0 energies so the
                // GPU device fn returns isotropic μ_cm.
                let ang = nuc.discrete_level_angles.get(li).and_then(|o| o.as_ref());
                match ang {
                    Some(ad) if !ad.energies.is_empty() => {
                        lev_ang_loff_vec.push(lev_ang_e_vec.len() as i32);
                        lev_ang_lne_vec.push(ad.energies.len() as i32);
                        for (ei, e) in ad.energies.iter().enumerate() {
                            lev_ang_e_vec.push(*e);
                            let dist = &ad.distributions[ei];
                            lev_ang_doff_vec.push(lev_ang_mu_vec.len() as i32);
                            lev_ang_dsz_vec.push(dist.mu.len() as i32);
                            lev_ang_mu_vec.extend_from_slice(&dist.mu);
                            lev_ang_cdf_vec.extend_from_slice(&dist.cdf);
                        }
                    }
                    _ => {
                        lev_ang_loff_vec.push(0);
                        lev_ang_lne_vec.push(0);
                    }
                }
            }
        }
        if lev_q_vec.is_empty() {
            lev_q_vec.push(0.0);
            lev_thr_vec.push(0.0);
            lev_mt_vec.push(0);
            lev_has_kernel_vec.push(0);
            lev_basis_off_vec.push(0);
            lev_coeffs_off_vec.push(0);
            lev_ang_loff_vec.push(0);
            lev_ang_lne_vec.push(0);
        }
        if lev_basis_vec.is_empty() {
            lev_basis_vec.push(0.0);
        }
        if lev_coeffs_vec.is_empty() {
            lev_coeffs_vec.push(0.0);
        }
        if lev_ang_e_vec.is_empty() {
            lev_ang_e_vec.push(0.0);
        }
        if lev_ang_mu_vec.is_empty() {
            lev_ang_mu_vec.push(0.0);
            lev_ang_cdf_vec.push(0.0);
        }
        if lev_ang_doff_vec.is_empty() {
            lev_ang_doff_vec.push(0);
            lev_ang_dsz_vec.push(0);
        }

        let n_total_levels: usize = lev_cnt_vec.iter().map(|&c| c as usize).sum();
        println!(
            "  GPU: {} discrete levels, {:.1} MB level basis",
            n_total_levels,
            lev_basis_vec.len() as f64 * 4.0 / 1e6
        );

        // ── Pack angular distributions ──
        let mut ang_e_vec: Vec<f64> = Vec::new();
        let mut ang_mu_vec: Vec<f64> = Vec::new();
        let mut ang_cdf_vec: Vec<f64> = Vec::new();
        let mut ang_doff_vec: Vec<i32> = Vec::new();
        let mut ang_dsz_vec: Vec<i32> = Vec::new();
        let mut ang_noff_vec = vec![0_i32; n_nuc];
        let mut ang_nne_vec = vec![0_i32; n_nuc];
        let mut ang_cm_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref ad) = nuc.elastic_angle {
                ang_noff_vec[nuc_idx] = ang_e_vec.len() as i32;
                ang_nne_vec[nuc_idx] = ad.energies.len() as i32;
                ang_cm_vec[nuc_idx] = if ad.center_of_mass { 1 } else { 0 };
                for (i, e) in ad.energies.iter().enumerate() {
                    ang_e_vec.push(*e);
                    let dist = &ad.distributions[i];
                    ang_doff_vec.push(ang_mu_vec.len() as i32);
                    ang_dsz_vec.push(dist.mu.len() as i32);
                    ang_mu_vec.extend_from_slice(&dist.mu);
                    ang_cdf_vec.extend_from_slice(&dist.cdf);
                }
            }
        }
        if ang_e_vec.is_empty() {
            ang_e_vec.push(0.0);
        }
        if ang_mu_vec.is_empty() {
            ang_mu_vec.push(0.0);
            ang_cdf_vec.push(0.0);
        }
        if ang_doff_vec.is_empty() {
            ang_doff_vec.push(0);
            ang_dsz_vec.push(0);
        }

        // ── Pack nu-bar tables (flat with offsets) ──
        let mut nb_energies_vec: Vec<f64> = Vec::new();
        let mut nb_values_vec: Vec<f64> = Vec::new();
        let mut nb_offsets_vec = vec![0_i32; n_nuc];
        let mut nb_sizes_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref t) = nuc.nu_bar_table {
                if !t.energies.is_empty() {
                    nb_offsets_vec[nuc_idx] = nb_energies_vec.len() as i32;
                    nb_sizes_vec[nuc_idx] = t.energies.len() as i32;
                    nb_energies_vec.extend_from_slice(&t.energies);
                    nb_values_vec.extend_from_slice(&t.values);
                }
            }
        }
        if nb_energies_vec.is_empty() {
            nb_energies_vec.push(0.0);
            nb_values_vec.push(0.0);
        }

        // ── Pack delayed-only ν̄(E) tables — mirrors the prompt+delayed
        // packing above. The device emitter divides ν_delayed(E)/ν_total(E)
        // per banked neutron to sample β(E) on the fly; nuclides without a
        // delayed table leave `dnb_sizes_vec[i] = 0` and the GPU falls
        // through to the prompt χ path (existing sample_fission_energy).
        let mut dnb_energies_vec: Vec<f64> = Vec::new();
        let mut dnb_values_vec: Vec<f64> = Vec::new();
        let mut dnb_offsets_vec = vec![0_i32; n_nuc];
        let mut dnb_sizes_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref t) = nuc.delayed_nu_bar_table {
                if !t.energies.is_empty() {
                    dnb_offsets_vec[nuc_idx] = dnb_energies_vec.len() as i32;
                    dnb_sizes_vec[nuc_idx] = t.energies.len() as i32;
                    dnb_energies_vec.extend_from_slice(&t.energies);
                    dnb_values_vec.extend_from_slice(&t.values);
                }
            }
        }
        if dnb_energies_vec.is_empty() {
            dnb_energies_vec.push(0.0);
            dnb_values_vec.push(0.0);
        }

        // ── Pack fission energy distributions (flat CDFs with offsets) ──
        let mut fis_inc_e_vec: Vec<f64> = Vec::new();
        let mut fis_dist_off_vec: Vec<i32> = Vec::new();
        let mut fis_dist_sz_vec: Vec<i32> = Vec::new();
        let mut fis_eout_vec: Vec<f64> = Vec::new();
        let mut fis_cdf_vec: Vec<f64> = Vec::new();
        let mut fis_pdf_vec: Vec<f64> = Vec::new();
        let mut fis_nuc_off_vec = vec![0_i32; n_nuc];
        let mut fis_nuc_ninc_vec = vec![0_i32; n_nuc];

        // Per-nuclide Watt closed-form χ parameters. Populated only
        // for nuclides whose ENDF evaluation carries Watt (Law 11) —
        // the rest leave `watt_nuc_n_vec[i] = 0` and the device kernel
        // skips the Watt branch.
        let mut watt_inc_e_vec: Vec<f64> = Vec::new();
        let mut watt_a_vec: Vec<f64> = Vec::new();
        let mut watt_b_vec: Vec<f64> = Vec::new();
        let mut watt_u_vec = vec![0.0_f64; n_nuc];
        let mut watt_nuc_off_vec = vec![0_i32; n_nuc];
        let mut watt_nuc_n_vec = vec![0_i32; n_nuc];

        // Per-nuclide Maxwell (Law 7) / Evaporation (Law 9) θ(E_in)
        // table — single 1D, shared by both laws; the per-nuclide
        // `maxevap_law_vec[i]` ∈ {0=none, 7=Maxwell, 9=Evaporation}
        // selects the sampler on the device.
        let mut maxevap_inc_e_vec: Vec<f64> = Vec::new();
        let mut maxevap_theta_vec: Vec<f64> = Vec::new();
        let mut maxevap_u_vec = vec![0.0_f64; n_nuc];
        let mut maxevap_law_vec = vec![0_i32; n_nuc];
        let mut maxevap_nuc_off_vec = vec![0_i32; n_nuc];
        let mut maxevap_nuc_n_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref edist) = nuc.fission_energy_dist {
                use crate::hdf5_reader::FissionEnergyLaw;
                match &edist.closed_form {
                    None => {
                        // Tabular path — ENDF Law 4 / Law 61. Distributions vec
                        // is aligned 1:1 with the incident-energy grid.
                        fis_nuc_off_vec[nuc_idx] = fis_inc_e_vec.len() as i32;
                        fis_nuc_ninc_vec[nuc_idx] = edist.energies.len() as i32;
                        for (i, e_inc) in edist.energies.iter().enumerate() {
                            fis_inc_e_vec.push(*e_inc);
                            let dist = &edist.distributions[i];
                            fis_dist_off_vec.push(fis_eout_vec.len() as i32);
                            fis_dist_sz_vec.push(dist.e_out.len() as i32);
                            fis_eout_vec.extend_from_slice(&dist.e_out);
                            fis_cdf_vec.extend_from_slice(&dist.cdf);
                            // PDF aligned 1:1 with e_out/cdf when ENDF
                            // ships it; some Law 4 variants ship only
                            // a histogram CDF (no PDF). The device
                            // helper falls back to linear-CDF when
                            // PDF entries are zero, so an unconditional
                            // extend preserves alignment.
                            if dist.pdf.len() == dist.e_out.len() {
                                fis_pdf_vec.extend_from_slice(&dist.pdf);
                            } else {
                                fis_pdf_vec.extend(std::iter::repeat_n(0.0_f64, dist.e_out.len()));
                            }
                        }
                    }
                    Some(FissionEnergyLaw::Watt(w)) => {
                        // Closed-form Watt — distributions vec is empty.
                        // Resample a(E) and b(E) onto a SHARED incident-
                        // energy grid (union of both) so the device only
                        // does one binary search per fission event.
                        let mut shared: Vec<f64> = w
                            .a_energies
                            .iter()
                            .chain(w.b_energies.iter())
                            .copied()
                            .collect();
                        shared.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        shared.dedup_by(|a, b| (*a - *b).abs() < 1e-30 * a.abs().max(1.0));
                        let off = watt_inc_e_vec.len() as i32;
                        watt_nuc_off_vec[nuc_idx] = off;
                        watt_nuc_n_vec[nuc_idx] = shared.len() as i32;
                        watt_u_vec[nuc_idx] = w.u;
                        for e in &shared {
                            watt_inc_e_vec.push(*e);
                            watt_a_vec.push(crate::hdf5_reader::WattLaw::lookup_lin_lin_pub(
                                &w.a_energies,
                                &w.a_values,
                                *e,
                            ));
                            watt_b_vec.push(crate::hdf5_reader::WattLaw::lookup_lin_lin_pub(
                                &w.b_energies,
                                &w.b_values,
                                *e,
                            ));
                        }
                    }
                    Some(FissionEnergyLaw::Maxwell(m))
                    | Some(FissionEnergyLaw::Evaporation(m)) => {
                        // Maxwell (Law 7) and Evaporation (Law 9) both
                        // carry the same single 1D θ(E_in) table; the
                        // device-side sampler chooses between
                        //     χ(E) ∝ √E · exp(−E/θ)   (Maxwell)
                        //     χ(E) ∝   E · exp(−E/θ)  (Evaporation)
                        // based on `maxevap_law_vec[nuc_idx]`. The CPU
                        // reference samplers live in
                        // `hdf5_reader.rs::{sample_maxwell,sample_evaporation}`.
                        let law = match edist.closed_form {
                            Some(FissionEnergyLaw::Maxwell(_)) => 7,
                            Some(FissionEnergyLaw::Evaporation(_)) => 9,
                            _ => 0,
                        };
                        let off = maxevap_inc_e_vec.len() as i32;
                        let n = m.theta_energies.len();
                        maxevap_nuc_off_vec[nuc_idx] = off;
                        maxevap_nuc_n_vec[nuc_idx] = n as i32;
                        maxevap_u_vec[nuc_idx] = m.u;
                        maxevap_law_vec[nuc_idx] = law;
                        maxevap_inc_e_vec.extend_from_slice(&m.theta_energies);
                        maxevap_theta_vec.extend_from_slice(&m.theta_values);
                    }
                }
            }
        }
        if fis_inc_e_vec.is_empty() {
            fis_inc_e_vec.push(0.0);
        }
        if fis_eout_vec.is_empty() {
            fis_eout_vec.push(0.0);
            fis_cdf_vec.push(0.0);
            fis_pdf_vec.push(0.0);
        }
        if fis_dist_off_vec.is_empty() {
            fis_dist_off_vec.push(0);
            fis_dist_sz_vec.push(0);
        }

        // ── Pack MT=91 continuum-inelastic outgoing-energy
        // distributions. Layout 1:1 with the fission spectrum
        // packing above. Closes the +400 keV ⟨E_out⟩ gap that the
        // evaporation fallback was producing on Godiva / Jezebel.
        let mut inel91_inc_e_vec: Vec<f64> = Vec::new();
        let mut inel91_dist_off_vec: Vec<i32> = Vec::new();
        let mut inel91_dist_sz_vec: Vec<i32> = Vec::new();
        let mut inel91_eout_vec: Vec<f64> = Vec::new();
        let mut inel91_cdf_vec: Vec<f64> = Vec::new();
        let mut inel91_pdf_vec: Vec<f64> = Vec::new();
        let mut inel91_nuc_off_vec = vec![0_i32; n_nuc];
        let mut inel91_nuc_ninc_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref edist) = nuc.inelastic_continuum_edist {
                // MT=91 is always tabular (ENDF Law 4); no closed-form
                // variant to dispatch on.
                if edist.energies.is_empty() || edist.distributions.is_empty() {
                    continue;
                }
                inel91_nuc_off_vec[nuc_idx] = inel91_inc_e_vec.len() as i32;
                inel91_nuc_ninc_vec[nuc_idx] = edist.energies.len() as i32;
                for (i, e_inc) in edist.energies.iter().enumerate() {
                    inel91_inc_e_vec.push(*e_inc);
                    let dist = &edist.distributions[i];
                    inel91_dist_off_vec.push(inel91_eout_vec.len() as i32);
                    inel91_dist_sz_vec.push(dist.e_out.len() as i32);
                    inel91_eout_vec.extend_from_slice(&dist.e_out);
                    inel91_cdf_vec.extend_from_slice(&dist.cdf);
                    if dist.pdf.len() == dist.e_out.len() {
                        inel91_pdf_vec.extend_from_slice(&dist.pdf);
                    } else {
                        inel91_pdf_vec
                            .extend(std::iter::repeat_n(0.0_f64, dist.e_out.len()));
                    }
                }
            }
        }
        if inel91_inc_e_vec.is_empty() {
            inel91_inc_e_vec.push(0.0);
        }
        if inel91_eout_vec.is_empty() {
            inel91_eout_vec.push(0.0);
            inel91_cdf_vec.push(0.0);
            inel91_pdf_vec.push(0.0);
        }
        if inel91_dist_off_vec.is_empty() {
            inel91_dist_off_vec.push(0);
            inel91_dist_sz_vec.push(0);
        }
        let n_with_inel91 = inel91_nuc_ninc_vec.iter().filter(|&&n| n > 0).count();
        println!(
            "  GPU: MT=91 continuum table = {} pts ({} / {} nuclides covered)",
            inel91_eout_vec.len(),
            n_with_inel91,
            n_nuc,
        );
        // Watt buffers must have at least one entry so the CUDA
        // device buffers are non-empty (`clone_htod` accepts empty
        // slices on most drivers but emitting a sentinel keeps the
        // hot path branch-free at the bounds check).
        if watt_inc_e_vec.is_empty() {
            watt_inc_e_vec.push(0.0);
            watt_a_vec.push(0.0);
            watt_b_vec.push(0.0);
        }
        // Same sentinel rule for the Maxwell / Evaporation θ(E_in) table.
        if maxevap_inc_e_vec.is_empty() {
            maxevap_inc_e_vec.push(0.0);
            maxevap_theta_vec.push(0.0);
        }
        let n_with_maxevap = maxevap_nuc_n_vec.iter().filter(|&&n| n > 0).count();
        if n_with_maxevap > 0 {
            println!(
                "  GPU: Maxwell/Evaporation χ uploaded for {n_with_maxevap} / {n_nuc} nuclide(s)"
            );
        }

        // ── Pack URR probability tables ──
        let mut urr_e_vec: Vec<f64> = Vec::new();
        let mut urr_cp_vec: Vec<f64> = Vec::new();
        let mut urr_tf_vec: Vec<f64> = Vec::new();
        let mut urr_ef_vec: Vec<f64> = Vec::new();
        let mut urr_ff_vec: Vec<f64> = Vec::new();
        let mut urr_cf_vec: Vec<f64> = Vec::new();
        let mut urr_off_vec = vec![0_i32; n_nuc];
        let mut urr_ne_vec = vec![0_i32; n_nuc];
        let mut urr_nb_vec = vec![0_i32; n_nuc];
        let mut urr_ms_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref urr) = nuc.urr_tables {
                urr_off_vec[nuc_idx] = urr_e_vec.len() as i32;
                urr_ne_vec[nuc_idx] = urr.energies.len() as i32;
                urr_nb_vec[nuc_idx] = urr.n_bands as i32;
                urr_ms_vec[nuc_idx] = if urr.multiply_smooth { 1 } else { 0 };
                urr_e_vec.extend_from_slice(&urr.energies);
                for row in &urr.cum_prob {
                    urr_cp_vec.extend_from_slice(row);
                }
                for row in &urr.total_factor {
                    urr_tf_vec.extend_from_slice(row);
                }
                for row in &urr.elastic_factor {
                    urr_ef_vec.extend_from_slice(row);
                }
                for row in &urr.fission_factor {
                    urr_ff_vec.extend_from_slice(row);
                }
                for row in &urr.capture_factor {
                    urr_cf_vec.extend_from_slice(row);
                }
            }
        }
        // Always have at least one element so device pointers are never null
        if urr_e_vec.is_empty() {
            urr_e_vec.push(0.0);
        }
        if urr_cp_vec.is_empty() {
            urr_cp_vec.push(0.0);
        }
        if urr_tf_vec.is_empty() {
            urr_tf_vec.push(0.0);
        }
        if urr_ef_vec.is_empty() {
            urr_ef_vec.push(0.0);
        }
        if urr_ff_vec.is_empty() {
            urr_ff_vec.push(0.0);
        }
        if urr_cf_vec.is_empty() {
            urr_cf_vec.push(0.0);
        }

        let n_with_dnb = dnb_sizes_vec.iter().filter(|&&s| s > 0).count();
        println!(
            "  GPU: basis={:.1} MB, grids={:.1} MB, nu-bar={} pts, fis_spec={} pts, delayed_nu_bar={} pts ({} nuclides)",
            all_basis_vec.len() as f64 * 8.0 / 1e6,
            all_grids_vec.len() as f64 * 8.0 / 1e6,
            nb_energies_vec.len(),
            fis_eout_vec.len(),
            dnb_energies_vec.len(),
            n_with_dnb,
        );

        Ok(GpuNuclideData {
            // Step-D pointer-array fields are not populated by the
            // legacy path; this function exists only as the byte-
            // equality reference for the assembled-path tests, which
            // don't compare per_nucs / basis_ptrs / coeffs_ptrs.
            per_nucs: Vec::new(),
            basis_ptrs: self.stream.clone_htod(&[0_u64])?,
            coeffs_ptrs: self.stream.clone_htod(&[0_u64])?,
            pw_xs_ptrs: self.stream.clone_htod(&[0_u64])?,
            total_xs_ptrs: self.stream.clone_htod(&[0_u64])?,
            nb_e_ptrs: self.stream.clone_htod(&[0_u64])?,
            nb_v_ptrs: self.stream.clone_htod(&[0_u64])?,
            dnb_e_ptrs: self.stream.clone_htod(&[0_u64])?,
            dnb_v_ptrs: self.stream.clone_htod(&[0_u64])?,
            urr_e_ptrs: self.stream.clone_htod(&[0_u64])?,
            urr_cp_ptrs: self.stream.clone_htod(&[0_u64])?,
            urr_tf_ptrs: self.stream.clone_htod(&[0_u64])?,
            urr_ef_ptrs: self.stream.clone_htod(&[0_u64])?,
            urr_ff_ptrs: self.stream.clone_htod(&[0_u64])?,
            urr_cf_ptrs: self.stream.clone_htod(&[0_u64])?,
            inel_cdf_ptrs: self.stream.clone_htod(&[0_u64])?,
            level_basis_ptrs: self.stream.clone_htod(&[0_u64])?,
            level_coeffs_ptrs: self.stream.clone_htod(&[0_u64])?,
            level_basis_local_off: self.stream.clone_htod(&[0_i32])?,
            level_coeffs_local_off: self.stream.clone_htod(&[0_i32])?,
            ang_e_ptrs: self.stream.clone_htod(&[0_u64])?,
            ang_mu_ptrs: self.stream.clone_htod(&[0_u64])?,
            ang_cdf_ptrs: self.stream.clone_htod(&[0_u64])?,
            lev_ang_e_ptrs: self.stream.clone_htod(&[0_u64])?,
            lev_ang_mu_ptrs: self.stream.clone_htod(&[0_u64])?,
            lev_ang_cdf_ptrs: self.stream.clone_htod(&[0_u64])?,
            ang_dist_local_off: self.stream.clone_htod(&[0_i32])?,
            lev_ang_lev_local_off: self.stream.clone_htod(&[0_i32])?,
            lev_ang_dist_local_off: self.stream.clone_htod(&[0_i32])?,
            all_basis: self.stream.clone_htod(&all_basis_vec)?,
            all_coeffs: self.stream.clone_htod(&all_coeffs_vec)?,
            all_energy_grids: self.stream.clone_htod(&all_grids_vec)?,
            basis_offsets: self.stream.clone_htod(&basis_offsets_vec)?,
            grid_offsets: self.stream.clone_htod(&grid_offsets_vec)?,
            n_energies: self.stream.clone_htod(&n_energies_vec)?,
            has_reaction: self.stream.clone_htod(&has_reaction_vec)?,
            coeffs_offsets: self.stream.clone_htod(&coeffs_offsets_vec)?,
            rank: rank as i32,
            total_xs: self.stream.clone_htod(&total_xs_vec)?,
            total_xs_offsets: self.stream.clone_htod(&total_xs_off_vec)?,
            has_total_xs: self.stream.clone_htod(&has_total_xs_vec)?,
            pointwise_xs: self.stream.clone_htod(&pw_xs_vec)?,
            pw_offsets: self.stream.clone_htod(&pw_off_vec)?,
            has_pw: self.stream.clone_htod(&has_pw_vec)?,
            level_q_values: self.stream.clone_htod(&lev_q_vec)?,
            level_thresholds: self.stream.clone_htod(&lev_thr_vec)?,
            level_offsets: self.stream.clone_htod(&lev_off_vec)?,
            level_counts: self.stream.clone_htod(&lev_cnt_vec)?,
            level_basis: self.stream.clone_htod(&lev_basis_vec)?,
            level_coeffs: self.stream.clone_htod(&lev_coeffs_vec)?,
            level_basis_offsets: self.stream.clone_htod(&lev_basis_off_vec)?,
            level_coeffs_offsets: self.stream.clone_htod(&lev_coeffs_off_vec)?,
            level_has_kernel: self.stream.clone_htod(&lev_has_kernel_vec)?,
            level_mt: self.stream.clone_htod(&lev_mt_vec)?,
            lev_ang_energies: self.stream.clone_htod(&lev_ang_e_vec)?,
            lev_ang_mu: self.stream.clone_htod(&lev_ang_mu_vec)?,
            lev_ang_cdf: self.stream.clone_htod(&lev_ang_cdf_vec)?,
            lev_ang_dist_off: self.stream.clone_htod(&lev_ang_doff_vec)?,
            lev_ang_dist_sz: self.stream.clone_htod(&lev_ang_dsz_vec)?,
            lev_ang_lev_off: self.stream.clone_htod(&lev_ang_loff_vec)?,
            lev_ang_lev_ne: self.stream.clone_htod(&lev_ang_lne_vec)?,
            ang_energies: self.stream.clone_htod(&ang_e_vec)?,
            ang_mu: self.stream.clone_htod(&ang_mu_vec)?,
            ang_cdf: self.stream.clone_htod(&ang_cdf_vec)?,
            ang_dist_offsets: self.stream.clone_htod(&ang_doff_vec)?,
            ang_dist_sizes: self.stream.clone_htod(&ang_dsz_vec)?,
            ang_nuc_offsets: self.stream.clone_htod(&ang_noff_vec)?,
            ang_nuc_n_energies: self.stream.clone_htod(&ang_nne_vec)?,
            ang_is_cm: self.stream.clone_htod(&ang_cm_vec)?,
            nu_bar_energies: self.stream.clone_htod(&nb_energies_vec)?,
            nu_bar_values: self.stream.clone_htod(&nb_values_vec)?,
            nu_bar_offsets: self.stream.clone_htod(&nb_offsets_vec)?,
            nu_bar_sizes: self.stream.clone_htod(&nb_sizes_vec)?,
            delayed_nu_bar_energies: self.stream.clone_htod(&dnb_energies_vec)?,
            delayed_nu_bar_values: self.stream.clone_htod(&dnb_values_vec)?,
            delayed_nu_bar_offsets: self.stream.clone_htod(&dnb_offsets_vec)?,
            delayed_nu_bar_sizes: self.stream.clone_htod(&dnb_sizes_vec)?,
            fis_inc_energies: self.stream.clone_htod(&fis_inc_e_vec)?,
            fis_dist_offsets: self.stream.clone_htod(&fis_dist_off_vec)?,
            fis_dist_sizes: self.stream.clone_htod(&fis_dist_sz_vec)?,
            fis_e_out: self.stream.clone_htod(&fis_eout_vec)?,
            fis_cdf: self.stream.clone_htod(&fis_cdf_vec)?,
            fis_pdf: self.stream.clone_htod(&fis_pdf_vec)?,
            fis_nuc_offsets: self.stream.clone_htod(&fis_nuc_off_vec)?,
            fis_nuc_n_inc: self.stream.clone_htod(&fis_nuc_ninc_vec)?,
            inel91_inc_energies: self.stream.clone_htod(&inel91_inc_e_vec)?,
            inel91_dist_offsets: self.stream.clone_htod(&inel91_dist_off_vec)?,
            inel91_dist_sizes: self.stream.clone_htod(&inel91_dist_sz_vec)?,
            inel91_e_out: self.stream.clone_htod(&inel91_eout_vec)?,
            inel91_cdf: self.stream.clone_htod(&inel91_cdf_vec)?,
            inel91_pdf: self.stream.clone_htod(&inel91_pdf_vec)?,
            inel91_nuc_offsets: self.stream.clone_htod(&inel91_nuc_off_vec)?,
            inel91_nuc_n_inc: self.stream.clone_htod(&inel91_nuc_ninc_vec)?,
            watt_inc_energies: self.stream.clone_htod(&watt_inc_e_vec)?,
            watt_a: self.stream.clone_htod(&watt_a_vec)?,
            watt_b: self.stream.clone_htod(&watt_b_vec)?,
            watt_u: self.stream.clone_htod(&watt_u_vec)?,
            watt_nuc_offsets: self.stream.clone_htod(&watt_nuc_off_vec)?,
            watt_nuc_n: self.stream.clone_htod(&watt_nuc_n_vec)?,
            maxevap_inc_energies: self.stream.clone_htod(&maxevap_inc_e_vec)?,
            maxevap_theta: self.stream.clone_htod(&maxevap_theta_vec)?,
            maxevap_u: self.stream.clone_htod(&maxevap_u_vec)?,
            maxevap_law: self.stream.clone_htod(&maxevap_law_vec)?,
            maxevap_nuc_offsets: self.stream.clone_htod(&maxevap_nuc_off_vec)?,
            maxevap_nuc_n: self.stream.clone_htod(&maxevap_nuc_n_vec)?,
            urr_energies: self.stream.clone_htod(&urr_e_vec)?,
            urr_cum_prob: self.stream.clone_htod(&urr_cp_vec)?,
            urr_total_f: self.stream.clone_htod(&urr_tf_vec)?,
            urr_elastic_f: self.stream.clone_htod(&urr_ef_vec)?,
            urr_fission_f: self.stream.clone_htod(&urr_ff_vec)?,
            urr_capture_f: self.stream.clone_htod(&urr_cf_vec)?,
            urr_offsets: self.stream.clone_htod(&urr_off_vec)?,
            urr_n_energies: self.stream.clone_htod(&urr_ne_vec)?,
            urr_n_bands: self.stream.clone_htod(&urr_nb_vec)?,
            urr_multiply_smooth: self.stream.clone_htod(&urr_ms_vec)?,
            inel_cdf_data: self.stream.clone_htod(&inel_cdf_data_vec)?,
            inel_cdf_off: self.stream.clone_htod(&inel_cdf_off_vec)?,
            inel_cdf_n_e: self.stream.clone_htod(&inel_cdf_n_e_vec)?,
            inel_cdf_n_t: self.stream.clone_htod(&inel_cdf_n_t_vec)?,
            inel_cdf_n_lev: self.stream.clone_htod(&inel_cdf_n_lev_vec)?,
            inel_cdf_log_e_min: self.stream.clone_htod(&inel_cdf_log_e_min_vec)?,
            inel_cdf_log_e_max: self.stream.clone_htod(&inel_cdf_log_e_max_vec)?,
        })
    }

    /// Upload material composition data to GPU.
    pub fn upload_material_data(
        &self,
        materials: &[crate::transport::material::Material],
        nuclide_awrs: &[f64],
        nuclide_nu_bars: &[f64],
    ) -> Result<GpuMaterialData, Box<dyn std::error::Error>> {
        // Single source of truth: `crate::MAX_NUCLIDES_PER_MATERIAL`.
        // The GPU sees the same value via the NVRTC `-DMAX_NUC_PER_MAT`
        // flag wired in `assemble_kernel_source` (gpu_recursive.rs) and
        // the transport_persistent compile site below.
        const MAX_NUC: usize = crate::MAX_NUCLIDES_PER_MATERIAL;
        let n_mat = materials.len();

        let mut n_nuclides = vec![0_i32; n_mat];
        let mut nuc_idx = vec![0_i32; n_mat * MAX_NUC];
        let mut atom_dens = vec![0.0_f64; n_mat * MAX_NUC];

        for (m, mat) in materials.iter().enumerate() {
            if mat.nuclides.len() > MAX_NUC {
                return Err(format!(
                    "upload_material_data: material {} has {} nuclides, GPU stride MAX_NUC = {}",
                    m, mat.nuclides.len(), MAX_NUC
                )
                .into());
            }
            n_nuclides[m] = mat.nuclides.len() as i32;
            for (i, nuc) in mat.nuclides.iter().enumerate() {
                nuc_idx[m * MAX_NUC + i] = nuc.xs_kernel_idx as i32;
                atom_dens[m * MAX_NUC + i] = nuc.atom_density;
            }
        }

        Ok(GpuMaterialData {
            mat_n_nuclides: self.stream.clone_htod(&n_nuclides)?,
            mat_nuclide_idx: self.stream.clone_htod(&nuc_idx)?,
            mat_atom_density: self.stream.clone_htod(&atom_dens)?,
            awr_table: self.stream.clone_htod(nuclide_awrs)?,
            nu_bar_const: self.stream.clone_htod(nuclide_nu_bars)?,
        })
    }

    /// Upload S(α,β) thermal scattering data for one nuclide.
    ///
    /// Convenience wrapper around [`upload_sab_data_multi`] for the
    /// common single-TSL case (PWR H-in-H₂O). `nuc_idx` is the index of
    /// the SAB-bearing nuclide inside the per-run nuclide table; `n_nuc`
    /// is the total nuclide count, used to size the `slot_per_nuc`
    /// lookup table.
    pub fn upload_sab_data(
        &self,
        tsl: &crate::thermal::ThermalScatteringData,
        temp_idx: usize,
        nuc_idx: usize,
        n_nuc: usize,
    ) -> Result<GpuSabData, Box<dyn std::error::Error>> {
        self.upload_sab_data_multi(&[(tsl, temp_idx, nuc_idx)], n_nuc)
    }

    /// Upload multiple S(α,β) libraries simultaneously, one per
    /// nuclide. Each tuple is `(tsl, temp_idx, nuc_idx)`. The kernel
    /// looks up the slot via `slot_per_nuc[nuc_idx]` at every collision
    /// site so different nuclides (H-in-H₂O, D-in-D₂O, C-in-graphite,
    /// …) each get the correct TSL routed by the device.
    ///
    /// Discrete-mode TSLs are currently uploaded as empty slots; the
    /// fast continuous-inelastic path is what the kernel consumes.
    pub fn upload_sab_data_multi(
        &self,
        slots: &[(
            &crate::thermal::ThermalScatteringData,
            usize, /* temp_idx */
            usize, /* nuc_idx */
        )],
        n_nuc: usize,
    ) -> Result<GpuSabData, Box<dyn std::error::Error>> {
        // Concatenated flat arrays.
        let mut inc_e_flat: Vec<f64> = Vec::new();
        let mut xs_flat: Vec<f64> = Vec::new();
        let mut eout_offsets_flat: Vec<i32> = Vec::new();
        let mut eout_sizes_flat: Vec<i32> = Vec::new();
        let mut e_out_flat: Vec<f64> = Vec::new();
        let mut cdf_e_flat: Vec<f64> = Vec::new();
        let mut pdf_e_flat: Vec<f64> = Vec::new();
        let mut mu_offsets_flat: Vec<i32> = Vec::new();
        let mut mu_sizes_flat: Vec<i32> = Vec::new();
        let mut mu_flat: Vec<f64> = Vec::new();
        let mut cdf_mu_flat: Vec<f64> = Vec::new();

        // Per-slot metadata.
        let mut slot_inc_e_off: Vec<i32> = Vec::new();
        let mut slot_n_inc: Vec<i32> = Vec::new();
        let mut slot_eout_table_off: Vec<i32> = Vec::new();
        let mut slot_mu_table_off: Vec<i32> = Vec::new();
        let mut slot_emax: Vec<f64> = Vec::new();

        // Per-nuclide → slot lookup. Default -1.
        let mut slot_per_nuc: Vec<i32> = vec![-1; n_nuc.max(1)];

        for (tsl, temp_idx, nuc_idx) in slots.iter().copied() {
            if nuc_idx >= n_nuc {
                return Err(format!(
                    "upload_sab_data_multi: nuc_idx {nuc_idx} >= n_nuc {n_nuc}"
                )
                .into());
            }
            if slot_per_nuc[nuc_idx] >= 0 {
                return Err(format!(
                    "upload_sab_data_multi: nuc_idx {nuc_idx} bound to multiple TSLs"
                )
                .into());
            }
            let slot_id = slot_inc_e_off.len() as i32;
            slot_per_nuc[nuc_idx] = slot_id;

            let inel = &tsl.inelastic[temp_idx];
            match &inel.dist {
                crate::thermal::InelasticDist::Continuous(c) => {
                    // Inc-energy block (and parallel xs).
                    let inc_e_off = inc_e_flat.len() as i32;
                    let n_inc_this = inel.energy.len() as i32;
                    inc_e_flat.extend_from_slice(&inel.energy);
                    xs_flat.extend_from_slice(&inel.xs);

                    // E_out block, with per-inc-energy table offsets.
                    let eout_table_off = eout_offsets_flat.len() as i32;
                    let e_out_base = e_out_flat.len() as i32;
                    for i in 0..c.n_inc {
                        let start = c.offsets[i];
                        let end = if i + 1 < c.offsets.len() {
                            c.offsets[i + 1]
                        } else {
                            c.e_out.len()
                        };
                        eout_offsets_flat.push(e_out_base + start as i32);
                        eout_sizes_flat.push((end - start) as i32);
                    }
                    e_out_flat.extend_from_slice(&c.e_out);
                    cdf_e_flat.extend_from_slice(&c.cdf_e);
                    pdf_e_flat.extend_from_slice(&c.pdf_e);

                    // Mu block, with per-eout-bin table offsets.
                    let mu_table_off = mu_offsets_flat.len() as i32;
                    let mu_base = mu_flat.len() as i32;
                    for i in 0..c.mu_offsets.len() {
                        let start = c.mu_offsets[i];
                        let end = if i + 1 < c.mu_offsets.len() {
                            c.mu_offsets[i + 1]
                        } else {
                            c.mu.len()
                        };
                        mu_offsets_flat.push(mu_base + start as i32);
                        mu_sizes_flat.push((end - start) as i32);
                    }
                    mu_flat.extend_from_slice(&c.mu);
                    cdf_mu_flat.extend_from_slice(&c.cdf_mu);

                    slot_inc_e_off.push(inc_e_off);
                    slot_n_inc.push(n_inc_this);
                    slot_eout_table_off.push(eout_table_off);
                    slot_mu_table_off.push(mu_table_off);
                    slot_emax.push(tsl.energy_max);

                    println!(
                        "  GPU S(α,β) slot {slot_id} (nuc {nuc_idx}): {n_inc_this} inc \
                         energies, {} E_out pts, {} mu pts",
                        c.e_out.len(),
                        c.mu.len()
                    );
                }
                crate::thermal::InelasticDist::Discrete(_) => {
                    println!(
                        "  GPU S(α,β) slot {slot_id} (nuc {nuc_idx}): discrete mode — \
                         empty placeholder"
                    );
                    let inc_e_off = inc_e_flat.len() as i32;
                    inc_e_flat.push(0.0);
                    xs_flat.push(0.0);
                    let eout_table_off = eout_offsets_flat.len() as i32;
                    eout_offsets_flat.push(0);
                    eout_sizes_flat.push(0);
                    let mu_table_off = mu_offsets_flat.len() as i32;
                    mu_offsets_flat.push(0);
                    mu_sizes_flat.push(0);

                    slot_inc_e_off.push(inc_e_off);
                    slot_n_inc.push(0);
                    slot_eout_table_off.push(eout_table_off);
                    slot_mu_table_off.push(mu_table_off);
                    slot_emax.push(0.0);
                }
            }
        }

        // Ensure no flat array is empty (cudarc rejects zero-sized
        // copies). The kernel never reads these padding bytes because
        // n_slots == 0 short-circuits the SAB branch.
        if inc_e_flat.is_empty() {
            inc_e_flat.push(0.0);
            xs_flat.push(0.0);
        }
        if eout_offsets_flat.is_empty() {
            eout_offsets_flat.push(0);
            eout_sizes_flat.push(0);
        }
        if e_out_flat.is_empty() {
            e_out_flat.push(0.0);
            cdf_e_flat.push(0.0);
            pdf_e_flat.push(0.0);
        }
        if mu_offsets_flat.is_empty() {
            mu_offsets_flat.push(0);
            mu_sizes_flat.push(0);
        }
        if mu_flat.is_empty() {
            mu_flat.push(0.0);
            cdf_mu_flat.push(0.0);
        }
        if slot_inc_e_off.is_empty() {
            slot_inc_e_off.push(0);
            slot_n_inc.push(0);
            slot_eout_table_off.push(0);
            slot_mu_table_off.push(0);
            slot_emax.push(0.0);
        }

        let n_slots = slots.len() as i32;
        // Legacy mirrors for the single-slot fast path in transport.cu.
        let (legacy_n_inc, legacy_emax) = if n_slots > 0 {
            (slot_n_inc[0], slot_emax[0])
        } else {
            (0, 0.0)
        };

        Ok(GpuSabData {
            inc_energies: self.stream.clone_htod(&inc_e_flat)?,
            eout_offsets: self.stream.clone_htod(&eout_offsets_flat)?,
            eout_sizes: self.stream.clone_htod(&eout_sizes_flat)?,
            e_out: self.stream.clone_htod(&e_out_flat)?,
            cdf_e: self.stream.clone_htod(&cdf_e_flat)?,
            pdf_e: self.stream.clone_htod(&pdf_e_flat)?,
            mu_offsets: self.stream.clone_htod(&mu_offsets_flat)?,
            mu_sizes: self.stream.clone_htod(&mu_sizes_flat)?,
            mu: self.stream.clone_htod(&mu_flat)?,
            cdf_mu: self.stream.clone_htod(&cdf_mu_flat)?,
            xs: self.stream.clone_htod(&xs_flat)?,

            n_slots,
            slot_per_nuc: self.stream.clone_htod(&slot_per_nuc)?,
            slot_inc_e_off: self.stream.clone_htod(&slot_inc_e_off)?,
            slot_n_inc: self.stream.clone_htod(&slot_n_inc)?,
            slot_eout_table_off: self.stream.clone_htod(&slot_eout_table_off)?,
            slot_mu_table_off: self.stream.clone_htod(&slot_mu_table_off)?,
            slot_emax: self.stream.clone_htod(&slot_emax)?,

            n_inc: legacy_n_inc,
            energy_max: legacy_emax,
        })
    }

    /// Create an empty S(α,β) placeholder. `n_nuc` is needed so the
    /// per-nuclide lookup table is sized correctly for the kernel.
    pub fn upload_sab_data_empty(
        &self,
        n_nuc: usize,
    ) -> Result<GpuSabData, Box<dyn std::error::Error>> {
        self.upload_sab_data_multi(&[], n_nuc)
    }

    /// Upload per-nuclide Windowed-Multipole data to the GPU. `wmps[i] = None`
    /// means nuclide `i` stays on the SVD/pointwise path in the transport
    /// kernel (parallels `HybridSvdWmpXsProvider` on the CPU).
    pub fn upload_wmp_data(
        &self,
        wmps: &[Option<(Arc<crate::wmp::WindowedMultipole>, f64)>],
    ) -> Result<GpuWmpData, Box<dyn std::error::Error>> {
        let n_nuc = wmps.len().max(1);
        let mut has_vec = vec![0_i32; n_nuc];
        let mut e_min_vec = vec![0.0_f64; n_nuc];
        let mut e_max_vec = vec![0.0_f64; n_nuc];
        let mut spacing_vec = vec![0.0_f64; n_nuc];
        let mut sqrt_awr_vec = vec![0.0_f64; n_nuc];
        let mut t_kelvin_vec = vec![0.0_f64; n_nuc];
        let mut fit_order_vec = vec![0_i32; n_nuc];
        let mut n_windows_vec = vec![0_i32; n_nuc];
        let mut fissionable_vec = vec![0_i32; n_nuc];

        let mut poles_vec: Vec<f64> = Vec::new();
        let mut pole_off_vec = vec![0_i32; n_nuc];
        let mut windows_vec: Vec<i32> = Vec::new();
        let mut win_off_vec = vec![0_i32; n_nuc];
        let mut broaden_vec: Vec<i8> = Vec::new();
        let mut bro_off_vec = vec![0_i32; n_nuc];
        let mut curvefit_vec: Vec<f64> = Vec::new();
        let mut cf_off_vec = vec![0_i32; n_nuc];

        let mut covered = 0usize;
        for (i, wmp_opt) in wmps.iter().enumerate() {
            if let Some((wmp, t_k)) = wmp_opt {
                has_vec[i] = 1;
                e_min_vec[i] = wmp.e_min;
                e_max_vec[i] = wmp.e_max;
                spacing_vec[i] = wmp.spacing;
                sqrt_awr_vec[i] = wmp.sqrt_awr;
                t_kelvin_vec[i] = *t_k;
                fit_order_vec[i] = wmp.fit_order as i32;
                n_windows_vec[i] = wmp.n_windows as i32;
                fissionable_vec[i] = if wmp.fissionable { 1 } else { 0 };

                // Poles: flattened to (re, im) pairs; offset in complex units
                // so `double2*` pointer arithmetic in the kernel is straight.
                pole_off_vec[i] = (poles_vec.len() / 2) as i32;
                for c in &wmp.poles {
                    poles_vec.push(c.re);
                    poles_vec.push(c.im);
                }

                win_off_vec[i] = windows_vec.len() as i32;
                windows_vec.extend_from_slice(&wmp.windows);

                bro_off_vec[i] = broaden_vec.len() as i32;
                broaden_vec.extend(wmp.broaden_poly.iter().map(|&b| b as i8));

                cf_off_vec[i] = curvefit_vec.len() as i32;
                curvefit_vec.extend_from_slice(&wmp.curvefit);

                covered += 1;
            }
        }

        // Keep all device buffers non-empty so device pointers are valid.
        if poles_vec.is_empty() {
            poles_vec.extend_from_slice(&[0.0_f64, 0.0_f64]);
        }
        if windows_vec.is_empty() {
            windows_vec.push(0);
            windows_vec.push(0);
        }
        if broaden_vec.is_empty() {
            broaden_vec.push(0);
        }
        if curvefit_vec.is_empty() {
            curvefit_vec.push(0.0);
        }

        println!(
            "  GPU: WMP payload = {:.1} KB ({} / {} nuclides covered)",
            (poles_vec.len() * 8
                + windows_vec.len() * 4
                + broaden_vec.len()
                + curvefit_vec.len() * 8) as f64
                / 1024.0,
            covered,
            wmps.len()
        );

        Ok(GpuWmpData {
            has: self.stream.clone_htod(&has_vec)?,
            e_min: self.stream.clone_htod(&e_min_vec)?,
            e_max: self.stream.clone_htod(&e_max_vec)?,
            spacing: self.stream.clone_htod(&spacing_vec)?,
            sqrt_awr: self.stream.clone_htod(&sqrt_awr_vec)?,
            t_kelvin: self.stream.clone_htod(&t_kelvin_vec)?,
            fit_order: self.stream.clone_htod(&fit_order_vec)?,
            n_windows: self.stream.clone_htod(&n_windows_vec)?,
            fissionable: self.stream.clone_htod(&fissionable_vec)?,
            poles: self.stream.clone_htod(&poles_vec)?,
            pole_offsets: self.stream.clone_htod(&pole_off_vec)?,
            windows: self.stream.clone_htod(&windows_vec)?,
            window_offsets: self.stream.clone_htod(&win_off_vec)?,
            broaden: self.stream.clone_htod(&broaden_vec)?,
            broaden_offsets: self.stream.clone_htod(&bro_off_vec)?,
            curvefit: self.stream.clone_htod(&curvefit_vec)?,
            curvefit_offsets: self.stream.clone_htod(&cf_off_vec)?,
        })
    }

    /// Create an empty WMP placeholder (no nuclide covered). Used by the
    /// SVD-only and pointwise paths to keep the kernel ABI uniform.
    pub fn upload_wmp_data_empty(
        &self,
        n_nuc: usize,
    ) -> Result<GpuWmpData, Box<dyn std::error::Error>> {
        let wmps: Vec<Option<(Arc<crate::wmp::WindowedMultipole>, f64)>> =
            (0..n_nuc).map(|_| None).collect();
        self.upload_wmp_data(&wmps)
    }

    /// Run one batch of transport on GPU.
    ///
    /// geom_type: 0=PWR pin cell, 1=Godiva bare sphere.
    pub fn run_batch(
        &self,
        source_bank: &[(f64, f64, f64, f64)],
        batch: u32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        wmp_data: &GpuWmpData,
        max_steps: u32,
        geom_type: i32,
    ) -> Result<GpuBatchResult, Box<dyn std::error::Error>> {
        let n = source_bank.len();
        let n_i32 = n as i32;
        let grid_full = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg_full = LaunchConfig {
            grid_dim: (grid_full, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        // Unpack source bank into SoA
        let mut sx = Vec::with_capacity(n);
        let mut sy = Vec::with_capacity(n);
        let mut sz = Vec::with_capacity(n);
        let mut se = Vec::with_capacity(n);
        for &(x, y, z, e) in source_bank {
            sx.push(x);
            sy.push(y);
            sz.push(z);
            se.push(e);
        }

        let d_src_x = self.stream.clone_htod(&sx)?;
        let d_src_y = self.stream.clone_htod(&sy)?;
        let d_src_z = self.stream.clone_htod(&sz)?;
        let d_src_e = self.stream.clone_htod(&se)?;

        // Pre-allocate all particle state arrays (reused across steps)
        let mut d_pos_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_energy: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_cell: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_alive: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_rng_state: CudaSlice<u64> = self.stream.alloc_zeros(n)?;
        let mut d_rng_inc: CudaSlice<u64> = self.stream.alloc_zeros(n)?;

        // Compaction + sort buffers
        let mut d_compact_idx: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_compact_idx_sorted: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let n_bins = 256;
        // Fission bank
        let max_fission = (n * 3) as i32;
        let mut d_fis_x: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_y: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_z: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_e: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_w: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        // Counters
        let mut d_cnt_coll: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_fis: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_leak: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_surf: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        // Initialize source
        let batch_seed = batch as u64 * 1_000_000;
        unsafe {
            self.stream
                .launch_builder(&self.k_init_source)
                .arg(&mut d_pos_x)
                .arg(&mut d_pos_y)
                .arg(&mut d_pos_z)
                .arg(&mut d_dir_x)
                .arg(&mut d_dir_y)
                .arg(&mut d_dir_z)
                .arg(&mut d_energy)
                .arg(&mut d_cell)
                .arg(&mut d_alive)
                .arg(&d_src_x)
                .arg(&d_src_y)
                .arg(&d_src_z)
                .arg(&d_src_e)
                .arg(&n_i32)
                .arg(&batch_seed)
                .arg(&mut d_rng_state)
                .arg(&mut d_rng_inc)
                .arg(&geom_type)
                .launch(cfg_full)?;
        }

        // Build packed TransportParams buffer (N_PARAMS u64 values)
        // Extract raw device pointers from each CudaSlice
        macro_rules! dptr {
            ($slice:expr) => {{
                let (ptr, _guard) = $slice.device_ptr(&self.stream);
                ptr // CUdeviceptr = u64
            }};
        }
        let params_vec: Vec<u64> = vec![
            dptr!(&nuc_data.all_basis),            //  0 P_BASIS
            dptr!(&nuc_data.all_coeffs),           //  1 P_COEFFS
            dptr!(&nuc_data.all_energy_grids),     //  2 P_ENERGY_GRIDS
            dptr!(&nuc_data.basis_offsets),        //  3 P_BASIS_OFFSETS
            dptr!(&nuc_data.grid_offsets),         //  4 P_GRID_OFFSETS
            dptr!(&nuc_data.n_energies),           //  5 P_N_ENERGIES
            dptr!(&nuc_data.has_reaction),         //  6 P_HAS_REACTION
            dptr!(&nuc_data.coeffs_offsets),       //  7 P_COEFFS_OFFSETS
            nuc_data.rank as u64,                  //  8 P_RANK
            dptr!(&mat_data.mat_n_nuclides),       //  9 P_MAT_N_NUC
            dptr!(&mat_data.mat_nuclide_idx),      // 10 P_MAT_NUC_IDX
            dptr!(&mat_data.mat_atom_density),     // 11 P_MAT_ATOM_DENS
            dptr!(&mat_data.awr_table),            // 12 P_AWR_TABLE
            dptr!(&mat_data.nu_bar_const),         // 13 P_NU_BAR_CONST
            dptr!(&nuc_data.nu_bar_energies),      // 14 P_NB_ENERGIES
            dptr!(&nuc_data.nu_bar_values),        // 15 P_NB_VALUES
            dptr!(&nuc_data.nu_bar_offsets),       // 16 P_NB_OFFSETS
            dptr!(&nuc_data.nu_bar_sizes),         // 17 P_NB_SIZES
            dptr!(&nuc_data.fis_inc_energies),     // 18 P_FIS_INC_E
            dptr!(&nuc_data.fis_dist_offsets),     // 19 P_FIS_DIST_OFF
            dptr!(&nuc_data.fis_dist_sizes),       // 20 P_FIS_DIST_SZ
            dptr!(&nuc_data.fis_e_out),            // 21 P_FIS_E_OUT
            dptr!(&nuc_data.fis_cdf),              // 22 P_FIS_CDF
            dptr!(&nuc_data.fis_nuc_offsets),      // 23 P_FIS_NUC_OFF
            dptr!(&nuc_data.fis_nuc_n_inc),        // 24 P_FIS_NUC_NINC
            dptr!(&nuc_data.level_q_values),       // 25 P_LEVEL_Q
            dptr!(&nuc_data.level_thresholds),     // 26 P_LEVEL_THR
            dptr!(&nuc_data.level_offsets),        // 27 P_LEVEL_OFFSETS
            dptr!(&nuc_data.level_counts),         // 28 P_LEVEL_COUNTS
            dptr!(&nuc_data.level_basis),          // 29 P_LEVEL_BASIS
            dptr!(&nuc_data.level_coeffs),         // 30 P_LEVEL_COEFFS
            dptr!(&nuc_data.level_basis_offsets),  // 31 P_LEVEL_BOFF
            dptr!(&nuc_data.level_coeffs_offsets), // 32 P_LEVEL_COFF
            dptr!(&nuc_data.level_has_kernel),     // 33 P_LEVEL_HAS_K
            dptr!(&nuc_data.level_mt),             // 34 P_LEVEL_MT
            dptr!(&nuc_data.ang_energies),         // 35 P_ANG_ENERGIES
            dptr!(&nuc_data.ang_mu),               // 36 P_ANG_MU
            dptr!(&nuc_data.ang_cdf),              // 37 P_ANG_CDF
            dptr!(&nuc_data.ang_dist_offsets),     // 38 P_ANG_DIST_OFF
            dptr!(&nuc_data.ang_dist_sizes),       // 39 P_ANG_DIST_SZ
            dptr!(&nuc_data.ang_nuc_offsets),      // 40 P_ANG_NUC_OFF
            dptr!(&nuc_data.ang_nuc_n_energies),   // 41 P_ANG_NUC_NE
            dptr!(&nuc_data.ang_is_cm),            // 42 P_ANG_IS_CM
            dptr!(&sab_data.inc_energies),         // 43 P_SAB_INC_E
            sab_data.n_inc as u64,                 // 44 P_SAB_N_INC
            dptr!(&sab_data.eout_offsets),         // 45 P_SAB_EOUT_OFF
            dptr!(&sab_data.eout_sizes),           // 46 P_SAB_EOUT_SZ
            dptr!(&sab_data.e_out),                // 47 P_SAB_E_OUT
            dptr!(&sab_data.cdf_e),                // 48 P_SAB_CDF_E
            dptr!(&sab_data.mu_offsets),           // 49 P_SAB_MU_OFF
            dptr!(&sab_data.mu_sizes),             // 50 P_SAB_MU_SZ
            dptr!(&sab_data.mu),                   // 51 P_SAB_MU
            dptr!(&sab_data.cdf_mu),               // 52 P_SAB_CDF_MU
            dptr!(&sab_data.xs),                   // 53 P_SAB_XS
            sab_data.energy_max.to_bits(),         // 54 P_SAB_EMAX (f64 as bits)
            dptr!(&sab_data.pdf_e),                // 55 P_SAB_PDF_E
            dptr!(&nuc_data.urr_energies),         // 56 P_URR_ENERGIES
            dptr!(&nuc_data.urr_cum_prob),         // 57 P_URR_CUM_PROB
            dptr!(&nuc_data.urr_total_f),          // 58 P_URR_TOTAL_F
            dptr!(&nuc_data.urr_elastic_f),        // 59 P_URR_ELASTIC_F
            dptr!(&nuc_data.urr_fission_f),        // 60 P_URR_FISSION_F
            dptr!(&nuc_data.urr_capture_f),        // 61 P_URR_CAPTURE_F
            dptr!(&nuc_data.urr_offsets),          // 62 P_URR_OFFSETS
            dptr!(&nuc_data.urr_n_energies),       // 63 P_URR_N_ENERGIES
            dptr!(&nuc_data.urr_n_bands),          // 64 P_URR_N_BANDS
            dptr!(&nuc_data.urr_multiply_smooth),  // 65 P_URR_MULT_SM
            geom_type as u64,                      // 66 P_GEOM_TYPE
            dptr!(&nuc_data.total_xs),             // 67 P_TOTAL_XS
            dptr!(&nuc_data.total_xs_offsets),     // 68 P_TOTAL_XS_OFF
            dptr!(&nuc_data.has_total_xs),         // 69 P_HAS_TOTAL_XS
            dptr!(&nuc_data.pointwise_xs),         // 70 P_PW_XS
            dptr!(&nuc_data.pw_offsets),           // 71 P_PW_OFF
            dptr!(&nuc_data.has_pw),               // 72 P_HAS_PW
            dptr!(&wmp_data.has),                  // 73 P_WMP_HAS
            dptr!(&wmp_data.e_min),                // 74 P_WMP_E_MIN
            dptr!(&wmp_data.e_max),                // 75 P_WMP_E_MAX
            dptr!(&wmp_data.spacing),              // 76 P_WMP_SPACING
            dptr!(&wmp_data.sqrt_awr),             // 77 P_WMP_SQRT_AWR
            dptr!(&wmp_data.t_kelvin),             // 78 P_WMP_T_KELVIN
            dptr!(&wmp_data.fit_order),            // 79 P_WMP_FIT_ORDER
            dptr!(&wmp_data.n_windows),            // 80 P_WMP_N_WINDOWS
            dptr!(&wmp_data.fissionable),          // 81 P_WMP_FISSIONABLE
            dptr!(&wmp_data.poles),                // 82 P_WMP_POLES
            dptr!(&wmp_data.pole_offsets),         // 83 P_WMP_POLE_OFF
            dptr!(&wmp_data.windows),              // 84 P_WMP_WINDOWS
            dptr!(&wmp_data.window_offsets),       // 85 P_WMP_WIN_OFF
            dptr!(&wmp_data.broaden),              // 86 P_WMP_BROADEN
            dptr!(&wmp_data.broaden_offsets),      // 87 P_WMP_BROADEN_OFF
            dptr!(&wmp_data.curvefit),             // 88 P_WMP_CURVEFIT
            dptr!(&wmp_data.curvefit_offsets),     // 89 P_WMP_CF_OFF
            dptr!(&nuc_data.lev_ang_energies),     // 90 P_LEV_ANG_ENERGIES
            dptr!(&nuc_data.lev_ang_mu),           // 91 P_LEV_ANG_MU
            dptr!(&nuc_data.lev_ang_cdf),          // 92 P_LEV_ANG_CDF
            dptr!(&nuc_data.lev_ang_dist_off),     // 93 P_LEV_ANG_DIST_OFF
            dptr!(&nuc_data.lev_ang_dist_sz),      // 94 P_LEV_ANG_DIST_SZ
            dptr!(&nuc_data.lev_ang_lev_off),      // 95 P_LEV_ANG_LEV_OFF
            dptr!(&nuc_data.lev_ang_lev_ne),       // 96 P_LEV_ANG_LEV_NE
            dptr!(&nuc_data.inel_cdf_data),        // 97 P_INEL_CDF_DATA
            dptr!(&nuc_data.inel_cdf_off),         // 98 P_INEL_CDF_OFF
            dptr!(&nuc_data.inel_cdf_n_e),         // 99 P_INEL_CDF_N_E
            dptr!(&nuc_data.inel_cdf_n_t),         //100 P_INEL_CDF_N_T
            dptr!(&nuc_data.inel_cdf_n_lev),       //101 P_INEL_CDF_N_LEV
            dptr!(&nuc_data.inel_cdf_log_e_min),   //102 P_INEL_CDF_LOG_EMIN
            dptr!(&nuc_data.inel_cdf_log_e_max),   //103 P_INEL_CDF_LOG_EMAX
            dptr!(&nuc_data.watt_inc_energies),    //104 P_WATT_INC_E
            dptr!(&nuc_data.watt_a),               //105 P_WATT_A
            dptr!(&nuc_data.watt_b),               //106 P_WATT_B
            dptr!(&nuc_data.watt_u),               //107 P_WATT_U
            dptr!(&nuc_data.watt_nuc_offsets),     //108 P_WATT_NUC_OFF
            dptr!(&nuc_data.watt_nuc_n),           //109 P_WATT_NUC_N
            dptr!(&nuc_data.delayed_nu_bar_energies), //110 P_DNB_ENERGIES
            dptr!(&nuc_data.delayed_nu_bar_values),   //111 P_DNB_VALUES
            dptr!(&nuc_data.delayed_nu_bar_offsets),  //112 P_DNB_OFFSETS
            dptr!(&nuc_data.delayed_nu_bar_sizes),    //113 P_DNB_SIZES
            dptr!(&nuc_data.fis_pdf),                 //114 P_FIS_PDF
            dptr!(&nuc_data.inel91_inc_energies),     //115 P_INEL91_INC_E
            dptr!(&nuc_data.inel91_dist_offsets),     //116 P_INEL91_DIST_OFF
            dptr!(&nuc_data.inel91_dist_sizes),       //117 P_INEL91_DIST_SZ
            dptr!(&nuc_data.inel91_e_out),            //118 P_INEL91_E_OUT
            dptr!(&nuc_data.inel91_cdf),              //119 P_INEL91_CDF
            dptr!(&nuc_data.inel91_pdf),              //120 P_INEL91_PDF
            dptr!(&nuc_data.inel91_nuc_offsets),      //121 P_INEL91_NUC_OFF
            dptr!(&nuc_data.inel91_nuc_n_inc),        //122 P_INEL91_NUC_NINC
            sab_data.n_slots as u64,                  //123 P_SAB_N_SLOTS
            dptr!(&sab_data.slot_per_nuc),            //124 P_SAB_SLOT_PER_NUC
            dptr!(&sab_data.slot_inc_e_off),          //125 P_SAB_SLOT_INC_E_OFF
            dptr!(&sab_data.slot_n_inc),              //126 P_SAB_SLOT_N_INC
            dptr!(&sab_data.slot_eout_table_off),     //127 P_SAB_SLOT_EOUT_TABLE_OFF
            dptr!(&sab_data.slot_mu_table_off),       //128 P_SAB_SLOT_MU_TABLE_OFF
            dptr!(&sab_data.slot_emax),               //129 P_SAB_SLOT_EMAX
            dptr!(&nuc_data.maxevap_inc_energies),    //130 P_MAXEVAP_INC_E
            dptr!(&nuc_data.maxevap_theta),           //131 P_MAXEVAP_THETA
            dptr!(&nuc_data.maxevap_u),               //132 P_MAXEVAP_U
            dptr!(&nuc_data.maxevap_law),             //133 P_MAXEVAP_LAW
            dptr!(&nuc_data.maxevap_nuc_offsets),     //134 P_MAXEVAP_NUC_OFF
            dptr!(&nuc_data.maxevap_nuc_n),           //135 P_MAXEVAP_NUC_N
        ];
        assert_eq!(params_vec.len(), N_PARAMS);
        let d_params = self.stream.clone_htod(&params_vec)?;

        let mut n_alive = n as i32;
        let compact_interval = 10; // Re-compact every N steps

        let mut step = 0_u32;
        while step < max_steps && n_alive > 0 {
            // 1. Compact: build dense list of alive particle indices
            let mut d_compact_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
            let compact_grid = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            let compact_cfg = LaunchConfig {
                grid_dim: (compact_grid, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.k_compact_alive)
                    .arg(&d_alive)
                    .arg(&n_i32)
                    .arg(&mut d_compact_idx)
                    .arg(&mut d_compact_count)
                    .launch(compact_cfg)?;
            }
            let count = self.stream.clone_dtoh(&d_compact_count)?;
            n_alive = count[0];
            if n_alive <= 0 {
                break;
            }

            // 2. Energy sort: bin count → prefix sum → scatter
            let alive_grid = (n_alive as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            let alive_cfg = LaunchConfig {
                grid_dim: (alive_grid, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };

            // 2a. Count particles per energy bin
            let mut d_bin_counts: CudaSlice<i32> = self.stream.alloc_zeros(n_bins)?;
            unsafe {
                self.stream
                    .launch_builder(&self.k_energy_bin_count)
                    .arg(&d_energy)
                    .arg(&d_compact_idx)
                    .arg(&n_alive)
                    .arg(&mut d_bin_counts)
                    .launch(alive_cfg)?;
            }

            // 2b. Prefix sum on CPU (256 ints — trivial)
            let counts = self.stream.clone_dtoh(&d_bin_counts)?;
            let mut offsets = vec![0_i32; n_bins];
            let mut running = 0_i32;
            for i in 0..n_bins {
                offsets[i] = running;
                running += counts[i];
            }
            let d_bin_offsets = self.stream.clone_htod(&offsets)?;

            // 2c. Scatter compact indices into energy-sorted order
            unsafe {
                self.stream
                    .launch_builder(&self.k_energy_bin_scatter)
                    .arg(&d_energy)
                    .arg(&d_compact_idx)
                    .arg(&n_alive)
                    .arg(&mut d_compact_idx_sorted)
                    .arg(&d_bin_offsets)
                    .launch(alive_cfg)?;
            }

            // Swap: sorted becomes the active compact index
            std::mem::swap(&mut d_compact_idx, &mut d_compact_idx_sorted);

            // Launch persistent kernel: N steps in one kernel call
            let alive_grid = (n_alive as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            let alive_cfg = LaunchConfig {
                grid_dim: (alive_grid, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };
            let steps_this_launch = compact_interval as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k_transport_persistent)
                    .arg(&d_params)
                    .arg(&d_compact_idx)
                    .arg(&n_alive)
                    .arg(&mut d_pos_x)
                    .arg(&mut d_pos_y)
                    .arg(&mut d_pos_z)
                    .arg(&mut d_dir_x)
                    .arg(&mut d_dir_y)
                    .arg(&mut d_dir_z)
                    .arg(&mut d_energy)
                    .arg(&mut d_cell)
                    .arg(&mut d_alive)
                    .arg(&mut d_rng_state)
                    .arg(&mut d_rng_inc)
                    .arg(&mut d_fis_x)
                    .arg(&mut d_fis_y)
                    .arg(&mut d_fis_z)
                    .arg(&mut d_fis_e)
                    .arg(&mut d_fis_w)
                    .arg(&mut d_fis_count)
                    .arg(&max_fission)
                    .arg(&mut d_cnt_coll)
                    .arg(&mut d_cnt_fis)
                    .arg(&mut d_cnt_leak)
                    .arg(&mut d_cnt_surf)
                    .arg(&steps_this_launch)
                    .launch(alive_cfg)?;
            }

            step += compact_interval; // persistent kernel did N steps
        }

        // Download results
        let fis_count = self.stream.clone_dtoh(&d_fis_count)?[0] as usize;
        let cnt_coll = self.stream.clone_dtoh(&d_cnt_coll)?[0] as u32;
        let cnt_fis = self.stream.clone_dtoh(&d_cnt_fis)?[0] as u32;
        let cnt_leak = self.stream.clone_dtoh(&d_cnt_leak)?[0] as u32;
        let cnt_surf = self.stream.clone_dtoh(&d_cnt_surf)?[0] as u32;

        let fis_count_clamped = fis_count.min(max_fission as usize);
        let fission_bank = if fis_count_clamped > 0 {
            let fx = self.stream.clone_dtoh(&d_fis_x)?;
            let fy = self.stream.clone_dtoh(&d_fis_y)?;
            let fz = self.stream.clone_dtoh(&d_fis_z)?;
            let fe = self.stream.clone_dtoh(&d_fis_e)?;
            (0..fis_count_clamped)
                .map(|i| (fx[i], fy[i], fz[i], fe[i]))
                .collect()
        } else {
            vec![]
        };

        let k_eff = fission_bank.len() as f64 / n as f64;

        Ok(GpuBatchResult {
            k_eff,
            collisions: cnt_coll,
            fissions: cnt_fis,
            leakage: cnt_leak,
            surface_crossings: cnt_surf,
            fission_bank,
        })
    }

    pub fn run_debug_trace(
        &self,
        source_bank: &[(f64, f64, f64, f64)],
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        wmp_data: &GpuWmpData,
        max_steps: u32,
        geom_type: i32,
    ) -> Result<GpuTraceResult, Box<dyn std::error::Error>> {
        use cudarc::driver::LaunchConfig;
        use cudarc::nvrtc;

        let n = source_bank.len();
        let trace_cols = 17_usize;
        let trace_size = n * max_steps as usize * trace_cols;

        let sx: Vec<f64> = source_bank.iter().map(|s| s.0).collect();
        let sy: Vec<f64> = source_bank.iter().map(|s| s.1).collect();
        let sz: Vec<f64> = source_bank.iter().map(|s| s.2).collect();
        let se: Vec<f64> = source_bank.iter().map(|s| s.3).collect();
        let d_sx = self.stream.clone_htod(&sx)?;
        let d_sy = self.stream.clone_htod(&sy)?;
        let d_sz = self.stream.clone_htod(&sz)?;
        let d_se = self.stream.clone_htod(&se)?;

        let mut d_pos_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_energy: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_cell: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_alive: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_rng_state: CudaSlice<u64> = self.stream.alloc_zeros(n)?;
        let mut d_rng_inc: CudaSlice<u64> = self.stream.alloc_zeros(n)?;

        let max_fis = n * 3;
        let mut d_fis_x: CudaSlice<f64> = self.stream.alloc_zeros(max_fis)?;
        let mut d_fis_y: CudaSlice<f64> = self.stream.alloc_zeros(max_fis)?;
        let mut d_fis_z: CudaSlice<f64> = self.stream.alloc_zeros(max_fis)?;
        let mut d_fis_e: CudaSlice<f64> = self.stream.alloc_zeros(max_fis)?;
        let mut d_fis_w: CudaSlice<f64> = self.stream.alloc_zeros(max_fis)?;
        let mut d_fis_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        let mut d_trace: CudaSlice<f64> = self.stream.alloc_zeros(trace_size)?;
        let mut d_step_counts: CudaSlice<i32> = self.stream.alloc_zeros(n)?;

        let ptx = nvrtc::compile_ptx_with_opts(TRANSPORT_KERNELS, transport_kernel_options())?;
        let module = self._ctx.load_module(ptx)?;
        let k_init = module.load_function("init_source")?;
        let k_trace = module.load_function("debug_transport_trace")?;

        let block = 256_u32;
        let grid = ((n as u32 + block - 1) / block, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_seed = 42_u64;
        let n_i32 = n as i32;
        unsafe {
            self.stream
                .launch_builder(&k_init)
                .arg(&mut d_pos_x)
                .arg(&mut d_pos_y)
                .arg(&mut d_pos_z)
                .arg(&mut d_dir_x)
                .arg(&mut d_dir_y)
                .arg(&mut d_dir_z)
                .arg(&mut d_energy)
                .arg(&mut d_cell)
                .arg(&mut d_alive)
                .arg(&d_sx)
                .arg(&d_sy)
                .arg(&d_sz)
                .arg(&d_se)
                .arg(&n_i32)
                .arg(&batch_seed)
                .arg(&mut d_rng_state)
                .arg(&mut d_rng_inc)
                .arg(&geom_type)
                .launch(cfg)?;
        }

        macro_rules! dptr {
            ($slice:expr) => {{
                let (ptr, _guard) = $slice.device_ptr(&self.stream);
                ptr
            }};
        }
        let params_vec: Vec<u64> = vec![
            dptr!(&nuc_data.all_basis),
            dptr!(&nuc_data.all_coeffs),
            dptr!(&nuc_data.all_energy_grids),
            dptr!(&nuc_data.basis_offsets),
            dptr!(&nuc_data.grid_offsets),
            dptr!(&nuc_data.n_energies),
            dptr!(&nuc_data.has_reaction),
            dptr!(&nuc_data.coeffs_offsets),
            nuc_data.rank as u64,
            dptr!(&mat_data.mat_n_nuclides),
            dptr!(&mat_data.mat_nuclide_idx),
            dptr!(&mat_data.mat_atom_density),
            dptr!(&mat_data.awr_table),
            dptr!(&mat_data.nu_bar_const),
            dptr!(&nuc_data.nu_bar_energies),
            dptr!(&nuc_data.nu_bar_values),
            dptr!(&nuc_data.nu_bar_offsets),
            dptr!(&nuc_data.nu_bar_sizes),
            dptr!(&nuc_data.fis_inc_energies),
            dptr!(&nuc_data.fis_dist_offsets),
            dptr!(&nuc_data.fis_dist_sizes),
            dptr!(&nuc_data.fis_e_out),
            dptr!(&nuc_data.fis_cdf),
            dptr!(&nuc_data.fis_nuc_offsets),
            dptr!(&nuc_data.fis_nuc_n_inc),
            dptr!(&nuc_data.level_q_values),
            dptr!(&nuc_data.level_thresholds),
            dptr!(&nuc_data.level_offsets),
            dptr!(&nuc_data.level_counts),
            dptr!(&nuc_data.level_basis),
            dptr!(&nuc_data.level_coeffs),
            dptr!(&nuc_data.level_basis_offsets),
            dptr!(&nuc_data.level_coeffs_offsets),
            dptr!(&nuc_data.level_has_kernel),
            dptr!(&nuc_data.level_mt),
            dptr!(&nuc_data.ang_energies),
            dptr!(&nuc_data.ang_mu),
            dptr!(&nuc_data.ang_cdf),
            dptr!(&nuc_data.ang_dist_offsets),
            dptr!(&nuc_data.ang_dist_sizes),
            dptr!(&nuc_data.ang_nuc_offsets),
            dptr!(&nuc_data.ang_nuc_n_energies),
            dptr!(&nuc_data.ang_is_cm),
            dptr!(&sab_data.inc_energies),
            sab_data.n_inc as u64,
            dptr!(&sab_data.eout_offsets),
            dptr!(&sab_data.eout_sizes),
            dptr!(&sab_data.e_out),
            dptr!(&sab_data.cdf_e),
            dptr!(&sab_data.mu_offsets),
            dptr!(&sab_data.mu_sizes),
            dptr!(&sab_data.mu),
            dptr!(&sab_data.cdf_mu),
            dptr!(&sab_data.xs),
            sab_data.energy_max.to_bits(),
            dptr!(&sab_data.pdf_e),
            dptr!(&nuc_data.urr_energies),
            dptr!(&nuc_data.urr_cum_prob),
            dptr!(&nuc_data.urr_total_f),
            dptr!(&nuc_data.urr_elastic_f),
            dptr!(&nuc_data.urr_fission_f),
            dptr!(&nuc_data.urr_capture_f),
            dptr!(&nuc_data.urr_offsets),
            dptr!(&nuc_data.urr_n_energies),
            dptr!(&nuc_data.urr_n_bands),
            dptr!(&nuc_data.urr_multiply_smooth),
            geom_type as u64,
            dptr!(&nuc_data.total_xs),
            dptr!(&nuc_data.total_xs_offsets),
            dptr!(&nuc_data.has_total_xs),
            dptr!(&nuc_data.pointwise_xs),
            dptr!(&nuc_data.pw_offsets),
            dptr!(&nuc_data.has_pw),
            dptr!(&wmp_data.has),
            dptr!(&wmp_data.e_min),
            dptr!(&wmp_data.e_max),
            dptr!(&wmp_data.spacing),
            dptr!(&wmp_data.sqrt_awr),
            dptr!(&wmp_data.t_kelvin),
            dptr!(&wmp_data.fit_order),
            dptr!(&wmp_data.n_windows),
            dptr!(&wmp_data.fissionable),
            dptr!(&wmp_data.poles),
            dptr!(&wmp_data.pole_offsets),
            dptr!(&wmp_data.windows),
            dptr!(&wmp_data.window_offsets),
            dptr!(&wmp_data.broaden),
            dptr!(&wmp_data.broaden_offsets),
            dptr!(&wmp_data.curvefit),
            dptr!(&wmp_data.curvefit_offsets),
            dptr!(&nuc_data.lev_ang_energies),
            dptr!(&nuc_data.lev_ang_mu),
            dptr!(&nuc_data.lev_ang_cdf),
            dptr!(&nuc_data.lev_ang_dist_off),
            dptr!(&nuc_data.lev_ang_dist_sz),
            dptr!(&nuc_data.lev_ang_lev_off),
            dptr!(&nuc_data.lev_ang_lev_ne),
        ];
        // Debug-trace kernel doesn't reference the inel_cdf / Watt /
        // delayed-ν̄ / fis_pdf / MT=91-inelastic slots. Pad to
        // N_PARAMS so the assert + TransportParams layout match.
        let mut params_vec = params_vec;
        while params_vec.len() < N_PARAMS {
            params_vec.push(0_u64);
        }
        assert_eq!(params_vec.len(), N_PARAMS);
        let d_params = self.stream.clone_htod(&params_vec)?;

        let max_fis_i32 = max_fis as i32;
        let max_steps_i32 = max_steps as i32;

        unsafe {
            self.stream
                .launch_builder(&k_trace)
                .arg(&d_params)
                .arg(&mut d_pos_x)
                .arg(&mut d_pos_y)
                .arg(&mut d_pos_z)
                .arg(&mut d_dir_x)
                .arg(&mut d_dir_y)
                .arg(&mut d_dir_z)
                .arg(&mut d_energy)
                .arg(&mut d_cell)
                .arg(&mut d_alive)
                .arg(&mut d_rng_state)
                .arg(&mut d_rng_inc)
                .arg(&mut d_fis_x)
                .arg(&mut d_fis_y)
                .arg(&mut d_fis_z)
                .arg(&mut d_fis_e)
                .arg(&mut d_fis_w)
                .arg(&mut d_fis_count)
                .arg(&max_fis_i32)
                .arg(&mut d_trace)
                .arg(&mut d_step_counts)
                .arg(&n_i32)
                .arg(&max_steps_i32)
                .launch(cfg)?;
        }

        let data = self.stream.clone_dtoh(&d_trace)?;
        let step_counts = self.stream.clone_dtoh(&d_step_counts)?;

        Ok(GpuTraceResult { data, step_counts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::xs_provider::NuclideKernels;

    /// `GpuTransportContext::shared()` must return the same `Arc` on
    /// every call within one process — that's what makes the
    /// per-context `per_nuclide_cache` survive across ICSBEP cases.
    /// Verifies pointer identity, not just value equality.
    #[test]
    fn shared_singleton_returns_same_arc() {
        let a = match GpuTransportContext::shared() {
            Ok(x) => x,
            // No CUDA device on this machine — the singleton path is
            // still correct by construction; skip the runtime check.
            Err(_) => return,
        };
        let b = GpuTransportContext::shared().expect("second call must succeed once first did");
        assert!(
            Arc::ptr_eq(&a, &b),
            "shared() must return the same Arc across calls"
        );
    }

    /// The per-nuclide cache key collides iff the same Arc + rank.
    /// Different nuclide Arcs or different rank values must produce
    /// distinct keys — pointer identity from upstream `nuclide_cache`
    /// guarantees byte-identical content per pointer.
    #[test]
    fn per_nuclide_key_collides_iff_arc_and_rank_match() {
        let a: Arc<NuclideKernels> = Arc::new(NuclideKernels::empty(1.0, 2.43));
        let b: Arc<NuclideKernels> = Arc::new(NuclideKernels::empty(16.0, 2.43));

        let k_a5 = PerNuclideCacheKey {
            nuc_ptr: Arc::as_ptr(&a) as usize,
            rank: 5,
        };
        let k_a5_again = PerNuclideCacheKey {
            nuc_ptr: Arc::as_ptr(&a) as usize,
            rank: 5,
        };
        assert_eq!(k_a5, k_a5_again, "same Arc + same rank must collide");

        let k_a7 = PerNuclideCacheKey {
            nuc_ptr: Arc::as_ptr(&a) as usize,
            rank: 7,
        };
        assert_ne!(k_a5, k_a7, "different rank must produce different key");

        let k_b5 = PerNuclideCacheKey {
            nuc_ptr: Arc::as_ptr(&b) as usize,
            rank: 5,
        };
        assert_ne!(k_a5, k_b5, "different Arc must produce different key");

        // Distinct Arc with same contents — pointer identity, not
        // value identity. Upstream `nuclide_cache` guarantees same
        // (path, blake3, policy) → same Arc, so pointer-collision
        // implies byte-identical content; this test confirms we
        // honour pointer identity strictly.
        let c: Arc<NuclideKernels> = Arc::new(NuclideKernels::empty(1.0, 2.43));
        let k_c5 = PerNuclideCacheKey {
            nuc_ptr: Arc::as_ptr(&c) as usize,
            rank: 5,
        };
        assert_ne!(k_a5, k_c5);
    }

    /// PerNuclideCache::new is empty; EvictionStats round-trips bytes
    /// / counter. Actual eviction semantics covered by
    /// `nuclide_cache::eviction::tests` and L1MemoryStore tests.
    #[test]
    fn per_nuclide_cache_initialises_empty() {
        let cache = PerNuclideCache::new();
        assert_eq!(cache.entries.len(), 0);
        assert_eq!(cache.total_bytes, 0);
        assert_eq!(cache.counter, 0);
        let s = EvictionStats::new(100, 0);
        assert_eq!(s.bytes, 100);
        assert_eq!(s.last_touch, 0);
        assert_eq!(s.hits, 0);
    }
}
