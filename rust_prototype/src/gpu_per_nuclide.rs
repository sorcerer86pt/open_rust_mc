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

use crate::transport::xs_provider::{NuclideKernels, ReactionKernel};

/// Fixed reaction-slot count, matching the bundle layout in
/// `gpu_transport::upload_nuclide_data_uncached`: elastic, inelastic,
/// n2n, n3n, fission, capture, total. Slot 6 (`RXN_TOTAL`) is always
/// `None` on the per-reaction arrays — the total XS lives on the
/// pointwise tables instead.
pub const N_RXN_SLOTS: usize = 7;

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

/// Per-nuclide elastic angular distribution (category A.4) — also
/// reused by the per-level angular slot of `LevelSlicesGpu` (same
/// schema, different host source).
///
/// Layout matches `gpu_transport.rs` bundle: per (e_inc_idx) the CDF
/// span is `mu[dist_local_off..+dist_sz]` / `cdf[…]`. Bundle assembly
/// shifts every `dist_local_off` by the running global offset; this
/// per-nuclide copy starts at 0.
pub struct AngularSlicesGpu {
    pub n_energies: i32,
    pub is_cm: i32,
    pub energies: CudaSlice<f64>,
    pub mu: CudaSlice<f64>,
    pub cdf: CudaSlice<f64>,
    /// `[n_energies]` host-side; offset into this nuclide's `mu` /
    /// `cdf` buffers (starts at 0 per nuclide).
    pub dist_local_off: Vec<i32>,
    /// `[n_energies]`.
    pub dist_sz: Vec<i32>,
}

impl AngularSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.energies.num_bytes()
            + self.mu.num_bytes()
            + self.cdf.num_bytes()
            + (self.dist_local_off.len() + self.dist_sz.len())
                * std::mem::size_of::<i32>()
    }
}

/// Per-nuclide discrete inelastic level bundle (category C).
///
/// One `LevelSlicesGpu` per nuclide carries every MT=51-91 level's
/// XS data plus per-level CM-frame angular distributions. Per-level
/// basis arrays are pre-padded to the bundle's global `rank` stride
/// — the rank-padding invariant from commit `1654c4d` (skipping the
/// padding silently reads adjacent levels' bytes → +500-700 pcm
/// fast-metal hot bias).
///
/// Offsets stored on the host side. The bundle assembly stage shifts
/// them by the running per-nuclide global offset to produce the
/// flat-pack `level_basis_offsets` / `lev_ang_lev_off` / etc.
/// device arrays the kernel expects.
pub struct LevelSlicesGpu {
    pub n_levels: i32,

    /// `[n_levels]` — Q-values, thresholds, ENDF MT numbers (51-91),
    /// and 0/1 kernel-presence flags. All parallel.
    pub q_values: CudaSlice<f64>,
    pub thresholds: CudaSlice<f64>,
    pub mt: CudaSlice<i32>,
    pub has_kernel: CudaSlice<i32>,

    /// Rank-padded basis, concatenated across this nuclide's levels.
    /// Each level contributes `[n_e_l × rank]` doubles. Layout matches
    /// the legacy bundle's `level_basis` per-nuclide slice exactly.
    pub basis: CudaSlice<f64>,
    pub coeffs: CudaSlice<f64>,
    /// `[n_levels]` host-side; `basis_local_off[l]` is the level's
    /// offset into this nuclide's `basis` buffer (starts at 0). Bundle
    /// assembly adds the running global offset to produce
    /// `level_basis_offsets[global_level_idx]`.
    pub basis_local_off: Vec<i32>,
    pub coeffs_local_off: Vec<i32>,

    /// Per-level CM-frame angular CDFs. Layout:
    /// `ang_energies[global_e_idx_within_nuclide]`, with per-level
    /// span `[ang_lev_local_off[l] .. + ang_lev_ne[l]]`. The (level,
    /// e_idx) pair locates a CDF run inside `ang_mu` / `ang_cdf` via
    /// `ang_dist_local_off[global_e_idx] / ang_dist_sz[global_e_idx]`.
    pub ang_energies: CudaSlice<f64>,
    pub ang_mu: CudaSlice<f64>,
    pub ang_cdf: CudaSlice<f64>,
    pub ang_lev_local_off: Vec<i32>,
    pub ang_lev_ne: Vec<i32>,
    pub ang_dist_local_off: Vec<i32>,
    pub ang_dist_sz: Vec<i32>,
}

impl LevelSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.q_values.num_bytes()
            + self.thresholds.num_bytes()
            + self.mt.num_bytes()
            + self.has_kernel.num_bytes()
            + self.basis.num_bytes()
            + self.coeffs.num_bytes()
            + self.ang_energies.num_bytes()
            + self.ang_mu.num_bytes()
            + self.ang_cdf.num_bytes()
            + (self.basis_local_off.len()
                + self.coeffs_local_off.len()
                + self.ang_lev_local_off.len()
                + self.ang_lev_ne.len()
                + self.ang_dist_local_off.len()
                + self.ang_dist_sz.len())
                * std::mem::size_of::<i32>()
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

    // ── Category B — per-reaction SVD basis / coeffs ──
    /// 0 / 1 flag per reaction slot (elastic .. capture .. total). Slot
    /// `RXN_TOTAL = 6` is always `0` — the total XS is handled via
    /// `pointwise_xs` / `total_xs`.
    pub has_reaction: [i32; N_RXN_SLOTS],
    /// `[n_e × rank]` per slot. `None` when `has_reaction[slot] == 0`.
    /// Table-variant reactions are pre-padded to rank stride here
    /// (basis row = `[log10(xs), 0, 0, …]`) so the device kernel sees
    /// the uniform layout.
    pub basis: [Option<CudaSlice<f64>>; N_RXN_SLOTS],
    /// `[rank]` per slot, paired with `basis`. Table-variant slots
    /// carry `[1.0, 0.0, 0.0, …]` so the dot product reconstructs the
    /// pointwise value.
    pub coeffs: [Option<CudaSlice<f64>>; N_RXN_SLOTS],

    // ── Category C — discrete inelastic levels (MT=51-91) ──
    pub levels: LevelSlicesGpu,

    // ── Category A.4 — elastic angular distribution ──
    pub elastic_angle: Option<AngularSlicesGpu>,
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
        for s in self.basis.iter().flatten() {
            total += s.num_bytes();
        }
        for s in self.coeffs.iter().flatten() {
            total += s.num_bytes();
        }
        total += self.levels.device_bytes();
        if let Some(a) = &self.elastic_angle {
            total += a.device_bytes();
        }
        total
    }
}

/// Pack a `ReactionKernel` into the uniform rank-padded
/// `(basis, coeffs)` layout the device kernel expects. The Svd
/// variant passes through unchanged; Table variants get
/// `basis_row = [log10(xs), 0, 0, …]` and `coeffs = [1.0, 0.0, …]`
/// — same convention as the legacy whole-bundle packer at
/// `gpu_transport.rs::upload_nuclide_data_uncached`, slot 1278-1318.
fn pack_reaction_to_rank(rxn: &ReactionKernel, rank: usize) -> (Vec<f64>, Vec<f64>) {
    match rxn {
        ReactionKernel::Svd { kernel, coeffs } => {
            (kernel.basis_f64().to_vec(), coeffs.clone())
        }
        ReactionKernel::Table { xs, .. } => {
            // Synthesize a rank-`rank` SVD layout from the pointwise
            // table. Mirrors the bundle path; if you change one,
            // change the other.
            let mut basis = Vec::with_capacity(xs.len() * rank);
            for &v in xs {
                let log10_v = if v > 0.0 { v.log10() } else { -300.0 };
                basis.push(log10_v);
                for _ in 1..rank {
                    basis.push(0.0);
                }
            }
            let mut coeffs = Vec::with_capacity(rank);
            coeffs.push(1.0);
            for _ in 1..rank {
                coeffs.push(0.0);
            }
            (basis, coeffs)
        }
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

    // Category B — per-reaction SVD basis / coeffs. Slot order must
    // match `gpu_transport::upload_nuclide_data_uncached` (slot 6 =
    // total is intentionally None).
    let reactions: [Option<&ReactionKernel>; N_RXN_SLOTS] = [
        nuc.elastic.as_ref(),
        nuc.inelastic.as_ref(),
        nuc.n2n.as_ref(),
        nuc.n3n.as_ref(),
        nuc.fission.as_ref(),
        nuc.capture.as_ref(),
        None,
    ];
    let mut has_reaction = [0_i32; N_RXN_SLOTS];
    let mut basis: [Option<CudaSlice<f64>>; N_RXN_SLOTS] =
        std::array::from_fn(|_| None);
    let mut coeffs: [Option<CudaSlice<f64>>; N_RXN_SLOTS] =
        std::array::from_fn(|_| None);
    for (slot, rxn_opt) in reactions.iter().enumerate() {
        if let Some(rxn) = rxn_opt {
            has_reaction[slot] = 1;
            let (basis_vec, coeffs_vec) = pack_reaction_to_rank(rxn, rank);
            basis[slot] = Some(stream.clone_htod(&basis_vec)?);
            coeffs[slot] = Some(stream.clone_htod(&coeffs_vec)?);
        }
    }

    // Category C — discrete inelastic levels with per-level rank
    // padding (commit `1654c4d` invariant).
    let levels = build_level_slices(stream, nuc, rank)?;

    // Category A.4 — elastic angular distribution.
    let elastic_angle = nuc
        .elastic_angle
        .as_ref()
        .filter(|ad| !ad.energies.is_empty())
        .map(|ad| build_angular_slices(stream, ad))
        .transpose()?;

    Ok(PerNuclideGpu {
        rank: rank as i32,
        n_energy,
        energy_grid,
        total_xs,
        pointwise_xs,
        nu_bar,
        delayed_nu_bar,
        has_reaction,
        basis,
        coeffs,
        levels,
        elastic_angle,
    })
}

/// Pack an `AngularDistribution` into per-nuclide GPU layout. Mirrors
/// the legacy bundle's elastic-angular packing at
/// `gpu_transport.rs::upload_nuclide_data_uncached` slot 1588-1602.
/// Per-energy `dist_local_off` is nuclide-local; bundle assembly
/// shifts it.
fn build_angular_slices(
    stream: &Arc<CudaStream>,
    ad: &crate::hdf5_reader::AngularDistribution,
) -> Result<AngularSlicesGpu, Box<dyn std::error::Error>> {
    let n_energies = ad.energies.len();
    let mut mu_buf: Vec<f64> = Vec::new();
    let mut cdf_buf: Vec<f64> = Vec::new();
    let mut dist_local_off: Vec<i32> = Vec::with_capacity(n_energies);
    let mut dist_sz: Vec<i32> = Vec::with_capacity(n_energies);
    for (i, _e) in ad.energies.iter().enumerate() {
        let dist = &ad.distributions[i];
        dist_local_off.push(mu_buf.len() as i32);
        dist_sz.push(dist.mu.len() as i32);
        mu_buf.extend_from_slice(&dist.mu);
        cdf_buf.extend_from_slice(&dist.cdf);
    }
    if mu_buf.is_empty() {
        mu_buf.push(0.0);
        cdf_buf.push(0.0);
    }
    Ok(AngularSlicesGpu {
        n_energies: n_energies as i32,
        is_cm: if ad.center_of_mass { 1 } else { 0 },
        energies: stream.clone_htod(&ad.energies)?,
        mu: stream.clone_htod(&mu_buf)?,
        cdf: stream.clone_htod(&cdf_buf)?,
        dist_local_off,
        dist_sz,
    })
}

/// Build the per-nuclide `LevelSlicesGpu`, applying the per-level
/// rank-padding invariant (`1654c4d`).
///
/// Each discrete-level SVD may have `level_rank < global rank` on
/// sparse HDF5 grids. The device kernel reads
/// `basis[e_idx * P_RANK + j]` for the full `j ∈ [0, P_RANK)` range;
/// the raw narrower-stride basis would silently read adjacent levels'
/// bytes. Pad each level's basis with zero columns up to global rank,
/// and pad coeffs to length rank with zeros — the dot product is
/// unchanged (extra × 0 = 0). Mirrors
/// `gpu_transport.rs::upload_nuclide_data_uncached` slot 1457-1518.
fn build_level_slices(
    stream: &Arc<CudaStream>,
    nuc: &NuclideKernels,
    rank: usize,
) -> Result<LevelSlicesGpu, Box<dyn std::error::Error>> {
    let n_levels = nuc.discrete_levels.len();

    let mut q_values = Vec::with_capacity(n_levels);
    let mut thresholds = Vec::with_capacity(n_levels);
    let mut mts = Vec::with_capacity(n_levels);
    let mut has_kernel = Vec::with_capacity(n_levels);
    let mut basis_buf: Vec<f64> = Vec::new();
    let mut coeffs_buf: Vec<f64> = Vec::new();
    let mut basis_local_off: Vec<i32> = Vec::with_capacity(n_levels);
    let mut coeffs_local_off: Vec<i32> = Vec::with_capacity(n_levels);

    let mut ang_e_buf: Vec<f64> = Vec::new();
    let mut ang_mu_buf: Vec<f64> = Vec::new();
    let mut ang_cdf_buf: Vec<f64> = Vec::new();
    let mut ang_lev_local_off: Vec<i32> = Vec::with_capacity(n_levels);
    let mut ang_lev_ne: Vec<i32> = Vec::with_capacity(n_levels);
    let mut ang_dist_local_off: Vec<i32> = Vec::new();
    let mut ang_dist_sz: Vec<i32> = Vec::new();

    for (li, lev) in nuc.discrete_levels.iter().enumerate() {
        q_values.push(lev.info.q_value);
        thresholds.push(lev.info.threshold);
        mts.push(lev.info.mt as i32);

        match lev.kernel.as_ref() {
            Some(ReactionKernel::Svd { kernel, coeffs }) => {
                has_kernel.push(1);
                basis_local_off.push(basis_buf.len() as i32);
                let level_rank = kernel.rank();
                let n_e = kernel.n_energy();
                let raw_basis = kernel.basis_f64();
                if level_rank == rank {
                    basis_buf.extend_from_slice(raw_basis);
                } else {
                    // Pad rows to global rank with zero columns —
                    // the rank-padding fix from `1654c4d`.
                    for i in 0..n_e {
                        let src = &raw_basis[i * level_rank..(i + 1) * level_rank];
                        basis_buf.extend_from_slice(src);
                        for _ in level_rank..rank {
                            basis_buf.push(0.0);
                        }
                    }
                }
                coeffs_local_off.push(coeffs_buf.len() as i32);
                coeffs_buf.extend_from_slice(coeffs);
                for _ in coeffs.len()..rank {
                    coeffs_buf.push(0.0);
                }
            }
            Some(ReactionKernel::Table { xs, .. }) => {
                has_kernel.push(1);
                basis_local_off.push(basis_buf.len() as i32);
                for &v in xs {
                    let log10_v = if v > 0.0 { v.log10() } else { -300.0 };
                    basis_buf.push(log10_v);
                    for _ in 1..rank {
                        basis_buf.push(0.0);
                    }
                }
                coeffs_local_off.push(coeffs_buf.len() as i32);
                coeffs_buf.push(1.0);
                for _ in 1..rank {
                    coeffs_buf.push(0.0);
                }
            }
            None => {
                has_kernel.push(0);
                basis_local_off.push(0);
                coeffs_local_off.push(0);
            }
        }

        // Per-level angular CDF. Missing → mark `ne = 0` so the device
        // returns isotropic μ_cm.
        let ang = nuc.discrete_level_angles.get(li).and_then(|o| o.as_ref());
        match ang {
            Some(ad) if !ad.energies.is_empty() => {
                ang_lev_local_off.push(ang_e_buf.len() as i32);
                ang_lev_ne.push(ad.energies.len() as i32);
                for (ei, e) in ad.energies.iter().enumerate() {
                    ang_e_buf.push(*e);
                    let dist = &ad.distributions[ei];
                    ang_dist_local_off.push(ang_mu_buf.len() as i32);
                    ang_dist_sz.push(dist.mu.len() as i32);
                    ang_mu_buf.extend_from_slice(&dist.mu);
                    ang_cdf_buf.extend_from_slice(&dist.cdf);
                }
            }
            _ => {
                ang_lev_local_off.push(0);
                ang_lev_ne.push(0);
            }
        }
    }

    // Sentinel for empty payloads — `clone_htod` of a zero-length
    // slice would leave a null device pointer on some drivers; the
    // legacy bundle path uses the same `[0.0]` / `[0]` rule.
    if q_values.is_empty() {
        q_values.push(0.0);
        thresholds.push(0.0);
        mts.push(0);
        has_kernel.push(0);
    }
    if basis_buf.is_empty() {
        basis_buf.push(0.0);
    }
    if coeffs_buf.is_empty() {
        coeffs_buf.push(0.0);
    }
    if ang_e_buf.is_empty() {
        ang_e_buf.push(0.0);
    }
    if ang_mu_buf.is_empty() {
        ang_mu_buf.push(0.0);
        ang_cdf_buf.push(0.0);
    }
    if ang_dist_local_off.is_empty() {
        ang_dist_local_off.push(0);
        ang_dist_sz.push(0);
    }
    if ang_lev_local_off.is_empty() {
        ang_lev_local_off.push(0);
        ang_lev_ne.push(0);
    }

    Ok(LevelSlicesGpu {
        n_levels: n_levels as i32,
        q_values: stream.clone_htod(&q_values)?,
        thresholds: stream.clone_htod(&thresholds)?,
        mt: stream.clone_htod(&mts)?,
        has_kernel: stream.clone_htod(&has_kernel)?,
        basis: stream.clone_htod(&basis_buf)?,
        coeffs: stream.clone_htod(&coeffs_buf)?,
        basis_local_off,
        coeffs_local_off,
        ang_energies: stream.clone_htod(&ang_e_buf)?,
        ang_mu: stream.clone_htod(&ang_mu_buf)?,
        ang_cdf: stream.clone_htod(&ang_cdf_buf)?,
        ang_lev_local_off,
        ang_lev_ne,
        ang_dist_local_off,
        ang_dist_sz,
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

    #[test]
    fn level_basis_padded_to_global_rank() {
        let Some(stream) = try_cuda_stream() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // Build a nuclide with one discrete Table-variant level — the
        // legacy bundle pads Table-variant levels to rank exactly like
        // it pads `level_rank < rank` SVD levels (`1654c4d`).
        use crate::hdf5_reader::DiscreteLevelInfo;
        use crate::transport::xs_provider::DiscreteLevel;

        let mut nuc = NuclideKernels::empty(1.0, 0.0);
        let level_kernel =
            ReactionKernel::from_table(vec![1.0e4, 1.0e6], vec![2.0, 4.0]);
        nuc.discrete_levels.push(DiscreteLevel {
            info: DiscreteLevelInfo {
                mt: 51,
                q_value: -1.0e5,
                threshold: 1.0e5,
            },
            kernel: Some(level_kernel),
        });
        // discrete_level_angles must be parallel even when individual
        // entries are absent.
        nuc.discrete_level_angles.push(None);

        // Need at least one reaction so `upload_one_nuclide` resolves
        // the energy grid.
        nuc.elastic = Some(ReactionKernel::from_table(
            vec![1.0e-5, 1.0, 2.0e7],
            vec![10.0, 5.0, 1.0],
        ));

        let rank = 5;
        let per_nuc =
            upload_one_nuclide(&stream, &nuc, rank).expect("upload_one_nuclide failed");

        // One level, one populated basis run of [n_e × rank] doubles.
        assert_eq!(per_nuc.levels.n_levels, 1);
        assert_eq!(per_nuc.levels.basis_local_off, vec![0]);
        assert_eq!(per_nuc.levels.coeffs_local_off, vec![0]);
        // Basis: [log10(2), 0,0,0,0, log10(4), 0,0,0,0] — n_e=2, rank=5.
        let mut host_basis = vec![0.0_f64; 2 * rank];
        stream
            .memcpy_dtoh(&per_nuc.levels.basis, &mut host_basis)
            .unwrap();
        let expected = vec![
            2.0_f64.log10(), 0.0, 0.0, 0.0, 0.0,
            4.0_f64.log10(), 0.0, 0.0, 0.0, 0.0,
        ];
        for (i, (h, e)) in host_basis.iter().zip(expected.iter()).enumerate() {
            assert!(
                (h - e).abs() < 1e-12,
                "level basis[{i}] padding: got {h}, want {e}"
            );
        }
        // Coeffs: [1.0, 0, 0, 0, 0]. Length = rank, not 1.
        let mut host_coeffs = vec![0.0_f64; rank];
        stream
            .memcpy_dtoh(&per_nuc.levels.coeffs, &mut host_coeffs)
            .unwrap();
        assert_eq!(host_coeffs, vec![1.0, 0.0, 0.0, 0.0, 0.0]);

        // Angular dist absent → sentinels in the host-side offsets.
        assert_eq!(per_nuc.levels.ang_lev_local_off, vec![0]);
        assert_eq!(per_nuc.levels.ang_lev_ne, vec![0]);
    }

    #[test]
    fn upload_one_nuclide_packs_table_reaction_to_rank() {
        let Some(stream) = try_cuda_stream() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let nuc = minimal_nuclide();
        let rank = 5;
        let per_nuc =
            upload_one_nuclide(&stream, &nuc, rank).expect("upload_one_nuclide failed");

        // Slot 0 = elastic (Table). Others = None / not loaded.
        assert_eq!(per_nuc.has_reaction[0], 1);
        for slot in 1..N_RXN_SLOTS {
            assert_eq!(per_nuc.has_reaction[slot], 0, "slot {slot}");
            assert!(per_nuc.basis[slot].is_none(), "slot {slot}");
            assert!(per_nuc.coeffs[slot].is_none(), "slot {slot}");
        }

        // Basis = [log10(10), 0,0,0,0, log10(5), 0,0,0,0, log10(1), 0,0,0,0].
        let basis = per_nuc.basis[0].as_ref().expect("elastic basis missing");
        let mut host_basis = vec![0.0_f64; 3 * rank];
        stream.memcpy_dtoh(basis, &mut host_basis).unwrap();
        let expected = vec![
            10.0_f64.log10(), 0.0, 0.0, 0.0, 0.0,
            5.0_f64.log10(),  0.0, 0.0, 0.0, 0.0,
            1.0_f64.log10(),  0.0, 0.0, 0.0, 0.0,
        ];
        for (i, (h, e)) in host_basis.iter().zip(expected.iter()).enumerate() {
            assert!((h - e).abs() < 1e-12, "basis[{i}]: got {h}, want {e}");
        }

        // Coeffs = [1, 0, 0, 0, 0].
        let coeffs = per_nuc.coeffs[0].as_ref().expect("elastic coeffs missing");
        let mut host_coeffs = vec![0.0_f64; rank];
        stream.memcpy_dtoh(coeffs, &mut host_coeffs).unwrap();
        assert_eq!(host_coeffs, vec![1.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
