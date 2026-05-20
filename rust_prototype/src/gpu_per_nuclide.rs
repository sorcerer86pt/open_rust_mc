// SPDX-License-Identifier: MIT
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
//! into `gpu_transport.rs::upload_nuclide_data`. The kernel
//! ABI stays unchanged until Stage 4 (separate commit, gated on
//! `metal_stats_diag` 3-way passing).

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

use crate::transport::xs_provider::{NuclideKernels, ReactionKernel};

/// Fixed reaction-slot count, matching the bundle layout in
/// `gpu_transport::upload_nuclide_data`: elastic, inelastic,
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

/// Per-(inc_energy_idx) outgoing-energy CDF run — shared shape used
/// by tabular fission χ (ENDF Law 4 / 61) and MT=91 continuum
/// inelastic. Per-inc offset is nuclide-local; bundle assembly
/// shifts.
pub struct TabularEdistSlicesGpu {
    pub n_inc: i32,
    pub inc_energies: CudaSlice<f64>,
    pub e_out: CudaSlice<f64>,
    pub cdf: CudaSlice<f64>,
    /// PDF samples aligned 1:1 with `e_out` / `cdf`; populated only
    /// when the ENDF evaluation shipped them, zeros otherwise (the
    /// device kernel falls back to linear-CDF interpolation when
    /// every PDF sample is zero — see fis_pdf comment on
    /// `GpuNuclideData`).
    pub pdf: CudaSlice<f64>,
    /// `[n_inc]` host-side — local offset into `e_out`/`cdf`/`pdf`
    /// (starts at 0 per nuclide).
    pub dist_local_off: Vec<i32>,
    pub dist_sz: Vec<i32>,
}

impl TabularEdistSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.inc_energies.num_bytes()
            + self.e_out.num_bytes()
            + self.cdf.num_bytes()
            + self.pdf.num_bytes()
            + (self.dist_local_off.len() + self.dist_sz.len())
                * std::mem::size_of::<i32>()
    }
}

/// Per-nuclide closed-form Watt fission χ parameters (ENDF Law 11).
/// `a(E_in)` and `b(E_in)` are pre-resampled onto a shared inc-energy
/// grid (union of the original a_energies + b_energies, sorted +
/// deduped) so the device samples via one binary search per fission
/// event. Mirrors upload_nuclide_data:1722-1751.
pub struct WattSlicesGpu {
    pub n_inc: i32,
    pub u: f64,
    pub inc_energies: CudaSlice<f64>,
    pub a: CudaSlice<f64>,
    pub b: CudaSlice<f64>,
}

impl WattSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.inc_energies.num_bytes() + self.a.num_bytes() + self.b.num_bytes()
    }
}

/// Per-nuclide closed-form Maxwell (Law 7) / Evaporation (Law 9)
/// fission χ parameters. Both laws share a single θ(E_in) table; the
/// device dispatches on `law ∈ {7, 9}` at collision time.
pub struct MaxEvapSlicesGpu {
    pub n_inc: i32,
    pub u: f64,
    /// 7 = Maxwell, 9 = Evaporation. 0 means "no maxevap data" — but
    /// in practice the slice is None in that case.
    pub law: i32,
    pub inc_energies: CudaSlice<f64>,
    pub theta: CudaSlice<f64>,
}

impl MaxEvapSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.inc_energies.num_bytes() + self.theta.num_bytes()
    }
}

/// Per-nuclide URR probability tables. Each cross-section factor
/// row is `[n_energy × n_bands]` flat (row-major), matching the
/// bundle's flat-pack layout.
pub struct UrrSlicesGpu {
    pub n_energies: i32,
    pub n_bands: i32,
    pub multiply_smooth: i32,
    /// ENDF interpolation code: 2 = lin-lin (default), 5 = log-log.
    /// Used by the GPU `apply_urr` to interpolate URR factors between
    /// the two bracketing energy bins — mirrors CPU's
    /// `UrrProbabilityTables::sample` (hdf5_reader.rs:1948).
    pub interpolation: i32,
    pub energies: CudaSlice<f64>,
    pub cum_prob: CudaSlice<f64>,
    pub total_factor: CudaSlice<f64>,
    pub elastic_factor: CudaSlice<f64>,
    pub fission_factor: CudaSlice<f64>,
    pub capture_factor: CudaSlice<f64>,
}

impl UrrSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.energies.num_bytes()
            + self.cum_prob.num_bytes()
            + self.total_factor.num_bytes()
            + self.elastic_factor.num_bytes()
            + self.fission_factor.num_bytes()
            + self.capture_factor.num_bytes()
    }
}

/// Per-nuclide synthesized MT=4 CDF (Zr-90..94, U-238). Flat tensor
/// `cdf[e_dec * n_t * n_lev + t * n_lev + l]`. When None, the device
/// falls back to the legacy per-level walk in `do_inelastic`.
pub struct InelCdfSlicesGpu {
    pub n_e: i32,
    pub n_t: i32,
    pub n_lev: i32,
    pub log_e_min: f64,
    pub log_e_max: f64,
    pub data: CudaSlice<f64>,
}

impl InelCdfSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.data.num_bytes()
    }
}

/// Tagged union over the four ENDF fission-spectrum laws supported by
/// the device. A nuclide can carry at most one variant; `None`
/// falls through to the host emitter's default χ.
pub enum FissionEdistGpu {
    None,
    Tabular(TabularEdistSlicesGpu),
    Watt(WattSlicesGpu),
    MaxEvap(MaxEvapSlicesGpu),
}

impl FissionEdistGpu {
    pub fn device_bytes(&self) -> usize {
        match self {
            FissionEdistGpu::None => 0,
            FissionEdistGpu::Tabular(s) => s.device_bytes(),
            FissionEdistGpu::Watt(s) => s.device_bytes(),
            FissionEdistGpu::MaxEvap(s) => s.device_bytes(),
        }
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
    /// PDF aligned 1:1 with `mu` / `cdf`. Enables the quadratic lin-
    /// lin CDF inversion in `sample_mu_bin` — without this the kernel
    /// falls back to a linear-CDF / histogram-PDF approximation that
    /// biases forward-peaked angular distributions (Al-27, Mg, Cr,
    /// Mn, W) — the +500-700 pcm CPU↔GPU gap on multi-nuclide fast-
    /// metal benchmarks (ieu-met-fast-001, heu-met-fast-011). Mirrors
    /// the `fis_pdf` / `inel91_pdf` fixes that closed the analogous
    /// gap on the χ outgoing spectrum.
    pub pdf: CudaSlice<f64>,
    /// `[n_energies]` host-side; offset into this nuclide's `mu` /
    /// `cdf` / `pdf` buffers (starts at 0 per nuclide).
    pub dist_local_off: Vec<i32>,
    /// `[n_energies]`.
    pub dist_sz: Vec<i32>,
    /// Real data length of `mu` / `cdf` / `pdf` excluding the trailing
    /// sentinel that `build_angular_slices` inserts when every
    /// distribution is empty. Bundle assembly slices to this length
    /// so per-nuclide sentinels aren't concatenated.
    pub mu_real_len: usize,
}

impl AngularSlicesGpu {
    pub fn device_bytes(&self) -> usize {
        self.energies.num_bytes()
            + self.mu.num_bytes()
            + self.cdf.num_bytes()
            + self.pdf.num_bytes()
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
    /// PDF aligned 1:1 with `ang_mu` / `ang_cdf`. Same role as
    /// `AngularSlicesGpu::pdf` — enables quadratic lin-lin CDF
    /// inversion in `sample_mu_bin`.
    pub ang_pdf: CudaSlice<f64>,
    pub ang_lev_local_off: Vec<i32>,
    pub ang_lev_ne: Vec<i32>,
    pub ang_dist_local_off: Vec<i32>,
    pub ang_dist_sz: Vec<i32>,

    /// Real data length of `basis` excluding the trailing `[0.0]`
    /// sentinel (`build_level_slices` always inserts one so the
    /// device pointer is non-null when used standalone). Equals
    /// `basis.len()` when there's at least one populated level kernel;
    /// `0` when `n_levels == 0` or every level's kernel was `None`.
    /// Bundle assembly slices to this length to skip per-nuclide
    /// sentinels.
    pub basis_real_len: usize,
    pub coeffs_real_len: usize,
    pub ang_e_real_len: usize,
    pub ang_mu_real_len: usize,
    pub ang_dist_real_len: usize,
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
            + self.ang_pdf.num_bytes()
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
/// `upload_nuclide_data` packing path for that category.
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

    // ── Category A.5 — fission outgoing-energy distribution ──
    pub fission_edist: FissionEdistGpu,

    // ── Category A.6 — MT=91 continuum inelastic outgoing-energy ──
    pub inel91: Option<TabularEdistSlicesGpu>,

    // ── Category A.7 — URR probability tables ──
    pub urr: Option<UrrSlicesGpu>,

    // ── Category A.8 — synthesized MT=4 CDF ──
    pub inel_cdf: Option<InelCdfSlicesGpu>,
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
        total += self.fission_edist.device_bytes();
        if let Some(i) = &self.inel91 {
            total += i.device_bytes();
        }
        if let Some(u) = &self.urr {
            total += u.device_bytes();
        }
        if let Some(c) = &self.inel_cdf {
            total += c.device_bytes();
        }
        total
    }
}

/// Pack a `ReactionKernel` into the uniform rank-padded
/// `(basis, coeffs)` layout the device kernel expects. The Svd
/// variant passes through unchanged; Table variants get
/// `basis_row = [log10(xs), 0, 0, …]` and `coeffs = [1.0, 0.0, …]`
/// — same convention as the legacy whole-bundle packer at
/// `gpu_transport.rs::upload_nuclide_data`, slot 1278-1318.
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
    // used by `gpu_transport::upload_nuclide_data`.
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
    // match `gpu_transport::upload_nuclide_data` (slot 6 =
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

    // Category A.5 — fission outgoing-energy distribution.
    let fission_edist = build_fission_edist(stream, nuc.fission_energy_dist.as_ref())?;

    // Category A.6 — MT=91 continuum inelastic outgoing-energy.
    let inel91 = match nuc.inelastic_continuum_edist.as_ref() {
        Some(edist) if !edist.energies.is_empty() && !edist.distributions.is_empty() => {
            Some(build_tabular_edist(stream, &edist.energies, &edist.distributions)?)
        }
        _ => None,
    };

    // Category A.7 — URR probability tables.
    let urr = nuc
        .urr_tables
        .as_ref()
        .filter(|u| !u.energies.is_empty())
        .map(|u| build_urr_slices(stream, u))
        .transpose()?;

    // Category A.8 — synthesized MT=4 CDF (Zr-90..94, U-238).
    let inel_cdf = nuc
        .inelastic_cdf
        .as_ref()
        .filter(|c| !c.cdf_flat.is_empty())
        .map(|c| -> Result<InelCdfSlicesGpu, Box<dyn std::error::Error>> {
            Ok(InelCdfSlicesGpu {
                n_e: c.n_energy as i32,
                n_t: c.n_temp as i32,
                n_lev: c.n_levels as i32,
                log_e_min: c.log_e_min,
                log_e_max: c.log_e_max,
                data: stream.clone_htod(&c.cdf_flat)?,
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
        has_reaction,
        basis,
        coeffs,
        levels,
        elastic_angle,
        fission_edist,
        inel91,
        urr,
        inel_cdf,
    })
}

/// Concatenate a per-nuclide sequence of optional device slices into
/// a single flat bundle slice. Returns `(bundle, per_nuc_offsets,
/// per_nuc_lengths, has_flags)`:
///   - `bundle[off..off+len]` is `src[i]` for each populated entry
///   - `per_nuc_offsets[i] = 0` and `has_flags[i] = 0` when `src[i] = None`
///   - sentinel `alloc_zeros(1)` returned when no nuclide contributes
///
/// All copies are direct device-to-device — the per-nuclide slices
/// stay on the device and are never round-tripped through host
/// memory.
fn concat_dtod_optional(
    stream: &Arc<CudaStream>,
    src: &[Option<&CudaSlice<f64>>],
) -> Result<(CudaSlice<f64>, Vec<i32>, Vec<i32>), Box<dyn std::error::Error>> {
    let total_len: usize = src.iter().filter_map(|o| o.map(|s| s.len())).sum();
    let alloc_len = total_len.max(1);
    let mut bundle = unsafe { stream.alloc::<f64>(alloc_len)? };
    if total_len == 0 {
        stream.memset_zeros(&mut bundle)?;
    }
    let mut offsets = vec![0_i32; src.len()];
    let mut has = vec![0_i32; src.len()];
    let mut running = 0_usize;
    for (i, slot) in src.iter().enumerate() {
        if let Some(s) = slot {
            offsets[i] = running as i32;
            has[i] = 1;
            let len = s.len();
            let mut view = bundle.slice_mut(running..running + len);
            stream.memcpy_dtod(*s, &mut view)?;
            running += len;
        }
    }
    Ok((bundle, offsets, has))
}

/// Concatenate a per-nuclide sequence of *always-present* device
/// slices (e.g. energy grids — every nuclide carries one). Same shape
/// as `concat_dtod_optional` minus the `has` flag.
fn concat_dtod_required(
    stream: &Arc<CudaStream>,
    src: &[&CudaSlice<f64>],
) -> Result<(CudaSlice<f64>, Vec<i32>), Box<dyn std::error::Error>> {
    let total_len: usize = src.iter().map(|s| s.len()).sum();
    let alloc_len = total_len.max(1);
    let mut bundle = unsafe { stream.alloc::<f64>(alloc_len)? };
    if total_len == 0 {
        stream.memset_zeros(&mut bundle)?;
    }
    let mut offsets = Vec::with_capacity(src.len());
    let mut running = 0_usize;
    for s in src {
        offsets.push(running as i32);
        let len = s.len();
        if len > 0 {
            let mut view = bundle.slice_mut(running..running + len);
            stream.memcpy_dtod(*s, &mut view)?;
        }
        running += len;
    }
    Ok((bundle, offsets))
}

/// Assemble simple A-cat bundle fields directly from per-nuclide
/// device slices via `cuMemcpyDtoD`. Returns the four GPU buffers
/// for category A.1 (energy grids + indexing) plus categories A.2
/// (total_xs / pointwise) and A.3 (ν̄ / delayed-ν̄).
///
/// The returned shape is byte-identical to the corresponding fields
/// in `GpuNuclideData` as built by
/// `gpu_transport.rs::upload_nuclide_data`. Sentinels match:
/// when no nuclide contributes data the result is a 1-element
/// zero slice (legacy path's `if vec.is_empty() { vec.push(0.0); }`
/// convention).
pub struct AssembledBundleACat {
    pub all_energy_grids: CudaSlice<f64>,
    pub grid_offsets_vec: Vec<i32>,
    pub n_energies_vec: Vec<i32>,

    pub total_xs: CudaSlice<f64>,
    pub total_xs_off_vec: Vec<i32>,
    pub has_total_xs_vec: Vec<i32>,

    pub pointwise_xs: CudaSlice<f64>,
    pub pw_off_vec: Vec<i32>,
    pub has_pw_vec: Vec<i32>,

    pub nu_bar_energies: CudaSlice<f64>,
    pub nu_bar_values: CudaSlice<f64>,
    pub nu_bar_offsets_vec: Vec<i32>,
    pub nu_bar_sizes_vec: Vec<i32>,

    pub delayed_nu_bar_energies: CudaSlice<f64>,
    pub delayed_nu_bar_values: CudaSlice<f64>,
    pub delayed_nu_bar_offsets_vec: Vec<i32>,
    pub delayed_nu_bar_sizes_vec: Vec<i32>,
}

pub fn assemble_a_cat(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<AssembledBundleACat, Box<dyn std::error::Error>> {
    // A.1 — energy grids (always populated; 1-point sentinel when the
    // nuclide carries no kernels at all, which mirrors the legacy
    // path's `None` branch).
    let grid_refs: Vec<&CudaSlice<f64>> =
        per_nucs.iter().map(|p| &p.energy_grid).collect();
    let (all_energy_grids, grid_offsets_vec) = concat_dtod_required(stream, &grid_refs)?;
    let n_energies_vec: Vec<i32> = per_nucs.iter().map(|p| p.n_energy).collect();

    // A.2 — total_xs + pointwise.
    let total_refs: Vec<Option<&CudaSlice<f64>>> =
        per_nucs.iter().map(|p| p.total_xs.as_ref()).collect();
    let (total_xs, total_xs_off_vec, has_total_xs_vec) =
        concat_dtod_optional(stream, &total_refs)?;

    let pw_refs: Vec<Option<&CudaSlice<f64>>> =
        per_nucs.iter().map(|p| p.pointwise_xs.as_ref()).collect();
    let (pointwise_xs, pw_off_vec, has_pw_vec) =
        concat_dtod_optional(stream, &pw_refs)?;

    // A.3 — ν̄ + delayed-ν̄. Encoded as (offset, size) pairs in legacy
    // path; `has` flag isn't materialised separately.
    let nb_e_refs: Vec<Option<&CudaSlice<f64>>> = per_nucs
        .iter()
        .map(|p| p.nu_bar.as_ref().map(|nb| &nb.energies))
        .collect();
    let nb_v_refs: Vec<Option<&CudaSlice<f64>>> = per_nucs
        .iter()
        .map(|p| p.nu_bar.as_ref().map(|nb| &nb.values))
        .collect();
    let (nu_bar_energies, nu_bar_offsets_vec, _) =
        concat_dtod_optional(stream, &nb_e_refs)?;
    let (nu_bar_values, _, _) = concat_dtod_optional(stream, &nb_v_refs)?;
    let nu_bar_sizes_vec: Vec<i32> = per_nucs
        .iter()
        .map(|p| p.nu_bar.as_ref().map(|nb| nb.n_points).unwrap_or(0))
        .collect();

    let dnb_e_refs: Vec<Option<&CudaSlice<f64>>> = per_nucs
        .iter()
        .map(|p| p.delayed_nu_bar.as_ref().map(|nb| &nb.energies))
        .collect();
    let dnb_v_refs: Vec<Option<&CudaSlice<f64>>> = per_nucs
        .iter()
        .map(|p| p.delayed_nu_bar.as_ref().map(|nb| &nb.values))
        .collect();
    let (delayed_nu_bar_energies, delayed_nu_bar_offsets_vec, _) =
        concat_dtod_optional(stream, &dnb_e_refs)?;
    let (delayed_nu_bar_values, _, _) = concat_dtod_optional(stream, &dnb_v_refs)?;
    let delayed_nu_bar_sizes_vec: Vec<i32> = per_nucs
        .iter()
        .map(|p| p.delayed_nu_bar.as_ref().map(|nb| nb.n_points).unwrap_or(0))
        .collect();

    Ok(AssembledBundleACat {
        all_energy_grids,
        grid_offsets_vec,
        n_energies_vec,
        total_xs,
        total_xs_off_vec,
        has_total_xs_vec,
        pointwise_xs,
        pw_off_vec,
        has_pw_vec,
        nu_bar_energies,
        nu_bar_values,
        nu_bar_offsets_vec,
        nu_bar_sizes_vec,
        delayed_nu_bar_energies,
        delayed_nu_bar_values,
        delayed_nu_bar_offsets_vec,
        delayed_nu_bar_sizes_vec,
    })
}

/// Assembled cat-B (per-reaction SVD) bundle slices.
pub struct AssembledBundleBCat {
    pub all_basis: CudaSlice<f64>,
    pub all_coeffs: CudaSlice<f64>,
    /// `[n_nuc × N_RXN_SLOTS]` row-major. Slot 6 (total) is `0` since
    /// `has_reaction[*, 6]` is always 0.
    pub basis_offsets_vec: Vec<i32>,
    pub coeffs_offsets_vec: Vec<i32>,
    pub has_reaction_vec: Vec<i32>,
}

/// Concatenate per-(nuclide × reaction) basis / coeffs from
/// `[Arc<PerNuclideGpu>]` into the bundle's flat layout. Slot order
/// (elastic, inelastic, n2n, n3n, fission, capture, total) matches
/// `gpu_transport::upload_nuclide_data` slot 1274-1325.
pub fn assemble_b_cat(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<AssembledBundleBCat, Box<dyn std::error::Error>> {
    let n_nuc = per_nucs.len();
    let n_rxn = N_RXN_SLOTS;

    let total_basis: usize = per_nucs
        .iter()
        .flat_map(|p| p.basis.iter().filter_map(|b| b.as_ref()))
        .map(|s| s.len())
        .sum();
    let total_coeffs: usize = per_nucs
        .iter()
        .flat_map(|p| p.coeffs.iter().filter_map(|c| c.as_ref()))
        .map(|s| s.len())
        .sum();

    let mut all_basis = unsafe { stream.alloc::<f64>(total_basis.max(1))? };
    let mut all_coeffs = unsafe { stream.alloc::<f64>(total_coeffs.max(1))? };
    if total_basis == 0 {
        stream.memset_zeros(&mut all_basis)?;
    }
    if total_coeffs == 0 {
        stream.memset_zeros(&mut all_coeffs)?;
    }

    let mut basis_offsets_vec = vec![0_i32; n_nuc * n_rxn];
    let mut coeffs_offsets_vec = vec![0_i32; n_nuc * n_rxn];
    let mut has_reaction_vec = vec![0_i32; n_nuc * n_rxn];
    let mut running_basis = 0_usize;
    let mut running_coeffs = 0_usize;
    for (nuc_idx, p) in per_nucs.iter().enumerate() {
        for slot in 0..n_rxn {
            let key = nuc_idx * n_rxn + slot;
            has_reaction_vec[key] = p.has_reaction[slot];
            if let Some(b) = &p.basis[slot] {
                basis_offsets_vec[key] = running_basis as i32;
                let len = b.len();
                let mut view = all_basis.slice_mut(running_basis..running_basis + len);
                stream.memcpy_dtod(b, &mut view)?;
                running_basis += len;
            }
            if let Some(c) = &p.coeffs[slot] {
                coeffs_offsets_vec[key] = running_coeffs as i32;
                let len = c.len();
                let mut view = all_coeffs.slice_mut(running_coeffs..running_coeffs + len);
                stream.memcpy_dtod(c, &mut view)?;
                running_coeffs += len;
            }
        }
    }

    Ok(AssembledBundleBCat {
        all_basis,
        all_coeffs,
        basis_offsets_vec,
        coeffs_offsets_vec,
        has_reaction_vec,
    })
}

/// Assembled cat-C (discrete inelastic levels) bundle slices.
pub struct AssembledBundleCCat {
    pub level_q_values: CudaSlice<f64>,
    pub level_thresholds: CudaSlice<f64>,
    pub level_mt: CudaSlice<i32>,
    pub level_has_kernel: CudaSlice<i32>,
    pub level_offsets_vec: Vec<i32>,
    pub level_counts_vec: Vec<i32>,
    pub level_basis: CudaSlice<f64>,
    pub level_coeffs: CudaSlice<f64>,
    pub level_basis_offsets_vec: Vec<i32>,
    pub level_coeffs_offsets_vec: Vec<i32>,
    pub lev_ang_energies: CudaSlice<f64>,
    pub lev_ang_mu: CudaSlice<f64>,
    pub lev_ang_cdf: CudaSlice<f64>,
    pub lev_ang_pdf: CudaSlice<f64>,
    pub lev_ang_dist_off_vec: Vec<i32>,
    pub lev_ang_dist_sz_vec: Vec<i32>,
    pub lev_ang_lev_off_vec: Vec<i32>,
    pub lev_ang_lev_ne_vec: Vec<i32>,
    /// `[total_ang_dist]` — un-shifted within-nuc ang_mu offsets.
    /// Indexed by global ang_energy idx (same indexing as
    /// `lev_ang_dist_off_vec`); value is the per-nuclide-local
    /// offset into `LevelSlicesGpu::ang_mu` / `.ang_cdf`. Step D
    /// pairs with `P_LEV_ANG_MU_PTRS[hit_nuc]`.
    pub lev_ang_dist_local_off_vec: Vec<i32>,
    /// `[total_levels]` — un-shifted within-nuc ang_energy offsets.
    /// Indexed by global level idx. Step D pairs with
    /// `P_LEV_ANG_E_PTRS[hit_nuc]`.
    pub lev_ang_lev_local_off_vec: Vec<i32>,
}

/// Assemble cat-C discrete inelastic level data from per-nuclide
/// slices via DtoD copy plus host-side offset shifting. Mirrors
/// `gpu_transport::upload_nuclide_data` slots 1399-1571 and
/// the sentinel rules at 1543-1569.
///
/// Tricky points:
/// - per-nuclide `LevelSlicesGpu` carries `*_real_len` so the
///   assembler slices past per-nuclide sentinels;
/// - per-level offsets need *two* shifts to become global: by the
///   running global offset into the bundle's flat buffer, AND
///   nothing else (the per-nuclide ones are already 0-based within
///   the nuclide);
/// - `lev_ang_dist_*` and `lev_ang_lev_*` are indexed by separate
///   per-nuclide running counts (per-level vs per-(level, e_inc)).
pub fn assemble_c_cat(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<AssembledBundleCCat, Box<dyn std::error::Error>> {
    // ── Step 1: totals (host-side prefix-sum reconnaissance) ──
    let total_levels: usize = per_nucs.iter().map(|p| p.levels.n_levels as usize).sum();
    let total_basis: usize = per_nucs.iter().map(|p| p.levels.basis_real_len).sum();
    let total_coeffs: usize = per_nucs.iter().map(|p| p.levels.coeffs_real_len).sum();
    let total_ang_e: usize = per_nucs.iter().map(|p| p.levels.ang_e_real_len).sum();
    let total_ang_mu: usize = per_nucs.iter().map(|p| p.levels.ang_mu_real_len).sum();
    let total_ang_dist: usize = per_nucs.iter().map(|p| p.levels.ang_dist_real_len).sum();

    // ── Step 2: allocate destination CudaSlices (with sentinel rules
    //    matching the legacy bundle). When the bundle is empty we
    //    must emit a 1-element sentinel — clone_htod of an empty Vec
    //    leaves a null device pointer on some drivers and the legacy
    //    path pushes [0.0] / [0] in that case. ──
    let level_n = total_levels.max(1);
    let basis_n = total_basis.max(1);
    let coeffs_n = total_coeffs.max(1);
    let ang_e_n = total_ang_e.max(1);
    let ang_mu_n = total_ang_mu.max(1);
    let _ang_dist_n = total_ang_dist.max(1);

    let mut level_q_values = unsafe { stream.alloc::<f64>(level_n)? };
    let mut level_thresholds = unsafe { stream.alloc::<f64>(level_n)? };
    let mut level_mt = unsafe { stream.alloc::<i32>(level_n)? };
    let mut level_has_kernel = unsafe { stream.alloc::<i32>(level_n)? };
    let mut level_basis = unsafe { stream.alloc::<f64>(basis_n)? };
    let mut level_coeffs = unsafe { stream.alloc::<f64>(coeffs_n)? };
    let mut lev_ang_energies = unsafe { stream.alloc::<f64>(ang_e_n)? };
    let mut lev_ang_mu = unsafe { stream.alloc::<f64>(ang_mu_n)? };
    let mut lev_ang_cdf = unsafe { stream.alloc::<f64>(ang_mu_n)? };
    let mut lev_ang_pdf = unsafe { stream.alloc::<f64>(ang_mu_n)? };

    // Zero out sentinel slots (only relevant when total = 0 → alloc is
    // size 1 sentinel; otherwise the slot is overwritten below).
    if total_levels == 0 {
        stream.memset_zeros(&mut level_q_values)?;
        stream.memset_zeros(&mut level_thresholds)?;
        stream.memset_zeros(&mut level_mt)?;
        stream.memset_zeros(&mut level_has_kernel)?;
    }
    if total_basis == 0 {
        stream.memset_zeros(&mut level_basis)?;
    }
    if total_coeffs == 0 {
        stream.memset_zeros(&mut level_coeffs)?;
    }
    if total_ang_e == 0 {
        stream.memset_zeros(&mut lev_ang_energies)?;
    }
    if total_ang_mu == 0 {
        stream.memset_zeros(&mut lev_ang_mu)?;
        stream.memset_zeros(&mut lev_ang_cdf)?;
        stream.memset_zeros(&mut lev_ang_pdf)?;
    }

    // ── Step 3: walk nuclides; DtoD the per-nuclide real-length
    //    slices and emit host-side shifted offsets. ──
    let mut level_offsets_vec = Vec::with_capacity(per_nucs.len());
    let mut level_counts_vec = Vec::with_capacity(per_nucs.len());
    let mut level_basis_offsets_vec: Vec<i32> = Vec::with_capacity(total_levels);
    let mut level_coeffs_offsets_vec: Vec<i32> = Vec::with_capacity(total_levels);
    let mut lev_ang_dist_off_vec: Vec<i32> = Vec::with_capacity(total_ang_dist);
    let mut lev_ang_dist_local_off_vec: Vec<i32> = Vec::with_capacity(total_ang_dist);
    let mut lev_ang_dist_sz_vec: Vec<i32> = Vec::with_capacity(total_ang_dist);
    let mut lev_ang_lev_off_vec: Vec<i32> = Vec::with_capacity(total_levels);
    let mut lev_ang_lev_local_off_vec: Vec<i32> = Vec::with_capacity(total_levels);
    let mut lev_ang_lev_ne_vec: Vec<i32> = Vec::with_capacity(total_levels);

    let mut run_level = 0_usize;
    let mut run_basis = 0_usize;
    let mut run_coeffs = 0_usize;
    let mut run_ang_e = 0_usize;
    let mut run_ang_mu = 0_usize;

    for p in per_nucs.iter() {
        let nl = p.levels.n_levels as usize;
        level_offsets_vec.push(run_level as i32);
        level_counts_vec.push(nl as i32);

        if nl > 0 {
            // Copy first `nl` elements of the per-nuclide level
            // scalars into the bundle. `slice(0..nl)` skips per-
            // nuclide sentinel padding (which is the only entry when
            // nl == 0 — and we just don't enter this branch).
            let mut dst_q = level_q_values.slice_mut(run_level..run_level + nl);
            stream.memcpy_dtod(&p.levels.q_values.slice(0..nl), &mut dst_q)?;
            let mut dst_t = level_thresholds.slice_mut(run_level..run_level + nl);
            stream.memcpy_dtod(&p.levels.thresholds.slice(0..nl), &mut dst_t)?;
            let mut dst_m = level_mt.slice_mut(run_level..run_level + nl);
            stream.memcpy_dtod(&p.levels.mt.slice(0..nl), &mut dst_m)?;
            let mut dst_h = level_has_kernel.slice_mut(run_level..run_level + nl);
            stream.memcpy_dtod(&p.levels.has_kernel.slice(0..nl), &mut dst_h)?;
        }

        let blen = p.levels.basis_real_len;
        let clen = p.levels.coeffs_real_len;
        if blen > 0 {
            let mut dst = level_basis.slice_mut(run_basis..run_basis + blen);
            stream.memcpy_dtod(&p.levels.basis.slice(0..blen), &mut dst)?;
        }
        if clen > 0 {
            let mut dst = level_coeffs.slice_mut(run_coeffs..run_coeffs + clen);
            stream.memcpy_dtod(&p.levels.coeffs.slice(0..clen), &mut dst)?;
        }

        // Per-level basis / coeffs offsets — shift the per-nuclide
        // local offset by the running global byte counters.
        for li in 0..nl {
            level_basis_offsets_vec
                .push(p.levels.basis_local_off[li] + run_basis as i32);
            level_coeffs_offsets_vec
                .push(p.levels.coeffs_local_off[li] + run_coeffs as i32);
        }
        // Per-level angular-energy locator — shift by global ang_e
        // running offset.
        for li in 0..nl {
            lev_ang_lev_off_vec.push(p.levels.ang_lev_local_off[li] + run_ang_e as i32);
            lev_ang_lev_local_off_vec.push(p.levels.ang_lev_local_off[li]);
            lev_ang_lev_ne_vec.push(p.levels.ang_lev_ne[li]);
        }

        // Per-(level, e_inc) angular distribution locator — shift by
        // global ang_mu running offset. The `_local_off` parallel
        // version stores within-nuc offsets (un-shifted) so Step D
        // can pair it with the per-nuc `lev_ang_mu` base pointer.
        let adlen = p.levels.ang_dist_real_len;
        for di in 0..adlen {
            lev_ang_dist_off_vec
                .push(p.levels.ang_dist_local_off[di] + run_ang_mu as i32);
            lev_ang_dist_local_off_vec.push(p.levels.ang_dist_local_off[di]);
            lev_ang_dist_sz_vec.push(p.levels.ang_dist_sz[di]);
        }

        // Copy per-nuclide angular-energy and (mu, cdf) bytes into
        // the bundle.
        let aelen = p.levels.ang_e_real_len;
        if aelen > 0 {
            let mut dst_e = lev_ang_energies.slice_mut(run_ang_e..run_ang_e + aelen);
            stream.memcpy_dtod(&p.levels.ang_energies.slice(0..aelen), &mut dst_e)?;
        }
        let amlen = p.levels.ang_mu_real_len;
        if amlen > 0 {
            let mut dst_mu = lev_ang_mu.slice_mut(run_ang_mu..run_ang_mu + amlen);
            stream.memcpy_dtod(&p.levels.ang_mu.slice(0..amlen), &mut dst_mu)?;
            let mut dst_cdf = lev_ang_cdf.slice_mut(run_ang_mu..run_ang_mu + amlen);
            stream.memcpy_dtod(&p.levels.ang_cdf.slice(0..amlen), &mut dst_cdf)?;
            let mut dst_pdf = lev_ang_pdf.slice_mut(run_ang_mu..run_ang_mu + amlen);
            stream.memcpy_dtod(&p.levels.ang_pdf.slice(0..amlen), &mut dst_pdf)?;
        }

        run_level += nl;
        run_basis += blen;
        run_coeffs += clen;
        run_ang_e += aelen;
        run_ang_mu += amlen;
    }

    // ── Bundle-level sentinels for the offset Vecs. When the bundle
    //    has zero levels the legacy path pushes single-element
    //    [0]-valued sentinels (slot 1543-1551) so the device pointers
    //    remain valid even when the kernel never reads them. ──
    if level_basis_offsets_vec.is_empty() {
        level_basis_offsets_vec.push(0);
        level_coeffs_offsets_vec.push(0);
        lev_ang_lev_off_vec.push(0);
        lev_ang_lev_local_off_vec.push(0);
        lev_ang_lev_ne_vec.push(0);
    }
    if lev_ang_dist_off_vec.is_empty() {
        lev_ang_dist_off_vec.push(0);
        lev_ang_dist_local_off_vec.push(0);
        lev_ang_dist_sz_vec.push(0);
    }

    Ok(AssembledBundleCCat {
        level_q_values,
        level_thresholds,
        level_mt,
        level_has_kernel,
        level_offsets_vec,
        level_counts_vec,
        level_basis,
        level_coeffs,
        level_basis_offsets_vec,
        level_coeffs_offsets_vec,
        lev_ang_energies,
        lev_ang_mu,
        lev_ang_cdf,
        lev_ang_pdf,
        lev_ang_dist_off_vec,
        lev_ang_dist_sz_vec,
        lev_ang_lev_off_vec,
        lev_ang_lev_ne_vec,
        lev_ang_dist_local_off_vec,
        lev_ang_lev_local_off_vec,
    })
}

/// Assembled cat-A.8 (synthesized MT=4 inelastic CDF) bundle slices.
/// Per-nuclide flat CDF tensors are concatenated; absent nuclides
/// get `inel_cdf_off = -1` (the device's "no CDF, use legacy per-
/// level walk" sentinel — matches legacy slot 1378's initial value).
pub struct AssembledBundleA8Cat {
    pub inel_cdf_data: CudaSlice<f64>,
    pub inel_cdf_off_vec: Vec<i32>,
    pub inel_cdf_n_e_vec: Vec<i32>,
    pub inel_cdf_n_t_vec: Vec<i32>,
    pub inel_cdf_n_lev_vec: Vec<i32>,
    pub inel_cdf_log_e_min_vec: Vec<f64>,
    pub inel_cdf_log_e_max_vec: Vec<f64>,
}

pub fn assemble_a8_cat(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<AssembledBundleA8Cat, Box<dyn std::error::Error>> {
    let n_nuc = per_nucs.len();
    let total: usize = per_nucs
        .iter()
        .filter_map(|p| p.inel_cdf.as_ref().map(|c| c.data.len()))
        .sum();
    let mut inel_cdf_data = unsafe { stream.alloc::<f64>(total.max(1))? };
    if total == 0 {
        stream.memset_zeros(&mut inel_cdf_data)?;
    }

    let mut inel_cdf_off_vec = vec![-1_i32; n_nuc];
    let mut inel_cdf_n_e_vec = vec![0_i32; n_nuc];
    let mut inel_cdf_n_t_vec = vec![0_i32; n_nuc];
    let mut inel_cdf_n_lev_vec = vec![0_i32; n_nuc];
    let mut inel_cdf_log_e_min_vec = vec![0.0_f64; n_nuc];
    let mut inel_cdf_log_e_max_vec = vec![0.0_f64; n_nuc];

    let mut run = 0_usize;
    for (nuc_idx, p) in per_nucs.iter().enumerate() {
        let Some(c) = p.inel_cdf.as_ref() else { continue };
        inel_cdf_off_vec[nuc_idx] = run as i32;
        inel_cdf_n_e_vec[nuc_idx] = c.n_e;
        inel_cdf_n_t_vec[nuc_idx] = c.n_t;
        inel_cdf_n_lev_vec[nuc_idx] = c.n_lev;
        inel_cdf_log_e_min_vec[nuc_idx] = c.log_e_min;
        inel_cdf_log_e_max_vec[nuc_idx] = c.log_e_max;
        let len = c.data.len();
        if len > 0 {
            let mut dst = inel_cdf_data.slice_mut(run..run + len);
            stream.memcpy_dtod(&c.data, &mut dst)?;
        }
        run += len;
    }

    Ok(AssembledBundleA8Cat {
        inel_cdf_data,
        inel_cdf_off_vec,
        inel_cdf_n_e_vec,
        inel_cdf_n_t_vec,
        inel_cdf_n_lev_vec,
        inel_cdf_log_e_min_vec,
        inel_cdf_log_e_max_vec,
    })
}

/// Assembled cat-A.7 (URR probability tables) bundle slices.
pub struct AssembledBundleA7Cat {
    pub urr_energies: CudaSlice<f64>,
    pub urr_cum_prob: CudaSlice<f64>,
    pub urr_total_f: CudaSlice<f64>,
    pub urr_elastic_f: CudaSlice<f64>,
    pub urr_fission_f: CudaSlice<f64>,
    pub urr_capture_f: CudaSlice<f64>,
    pub urr_offsets_vec: Vec<i32>,
    pub urr_n_energies_vec: Vec<i32>,
    pub urr_n_bands_vec: Vec<i32>,
    pub urr_multiply_smooth_vec: Vec<i32>,
    /// Per-nuclide URR interpolation code (2 = lin-lin, 5 = log-log).
    /// `0` when no URR for this nuclide (kernel guards on
    /// `urr_n_energies_vec[ni] > 0` anyway).
    pub urr_interpolation_vec: Vec<i32>,
}

pub fn assemble_a7_cat(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<AssembledBundleA7Cat, Box<dyn std::error::Error>> {
    let n_nuc = per_nucs.len();
    let total_e: usize = per_nucs
        .iter()
        .filter_map(|p| p.urr.as_ref().map(|u| u.n_energies as usize))
        .sum();
    let total_fac: usize = per_nucs
        .iter()
        .filter_map(|p| p.urr.as_ref())
        .map(|u| (u.n_energies as usize) * (u.n_bands as usize))
        .sum();

    let mut urr_energies = unsafe { stream.alloc::<f64>(total_e.max(1))? };
    let mut urr_cum_prob = unsafe { stream.alloc::<f64>(total_fac.max(1))? };
    let mut urr_total_f = unsafe { stream.alloc::<f64>(total_fac.max(1))? };
    let mut urr_elastic_f = unsafe { stream.alloc::<f64>(total_fac.max(1))? };
    let mut urr_fission_f = unsafe { stream.alloc::<f64>(total_fac.max(1))? };
    let mut urr_capture_f = unsafe { stream.alloc::<f64>(total_fac.max(1))? };
    if total_e == 0 {
        stream.memset_zeros(&mut urr_energies)?;
    }
    if total_fac == 0 {
        stream.memset_zeros(&mut urr_cum_prob)?;
        stream.memset_zeros(&mut urr_total_f)?;
        stream.memset_zeros(&mut urr_elastic_f)?;
        stream.memset_zeros(&mut urr_fission_f)?;
        stream.memset_zeros(&mut urr_capture_f)?;
    }

    let mut urr_offsets_vec = vec![0_i32; n_nuc];
    let mut urr_n_energies_vec = vec![0_i32; n_nuc];
    let mut urr_n_bands_vec = vec![0_i32; n_nuc];
    let mut urr_multiply_smooth_vec = vec![0_i32; n_nuc];
    let mut urr_interpolation_vec = vec![0_i32; n_nuc];

    let mut run_e = 0_usize;
    let mut run_fac = 0_usize;
    for (nuc_idx, p) in per_nucs.iter().enumerate() {
        let Some(u) = p.urr.as_ref() else { continue };
        let ne = u.n_energies as usize;
        let nb = u.n_bands as usize;
        let fac_len = ne * nb;
        urr_offsets_vec[nuc_idx] = run_e as i32;
        urr_n_energies_vec[nuc_idx] = u.n_energies;
        urr_n_bands_vec[nuc_idx] = u.n_bands;
        urr_multiply_smooth_vec[nuc_idx] = u.multiply_smooth;
        urr_interpolation_vec[nuc_idx] = u.interpolation;
        if ne > 0 {
            let mut dst = urr_energies.slice_mut(run_e..run_e + ne);
            stream.memcpy_dtod(&u.energies, &mut dst)?;
        }
        if fac_len > 0 {
            let mut v = urr_cum_prob.slice_mut(run_fac..run_fac + fac_len);
            stream.memcpy_dtod(&u.cum_prob, &mut v)?;
            let mut v = urr_total_f.slice_mut(run_fac..run_fac + fac_len);
            stream.memcpy_dtod(&u.total_factor, &mut v)?;
            let mut v = urr_elastic_f.slice_mut(run_fac..run_fac + fac_len);
            stream.memcpy_dtod(&u.elastic_factor, &mut v)?;
            let mut v = urr_fission_f.slice_mut(run_fac..run_fac + fac_len);
            stream.memcpy_dtod(&u.fission_factor, &mut v)?;
            let mut v = urr_capture_f.slice_mut(run_fac..run_fac + fac_len);
            stream.memcpy_dtod(&u.capture_factor, &mut v)?;
        }
        run_e += ne;
        run_fac += fac_len;
    }

    Ok(AssembledBundleA7Cat {
        urr_energies,
        urr_cum_prob,
        urr_total_f,
        urr_elastic_f,
        urr_fission_f,
        urr_capture_f,
        urr_offsets_vec,
        urr_n_energies_vec,
        urr_n_bands_vec,
        urr_multiply_smooth_vec,
        urr_interpolation_vec,
    })
}

/// Assembled cat-A.6 (MT=91 continuum inelastic outgoing-energy)
/// bundle slices. Tabular layout, identical to the fission Tabular
/// branch but indexed against `inel91_*` fields. When no nuclide
/// carries MT=91 data the bundle gets the standard 1-element
/// sentinel slices (legacy slot 1831-1842).
pub struct AssembledBundleA6Cat {
    pub inel91_inc_energies: CudaSlice<f64>,
    pub inel91_dist_offsets_vec: Vec<i32>,
    pub inel91_dist_local_off_vec: Vec<i32>,
    pub inel91_dist_sizes_vec: Vec<i32>,
    pub inel91_e_out: CudaSlice<f64>,
    pub inel91_cdf: CudaSlice<f64>,
    pub inel91_pdf: CudaSlice<f64>,
    pub inel91_nuc_offsets_vec: Vec<i32>,
    pub inel91_nuc_n_inc_vec: Vec<i32>,
}

pub fn assemble_a6_cat(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<AssembledBundleA6Cat, Box<dyn std::error::Error>> {
    let n_nuc = per_nucs.len();
    let total_inc: usize = per_nucs
        .iter()
        .filter_map(|p| p.inel91.as_ref().map(|t| t.n_inc as usize))
        .sum();
    let total_eout: usize = per_nucs
        .iter()
        .filter_map(|p| p.inel91.as_ref().map(|t| t.e_out.len()))
        .sum();

    let mut inel91_inc_energies = unsafe { stream.alloc::<f64>(total_inc.max(1))? };
    let mut inel91_e_out = unsafe { stream.alloc::<f64>(total_eout.max(1))? };
    let mut inel91_cdf = unsafe { stream.alloc::<f64>(total_eout.max(1))? };
    let mut inel91_pdf = unsafe { stream.alloc::<f64>(total_eout.max(1))? };
    if total_inc == 0 {
        stream.memset_zeros(&mut inel91_inc_energies)?;
    }
    if total_eout == 0 {
        stream.memset_zeros(&mut inel91_e_out)?;
        stream.memset_zeros(&mut inel91_cdf)?;
        stream.memset_zeros(&mut inel91_pdf)?;
    }

    let mut inel91_dist_offsets_vec: Vec<i32> = Vec::with_capacity(total_inc);
    let mut inel91_dist_local_off_vec: Vec<i32> = Vec::with_capacity(total_inc);
    let mut inel91_dist_sizes_vec: Vec<i32> = Vec::with_capacity(total_inc);
    let mut inel91_nuc_offsets_vec = vec![0_i32; n_nuc];
    let mut inel91_nuc_n_inc_vec = vec![0_i32; n_nuc];

    let mut run_inc = 0_usize;
    let mut run_eout = 0_usize;
    for (nuc_idx, p) in per_nucs.iter().enumerate() {
        let Some(t) = p.inel91.as_ref() else { continue };
        let ni = t.n_inc as usize;
        inel91_nuc_offsets_vec[nuc_idx] = run_inc as i32;
        inel91_nuc_n_inc_vec[nuc_idx] = t.n_inc;
        if ni > 0 {
            let mut dst = inel91_inc_energies.slice_mut(run_inc..run_inc + ni);
            stream.memcpy_dtod(&t.inc_energies.slice(0..ni), &mut dst)?;
        }
        let elen = t.e_out.len();
        if elen > 0 {
            let mut dst_e = inel91_e_out.slice_mut(run_eout..run_eout + elen);
            stream.memcpy_dtod(&t.e_out, &mut dst_e)?;
            let mut dst_c = inel91_cdf.slice_mut(run_eout..run_eout + elen);
            stream.memcpy_dtod(&t.cdf, &mut dst_c)?;
            let mut dst_p = inel91_pdf.slice_mut(run_eout..run_eout + elen);
            stream.memcpy_dtod(&t.pdf, &mut dst_p)?;
        }
        for di in 0..ni {
            inel91_dist_offsets_vec.push(t.dist_local_off[di] + run_eout as i32);
            inel91_dist_local_off_vec.push(t.dist_local_off[di]);
            inel91_dist_sizes_vec.push(t.dist_sz[di]);
        }
        run_inc += ni;
        run_eout += elen;
    }

    if inel91_dist_offsets_vec.is_empty() {
        inel91_dist_offsets_vec.push(0);
        inel91_dist_local_off_vec.push(0);
        inel91_dist_sizes_vec.push(0);
    }

    Ok(AssembledBundleA6Cat {
        inel91_inc_energies,
        inel91_dist_offsets_vec,
        inel91_dist_local_off_vec,
        inel91_dist_sizes_vec,
        inel91_e_out,
        inel91_cdf,
        inel91_pdf,
        inel91_nuc_offsets_vec,
        inel91_nuc_n_inc_vec,
    })
}

/// Assembled cat-A.6 — see `AssembledBundleA6Cat`. Step D needs a
/// `lev_ang_dist_local_off`-style un-shifted parallel for both
/// fission tabular and MT=91 paths; we extend the existing structs
/// rather than adding new ones.
///
/// Assembled cat-A.5 (fission outgoing-energy distribution) bundle
/// slices. Three exclusive branches: a nuclide contributes to at
/// most one of `fis_*` (Tabular), `watt_*` (Law 11), or `maxevap_*`
/// (Law 7 / Law 9). When a branch has zero contributing nuclides
/// the bundle still emits a 1-element sentinel slice so the device
/// pointer is non-null — matches upload_nuclide_data
/// slot 1780-1869.
pub struct AssembledBundleA5Cat {
    // Tabular (Law 4 / 61).
    pub fis_inc_energies: CudaSlice<f64>,
    pub fis_dist_offsets_vec: Vec<i32>,
    /// `[total_inc_tab]` — un-shifted within-nuc fis_e_out offsets,
    /// parallel to `fis_dist_offsets_vec`. Step D pairs with
    /// `P_FIS_E_OUT_PTRS[hit_nuc]`.
    pub fis_dist_local_off_vec: Vec<i32>,
    pub fis_dist_sizes_vec: Vec<i32>,
    pub fis_e_out: CudaSlice<f64>,
    pub fis_cdf: CudaSlice<f64>,
    pub fis_pdf: CudaSlice<f64>,
    pub fis_nuc_offsets_vec: Vec<i32>,
    pub fis_nuc_n_inc_vec: Vec<i32>,
    // Watt (Law 11).
    pub watt_inc_energies: CudaSlice<f64>,
    pub watt_a: CudaSlice<f64>,
    pub watt_b: CudaSlice<f64>,
    pub watt_u_vec: Vec<f64>,
    pub watt_nuc_offsets_vec: Vec<i32>,
    pub watt_nuc_n_vec: Vec<i32>,
    // Maxwell (Law 7) / Evaporation (Law 9).
    pub maxevap_inc_energies: CudaSlice<f64>,
    pub maxevap_theta: CudaSlice<f64>,
    pub maxevap_u_vec: Vec<f64>,
    pub maxevap_law_vec: Vec<i32>,
    pub maxevap_nuc_offsets_vec: Vec<i32>,
    pub maxevap_nuc_n_vec: Vec<i32>,
}

pub fn assemble_a5_cat(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<AssembledBundleA5Cat, Box<dyn std::error::Error>> {
    let n_nuc = per_nucs.len();

    // ── Tabular branch ──
    let total_inc_tab: usize = per_nucs
        .iter()
        .filter_map(|p| match &p.fission_edist {
            FissionEdistGpu::Tabular(t) => Some(t.n_inc as usize),
            _ => None,
        })
        .sum();
    let total_eout_tab: usize = per_nucs
        .iter()
        .filter_map(|p| match &p.fission_edist {
            FissionEdistGpu::Tabular(t) => Some(t.e_out.len()),
            _ => None,
        })
        .sum();

    let mut fis_inc_energies = unsafe { stream.alloc::<f64>(total_inc_tab.max(1))? };
    let mut fis_e_out = unsafe { stream.alloc::<f64>(total_eout_tab.max(1))? };
    let mut fis_cdf = unsafe { stream.alloc::<f64>(total_eout_tab.max(1))? };
    let mut fis_pdf = unsafe { stream.alloc::<f64>(total_eout_tab.max(1))? };
    if total_inc_tab == 0 {
        stream.memset_zeros(&mut fis_inc_energies)?;
    }
    if total_eout_tab == 0 {
        stream.memset_zeros(&mut fis_e_out)?;
        stream.memset_zeros(&mut fis_cdf)?;
        stream.memset_zeros(&mut fis_pdf)?;
    }

    let mut fis_dist_offsets_vec: Vec<i32> = Vec::with_capacity(total_inc_tab);
    let mut fis_dist_local_off_vec: Vec<i32> = Vec::with_capacity(total_inc_tab);
    let mut fis_dist_sizes_vec: Vec<i32> = Vec::with_capacity(total_inc_tab);
    let mut fis_nuc_offsets_vec = vec![0_i32; n_nuc];
    let mut fis_nuc_n_inc_vec = vec![0_i32; n_nuc];

    // ── Watt branch ──
    let total_inc_watt: usize = per_nucs
        .iter()
        .filter_map(|p| match &p.fission_edist {
            FissionEdistGpu::Watt(w) => Some(w.n_inc as usize),
            _ => None,
        })
        .sum();
    let mut watt_inc_energies = unsafe { stream.alloc::<f64>(total_inc_watt.max(1))? };
    let mut watt_a = unsafe { stream.alloc::<f64>(total_inc_watt.max(1))? };
    let mut watt_b = unsafe { stream.alloc::<f64>(total_inc_watt.max(1))? };
    if total_inc_watt == 0 {
        stream.memset_zeros(&mut watt_inc_energies)?;
        stream.memset_zeros(&mut watt_a)?;
        stream.memset_zeros(&mut watt_b)?;
    }
    let mut watt_u_vec = vec![0.0_f64; n_nuc];
    let mut watt_nuc_offsets_vec = vec![0_i32; n_nuc];
    let mut watt_nuc_n_vec = vec![0_i32; n_nuc];

    // ── Maxwell / Evaporation branch ──
    let total_inc_me: usize = per_nucs
        .iter()
        .filter_map(|p| match &p.fission_edist {
            FissionEdistGpu::MaxEvap(m) => Some(m.n_inc as usize),
            _ => None,
        })
        .sum();
    let mut maxevap_inc_energies = unsafe { stream.alloc::<f64>(total_inc_me.max(1))? };
    let mut maxevap_theta = unsafe { stream.alloc::<f64>(total_inc_me.max(1))? };
    if total_inc_me == 0 {
        stream.memset_zeros(&mut maxevap_inc_energies)?;
        stream.memset_zeros(&mut maxevap_theta)?;
    }
    let mut maxevap_u_vec = vec![0.0_f64; n_nuc];
    let mut maxevap_law_vec = vec![0_i32; n_nuc];
    let mut maxevap_nuc_offsets_vec = vec![0_i32; n_nuc];
    let mut maxevap_nuc_n_vec = vec![0_i32; n_nuc];

    // ── Walk per_nucs and dispatch to the appropriate branch ──
    let mut run_tab_inc = 0_usize;
    let mut run_tab_eout = 0_usize;
    let mut run_watt = 0_usize;
    let mut run_me = 0_usize;
    for (nuc_idx, p) in per_nucs.iter().enumerate() {
        match &p.fission_edist {
            FissionEdistGpu::None => {}
            FissionEdistGpu::Tabular(t) => {
                let ni = t.n_inc as usize;
                fis_nuc_offsets_vec[nuc_idx] = run_tab_inc as i32;
                fis_nuc_n_inc_vec[nuc_idx] = t.n_inc;
                if ni > 0 {
                    let mut dst = fis_inc_energies.slice_mut(run_tab_inc..run_tab_inc + ni);
                    stream.memcpy_dtod(&t.inc_energies.slice(0..ni), &mut dst)?;
                }
                let elen = t.e_out.len();
                if elen > 0 {
                    let mut dst_e = fis_e_out.slice_mut(run_tab_eout..run_tab_eout + elen);
                    stream.memcpy_dtod(&t.e_out, &mut dst_e)?;
                    let mut dst_c = fis_cdf.slice_mut(run_tab_eout..run_tab_eout + elen);
                    stream.memcpy_dtod(&t.cdf, &mut dst_c)?;
                    let mut dst_p = fis_pdf.slice_mut(run_tab_eout..run_tab_eout + elen);
                    stream.memcpy_dtod(&t.pdf, &mut dst_p)?;
                }
                for di in 0..ni {
                    fis_dist_offsets_vec.push(t.dist_local_off[di] + run_tab_eout as i32);
                    fis_dist_local_off_vec.push(t.dist_local_off[di]);
                    fis_dist_sizes_vec.push(t.dist_sz[di]);
                }
                run_tab_inc += ni;
                run_tab_eout += elen;
            }
            FissionEdistGpu::Watt(w) => {
                let ni = w.n_inc as usize;
                watt_nuc_offsets_vec[nuc_idx] = run_watt as i32;
                watt_nuc_n_vec[nuc_idx] = w.n_inc;
                watt_u_vec[nuc_idx] = w.u;
                if ni > 0 {
                    let mut dst_e = watt_inc_energies.slice_mut(run_watt..run_watt + ni);
                    stream.memcpy_dtod(&w.inc_energies, &mut dst_e)?;
                    let mut dst_a = watt_a.slice_mut(run_watt..run_watt + ni);
                    stream.memcpy_dtod(&w.a, &mut dst_a)?;
                    let mut dst_b = watt_b.slice_mut(run_watt..run_watt + ni);
                    stream.memcpy_dtod(&w.b, &mut dst_b)?;
                }
                run_watt += ni;
            }
            FissionEdistGpu::MaxEvap(m) => {
                let ni = m.n_inc as usize;
                maxevap_nuc_offsets_vec[nuc_idx] = run_me as i32;
                maxevap_nuc_n_vec[nuc_idx] = m.n_inc;
                maxevap_u_vec[nuc_idx] = m.u;
                maxevap_law_vec[nuc_idx] = m.law;
                if ni > 0 {
                    let mut dst_e = maxevap_inc_energies.slice_mut(run_me..run_me + ni);
                    stream.memcpy_dtod(&m.inc_energies, &mut dst_e)?;
                    let mut dst_t = maxevap_theta.slice_mut(run_me..run_me + ni);
                    stream.memcpy_dtod(&m.theta, &mut dst_t)?;
                }
                run_me += ni;
            }
        }
    }

    // ── Bundle-level sentinels. Legacy paths (1788-1791, 1839-1842)
    //    push [0]/[0.0] for empty dist_off / dist_sz / fis_eout
    //    families. fis_inc_energies and fis_e_out / cdf / pdf are
    //    already allocated to size 1 above when totals were 0. ──
    if fis_dist_offsets_vec.is_empty() {
        fis_dist_offsets_vec.push(0);
        fis_dist_local_off_vec.push(0);
        fis_dist_sizes_vec.push(0);
    }

    Ok(AssembledBundleA5Cat {
        fis_inc_energies,
        fis_dist_offsets_vec,
        fis_dist_local_off_vec,
        fis_dist_sizes_vec,
        fis_e_out,
        fis_cdf,
        fis_pdf,
        fis_nuc_offsets_vec,
        fis_nuc_n_inc_vec,
        watt_inc_energies,
        watt_a,
        watt_b,
        watt_u_vec,
        watt_nuc_offsets_vec,
        watt_nuc_n_vec,
        maxevap_inc_energies,
        maxevap_theta,
        maxevap_u_vec,
        maxevap_law_vec,
        maxevap_nuc_offsets_vec,
        maxevap_nuc_n_vec,
    })
}

/// Assembled cat-A.4 (elastic angular) bundle slices.
pub struct AssembledBundleA4Cat {
    pub ang_energies: CudaSlice<f64>,
    pub ang_mu: CudaSlice<f64>,
    pub ang_cdf: CudaSlice<f64>,
    pub ang_pdf: CudaSlice<f64>,
    pub ang_dist_offsets_vec: Vec<i32>,
    pub ang_dist_sizes_vec: Vec<i32>,
    pub ang_nuc_offsets_vec: Vec<i32>,
    pub ang_nuc_n_energies_vec: Vec<i32>,
    pub ang_is_cm_vec: Vec<i32>,
    /// `[total_e]` — same shape and indexing as `ang_dist_offsets_vec`
    /// but un-shifted: each value is a **within-nuc** offset into
    /// the per-nuclide `AngularSlicesGpu::mu` / `.cdf`. Step D pairs
    /// this with `P_ANG_MU_PTRS[hit_nuc]` for per-nuclide pointer
    /// loading.
    pub ang_dist_local_off_vec: Vec<i32>,
}

/// Bundle assembly for category A.4 (elastic angular distribution).
/// Mirrors upload_nuclide_data:1578-1613. Per-nuclide
/// elastic_angle is optional; when absent the per-nuclide offset /
/// n_energies / is_cm slots stay zero.
pub fn assemble_a4_cat(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<AssembledBundleA4Cat, Box<dyn std::error::Error>> {
    let total_e: usize = per_nucs
        .iter()
        .filter_map(|p| p.elastic_angle.as_ref())
        .map(|a| a.n_energies as usize)
        .sum();
    let total_mu: usize = per_nucs
        .iter()
        .filter_map(|p| p.elastic_angle.as_ref())
        .map(|a| a.mu_real_len)
        .sum();

    let mut ang_energies = unsafe { stream.alloc::<f64>(total_e.max(1))? };
    let mut ang_mu = unsafe { stream.alloc::<f64>(total_mu.max(1))? };
    let mut ang_cdf = unsafe { stream.alloc::<f64>(total_mu.max(1))? };
    let mut ang_pdf = unsafe { stream.alloc::<f64>(total_mu.max(1))? };
    if total_e == 0 {
        stream.memset_zeros(&mut ang_energies)?;
    }
    if total_mu == 0 {
        stream.memset_zeros(&mut ang_mu)?;
        stream.memset_zeros(&mut ang_cdf)?;
        stream.memset_zeros(&mut ang_pdf)?;
    }

    let mut ang_dist_offsets_vec: Vec<i32> = Vec::with_capacity(total_e);
    let mut ang_dist_local_off_vec: Vec<i32> = Vec::with_capacity(total_e);
    let mut ang_dist_sizes_vec: Vec<i32> = Vec::with_capacity(total_e);
    let mut ang_nuc_offsets_vec = vec![0_i32; per_nucs.len()];
    let mut ang_nuc_n_energies_vec = vec![0_i32; per_nucs.len()];
    let mut ang_is_cm_vec = vec![0_i32; per_nucs.len()];

    let mut run_e = 0_usize;
    let mut run_mu = 0_usize;
    for (nuc_idx, p) in per_nucs.iter().enumerate() {
        let Some(ang) = p.elastic_angle.as_ref() else {
            continue;
        };
        let ne = ang.n_energies as usize;
        ang_nuc_offsets_vec[nuc_idx] = run_e as i32;
        ang_nuc_n_energies_vec[nuc_idx] = ang.n_energies;
        ang_is_cm_vec[nuc_idx] = ang.is_cm;

        if ne > 0 {
            let mut dst = ang_energies.slice_mut(run_e..run_e + ne);
            stream.memcpy_dtod(&ang.energies.slice(0..ne), &mut dst)?;
        }
        let ml = ang.mu_real_len;
        if ml > 0 {
            let mut dst_mu = ang_mu.slice_mut(run_mu..run_mu + ml);
            stream.memcpy_dtod(&ang.mu.slice(0..ml), &mut dst_mu)?;
            let mut dst_cdf = ang_cdf.slice_mut(run_mu..run_mu + ml);
            stream.memcpy_dtod(&ang.cdf.slice(0..ml), &mut dst_cdf)?;
            let mut dst_pdf = ang_pdf.slice_mut(run_mu..run_mu + ml);
            stream.memcpy_dtod(&ang.pdf.slice(0..ml), &mut dst_pdf)?;
        }
        for ei in 0..ne {
            ang_dist_offsets_vec.push(ang.dist_local_off[ei] + run_mu as i32);
            ang_dist_local_off_vec.push(ang.dist_local_off[ei]);
            ang_dist_sizes_vec.push(ang.dist_sz[ei]);
        }
        run_e += ne;
        run_mu += ml;
    }

    // Bundle-level sentinels (slot 1610-1613).
    if ang_dist_offsets_vec.is_empty() {
        ang_dist_offsets_vec.push(0);
        ang_dist_local_off_vec.push(0);
        ang_dist_sizes_vec.push(0);
    }

    Ok(AssembledBundleA4Cat {
        ang_energies,
        ang_mu,
        ang_cdf,
        ang_pdf,
        ang_dist_offsets_vec,
        ang_dist_sizes_vec,
        ang_nuc_offsets_vec,
        ang_nuc_n_energies_vec,
        ang_is_cm_vec,
        ang_dist_local_off_vec,
    })
}

/// Build per-nuclide pointer-array buffers for the basis / coeffs
/// reaction slots. Each slot stores the `CUdeviceptr` of the
/// corresponding `PerNuclideGpu::basis[slot]` / `coeffs[slot]`
/// `CudaSlice` as a `u64`; absent reactions get `0` (the kernel will
/// gate on `has_reaction[slot]` and never dereference a null
/// pointer).
///
/// Stage C step D groundwork — populate the pointer arrays so the
/// kernel can be migrated from `basis[basis_offsets[i] + …]` to
/// `((double*)basis_ptrs[i])[…]` per access site. The per-nuclide
/// CudaSlices that these pointers reference are pinned by
/// `GpuNuclideData::per_nucs` for the bundle's lifetime.
pub fn build_per_nuclide_ptr_arrays(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<(CudaSlice<u64>, CudaSlice<u64>), Box<dyn std::error::Error>> {
    let n_nuc = per_nucs.len();
    let mut basis_ptrs = vec![0_u64; n_nuc * N_RXN_SLOTS];
    let mut coeffs_ptrs = vec![0_u64; n_nuc * N_RXN_SLOTS];
    for (i, p) in per_nucs.iter().enumerate() {
        for r in 0..N_RXN_SLOTS {
            if let Some(b) = &p.basis[r] {
                let (ptr, _sync) = b.device_ptr(stream);
                basis_ptrs[i * N_RXN_SLOTS + r] = ptr;
            }
            if let Some(c) = &p.coeffs[r] {
                let (ptr, _sync) = c.device_ptr(stream);
                coeffs_ptrs[i * N_RXN_SLOTS + r] = ptr;
            }
        }
    }
    // Sentinel: clone_htod on `[0_u64; 0]` would null the pointer.
    // n_nuc is always ≥ 1 in production but defend against it anyway.
    if basis_ptrs.is_empty() {
        basis_ptrs.push(0);
        coeffs_ptrs.push(0);
    }
    Ok((
        stream.clone_htod(&basis_ptrs)?,
        stream.clone_htod(&coeffs_ptrs)?,
    ))
}

/// Build the per-nuclide discrete-level base pointer arrays
/// (`level_basis_ptrs[ni]` and `level_coeffs_ptrs[ni]`) plus
/// concatenated within-nuc local offset arrays
/// (`level_basis_local_off[gl]` and `level_coeffs_local_off[gl]`).
///
/// Pointer entries are `0` for nuclides with no discrete levels
/// (`n_levels == 0`); the kernel gates on
/// `P_LEVEL_COUNTS[ni] > 0` before dereferencing.
///
/// Local-offset arrays preserve the rank-padding invariant
/// (`1654c4d`) by construction — each per-nuclide's `basis_local_off`
/// is built against the rank-padded `[n_e × global_rank]` per-level
/// basis layout in `build_level_slices`.
pub fn build_per_nuc_level_ptr_and_offsets(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
) -> Result<
    (CudaSlice<u64>, CudaSlice<u64>, CudaSlice<i32>, CudaSlice<i32>),
    Box<dyn std::error::Error>,
> {
    let n_nuc = per_nucs.len();
    let mut basis_ptrs: Vec<u64> = Vec::with_capacity(n_nuc.max(1));
    let mut coeffs_ptrs: Vec<u64> = Vec::with_capacity(n_nuc.max(1));
    let mut basis_local_off: Vec<i32> = Vec::new();
    let mut coeffs_local_off: Vec<i32> = Vec::new();
    for p in per_nucs {
        if p.levels.n_levels > 0 && p.levels.basis_real_len > 0 {
            let (bp, _sync) = p.levels.basis.device_ptr(stream);
            basis_ptrs.push(bp);
            let (cp, _sync) = p.levels.coeffs.device_ptr(stream);
            coeffs_ptrs.push(cp);
        } else {
            basis_ptrs.push(0);
            coeffs_ptrs.push(0);
        }
        basis_local_off.extend_from_slice(&p.levels.basis_local_off);
        coeffs_local_off.extend_from_slice(&p.levels.coeffs_local_off);
    }
    // Sentinels.
    if basis_ptrs.is_empty() {
        basis_ptrs.push(0);
        coeffs_ptrs.push(0);
    }
    if basis_local_off.is_empty() {
        basis_local_off.push(0);
        coeffs_local_off.push(0);
    }
    Ok((
        stream.clone_htod(&basis_ptrs)?,
        stream.clone_htod(&coeffs_ptrs)?,
        stream.clone_htod(&basis_local_off)?,
        stream.clone_htod(&coeffs_local_off)?,
    ))
}

/// Build a per-nuclide `[n_nuc]` pointer array selecting one
/// `Option<CudaSlice<f64>>` field. Absent / empty per-nuclide slices
/// store `0`; the caller's kernel must gate on the corresponding
/// `has_*` flag before dereferencing.
pub fn build_per_nuc_optional_ptr_array<F>(
    stream: &Arc<CudaStream>,
    per_nucs: &[Arc<PerNuclideGpu>],
    pick: F,
) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>>
where
    F: Fn(&PerNuclideGpu) -> Option<&CudaSlice<f64>>,
{
    let mut ptrs: Vec<u64> = Vec::with_capacity(per_nucs.len().max(1));
    for p in per_nucs {
        if let Some(s) = pick(p) {
            let (ptr, _sync) = s.device_ptr(stream);
            ptrs.push(ptr);
        } else {
            ptrs.push(0);
        }
    }
    if ptrs.is_empty() {
        ptrs.push(0);
    }
    Ok(stream.clone_htod(&ptrs)?)
}

/// Flatten URR per-band 2D rows into the bundle's row-major layout.
/// Mirrors upload_nuclide_data:1883-1905.
fn build_urr_slices(
    stream: &Arc<CudaStream>,
    urr: &crate::hdf5_reader::UrrProbabilityTables,
) -> Result<UrrSlicesGpu, Box<dyn std::error::Error>> {
    fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.iter().map(|r| r.len()).sum());
        for row in rows {
            out.extend_from_slice(row);
        }
        out
    }
    let cp = flatten(&urr.cum_prob);
    let tf = flatten(&urr.total_factor);
    let ef = flatten(&urr.elastic_factor);
    let ff = flatten(&urr.fission_factor);
    let cf = flatten(&urr.capture_factor);
    Ok(UrrSlicesGpu {
        n_energies: urr.energies.len() as i32,
        n_bands: urr.n_bands as i32,
        multiply_smooth: if urr.multiply_smooth { 1 } else { 0 },
        interpolation: urr.interpolation as i32,
        energies: stream.clone_htod(&urr.energies)?,
        cum_prob: stream.clone_htod(&cp)?,
        total_factor: stream.clone_htod(&tf)?,
        elastic_factor: stream.clone_htod(&ef)?,
        fission_factor: stream.clone_htod(&ff)?,
        capture_factor: stream.clone_htod(&cf)?,
    })
}

/// Pack a tabular `EnergyDistribution`-style payload (1:1 per-inc
/// outgoing-energy CDF) into per-nuclide layout. Used for both
/// fission χ (Tabular branch) and MT=91 continuum inelastic. Per-inc
/// `dist_local_off` is nuclide-local; bundle assembly shifts.
fn build_tabular_edist(
    stream: &Arc<CudaStream>,
    inc_energies: &[f64],
    distributions: &[crate::hdf5_reader::TabularEnergyDist],
) -> Result<TabularEdistSlicesGpu, Box<dyn std::error::Error>> {
    let mut e_out_buf: Vec<f64> = Vec::new();
    let mut cdf_buf: Vec<f64> = Vec::new();
    let mut pdf_buf: Vec<f64> = Vec::new();
    let mut dist_local_off: Vec<i32> = Vec::with_capacity(inc_energies.len());
    let mut dist_sz: Vec<i32> = Vec::with_capacity(inc_energies.len());
    for dist in distributions {
        dist_local_off.push(e_out_buf.len() as i32);
        dist_sz.push(dist.e_out.len() as i32);
        e_out_buf.extend_from_slice(&dist.e_out);
        cdf_buf.extend_from_slice(&dist.cdf);
        // PDF aligned 1:1 with e_out when ENDF ships it; otherwise
        // zero-fill so the device's quadratic lin-lin sampler falls
        // back to linear-CDF — matches upload_nuclide_data
        // slot 1715-1719.
        if dist.pdf.len() == dist.e_out.len() {
            pdf_buf.extend_from_slice(&dist.pdf);
        } else {
            pdf_buf.extend(std::iter::repeat_n(0.0_f64, dist.e_out.len()));
        }
    }
    if e_out_buf.is_empty() {
        e_out_buf.push(0.0);
        cdf_buf.push(0.0);
        pdf_buf.push(0.0);
    }
    if dist_local_off.is_empty() {
        dist_local_off.push(0);
        dist_sz.push(0);
    }
    Ok(TabularEdistSlicesGpu {
        n_inc: inc_energies.len() as i32,
        inc_energies: stream.clone_htod(inc_energies)?,
        e_out: stream.clone_htod(&e_out_buf)?,
        cdf: stream.clone_htod(&cdf_buf)?,
        pdf: stream.clone_htod(&pdf_buf)?,
        dist_local_off,
        dist_sz,
    })
}

/// Per-nuclide fission χ extraction — dispatches on the four ENDF
/// laws (tabular Law 4/61, closed-form Watt Law 11, Maxwell Law 7,
/// Evaporation Law 9). Mirrors the bundle path at
/// upload_nuclide_data:1693-1779.
fn build_fission_edist(
    stream: &Arc<CudaStream>,
    edist: Option<&crate::hdf5_reader::EnergyDistribution>,
) -> Result<FissionEdistGpu, Box<dyn std::error::Error>> {
    use crate::hdf5_reader::FissionEnergyLaw;
    let Some(edist) = edist else {
        return Ok(FissionEdistGpu::None);
    };
    match &edist.closed_form {
        None => {
            if edist.energies.is_empty() || edist.distributions.is_empty() {
                return Ok(FissionEdistGpu::None);
            }
            Ok(FissionEdistGpu::Tabular(build_tabular_edist(
                stream,
                &edist.energies,
                &edist.distributions,
            )?))
        }
        Some(FissionEnergyLaw::Watt(w)) => {
            // Resample a(E_in) and b(E_in) onto a shared inc-energy
            // grid (union sorted+deduped) so the device runs ONE
            // binary search per fission event. Mirrors the bundle's
            // slot 1727-1751.
            let mut shared: Vec<f64> = w
                .a_energies
                .iter()
                .chain(w.b_energies.iter())
                .copied()
                .collect();
            shared.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            shared.dedup_by(|a, b| (*a - *b).abs() < 1e-30 * a.abs().max(1.0));
            let mut a_vec = Vec::with_capacity(shared.len());
            let mut b_vec = Vec::with_capacity(shared.len());
            for e in &shared {
                a_vec.push(crate::hdf5_reader::WattLaw::lookup_lin_lin_pub(
                    &w.a_energies,
                    &w.a_values,
                    *e,
                ));
                b_vec.push(crate::hdf5_reader::WattLaw::lookup_lin_lin_pub(
                    &w.b_energies,
                    &w.b_values,
                    *e,
                ));
            }
            Ok(FissionEdistGpu::Watt(WattSlicesGpu {
                n_inc: shared.len() as i32,
                u: w.u,
                inc_energies: stream.clone_htod(&shared)?,
                a: stream.clone_htod(&a_vec)?,
                b: stream.clone_htod(&b_vec)?,
            }))
        }
        Some(FissionEnergyLaw::Maxwell(m)) | Some(FissionEnergyLaw::Evaporation(m)) => {
            let law = match edist.closed_form {
                Some(FissionEnergyLaw::Maxwell(_)) => 7,
                Some(FissionEnergyLaw::Evaporation(_)) => 9,
                _ => 0,
            };
            Ok(FissionEdistGpu::MaxEvap(MaxEvapSlicesGpu {
                n_inc: m.theta_energies.len() as i32,
                u: m.u,
                law,
                inc_energies: stream.clone_htod(&m.theta_energies)?,
                theta: stream.clone_htod(&m.theta_values)?,
            }))
        }
    }
}

/// Pack an `AngularDistribution` into per-nuclide GPU layout. Mirrors
/// the legacy bundle's elastic-angular packing at
/// `gpu_transport.rs::upload_nuclide_data` slot 1588-1602.
/// Per-energy `dist_local_off` is nuclide-local; bundle assembly
/// shifts it.
fn build_angular_slices(
    stream: &Arc<CudaStream>,
    ad: &crate::hdf5_reader::AngularDistribution,
) -> Result<AngularSlicesGpu, Box<dyn std::error::Error>> {
    let n_energies = ad.energies.len();
    let mut mu_buf: Vec<f64> = Vec::new();
    let mut cdf_buf: Vec<f64> = Vec::new();
    let mut pdf_buf: Vec<f64> = Vec::new();
    let mut dist_local_off: Vec<i32> = Vec::with_capacity(n_energies);
    let mut dist_sz: Vec<i32> = Vec::with_capacity(n_energies);
    for (i, _e) in ad.energies.iter().enumerate() {
        let dist = &ad.distributions[i];
        dist_local_off.push(mu_buf.len() as i32);
        dist_sz.push(dist.mu.len() as i32);
        mu_buf.extend_from_slice(&dist.mu);
        cdf_buf.extend_from_slice(&dist.cdf);
        // Length-matched PDF — CPU's TabularMuDist always carries pdf
        // alongside mu / cdf. Pad if a malformed distribution has
        // fewer pdf entries than mu / cdf so the GPU stride stays
        // consistent and the kernel's `pd[i]` indexing is safe.
        if dist.pdf.len() == dist.mu.len() {
            pdf_buf.extend_from_slice(&dist.pdf);
        } else {
            let mut padded = dist.pdf.clone();
            padded.resize(dist.mu.len(), 0.0);
            pdf_buf.extend_from_slice(&padded);
        }
    }
    let mu_real_len = mu_buf.len();
    if mu_buf.is_empty() {
        mu_buf.push(0.0);
        cdf_buf.push(0.0);
        pdf_buf.push(0.0);
    }
    Ok(AngularSlicesGpu {
        n_energies: n_energies as i32,
        is_cm: if ad.center_of_mass { 1 } else { 0 },
        energies: stream.clone_htod(&ad.energies)?,
        mu: stream.clone_htod(&mu_buf)?,
        cdf: stream.clone_htod(&cdf_buf)?,
        pdf: stream.clone_htod(&pdf_buf)?,
        dist_local_off,
        dist_sz,
        mu_real_len,
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
/// `gpu_transport.rs::upload_nuclide_data` slot 1457-1518.
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
    let mut ang_pdf_buf: Vec<f64> = Vec::new();
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
                    // Length-matched PDF for the quadratic CDF
                    // inversion. Same padding rule as the elastic
                    // angular path.
                    if dist.pdf.len() == dist.mu.len() {
                        ang_pdf_buf.extend_from_slice(&dist.pdf);
                    } else {
                        let mut padded = dist.pdf.clone();
                        padded.resize(dist.mu.len(), 0.0);
                        ang_pdf_buf.extend_from_slice(&padded);
                    }
                }
            }
            _ => {
                ang_lev_local_off.push(0);
                ang_lev_ne.push(0);
            }
        }
    }

    // Capture real lengths BEFORE sentinel insertion so the bundle
    // assembly stage can slice past the per-nuclide padding.
    let basis_real_len = basis_buf.len();
    let coeffs_real_len = coeffs_buf.len();
    let ang_e_real_len = ang_e_buf.len();
    let ang_mu_real_len = ang_mu_buf.len();
    let ang_dist_real_len = ang_dist_local_off.len();

    // Sentinel for empty payloads — `clone_htod` of a zero-length
    // slice would leave a null device pointer on some drivers; the
    // legacy bundle path uses the same `[0.0]` / `[0]` rule. The
    // sentinel slot is NOT included in `*_real_len` above.
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
        ang_pdf_buf.push(0.0);
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
        ang_pdf: stream.clone_htod(&ang_pdf_buf)?,
        ang_lev_local_off,
        ang_lev_ne,
        ang_dist_local_off,
        ang_dist_sz,
        basis_real_len,
        coeffs_real_len,
        ang_e_real_len,
        ang_mu_real_len,
        ang_dist_real_len,
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
    fn fission_edist_watt_resamples_to_shared_grid() {
        let Some(stream) = try_cuda_stream() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use crate::hdf5_reader::{EnergyDistribution, FissionEnergyLaw, WattLaw};

        let mut nuc = NuclideKernels::empty(235.0, 2.5);
        nuc.elastic = Some(ReactionKernel::from_table(
            vec![1.0e-5, 1.0, 2.0e7],
            vec![10.0, 5.0, 1.0],
        ));
        nuc.fission_energy_dist = Some(EnergyDistribution {
            energies: vec![],
            distributions: vec![],
            closed_form: Some(FissionEnergyLaw::Watt(WattLaw {
                a_energies: vec![1.0e3, 1.0e6],
                a_values: vec![1.0e6, 1.5e6],
                b_energies: vec![1.0e4, 1.0e7],
                b_values: vec![2.0, 3.0],
                u: 0.0,
            })),
        });

        let per_nuc =
            upload_one_nuclide(&stream, &nuc, 5).expect("upload_one_nuclide failed");

        let watt = match &per_nuc.fission_edist {
            FissionEdistGpu::Watt(w) => w,
            _ => panic!("expected Watt edist"),
        };
        // Shared grid = sorted union {1e3, 1e4, 1e6, 1e7} → 4 points.
        assert_eq!(watt.n_inc, 4);
        let mut grid = vec![0.0_f64; 4];
        stream.memcpy_dtoh(&watt.inc_energies, &mut grid).unwrap();
        assert_eq!(grid, vec![1.0e3, 1.0e4, 1.0e6, 1.0e7]);
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
    fn ptr_arrays_match_per_nuclide_device_addresses() {
        let Some(stream) = try_cuda_stream() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use crate::gpu_transport::GpuTransportContext;
        let Ok(ctx) = GpuTransportContext::new() else {
            eprintln!("skipping: cannot construct GpuTransportContext");
            return;
        };

        // 2 nuclides with elastic + capture slots populated. Each
        // populated slot's basis_ptrs entry must equal the device
        // address of the corresponding PerNuclideGpu CudaSlice;
        // absent slots must be zero.
        let mut nuc_a = NuclideKernels::empty(235.0, 2.5);
        nuc_a.elastic = Some(ReactionKernel::from_table(
            vec![1.0e-5, 1.0, 2.0e7],
            vec![10.0, 5.0, 1.0],
        ));
        nuc_a.fission = Some(ReactionKernel::from_table(
            vec![1.0e-5, 1.0, 2.0e7],
            vec![0.0, 0.0, 1.5],
        ));
        let mut nuc_b = NuclideKernels::empty(16.0, 0.0);
        nuc_b.elastic = Some(ReactionKernel::from_table(
            vec![1.0e-5, 100.0, 2.0e7],
            vec![3.5, 3.6, 3.0],
        ));

        let nuclides: Vec<Arc<NuclideKernels>> =
            vec![Arc::new(nuc_a), Arc::new(nuc_b)];
        let bundle = ctx.upload_nuclide_data(&nuclides, 5).unwrap();

        assert_eq!(bundle.per_nucs.len(), 2, "per_nucs pin");

        let mut basis_ptrs_host = vec![0_u64; 2 * N_RXN_SLOTS];
        let mut coeffs_ptrs_host = vec![0_u64; 2 * N_RXN_SLOTS];
        stream
            .memcpy_dtoh(&bundle.basis_ptrs, &mut basis_ptrs_host)
            .unwrap();
        stream
            .memcpy_dtoh(&bundle.coeffs_ptrs, &mut coeffs_ptrs_host)
            .unwrap();

        for (nuc_idx, p) in bundle.per_nucs.iter().enumerate() {
            for slot in 0..N_RXN_SLOTS {
                let key = nuc_idx * N_RXN_SLOTS + slot;
                if let Some(b) = &p.basis[slot] {
                    let (ptr, _sync) = b.device_ptr(&stream);
                    assert_eq!(
                        basis_ptrs_host[key], ptr,
                        "basis_ptrs[{nuc_idx},{slot}] mismatch"
                    );
                    assert_eq!(p.has_reaction[slot], 1);
                } else {
                    assert_eq!(
                        basis_ptrs_host[key], 0,
                        "absent basis[{nuc_idx},{slot}] should be 0"
                    );
                }
                if let Some(c) = &p.coeffs[slot] {
                    let (ptr, _sync) = c.device_ptr(&stream);
                    assert_eq!(
                        coeffs_ptrs_host[key], ptr,
                        "coeffs_ptrs[{nuc_idx},{slot}] mismatch"
                    );
                } else {
                    assert_eq!(
                        coeffs_ptrs_host[key], 0,
                        "absent coeffs[{nuc_idx},{slot}] should be 0"
                    );
                }
            }
        }
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
