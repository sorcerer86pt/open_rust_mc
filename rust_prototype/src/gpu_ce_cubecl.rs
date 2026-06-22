// SPDX-License-Identifier: MIT
//! Continuous-energy cross-section lookup on CubeCL — foundation for a
//! CE transport kernel (target: HEU-COMP-INTER, multi-nuclide
//! fast/intermediate, no S(α,β)/URR).
//!
//! This module ports the genuinely new CE piece — per-nuclide pointwise
//! cross-section data on the device, with a single binary search on the
//! shared energy grid followed by log-log interpolation per reaction —
//! exactly mirroring `SvdXsProvider::lookup` (`xs_provider.rs`). The
//! transport loop on top reuses the geometry walk + collision structure
//! already validated by the const-XS A/B.
//!
//! Like the const-XS kernel, it runs in f64. Per cubecl#1336 the heavy
//! kernel only runs on the CUDA runtime (SPIR-V/Vulkan faults on large
//! private-Array state); this lookup foundation is small enough to test
//! on either, but the eventual transport kernel is CUDA-only until the
//! upstream fix lands.
//!
//! ── Device data layout ──────────────────────────────────────────────
//! One shared grid per nuclide. For each nuclide we upload:
//!   - `grid`   : N_e energies (eV, ascending)
//!   - per reaction (elastic, fission, capture, inelastic, n2n): N_e σ
//!     values (barns); a reaction absent for a nuclide gets an all-zero
//!     slab so the kernel can read it unconditionally.
//! All packed into one big f64 blob with per-nuclide offsets in an i32
//! header, plus per-(material,nuclide) atom densities for the
//! macroscopic sum.

use cubecl::prelude::*;

/// CE reaction channels carried per nuclide. Order matters: it's the
/// stride layout in the device blob.
pub const N_RX: usize = 5;
pub const RX_ELASTIC: usize = 0;
pub const RX_FISSION: usize = 1;
pub const RX_CAPTURE: usize = 2;
pub const RX_INELASTIC: usize = 3;
pub const RX_N2N: usize = 4;

/// Host-side per-nuclide CE data, extracted from the resolved provider.
#[derive(Clone)]
pub struct NuclideCe {
    /// Shared energy grid (eV, ascending).
    pub grid: Vec<f64>,
    /// `[N_RX][grid.len()]` reconstructed σ(E) per reaction (barns).
    /// Reactions absent for this nuclide are all-zero.
    pub xs: [Vec<f64>; N_RX],
    /// Mass ratio A (for elastic CM kinematics later).
    pub awr: f64,
    /// ν̄ at a representative energy (constant approx for the first cut;
    /// the transport kernel will interpolate a real ν̄(E) grid later).
    pub nu_bar: f64,
}

/// Extract per-nuclide CE data from a resolved provider's nuclide
/// kernels, reconstructing each reaction's σ on the nuclide's own grid.
/// Reactions absent for a nuclide become all-zero slabs. Mirrors the
/// reactions the device kernel carries (`N_RX`).
pub fn extract_ce(
    nuclides: &[std::sync::Arc<crate::transport::xs_provider::NuclideKernels>],
) -> Vec<NuclideCe> {
    use crate::transport::xs_provider::ReactionKernel;

    let recon = |k: &Option<ReactionKernel>, grid: &[f64]| -> Vec<f64> {
        match k {
            Some(rk) => (0..grid.len()).map(|i| rk.reconstruct_at_index(i)).collect(),
            None => vec![0.0; grid.len()],
        }
    };

    nuclides
        .iter()
        .map(|nuc| {
            // Shared grid = whichever reaction is present (elastic first).
            let grid: Vec<f64> = nuc
                .elastic
                .as_ref()
                .or(nuc.fission.as_ref())
                .or(nuc.capture.as_ref())
                .or(nuc.inelastic.as_ref())
                .or(nuc.n2n.as_ref())
                .map(|k| k.energies().to_vec())
                .unwrap_or_default();
            let xs = [
                recon(&nuc.elastic, &grid),
                recon(&nuc.fission, &grid),
                recon(&nuc.capture, &grid),
                recon(&nuc.inelastic, &grid),
                recon(&nuc.n2n, &grid),
            ];
            NuclideCe {
                grid,
                xs,
                awr: nuc.awr,
                nu_bar: nuc.nu_bar_const,
            }
        })
        .collect()
}

// ── Elastic angular distribution (tabulated μ) ──────────────────────
//
// CPU source: `NuclideKernels::elastic_angle: Option<AngularDistribution>`
// = { energies[N_e], distributions[N_e]: TabularMuDist{ mu, pdf, cdf,
// histogram } }. Flattened for the device so the CubeCL sampler mirrors
// `AngularDistribution::sample_mu` (bracket E, stochastic bin pick) +
// `TabularMuDist::sample_with_xi` (linear or quadratic CDF inversion).

/// Flat per-nuclide elastic angular data, device-ready.
#[derive(Clone, Default)]
pub struct AngularCe {
    /// Incident energy grid (eV). Empty ⇒ isotropic fallback.
    pub energies: Vec<f64>,
    /// Per-energy μ-distribution slices into the flat mu/cdf/pdf arrays:
    /// distribution `i` occupies `[dist_off[i] .. dist_off[i]+dist_len[i]]`.
    pub dist_off: Vec<i32>,
    pub dist_len: Vec<i32>,
    /// `1` if histogram interpolation (linear CDF), `0` if lin-lin
    /// (quadratic CDF inversion). Per energy point.
    pub histogram: Vec<i32>,
    /// Flat concatenated breakpoint arrays across all energy points.
    pub mu: Vec<f64>,
    pub cdf: Vec<f64>,
    pub pdf: Vec<f64>,
}

/// Extract elastic angular data per nuclide (parallel to `extract_ce`).
/// Nuclides with no elastic angular distribution get an empty
/// `AngularCe` (isotropic fallback at sample time).
pub fn extract_angular(
    nuclides: &[std::sync::Arc<crate::transport::xs_provider::NuclideKernels>],
) -> Vec<AngularCe> {
    nuclides
        .iter()
        .map(|nuc| {
            let Some(ang) = nuc.elastic_angle.as_ref() else {
                return AngularCe::default();
            };
            let mut out = AngularCe {
                energies: ang.energies.clone(),
                ..Default::default()
            };
            for d in &ang.distributions {
                out.dist_off.push(out.mu.len() as i32);
                out.dist_len.push(d.mu.len() as i32);
                out.histogram.push(i32::from(d.histogram));
                out.mu.extend_from_slice(&d.mu);
                out.cdf.extend_from_slice(&d.cdf);
                // pdf may be empty (reader left it out) → pad to mu len
                // with zeros so the device can read it unconditionally;
                // the sampler treats all-zero pdf as the linear path.
                if d.pdf.len() == d.mu.len() {
                    out.pdf.extend_from_slice(&d.pdf);
                } else {
                    out.pdf.extend(std::iter::repeat_n(0.0, d.mu.len()));
                }
            }
            out
        })
        .collect()
}

/// One material = a list of (nuclide index, atom density). Mirrors the
/// CPU `Material.nuclides` after resolution; the device sums
/// `Σ_t(E) = Σ_nuc n_d · σ_t,nuc(E)` over these.
#[derive(Clone)]
pub struct MaterialCe {
    /// `(nuclide_idx, atom_density [atoms/barn-cm])`.
    pub nuclides: Vec<(usize, f64)>,
}

/// CE scene packed for the device: flat blobs + offset header.
pub struct PackedCe {
    /// f64 blob: per-nuclide [grid, xs[0..N_RX]], then the flat
    /// material-nuclide atom-density list.
    pub fdata: Vec<f64>,
    /// i32 header: per-nuclide [grid_off, n_e, xs_off], then the
    /// material table (off/len into the mat-nuclide lists).
    pub idata: Vec<i32>,
    pub n_nuclides: usize,
    pub n_materials: usize,
    /// Mirror of the per-nuclide awr / nu_bar (host-side; also uploaded).
    pub awr: Vec<f64>,
    pub nu_bar: Vec<f64>,
    /// Device offsets (filled by `pack_ce_scene`) into idata/fdata for
    /// the material table and the per-nuclide awr/nu_bar arrays.
    pub mat_table_off: usize,  // i32: per material [list_off, list_len]
    pub mat_nuc_idx_off: usize, // i32: flat nuclide indices
    pub mat_nuc_den_off: usize, // f64: flat atom densities
    pub awr_off: usize,         // f64
    pub nu_bar_off: usize,      // f64
}

// Per-nuclide i32 header stride: [grid_off, n_e, xs_off].
const NUC_HDR: usize = 3;
const H_GRID_OFF: usize = 0;
const H_N_E: usize = 1;
const H_XS_OFF: usize = 2;

/// Pack per-nuclide CE data only (no material table). Used by the XS
/// lookup A/B, which compares one nuclide at a time. The material /
/// awr / nu_bar offsets are left at 0 with empty tables.
pub fn pack_ce(nuclides: &[NuclideCe]) -> PackedCe {
    pack_ce_scene(nuclides, &[])
}

/// Pack a full CE scene: per-nuclide grids+σ, then the material table
/// (per material: a list of (nuclide_idx, atom_density)), then the
/// per-nuclide awr / nu_bar arrays. Layout (offsets recorded in the
/// returned struct, in *element* units):
///   fdata: [ per-nuclide grid+xs … ][ mat atom densities ][ awr ][ nu_bar ]
///   idata: [ per-nuclide NUC_HDR … ][ mat [off,len] table ][ flat nuclide idxs ]
pub fn pack_ce_scene(nuclides: &[NuclideCe], materials: &[MaterialCe]) -> PackedCe {
    let mut fdata: Vec<f64> = Vec::new();
    let mut idata: Vec<i32> = vec![0; nuclides.len() * NUC_HDR];
    let mut awr = Vec::with_capacity(nuclides.len());
    let mut nu_bar = Vec::with_capacity(nuclides.len());

    for (n, nuc) in nuclides.iter().enumerate() {
        let n_e = nuc.grid.len();
        let grid_off = fdata.len();
        fdata.extend_from_slice(&nuc.grid);
        let xs_off = fdata.len();
        for r in 0..N_RX {
            debug_assert_eq!(nuc.xs[r].len(), n_e, "reaction {r} σ len != grid len");
            fdata.extend_from_slice(&nuc.xs[r]);
        }
        idata[n * NUC_HDR + H_GRID_OFF] = grid_off as i32;
        idata[n * NUC_HDR + H_N_E] = n_e as i32;
        idata[n * NUC_HDR + H_XS_OFF] = xs_off as i32;
        awr.push(nuc.awr);
        nu_bar.push(nuc.nu_bar);
    }

    // Material table: per material [list_off, list_len] into the flat
    // (nuclide_idx) i32 list + parallel (atom_density) f64 list.
    let mat_table_off = idata.len();
    idata.extend(std::iter::repeat_n(0, materials.len() * 2));
    let mut flat_idx: Vec<i32> = Vec::new();
    let mut flat_den: Vec<f64> = Vec::new();
    for (m, mat) in materials.iter().enumerate() {
        let off = flat_idx.len();
        for &(ni, den) in &mat.nuclides {
            flat_idx.push(ni as i32);
            flat_den.push(den);
        }
        idata[mat_table_off + m * 2] = off as i32;
        idata[mat_table_off + m * 2 + 1] = mat.nuclides.len() as i32;
    }
    let mat_nuc_idx_off = idata.len();
    idata.extend_from_slice(&flat_idx);

    let mat_nuc_den_off = fdata.len();
    fdata.extend_from_slice(&flat_den);
    let awr_off = fdata.len();
    fdata.extend_from_slice(&awr);
    let nu_bar_off = fdata.len();
    fdata.extend_from_slice(&nu_bar);

    if fdata.is_empty() {
        fdata.push(0.0);
    }
    if idata.is_empty() {
        idata.push(0);
    }
    PackedCe {
        fdata,
        idata,
        n_nuclides: nuclides.len(),
        n_materials: materials.len(),
        awr,
        nu_bar,
        mat_table_off,
        mat_nuc_idx_off,
        mat_nuc_den_off,
        awr_off,
        nu_bar_off,
    }
}

// ── Device CE lookup ────────────────────────────────────────────────

/// Lower-bracket grid index for `energy` by binary search over the
/// nuclide's grid slice `[grid_off .. grid_off + n_e]`. Returns an index
/// in `[0, n_e-2]` (clamped), matching `ReactionKernel::energy_index`.
#[cube]
fn energy_index(fdata: &Array<f64>, grid_off: u32, n_e: u32, energy: f64) -> u32 {
    let mut lo = u32::new(0);
    let mut hi = n_e - 1u32;
    // Standard binary search for the bracket; bounded loop (no while-cap
    // worries — grids are < 2^20 points so 24 steps suffice, but use a
    // generous fixed cap for the CubeCL frontend).
    for _i in 0..32u32 {
        if lo + 1u32 < hi {
            let mid = (lo + hi) / 2u32;
            if fdata[(grid_off + mid) as usize] <= energy {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    // Clamp to [0, n_e-2].
    select(lo > n_e - 2u32, n_e - 2u32, lo)
}

/// Log-log interpolate reaction `rx` of nuclide at header `hdr_base`,
/// at `energy`, given the precomputed bracket `idx`. Mirrors
/// `ReactionKernel::reconstruct_interp` for the `Table` case.
#[cube]
fn rx_interp(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    hdr_base: u32,
    rx: u32,
    idx: u32,
    energy: f64,
) -> f64 {
    let grid_off = u32::cast_from(idata[(hdr_base + H_GRID_OFF as u32) as usize]);
    let n_e = u32::cast_from(idata[(hdr_base + H_N_E as u32) as usize]);
    let xs_off = u32::cast_from(idata[(hdr_base + H_XS_OFF as u32) as usize]);
    let base = xs_off + rx * n_e;
    let e_lo = fdata[(grid_off + idx) as usize];
    let xs_lo = fdata[(base + idx) as usize];
    let mut out = xs_lo;
    if idx + 1u32 < n_e {
        let e_hi = fdata[(grid_off + idx + 1u32) as usize];
        let xs_hi = fdata[(base + idx + 1u32) as usize];
        if e_hi > e_lo {
            if xs_lo > f64::new(0.0) {
                if xs_hi > f64::new(0.0) {
                    // log-log
                    let f = (energy / e_lo).ln() / (e_hi / e_lo).ln();
                    let ratio = xs_hi / xs_lo;
                    out = xs_lo * (f * ratio.ln()).exp();
                } else {
                    let frac = (energy - e_lo) / (e_hi - e_lo);
                    out = xs_lo + frac * (xs_hi - xs_lo);
                }
            } else {
                let frac = (energy - e_lo) / (e_hi - e_lo);
                out = xs_lo + frac * (xs_hi - xs_lo);
            }
        }
    }
    out
}

/// Test kernel: for each input energy, look up the total microscopic σ
/// (sum of the N_RX reactions) of nuclide 0 and write it out. Validates
/// the grid search + log-log interp against the CPU provider.
#[cube(launch)]
fn ce_total_micro_kernel(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    energies: &Array<f64>,
    out: &mut Array<f64>,
) {
    let tid = ABSOLUTE_POS;
    if tid < out.len() {
        let e = energies[tid];
        let hdr_base = u32::new(0); // nuclide 0
        let grid_off = u32::cast_from(idata[(hdr_base + H_GRID_OFF as u32) as usize]);
        let n_e = u32::cast_from(idata[(hdr_base + H_N_E as u32) as usize]);
        let idx = energy_index(fdata, grid_off, n_e, e);
        let mut tot = f64::new(0.0);
        for r in 0..N_RX as u32 {
            tot = tot + rx_interp(idata, fdata, hdr_base, r, idx, e);
        }
        out[tid] = tot;
    }
}

/// Look up total microscopic σ(E) of nuclide 0 on the GPU at the given
/// energies. Convenience wrapper for the A/B test.
pub fn total_micro_xs<R: Runtime>(
    device: &R::Device,
    packed: &PackedCe,
    energies: &[f64],
) -> Vec<f64> {
    let client = R::client(device);
    let n = energies.len();
    let idata_h = client.create_from_slice(i32::as_bytes(&packed.idata));
    let fdata_h = client.create_from_slice(f64::as_bytes(&packed.fdata));
    let e_h = client.create_from_slice(f64::as_bytes(energies));
    let out_h = client.empty(n * core::mem::size_of::<f64>());

    let threads = 64u32;
    let blocks = n.div_ceil(threads as usize) as u32;
    unsafe {
        ce_total_micro_kernel::launch::<R>(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(idata_h, packed.idata.len()),
            ArrayArg::from_raw_parts(fdata_h, packed.fdata.len()),
            ArrayArg::from_raw_parts(e_h, n),
            ArrayArg::from_raw_parts(out_h.clone(), n),
        );
    }
    let bytes = client.read_one(out_h).expect("ce readback");
    f64::from_bytes(&bytes).to_vec()
}

// ── Packed elastic angular blob (one nuclide) ───────────────────────
//
// Layout mirrors the σ pack: an i32 header + i32/f64 blobs. For the
// A/B we pack a single nuclide at a time (the kernel samples "nuclide
// 0"), same as the σ lookup A/B.
//
// idata: [ n_e,
//          eoff (energies start in fdata),
//          doff (dist_off start in idata, len n_e),
//          dlen (dist_len start in idata, len n_e),
//          hoff (histogram start in idata, len n_e),
//          muoff (mu start in fdata),
//          cdfoff (cdf start in fdata),
//          pdfoff (pdf start in fdata) ]
//   then the dist_off / dist_len / histogram arrays inline.
// fdata: [ energies | mu | cdf | pdf ].

/// One-nuclide angular pack for the device sampler.
pub struct PackedAngular {
    pub idata: Vec<i32>,
    pub fdata: Vec<f64>,
}

const A_N_E: usize = 0;
const A_EOFF: usize = 1;
const A_DOFF: usize = 2;
const A_DLEN: usize = 3;
const A_HOFF: usize = 4;
const A_MUOFF: usize = 5;
const A_CDFOFF: usize = 6;
const A_PDFOFF: usize = 7;
const A_HDR: usize = 8;

/// Pack one nuclide's elastic angular data for the device.
pub fn pack_angular(ang: &AngularCe) -> PackedAngular {
    let mut idata = vec![0i32; A_HDR];
    let mut fdata: Vec<f64> = Vec::new();
    let n_e = ang.energies.len();
    idata[A_N_E] = n_e as i32;

    idata[A_EOFF] = fdata.len() as i32;
    fdata.extend_from_slice(&ang.energies);

    idata[A_DOFF] = idata.len() as i32;
    idata.extend_from_slice(&ang.dist_off);
    idata[A_DLEN] = idata.len() as i32;
    idata.extend_from_slice(&ang.dist_len);
    idata[A_HOFF] = idata.len() as i32;
    idata.extend_from_slice(&ang.histogram);

    idata[A_MUOFF] = fdata.len() as i32;
    fdata.extend_from_slice(&ang.mu);
    idata[A_CDFOFF] = fdata.len() as i32;
    fdata.extend_from_slice(&ang.cdf);
    idata[A_PDFOFF] = fdata.len() as i32;
    fdata.extend_from_slice(&ang.pdf);

    if fdata.is_empty() {
        fdata.push(0.0);
    }
    if idata.len() == A_HDR {
        idata.push(0); // ensure non-empty blob
    }
    PackedAngular { idata, fdata }
}

// ── Device μ sampler (mirrors TabularMuDist::sample_with_xi) ─────────

/// Invert the μ CDF of distribution at `[doff .. doff+dlen]` (slices
/// into mu/cdf/pdf) for a pre-drawn `xi`. `hist` selects linear (1) vs
/// quadratic (0) inversion. Mirrors `TabularMuDist::sample_with_xi`.
#[cube]
#[allow(unused_assignments)]
fn sample_mu_bin(
    fdata: &Array<f64>,
    mu_off: u32,
    cdf_off: u32,
    pdf_off: u32,
    doff: u32,
    dlen: u32,
    hist: i32,
    xi: f64,
) -> f64 {
    let mut out = 2.0 * xi - 1.0; // n<2 fallback
    if dlen >= 2u32 {
        // binary search cdf for bracket idx (local within the dist).
        let mut lo = u32::new(0);
        let mut hi = dlen - 1u32;
        for _i in 0..32u32 {
            if lo + 1u32 < hi {
                let mid = (lo + hi) / 2u32;
                if fdata[(cdf_off + doff + mid) as usize] <= xi {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        // clamp lo to dlen-2
        let idx = select(lo > dlen - 2u32, dlen - 2u32, lo);
        let cdf_lo = fdata[(cdf_off + doff + idx) as usize];
        let cdf_hi = fdata[(cdf_off + doff + idx + 1u32) as usize];
        let mu_lo = fdata[(mu_off + doff + idx) as usize];
        let mu_hi = fdata[(mu_off + doff + idx + 1u32) as usize];
        let dmu = mu_hi - mu_lo;
        let dc = cdf_hi - cdf_lo;
        if dc.abs() < f64::new(1e-15) {
            out = clamp_mu(mu_lo);
        } else {
            if dmu.abs() < f64::new(1e-15) {
                out = clamp_mu(mu_lo);
            } else {
                if hist == 1i32 {
                    let frac = (xi - cdf_lo) / dc;
                    out = clamp_mu(mu_lo + frac * dmu);
                } else {
                    // quadratic lin-lin inversion.
                    let pdf_lo = fdata[(pdf_off + doff + idx) as usize];
                    let pdf_hi = fdata[(pdf_off + doff + idx + 1u32) as usize];
                    let a = (pdf_hi - pdf_lo) / (2.0 * dmu);
                    let b = pdf_lo;
                    let c = cdf_lo - xi;
                    let mut x = f64::new(0.0);
                    if a.abs() < f64::new(1e-14) {
                        if b.abs() < f64::new(1e-30) {
                            x = (xi - cdf_lo) / dc * dmu;
                        } else {
                            x = -c / b;
                        }
                    } else {
                        let disc = b * b - 4.0 * a * c;
                        let sq = select(disc > f64::new(0.0), disc, f64::new(0.0)).sqrt();
                        // physical root in [0, dmu]
                        let x1 = (-b + sq) / (2.0 * a);
                        let x2 = (-b - sq) / (2.0 * a);
                        let x1_ok = x1 >= f64::new(0.0) && x1 <= dmu;
                        x = select(x1_ok, x1, x2);
                    }
                    out = clamp_mu(mu_lo + x);
                }
            }
        }
    }
    out
}

#[cube]
fn clamp_mu(m: f64) -> f64 {
    let lo = select(m > f64::new(-1.0), m, f64::new(-1.0));
    select(lo < f64::new(1.0), lo, f64::new(1.0))
}

/// Sample an elastic μ at incident `energy` for the packed nuclide,
/// given two uniforms (`xi_bin` for the stochastic energy-bin pick,
/// `xi_mu` for the CDF inversion). Mirrors `AngularDistribution::sample_mu`.
#[cube]
#[allow(unused_assignments)]
fn sample_mu_at(idata: &Array<i32>, fdata: &Array<f64>, energy: f64, xi_bin: f64, xi_mu: f64) -> f64 {
    let n_e = u32::cast_from(idata[A_N_E]);
    let mut out = 2.0 * xi_mu - 1.0; // isotropic fallback (no data)
    if n_e >= 1u32 {
        let eoff = u32::cast_from(idata[A_EOFF]);
        let doff_base = u32::cast_from(idata[A_DOFF]);
        let dlen_base = u32::cast_from(idata[A_DLEN]);
        let hoff_base = u32::cast_from(idata[A_HOFF]);
        let mu_off = u32::cast_from(idata[A_MUOFF]);
        let cdf_off = u32::cast_from(idata[A_CDFOFF]);
        let pdf_off = u32::cast_from(idata[A_PDFOFF]);

        // bracket energy: pick a distribution index `pick`.
        let e0 = fdata[eoff as usize];
        let elast = fdata[(eoff + n_e - 1u32) as usize];
        let mut pick = u32::new(0);
        if energy <= e0 {
            pick = u32::new(0);
        } else {
            if energy >= elast {
                pick = n_e - 1u32;
            } else {
                // binary search for lower bracket idx
                let mut lo = u32::new(0);
                let mut hi = n_e - 1u32;
                for _i in 0..32u32 {
                    if lo + 1u32 < hi {
                        let mid = (lo + hi) / 2u32;
                        if fdata[(eoff + mid) as usize] <= energy {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                }
                let e_lo = fdata[(eoff + lo) as usize];
                let e_hi = fdata[(eoff + lo + 1u32) as usize];
                let r = (energy - e_lo) / (e_hi - e_lo);
                pick = select(xi_bin < r, lo + 1u32, lo);
            }
        }
        let doff = u32::cast_from(idata[(doff_base + pick) as usize]);
        let dlen = u32::cast_from(idata[(dlen_base + pick) as usize]);
        let hist = idata[(hoff_base + pick) as usize];
        out = sample_mu_bin(fdata, mu_off, cdf_off, pdf_off, doff, dlen, hist, xi_mu);
    }
    out
}

/// Test kernel: sample one μ per thread at the given (energy, xi_bin,
/// xi_mu) triples. Lets the host drive a CPU-vs-GPU distribution A/B.
#[cube(launch)]
fn ce_sample_mu_kernel(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    energy: &Array<f64>,
    xi_bin: &Array<f64>,
    xi_mu: &Array<f64>,
    out: &mut Array<f64>,
) {
    let tid = ABSOLUTE_POS;
    if tid < out.len() {
        out[tid] = sample_mu_at(idata, fdata, energy[tid], xi_bin[tid], xi_mu[tid]);
    }
}

/// Sample μ on the GPU for one packed nuclide at the given inputs.
pub fn sample_mu_gpu<R: Runtime>(
    device: &R::Device,
    packed: &PackedAngular,
    energy: &[f64],
    xi_bin: &[f64],
    xi_mu: &[f64],
) -> Vec<f64> {
    let client = R::client(device);
    let n = energy.len();
    let idata_h = client.create_from_slice(i32::as_bytes(&packed.idata));
    let fdata_h = client.create_from_slice(f64::as_bytes(&packed.fdata));
    let e_h = client.create_from_slice(f64::as_bytes(energy));
    let xb_h = client.create_from_slice(f64::as_bytes(xi_bin));
    let xm_h = client.create_from_slice(f64::as_bytes(xi_mu));
    let out_h = client.empty(n * core::mem::size_of::<f64>());
    let threads = 64u32;
    let blocks = n.div_ceil(threads as usize) as u32;
    unsafe {
        ce_sample_mu_kernel::launch::<R>(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(idata_h, packed.idata.len()),
            ArrayArg::from_raw_parts(fdata_h, packed.fdata.len()),
            ArrayArg::from_raw_parts(e_h, n),
            ArrayArg::from_raw_parts(xb_h, n),
            ArrayArg::from_raw_parts(xm_h, n),
            ArrayArg::from_raw_parts(out_h.clone(), n),
        );
    }
    let bytes = client.read_one(out_h).expect("mu readback");
    f64::from_bytes(&bytes).to_vec()
}

// ── Fission outgoing-energy (χ) distribution ────────────────────────
//
// CPU source: `NuclideKernels::fission_energy_dist: Option<EnergyDistribution>`.
// Two device paths, selected by `law`:
//   law = 0  tabular ContinuousTabular (ENDF Law 4): per-incident-energy
//            (e_out, pdf, cdf) tables + OpenMC stochastic-bin pick +
//            scaled kinematic remap. Mirrors EnergyDistribution::sample.
//   law = 1/2/3  closed-form Watt / Maxwell / Evaporation: a(E)/b(E) or
//            θ(E) lin-lin tables, sampled by the Cranberg/Coveyou
//            rejection samplers. Mirrors WattLaw::sample etc.
//   law = -1 none → emit the incident energy unchanged (fallback).

/// Flat per-nuclide fission-χ data, device-ready.
#[derive(Clone, Default)]
pub struct FissionCe {
    /// -1 none, 0 tabular, 1 Watt, 2 Maxwell, 3 Evaporation.
    pub law: i32,
    pub u: f64,
    // Tabular (law 0): incident grid + per-E (e_out,pdf,cdf) slices.
    pub energies: Vec<f64>,
    pub dist_off: Vec<i32>,
    pub dist_len: Vec<i32>,
    pub e_out: Vec<f64>,
    pub pdf: Vec<f64>,
    pub cdf: Vec<f64>,
    // Closed-form param tables (law 1/2/3). Watt uses (a*, b*); Maxwell/
    // Evaporation use (a* = θ). b* empty for non-Watt.
    pub pa_e: Vec<f64>,
    pub pa_v: Vec<f64>,
    pub pb_e: Vec<f64>,
    pub pb_v: Vec<f64>,
}

/// Extract fission-χ per nuclide (parallel to extract_ce / extract_angular).
pub fn extract_fission(
    nuclides: &[std::sync::Arc<crate::transport::xs_provider::NuclideKernels>],
) -> Vec<FissionCe> {
    use crate::hdf5_reader::FissionEnergyLaw;
    nuclides
        .iter()
        .map(|nuc| {
            let Some(ed) = nuc.fission_energy_dist.as_ref() else {
                return FissionCe { law: -1, ..Default::default() };
            };
            match &ed.closed_form {
                Some(FissionEnergyLaw::Watt(w)) => FissionCe {
                    law: 1,
                    u: w.u,
                    pa_e: w.a_energies.clone(),
                    pa_v: w.a_values.clone(),
                    pb_e: w.b_energies.clone(),
                    pb_v: w.b_values.clone(),
                    ..Default::default()
                },
                Some(FissionEnergyLaw::Maxwell(m)) => FissionCe {
                    law: 2,
                    u: m.u,
                    pa_e: m.theta_energies.clone(),
                    pa_v: m.theta_values.clone(),
                    ..Default::default()
                },
                Some(FissionEnergyLaw::Evaporation(m)) => FissionCe {
                    law: 3,
                    u: m.u,
                    pa_e: m.theta_energies.clone(),
                    pa_v: m.theta_values.clone(),
                    ..Default::default()
                },
                None => {
                    // Tabular ContinuousTabular.
                    if ed.energies.is_empty() || ed.distributions.is_empty() {
                        return FissionCe { law: -1, ..Default::default() };
                    }
                    let mut f = FissionCe {
                        law: 0,
                        energies: ed.energies.clone(),
                        ..Default::default()
                    };
                    for d in &ed.distributions {
                        f.dist_off.push(f.e_out.len() as i32);
                        f.dist_len.push(d.e_out.len() as i32);
                        f.e_out.extend_from_slice(&d.e_out);
                        f.cdf.extend_from_slice(&d.cdf);
                        if d.pdf.len() == d.e_out.len() {
                            f.pdf.extend_from_slice(&d.pdf);
                        } else {
                            f.pdf.extend(std::iter::repeat_n(0.0, d.e_out.len()));
                        }
                    }
                    f
                }
            }
        })
        .collect()
}

// ── Packed fission blob (one nuclide) ───────────────────────────────
// idata: [ law, n_e, eoff, doff, dloff, e_out_off, pdf_off, cdf_off,
//          pa_e_off, pa_n, pa_v_off, pb_e_off, pb_n, pb_v_off ]
//   then dist_off / dist_len arrays inline (tabular only).
// fdata: [ u, energies | e_out | pdf | cdf | pa_e | pa_v | pb_e | pb_v ]

pub struct PackedFission {
    pub idata: Vec<i32>,
    pub fdata: Vec<f64>,
}

const F_LAW: usize = 0;
const F_N_E: usize = 1;
const F_EOFF: usize = 2;
const F_DOFF: usize = 3;
const F_DLOFF: usize = 4;
const F_EOUT_OFF: usize = 5;
const F_PDF_OFF: usize = 6;
const F_CDF_OFF: usize = 7;
const F_PA_E_OFF: usize = 8;
const F_PA_N: usize = 9;
const F_PA_V_OFF: usize = 10;
const F_PB_E_OFF: usize = 11;
const F_PB_N: usize = 12;
const F_PB_V_OFF: usize = 13;
const F_U_SLOT: usize = 0; // fdata[0] = u
const F_HDR: usize = 14;

pub fn pack_fission(f: &FissionCe) -> PackedFission {
    let mut idata = vec![0i32; F_HDR];
    let mut fdata: Vec<f64> = vec![f.u]; // F_U_SLOT
    idata[F_LAW] = f.law;
    idata[F_N_E] = f.energies.len() as i32;

    idata[F_EOFF] = fdata.len() as i32;
    fdata.extend_from_slice(&f.energies);
    idata[F_EOUT_OFF] = fdata.len() as i32;
    fdata.extend_from_slice(&f.e_out);
    idata[F_PDF_OFF] = fdata.len() as i32;
    fdata.extend_from_slice(&f.pdf);
    idata[F_CDF_OFF] = fdata.len() as i32;
    fdata.extend_from_slice(&f.cdf);
    idata[F_PA_E_OFF] = fdata.len() as i32;
    fdata.extend_from_slice(&f.pa_e);
    idata[F_PA_N] = f.pa_e.len() as i32;
    idata[F_PA_V_OFF] = fdata.len() as i32;
    fdata.extend_from_slice(&f.pa_v);
    idata[F_PB_E_OFF] = fdata.len() as i32;
    fdata.extend_from_slice(&f.pb_e);
    idata[F_PB_N] = f.pb_e.len() as i32;
    idata[F_PB_V_OFF] = fdata.len() as i32;
    fdata.extend_from_slice(&f.pb_v);

    idata[F_DOFF] = idata.len() as i32;
    idata.extend_from_slice(&f.dist_off);
    idata[F_DLOFF] = idata.len() as i32;
    idata.extend_from_slice(&f.dist_len);

    if fdata.is_empty() {
        fdata.push(0.0);
    }
    if idata.len() == F_HDR {
        idata.push(0);
    }
    PackedFission { idata, fdata }
}

// ── Device samplers ─────────────────────────────────────────────────

/// lin-lin lookup over a (grid, values) pair packed at given offsets.
/// Mirrors WattLaw::lookup_lin_lin.
#[cube]
fn fis_lin_lin(fdata: &Array<f64>, goff: u32, voff: u32, n: u32, e: f64) -> f64 {
    let mut out = f64::new(0.0);
    if n >= 1u32 {
        let g0 = fdata[goff as usize];
        let glast = fdata[(goff + n - 1u32) as usize];
        if e <= g0 {
            out = fdata[voff as usize];
        } else {
            if e >= glast {
                out = fdata[(voff + n - 1u32) as usize];
            } else {
                let mut lo = u32::new(0);
                let mut hi = n - 1u32;
                for _i in 0..32u32 {
                    if lo + 1u32 < hi {
                        let mid = (lo + hi) / 2u32;
                        if fdata[(goff + mid) as usize] <= e {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                }
                let gl = fdata[(goff + lo) as usize];
                let gh = fdata[(goff + lo + 1u32) as usize];
                let vl = fdata[(voff + lo) as usize];
                let vh = fdata[(voff + lo + 1u32) as usize];
                let frac = (e - gl) / (gh - gl);
                out = vl + frac * (vh - vl);
            }
        }
    }
    out
}

/// Quadratic/linear CDF inversion of a tabular (e_out, cdf, pdf) slice
/// `[doff .. doff+dlen]` for `xi`. Mirrors TabularEnergyDist::sample_with_xi.
#[cube]
#[allow(unused_assignments)]
fn fis_sample_eout_bin(
    fdata: &Array<f64>,
    eout_off: u32,
    cdf_off: u32,
    pdf_off: u32,
    doff: u32,
    dlen: u32,
    xi: f64,
) -> f64 {
    let mut out = f64::new(1.0e6);
    if dlen >= 2u32 {
        let mut lo = u32::new(0);
        let mut hi = dlen - 1u32;
        for _i in 0..32u32 {
            if lo + 1u32 < hi {
                let mid = (lo + hi) / 2u32;
                if fdata[(cdf_off + doff + mid) as usize] <= xi {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        let idx = select(lo > dlen - 2u32, dlen - 2u32, lo);
        let cdf_lo = fdata[(cdf_off + doff + idx) as usize];
        let cdf_hi = fdata[(cdf_off + doff + idx + 1u32) as usize];
        let e_lo = fdata[(eout_off + doff + idx) as usize];
        let e_hi = fdata[(eout_off + doff + idx + 1u32) as usize];
        let de = e_hi - e_lo;
        if (cdf_hi - cdf_lo).abs() < f64::new(1e-15) {
            out = e_lo;
        } else {
            let p_lo = fdata[(pdf_off + doff + idx) as usize];
            let p_hi = fdata[(pdf_off + doff + idx + 1u32) as usize];
            // quadratic when pdf present & de>0, else linear.
            let use_quad = de > f64::new(0.0) && (p_lo > f64::new(0.0) || p_hi > f64::new(0.0));
            if use_quad {
                let m = (p_hi - p_lo) / de;
                let dc = xi - cdf_lo;
                if m.abs() < f64::new(1e-30) {
                    out = e_lo + dc / p_lo;
                } else {
                    let disc = p_lo * p_lo + 2.0 * m * dc;
                    let sq = select(disc > f64::new(0.0), disc, f64::new(0.0)).sqrt();
                    out = e_lo + (sq - p_lo) / m;
                }
            } else {
                let frac = (xi - cdf_lo) / (cdf_hi - cdf_lo);
                out = e_lo + frac * de;
            }
        }
    }
    let floor = f64::new(1e-5);
    select(out > floor, out, floor)
}

/// Sample a fission outgoing energy for the packed nuclide at incident
/// `e_in`, drawing uniforms from the local rng. Mirrors
/// EnergyDistribution::sample (tabular path with stochastic bin +
/// kinematic remap) and the closed-form Watt/Maxwell/Evaporation
/// samplers. `rng` is a length-2 u64 [state, inc].
#[cube]
#[allow(unused_assignments)]
fn fis_sample_energy(idata: &Array<i32>, fdata: &Array<f64>, e_in: f64, rng: &mut Array<u64>) -> f64 {
    let law = idata[F_LAW];
    let u = fdata[F_U_SLOT];
    let mut out = e_in; // law -1 fallback
    if law == 0i32 {
        // Tabular ContinuousTabular.
        let n_e = u32::cast_from(idata[F_N_E]);
        let eoff = u32::cast_from(idata[F_EOFF]);
        let doff_base = u32::cast_from(idata[F_DOFF]);
        let dloff_base = u32::cast_from(idata[F_DLOFF]);
        let eout_off = u32::cast_from(idata[F_EOUT_OFF]);
        let pdf_off = u32::cast_from(idata[F_PDF_OFF]);
        let cdf_off = u32::cast_from(idata[F_CDF_OFF]);
        if n_e >= 1u32 {
            let e0 = fdata[eoff as usize];
            let elast = fdata[(eoff + n_e - 1u32) as usize];
            if e_in <= e0 {
                let xi = ce_uniform_f(rng);
                let doff = u32::cast_from(idata[(doff_base) as usize]);
                let dlen = u32::cast_from(idata[(dloff_base) as usize]);
                out = fis_sample_eout_bin(fdata, eout_off, cdf_off, pdf_off, doff, dlen, xi);
            } else {
                if e_in >= elast {
                    let xi = ce_uniform_f(rng);
                    let doff = u32::cast_from(idata[(doff_base + n_e - 1u32) as usize]);
                    let dlen = u32::cast_from(idata[(dloff_base + n_e - 1u32) as usize]);
                    out = fis_sample_eout_bin(fdata, eout_off, cdf_off, pdf_off, doff, dlen, xi);
                } else {
                    // bracket
                    let mut lo = u32::new(0);
                    let mut hi = n_e - 1u32;
                    for _i in 0..32u32 {
                        if lo + 1u32 < hi {
                            let mid = (lo + hi) / 2u32;
                            if fdata[(eoff + mid) as usize] <= e_in {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                    }
                    let e_lo = fdata[(eoff + lo) as usize];
                    let e_hi = fdata[(eoff + lo + 1u32) as usize];
                    let r = (e_in - e_lo) / (e_hi - e_lo);
                    let pick_hi = ce_uniform_f(rng) < r;
                    let l = select(pick_hi, lo + 1u32, lo);
                    let doff_l = u32::cast_from(idata[(doff_base + l) as usize]);
                    let dlen_l = u32::cast_from(idata[(dloff_base + l) as usize]);
                    let xi = ce_uniform_f(rng);
                    let e_out = fis_sample_eout_bin(fdata, eout_off, cdf_off, pdf_off, doff_l, dlen_l, xi);
                    // Scaled kinematic remap to interpolated [E1, EK].
                    let doff_a = u32::cast_from(idata[(doff_base + lo) as usize]);
                    let dlen_a = u32::cast_from(idata[(dloff_base + lo) as usize]);
                    let doff_b = u32::cast_from(idata[(doff_base + lo + 1u32) as usize]);
                    let dlen_b = u32::cast_from(idata[(dloff_base + lo + 1u32) as usize]);
                    let ea_lo = fdata[(eout_off + doff_a) as usize];
                    let ea_hi = fdata[(eout_off + doff_a + dlen_a - 1u32) as usize];
                    let eb_lo = fdata[(eout_off + doff_b) as usize];
                    let eb_hi = fdata[(eout_off + doff_b + dlen_b - 1u32) as usize];
                    let el_lo = fdata[(eout_off + doff_l) as usize];
                    let el_hi = fdata[(eout_off + doff_l + dlen_l - 1u32) as usize];
                    let e1 = (f64::new(1.0) - r) * ea_lo + r * eb_lo;
                    let ek = (f64::new(1.0) - r) * ea_hi + r * eb_hi;
                    let span = el_hi - el_lo;
                    let adjusted = select(
                        span.abs() < f64::new(1e-30),
                        e_out,
                        e1 + (e_out - el_lo) * (ek - e1) / span,
                    );
                    let floor = f64::new(1e-5);
                    out = select(adjusted > floor, adjusted, floor);
                }
            }
        }
    } else {
        let pa_e = u32::cast_from(idata[F_PA_E_OFF]);
        let pa_n = u32::cast_from(idata[F_PA_N]);
        let pa_v = u32::cast_from(idata[F_PA_V_OFF]);
        let max_e = select(e_in - u > f64::new(1e-5), e_in - u, f64::new(1e-5));
        if law == 1i32 {
            // Watt.
            let pb_e = u32::cast_from(idata[F_PB_E_OFF]);
            let pb_n = u32::cast_from(idata[F_PB_N]);
            let pb_v = u32::cast_from(idata[F_PB_V_OFF]);
            let a = fis_lin_lin(fdata, pa_e, pa_v, pa_n, e_in);
            let b = fis_lin_lin(fdata, pb_e, pb_v, pb_n, e_in);
            let mut acc = f64::new(2.0);
            let mut done = false;
            for _i in 0..128u32 {
                if !done {
                    let xi1 = fmax_small(ce_uniform_f(rng));
                    let xi2 = ce_uniform_f(rng);
                    let xi3 = fmax_small(ce_uniform_f(rng));
                    let xi4 = ce_uniform_f(rng);
                    let c = (f64::new(1.5707963267948966) * xi2).cos();
                    let w = -a * (xi1.ln() + c * c * xi3.ln());
                    let term = a * a * b / 4.0;
                    let e_out = w + term + (2.0 * xi4 - 1.0) * (a * a * b * w).sqrt();
                    if e_out > f64::new(0.0) && e_out <= max_e {
                        acc = select(e_out > f64::new(1e-5), e_out, f64::new(1e-5));
                        done = true;
                    }
                }
            }
            out = acc;
        } else {
            // Maxwell (law 2) or Evaporation (law 3): pa = θ table.
            let theta = fis_lin_lin(fdata, pa_e, pa_v, pa_n, e_in);
            let mut acc = f64::new(2.0);
            let mut done = false;
            for _i in 0..128u32 {
                if !done {
                    let xi1 = fmax_small(ce_uniform_f(rng));
                    let xi2 = ce_uniform_f(rng);
                    let xi3 = fmax_small(ce_uniform_f(rng));
                    let e_out = select(
                        law == 2i32,
                        -theta * (xi1.ln() + (f64::new(1.5707963267948966) * xi2).cos() * (f64::new(1.5707963267948966) * xi2).cos() * xi3.ln()),
                        -theta * (xi1 * xi2).ln(),
                    );
                    if e_out > f64::new(0.0) && e_out <= max_e {
                        acc = select(e_out > f64::new(1e-5), e_out, f64::new(1e-5));
                        done = true;
                    }
                }
            }
            out = acc;
        }
    }
    out
}

/// uniform in [0,1) from a local [state,inc] rng (fission-module copy).
#[cube]
fn ce_uniform_f(rng: &mut Array<u64>) -> f64 {
    let old0 = rng[0];
    rng[0] = old0 * 6364136223846793005u64 + rng[1];
    let xs0 = u32::cast_from(((old0 >> 18u64) ^ old0) >> 27u64);
    let rot0 = u32::cast_from(old0 >> 59u64);
    let a = u64::cast_from((xs0 >> rot0) | (xs0 << ((32u32 - rot0) & 31u32))) >> 5u64;
    let old1 = rng[0];
    rng[0] = old1 * 6364136223846793005u64 + rng[1];
    let xs1 = u32::cast_from(((old1 >> 18u64) ^ old1) >> 27u64);
    let rot1 = u32::cast_from(old1 >> 59u64);
    let b = u64::cast_from((xs1 >> rot1) | (xs1 << ((32u32 - rot1) & 31u32))) >> 6u64;
    f64::cast_from(a * 67108864u64 + b) * (1.0 / 9007199254740992.0)
}

#[cube]
fn fmax_small(x: f64) -> f64 {
    select(x > f64::new(1e-300), x, f64::new(1e-300))
}

/// Test kernel: one fission E_out per thread at the given incident
/// energies + seeds.
#[cube(launch)]
fn ce_sample_fis_kernel(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    e_in: &Array<f64>,
    rng: &mut Array<u64>,
    out: &mut Array<f64>,
) {
    let tid = ABSOLUTE_POS;
    if tid < out.len() {
        let mut lrng = Array::<u64>::new(2usize);
        lrng[0] = rng[tid * 2];
        lrng[1] = rng[tid * 2 + 1];
        out[tid] = fis_sample_energy(idata, fdata, e_in[tid], &mut lrng);
        rng[tid * 2] = lrng[0];
        rng[tid * 2 + 1] = lrng[1];
    }
}

/// Sample fission E_out on the GPU for one packed nuclide.
pub fn sample_fission_gpu<R: Runtime>(
    device: &R::Device,
    packed: &PackedFission,
    e_in: &[f64],
    seeds: &[(u64, u64)],
) -> Vec<f64> {
    let client = R::client(device);
    let n = e_in.len();
    let mut rng_flat = Vec::with_capacity(n * 2);
    for s in seeds {
        rng_flat.push(s.0);
        rng_flat.push(s.1 | 1);
    }
    let idata_h = client.create_from_slice(i32::as_bytes(&packed.idata));
    let fdata_h = client.create_from_slice(f64::as_bytes(&packed.fdata));
    let e_h = client.create_from_slice(f64::as_bytes(e_in));
    let rng_h = client.create_from_slice(u64::as_bytes(&rng_flat));
    let out_h = client.empty(n * core::mem::size_of::<f64>());
    let threads = 64u32;
    let blocks = n.div_ceil(threads as usize) as u32;
    unsafe {
        ce_sample_fis_kernel::launch::<R>(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(idata_h, packed.idata.len()),
            ArrayArg::from_raw_parts(fdata_h, packed.fdata.len()),
            ArrayArg::from_raw_parts(e_h, n),
            ArrayArg::from_raw_parts(rng_h, n * 2),
            ArrayArg::from_raw_parts(out_h.clone(), n),
        );
    }
    let bytes = client.read_one(out_h).expect("fis readback");
    f64::from_bytes(&bytes).to_vec()
}
