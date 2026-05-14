//! Custom binary format for cached `NuclideKernels`.
//!
//! Design goals:
//!
//! - **No serde derive cascade.** `NuclideKernels` plus its sub-types
//!   (`SvdKernel` from an external crate, `PointwiseTable`,
//!   `AngularDistribution`, `EnergyDistribution`, `NuBarTable`,
//!   `UrrProbabilityTables`, `PhotonProduct`, ...) total ~25 types.
//!   Adding `#[derive(Serialize, Deserialize)]` to each is an invasive
//!   refactor that ripples into the third-party `rust-mc-sim` crate.
//!   This module encodes each type by hand against its public API.
//! - **`Vec<f64>` blocks are `memcpy`.** `bytemuck::cast_slice` turns
//!   `&[f64]` into `&[u8]` in zero passes; the writer just dumps the
//!   length prefix + raw bytes. Faster than bincode, which traverses
//!   element-by-element.
//! - **Same wire format works for L3.** A future remote daemon ships
//!   the exact bytes a fresh L2 file would carry; the protocol is just
//!   a length-prefixed framing on top.
//! - **Version-stamped header.** Any layout change → bump
//!   [`FORMAT_VERSION`] → existing keys mismatch → cache rebuilds.
//!   Versions are not backward-compatible by design (a half-decoded
//!   `NuclideKernels` is more dangerous than a re-parse).
//!
//! ## Layout
//!
//! ```text
//! [magic           : 8B] = b"ORM_NK01"
//! [format_version  : u32 LE]
//! [payload_blake3  : 32B]                ← integrity check
//! [payload_len     : u64 LE]
//! [payload         : payload_len bytes]
//! ```
//!
//! Payload is the concatenation of every encoded field of
//! `NuclideKernels`, in the order the encoder visits them. The decoder
//! walks the same order. Each leaf type owns its `encode_*` / `decode_*`
//! pair below; no field is positional only by accident, every read has
//! a matching write.

use std::io::{self, Read, Write};
use std::sync::Arc;

use crate::hdf5_reader::{
    AngularDistribution, DiscreteLevelInfo, EnergyDistribution, FissionEnergyLaw, MaxwellLaw,
    NuBarTable, PhotonProduct, TabularEnergyDist, TabularMuDist, UrrProbabilityTables, WattLaw,
};
use crate::kernel::SvdKernel;
use crate::table::PointwiseTable;
use crate::transport::xs_provider::{
    DiscreteLevel, InelasticCdf, NuclideKernels, ReactionKernel,
};

/// Magic bytes identifying our binary cache files. Eight ASCII bytes
/// so simple grep finds them in mixed-content directories.
pub const MAGIC: &[u8; 8] = b"ORM_NK01";

/// Bump on ANY change to the encode/decode layout — including adding a
/// field, changing a sub-type's encoder, or changing the underlying
/// `NuclideKernels` struct. Old entries become unreachable and are
/// transparently rebuilt.
///
/// History:
/// - v1: scaffolded (encoders stubbed, returned `Unimplemented`).
/// - v2: full encoder implementation (SvdKernel via from_factors,
///   ReactionKernel + EnergyDistribution + AngularDistribution +
///   NuBarTable + DiscreteLevel + InelasticCdf + UrrProbabilityTables
///   + PhotonProduct + partial_kernels).
pub const FORMAT_VERSION: u32 = 2;

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("bad magic — file is not a NuclideKernels cache entry")]
    BadMagic,
    #[error("format version {found} does not match expected {expected}")]
    VersionMismatch { found: u32, expected: u32 },
    #[error("payload blake3 mismatch — cache file is corrupt")]
    PayloadHashMismatch,
    #[error("truncated payload at offset {0}")]
    Truncated(u64),
    #[error("unknown enum discriminant {0}")]
    BadDiscriminant(u32),
    #[error("invalid UTF-8 in cached string")]
    BadUtf8,
}

// ── Primitive writers ────────────────────────────────────────────────

pub fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn write_i32<W: Write>(w: &mut W, v: i32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn write_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn write_bool<W: Write>(w: &mut W, v: bool) -> io::Result<()> {
    w.write_all(&[v as u8])
}

/// Write `Vec<f64>` as `[len: u64 LE][raw bytes]`. The raw payload is
/// `bytemuck::cast_slice` over `&[f64]` — zero per-element traversal.
pub fn write_vec_f64<W: Write>(w: &mut W, v: &[f64]) -> io::Result<()> {
    write_u64(w, v.len() as u64)?;
    if !v.is_empty() {
        let bytes: &[u8] = bytemuck::cast_slice(v);
        w.write_all(bytes)?;
    }
    Ok(())
}

pub fn write_vec_i32<W: Write>(w: &mut W, v: &[i32]) -> io::Result<()> {
    write_u64(w, v.len() as u64)?;
    if !v.is_empty() {
        let bytes: &[u8] = bytemuck::cast_slice(v);
        w.write_all(bytes)?;
    }
    Ok(())
}

pub fn write_vec_u32<W: Write>(w: &mut W, v: &[u32]) -> io::Result<()> {
    write_u64(w, v.len() as u64)?;
    if !v.is_empty() {
        let bytes: &[u8] = bytemuck::cast_slice(v);
        w.write_all(bytes)?;
    }
    Ok(())
}

pub fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    write_u64(w, s.len() as u64)?;
    w.write_all(s.as_bytes())
}

/// Write a `Some(T)` / `None` discriminant + payload. The closure
/// encodes the inner value when `Some`. Discriminant is 1B.
pub fn write_option<W: Write, T, F: FnOnce(&mut W, &T) -> io::Result<()>>(
    w: &mut W,
    v: Option<&T>,
    f: F,
) -> io::Result<()> {
    match v {
        Some(inner) => {
            w.write_all(&[1])?;
            f(w, inner)
        }
        None => w.write_all(&[0]),
    }
}

// ── Primitive readers ────────────────────────────────────────────────

pub fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0_u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
pub fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0_u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
pub fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut b = [0_u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
pub fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0_u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
pub fn read_bool<R: Read>(r: &mut R) -> io::Result<bool> {
    let mut b = [0_u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0] != 0)
}

pub fn read_vec_f64<R: Read>(r: &mut R) -> io::Result<Vec<f64>> {
    let n = read_u64(r)? as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut v = vec![0_f64; n];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(v.as_mut_slice());
    r.read_exact(bytes)?;
    Ok(v)
}

pub fn read_vec_i32<R: Read>(r: &mut R) -> io::Result<Vec<i32>> {
    let n = read_u64(r)? as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut v = vec![0_i32; n];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(v.as_mut_slice());
    r.read_exact(bytes)?;
    Ok(v)
}

pub fn read_vec_u32<R: Read>(r: &mut R) -> io::Result<Vec<u32>> {
    let n = read_u64(r)? as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut v = vec![0_u32; n];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(v.as_mut_slice());
    r.read_exact(bytes)?;
    Ok(v)
}

pub fn read_string<R: Read>(r: &mut R) -> Result<String, DecodeError> {
    let n = read_u64(r)? as usize;
    let mut buf = vec![0_u8; n];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| DecodeError::BadUtf8)
}

pub fn read_option<R: Read, T, F: FnOnce(&mut R) -> Result<T, DecodeError>>(
    r: &mut R,
    f: F,
) -> Result<Option<T>, DecodeError> {
    let mut tag = [0_u8; 1];
    r.read_exact(&mut tag)?;
    match tag[0] {
        0 => Ok(None),
        1 => Ok(Some(f(r)?)),
        d => Err(DecodeError::BadDiscriminant(d as u32)),
    }
}

// ── Header ────────────────────────────────────────────────────────────

/// Write the file header. `payload` is the already-encoded body; the
/// header is computed from it (blake3, length). The wire format is
/// length-then-payload, but the writer here accepts the full payload
/// up front so the disk format and the future network format are
/// identical (no streaming-hash-yet-to-determine-length games).
pub fn write_header_and_payload<W: Write>(w: &mut W, payload: &[u8]) -> io::Result<()> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(payload);
    let hash = hasher.finalize();
    w.write_all(MAGIC)?;
    write_u32(w, FORMAT_VERSION)?;
    w.write_all(hash.as_bytes())?;
    write_u64(w, payload.len() as u64)?;
    w.write_all(payload)?;
    Ok(())
}

/// Read header + verify magic / version / blake3 / payload length.
/// Returns the validated payload bytes ready to feed into a typed
/// decoder.
pub fn read_header_and_payload<R: Read>(r: &mut R) -> Result<Vec<u8>, DecodeError> {
    let mut magic = [0_u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let version = read_u32(r)?;
    if version != FORMAT_VERSION {
        return Err(DecodeError::VersionMismatch {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let mut expected_hash = [0_u8; 32];
    r.read_exact(&mut expected_hash)?;
    let payload_len = read_u64(r)? as usize;
    let mut payload = vec![0_u8; payload_len];
    r.read_exact(&mut payload)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&payload);
    if hasher.finalize().as_bytes() != &expected_hash {
        return Err(DecodeError::PayloadHashMismatch);
    }
    Ok(payload)
}

// ── Per-type encoders ────────────────────────────────────────────────
//
// One `encode_<type>` / `decode_<type>` pair per sub-type of
// `NuclideKernels`. Each pair lives next to its sibling so the
// inverse-pair check is local. Adding a new field to one of these
// types means bumping `FORMAT_VERSION` and updating both halves.
//
// Reading order rule: the decoder MUST visit fields in the exact same
// order as the encoder. Any reordering is a silent-bias bug.

fn encode_svd_kernel<W: Write>(w: &mut W, k: &SvdKernel) -> io::Result<()> {
    // `SvdKernel` is from `rust-mc-sim` — no Clone, no Serialize.
    // Reconstructed on decode via `SvdKernel::from_factors(basis,
    // vt_coeffs, row_axis, rank, n_rows, n_cols)`. We dump its public
    // accessors verbatim.
    write_u64(w, k.rank() as u64)?;
    write_u64(w, k.n_rows() as u64)?;
    write_u64(w, k.n_cols() as u64)?;
    write_vec_f64(w, k.row_axis())?;
    write_vec_f64(w, k.basis())?;
    write_vec_f64(w, k.vt())?;
    Ok(())
}

fn decode_svd_kernel<R: Read>(r: &mut R) -> Result<SvdKernel, DecodeError> {
    let rank = read_u64(r)? as usize;
    let n_rows = read_u64(r)? as usize;
    let n_cols = read_u64(r)? as usize;
    let row_axis_vec = read_vec_f64(r)?;
    let basis = read_vec_f64(r)?;
    let vt = read_vec_f64(r)?;
    let row_axis: Arc<[f64]> = row_axis_vec.into();
    Ok(SvdKernel::from_factors(basis, vt, row_axis, rank, n_rows, n_cols))
}

fn encode_reaction_kernel<W: Write>(w: &mut W, k: &ReactionKernel) -> io::Result<()> {
    match k {
        ReactionKernel::Svd { kernel, coeffs } => {
            w.write_all(&[0])?;
            encode_svd_kernel(w, kernel)?;
            write_vec_f64(w, coeffs)?;
        }
        ReactionKernel::Table { energies, xs } => {
            w.write_all(&[1])?;
            write_vec_f64(w, energies)?;
            write_vec_f64(w, xs)?;
        }
    }
    Ok(())
}

fn decode_reaction_kernel<R: Read>(r: &mut R) -> Result<ReactionKernel, DecodeError> {
    let mut tag = [0_u8; 1];
    r.read_exact(&mut tag)?;
    match tag[0] {
        0 => {
            let kernel = decode_svd_kernel(r)?;
            let coeffs = read_vec_f64(r)?;
            Ok(ReactionKernel::Svd { kernel, coeffs })
        }
        1 => {
            let energies = read_vec_f64(r)?;
            let xs = read_vec_f64(r)?;
            Ok(ReactionKernel::Table { energies, xs })
        }
        d => Err(DecodeError::BadDiscriminant(d as u32)),
    }
}

fn encode_pointwise_table<W: Write>(w: &mut W, t: &PointwiseTable) -> io::Result<()> {
    write_vec_f64(w, t.energies_slice())?;
    write_vec_f64(w, t.xs_slice())?;
    Ok(())
}

fn decode_pointwise_table<R: Read>(r: &mut R) -> Result<PointwiseTable, DecodeError> {
    let energies = read_vec_f64(r)?;
    let xs = read_vec_f64(r)?;
    Ok(PointwiseTable::from_vecs(energies, xs))
}

fn encode_nu_bar_table<W: Write>(w: &mut W, n: &NuBarTable) -> io::Result<()> {
    write_vec_f64(w, &n.energies)?;
    write_vec_f64(w, &n.values)?;
    Ok(())
}

fn decode_nu_bar_table<R: Read>(r: &mut R) -> Result<NuBarTable, DecodeError> {
    let energies = read_vec_f64(r)?;
    let values = read_vec_f64(r)?;
    Ok(NuBarTable { energies, values })
}

fn encode_discrete_level_info<W: Write>(w: &mut W, i: &DiscreteLevelInfo) -> io::Result<()> {
    write_u32(w, i.mt)?;
    write_f64(w, i.q_value)?;
    write_f64(w, i.threshold)?;
    Ok(())
}

fn decode_discrete_level_info<R: Read>(r: &mut R) -> Result<DiscreteLevelInfo, DecodeError> {
    let mt = read_u32(r)?;
    let q_value = read_f64(r)?;
    let threshold = read_f64(r)?;
    Ok(DiscreteLevelInfo { mt, q_value, threshold })
}

fn encode_discrete_level<W: Write>(w: &mut W, d: &DiscreteLevel) -> io::Result<()> {
    encode_discrete_level_info(w, &d.info)?;
    write_option(w, d.kernel.as_ref(), encode_reaction_kernel)?;
    Ok(())
}

fn decode_discrete_level<R: Read>(r: &mut R) -> Result<DiscreteLevel, DecodeError> {
    let info = decode_discrete_level_info(r)?;
    let kernel = read_option(r, decode_reaction_kernel)?;
    Ok(DiscreteLevel { info, kernel })
}

fn encode_inelastic_cdf<W: Write>(w: &mut W, c: &InelasticCdf) -> io::Result<()> {
    write_u64(w, c.n_levels as u64)?;
    write_u64(w, c.n_temp as u64)?;
    write_u64(w, c.n_energy as u64)?;
    write_f64(w, c.log_e_min)?;
    write_f64(w, c.log_e_max)?;
    write_vec_f64(w, &c.cdf_flat)?;
    write_vec_u32(w, &c.level_mts)?;
    Ok(())
}

fn decode_inelastic_cdf<R: Read>(r: &mut R) -> Result<InelasticCdf, DecodeError> {
    let n_levels = read_u64(r)? as usize;
    let n_temp = read_u64(r)? as usize;
    let n_energy = read_u64(r)? as usize;
    let log_e_min = read_f64(r)?;
    let log_e_max = read_f64(r)?;
    let cdf_flat = read_vec_f64(r)?;
    let level_mts = read_vec_u32(r)?;
    Ok(InelasticCdf {
        n_levels,
        n_temp,
        n_energy,
        log_e_min,
        log_e_max,
        cdf_flat,
        level_mts,
    })
}

fn encode_tabular_mu_dist<W: Write>(w: &mut W, d: &TabularMuDist) -> io::Result<()> {
    write_vec_f64(w, &d.mu)?;
    write_vec_f64(w, &d.pdf)?;
    write_vec_f64(w, &d.cdf)?;
    write_bool(w, d.histogram)?;
    Ok(())
}

fn decode_tabular_mu_dist<R: Read>(r: &mut R) -> Result<TabularMuDist, DecodeError> {
    let mu = read_vec_f64(r)?;
    let pdf = read_vec_f64(r)?;
    let cdf = read_vec_f64(r)?;
    let histogram = read_bool(r)?;
    Ok(TabularMuDist { mu, pdf, cdf, histogram })
}

fn encode_angular_distribution<W: Write>(w: &mut W, a: &AngularDistribution) -> io::Result<()> {
    write_vec_f64(w, &a.energies)?;
    write_u64(w, a.distributions.len() as u64)?;
    for d in &a.distributions {
        encode_tabular_mu_dist(w, d)?;
    }
    write_bool(w, a.center_of_mass)?;
    Ok(())
}

fn decode_angular_distribution<R: Read>(r: &mut R) -> Result<AngularDistribution, DecodeError> {
    let energies = read_vec_f64(r)?;
    let n = read_u64(r)? as usize;
    let mut distributions = Vec::with_capacity(n);
    for _ in 0..n {
        distributions.push(decode_tabular_mu_dist(r)?);
    }
    let center_of_mass = read_bool(r)?;
    Ok(AngularDistribution {
        energies,
        distributions,
        center_of_mass,
    })
}

fn encode_tabular_energy_dist<W: Write>(w: &mut W, t: &TabularEnergyDist) -> io::Result<()> {
    write_vec_f64(w, &t.e_out)?;
    write_vec_f64(w, &t.pdf)?;
    write_vec_f64(w, &t.cdf)?;
    Ok(())
}

fn decode_tabular_energy_dist<R: Read>(r: &mut R) -> Result<TabularEnergyDist, DecodeError> {
    let e_out = read_vec_f64(r)?;
    let pdf = read_vec_f64(r)?;
    let cdf = read_vec_f64(r)?;
    Ok(TabularEnergyDist { e_out, pdf, cdf })
}

fn encode_watt_law<W: Write>(w: &mut W, l: &WattLaw) -> io::Result<()> {
    write_vec_f64(w, &l.a_energies)?;
    write_vec_f64(w, &l.a_values)?;
    write_vec_f64(w, &l.b_energies)?;
    write_vec_f64(w, &l.b_values)?;
    write_f64(w, l.u)?;
    Ok(())
}

fn decode_watt_law<R: Read>(r: &mut R) -> Result<WattLaw, DecodeError> {
    let a_energies = read_vec_f64(r)?;
    let a_values = read_vec_f64(r)?;
    let b_energies = read_vec_f64(r)?;
    let b_values = read_vec_f64(r)?;
    let u = read_f64(r)?;
    Ok(WattLaw {
        a_energies,
        a_values,
        b_energies,
        b_values,
        u,
    })
}

fn encode_maxwell_law<W: Write>(w: &mut W, l: &MaxwellLaw) -> io::Result<()> {
    write_vec_f64(w, &l.theta_energies)?;
    write_vec_f64(w, &l.theta_values)?;
    write_f64(w, l.u)?;
    Ok(())
}

fn decode_maxwell_law<R: Read>(r: &mut R) -> Result<MaxwellLaw, DecodeError> {
    let theta_energies = read_vec_f64(r)?;
    let theta_values = read_vec_f64(r)?;
    let u = read_f64(r)?;
    Ok(MaxwellLaw {
        theta_energies,
        theta_values,
        u,
    })
}

fn encode_fission_energy_law<W: Write>(w: &mut W, l: &FissionEnergyLaw) -> io::Result<()> {
    match l {
        FissionEnergyLaw::Watt(law) => {
            w.write_all(&[0])?;
            encode_watt_law(w, law)
        }
        FissionEnergyLaw::Maxwell(law) => {
            w.write_all(&[1])?;
            encode_maxwell_law(w, law)
        }
        FissionEnergyLaw::Evaporation(law) => {
            w.write_all(&[2])?;
            encode_maxwell_law(w, law)
        }
    }
}

fn decode_fission_energy_law<R: Read>(r: &mut R) -> Result<FissionEnergyLaw, DecodeError> {
    let mut tag = [0_u8; 1];
    r.read_exact(&mut tag)?;
    match tag[0] {
        0 => Ok(FissionEnergyLaw::Watt(decode_watt_law(r)?)),
        1 => Ok(FissionEnergyLaw::Maxwell(decode_maxwell_law(r)?)),
        2 => Ok(FissionEnergyLaw::Evaporation(decode_maxwell_law(r)?)),
        d => Err(DecodeError::BadDiscriminant(d as u32)),
    }
}

fn encode_energy_distribution<W: Write>(w: &mut W, e: &EnergyDistribution) -> io::Result<()> {
    write_vec_f64(w, &e.energies)?;
    write_u64(w, e.distributions.len() as u64)?;
    for d in &e.distributions {
        encode_tabular_energy_dist(w, d)?;
    }
    write_option(w, e.closed_form.as_ref(), encode_fission_energy_law)?;
    Ok(())
}

fn decode_energy_distribution<R: Read>(r: &mut R) -> Result<EnergyDistribution, DecodeError> {
    let energies = read_vec_f64(r)?;
    let n = read_u64(r)? as usize;
    let mut distributions = Vec::with_capacity(n);
    for _ in 0..n {
        distributions.push(decode_tabular_energy_dist(r)?);
    }
    let closed_form = read_option(r, decode_fission_energy_law)?;
    Ok(EnergyDistribution {
        energies,
        distributions,
        closed_form,
    })
}

fn encode_urr_tables<W: Write>(w: &mut W, u: &UrrProbabilityTables) -> io::Result<()> {
    write_vec_f64(w, &u.energies)?;
    write_u64(w, u.n_bands as u64)?;
    write_vec_vec_f64(w, &u.cum_prob)?;
    write_vec_vec_f64(w, &u.total_factor)?;
    write_vec_vec_f64(w, &u.elastic_factor)?;
    write_vec_vec_f64(w, &u.fission_factor)?;
    write_vec_vec_f64(w, &u.capture_factor)?;
    write_bool(w, u.multiply_smooth)?;
    w.write_all(&[u.interpolation])?;
    Ok(())
}

fn decode_urr_tables<R: Read>(r: &mut R) -> Result<UrrProbabilityTables, DecodeError> {
    let energies = read_vec_f64(r)?;
    let n_bands = read_u64(r)? as usize;
    let cum_prob = read_vec_vec_f64(r)?;
    let total_factor = read_vec_vec_f64(r)?;
    let elastic_factor = read_vec_vec_f64(r)?;
    let fission_factor = read_vec_vec_f64(r)?;
    let capture_factor = read_vec_vec_f64(r)?;
    let multiply_smooth = read_bool(r)?;
    let mut interp = [0_u8; 1];
    r.read_exact(&mut interp)?;
    Ok(UrrProbabilityTables {
        energies,
        n_bands,
        cum_prob,
        total_factor,
        elastic_factor,
        fission_factor,
        capture_factor,
        multiply_smooth,
        interpolation: interp[0],
    })
}

fn write_vec_vec_f64<W: Write>(w: &mut W, vv: &[Vec<f64>]) -> io::Result<()> {
    write_u64(w, vv.len() as u64)?;
    for inner in vv {
        write_vec_f64(w, inner)?;
    }
    Ok(())
}

fn read_vec_vec_f64<R: Read>(r: &mut R) -> Result<Vec<Vec<f64>>, DecodeError> {
    let n = read_u64(r)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_vec_f64(r)?);
    }
    Ok(out)
}

fn encode_photon_product<W: Write>(w: &mut W, p: &PhotonProduct) -> io::Result<()> {
    write_u32(w, p.mt)?;
    encode_nu_bar_table(w, &p.yield_table)?;
    encode_energy_distribution(w, &p.energy_dist)?;
    Ok(())
}

fn decode_photon_product<R: Read>(r: &mut R) -> Result<PhotonProduct, DecodeError> {
    let mt = read_u32(r)?;
    let yield_table = decode_nu_bar_table(r)?;
    let energy_dist = decode_energy_distribution(r)?;
    Ok(PhotonProduct {
        mt,
        yield_table,
        energy_dist,
    })
}

// ── Top-level NuclideKernels encode / decode ─────────────────────────

/// Encode a `NuclideKernels` into the binary wire / disk format.
/// Returns the full byte sequence (header + payload). Callers write
/// it to disk (L2) or socket (future L3) atomically.
pub fn encode_nuclide_kernels(kernel: &NuclideKernels) -> Result<Vec<u8>, EncodeError> {
    let mut payload: Vec<u8> = Vec::new();
    encode_body(&mut payload, kernel)?;
    let mut out = Vec::with_capacity(payload.len() + 64);
    write_header_and_payload(&mut out, &payload)?;
    Ok(out)
}

/// Decode a `NuclideKernels` from the binary wire / disk format.
/// Verifies header (magic, version, blake3 of payload, length) before
/// invoking the body decoder. Failures here are typed; callers can
/// distinguish corruption from format-mismatch and act accordingly.
pub fn decode_nuclide_kernels(bytes: &[u8]) -> Result<NuclideKernels, DecodeError> {
    let mut r: &[u8] = bytes;
    let payload = read_header_and_payload(&mut r)?;
    let mut pr: &[u8] = &payload;
    decode_body(&mut pr)
}

fn encode_body<W: Write>(w: &mut W, k: &NuclideKernels) -> Result<(), EncodeError> {
    // Field order MUST match decode_body. Adding / removing / reordering
    // a field requires bumping FORMAT_VERSION.
    write_option(w, k.elastic.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.total_table.as_ref(), encode_pointwise_table)?;
    write_option(w, k.total_xs_raw.as_ref(), |w, v| write_vec_f64(w, v))?;
    write_option(w, k.missing_xs.as_ref(), |w, v| write_vec_f64(w, v))?;
    write_option(w, k.pointwise_xs.as_ref(), |w, v| write_vec_f64(w, v))?;
    write_option(w, k.inelastic.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.n2n.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.n3n.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.n4n.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.n4n_edist.as_ref(), encode_energy_distribution)?;
    write_option(w, k.fission.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.capture.as_ref(), encode_reaction_kernel)?;
    write_f64(w, k.awr)?;
    write_f64(w, k.nu_bar_const)?;
    write_option(w, k.nu_bar_table.as_ref(), encode_nu_bar_table)?;
    write_option(w, k.delayed_nu_bar_table.as_ref(), encode_nu_bar_table)?;
    // discrete_levels: Vec<DiscreteLevel>
    write_u64(w, k.discrete_levels.len() as u64)?;
    for d in &k.discrete_levels {
        encode_discrete_level(w, d)?;
    }
    write_option(w, k.inelastic_cdf.as_ref(), encode_inelastic_cdf)?;
    // discrete_level_angles: Vec<Option<AngularDistribution>>
    write_u64(w, k.discrete_level_angles.len() as u64)?;
    for slot in &k.discrete_level_angles {
        write_option(w, slot.as_ref(), encode_angular_distribution)?;
    }
    write_bool(w, k.has_continuum_inelastic)?;
    write_option(w, k.elastic_angle.as_ref(), encode_angular_distribution)?;
    write_option(w, k.fission_energy_dist.as_ref(), encode_energy_distribution)?;
    write_option(w, k.inelastic_continuum_edist.as_ref(), encode_energy_distribution)?;
    write_option(w, k.n2n_edist.as_ref(), encode_energy_distribution)?;
    write_option(w, k.n3n_edist.as_ref(), encode_energy_distribution)?;
    // (n,nα) / (n,2nα) / (n,np) — kernels + edist + Q triples
    write_option(w, k.n_nalpha.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.n_nalpha_edist.as_ref(), encode_energy_distribution)?;
    write_f64(w, k.q_n_nalpha)?;
    write_option(w, k.n_2nalpha.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.n_2nalpha_edist.as_ref(), encode_energy_distribution)?;
    write_f64(w, k.q_n_2nalpha)?;
    write_option(w, k.n_np.as_ref(), encode_reaction_kernel)?;
    write_option(w, k.n_np_edist.as_ref(), encode_energy_distribution)?;
    write_f64(w, k.q_n_np)?;
    write_option(w, k.urr_tables.as_ref(), encode_urr_tables)?;
    // photon_products: Vec<(u32, PhotonProduct)>
    write_u64(w, k.photon_products.len() as u64)?;
    for (mt, pp) in &k.photon_products {
        write_u32(w, *mt)?;
        encode_photon_product(w, pp)?;
    }
    // partial_kernels: Vec<(u32, ReactionKernel)>
    write_u64(w, k.partial_kernels.len() as u64)?;
    for (mt, rk) in &k.partial_kernels {
        write_u32(w, *mt)?;
        encode_reaction_kernel(w, rk)?;
    }
    Ok(())
}

fn decode_body<R: Read>(r: &mut R) -> Result<NuclideKernels, DecodeError> {
    let elastic = read_option(r, decode_reaction_kernel)?;
    let total_table = read_option(r, decode_pointwise_table)?;
    let total_xs_raw = read_option(r, |r| read_vec_f64(r).map_err(DecodeError::from))?;
    let missing_xs = read_option(r, |r| read_vec_f64(r).map_err(DecodeError::from))?;
    let pointwise_xs = read_option(r, |r| read_vec_f64(r).map_err(DecodeError::from))?;
    let inelastic = read_option(r, decode_reaction_kernel)?;
    let n2n = read_option(r, decode_reaction_kernel)?;
    let n3n = read_option(r, decode_reaction_kernel)?;
    let n4n = read_option(r, decode_reaction_kernel)?;
    let n4n_edist = read_option(r, decode_energy_distribution)?;
    let fission = read_option(r, decode_reaction_kernel)?;
    let capture = read_option(r, decode_reaction_kernel)?;
    let awr = read_f64(r)?;
    let nu_bar_const = read_f64(r)?;
    let nu_bar_table = read_option(r, decode_nu_bar_table)?;
    let delayed_nu_bar_table = read_option(r, decode_nu_bar_table)?;
    let n_levels = read_u64(r)? as usize;
    let mut discrete_levels = Vec::with_capacity(n_levels);
    for _ in 0..n_levels {
        discrete_levels.push(decode_discrete_level(r)?);
    }
    let inelastic_cdf = read_option(r, decode_inelastic_cdf)?;
    let n_angles = read_u64(r)? as usize;
    let mut discrete_level_angles = Vec::with_capacity(n_angles);
    for _ in 0..n_angles {
        discrete_level_angles.push(read_option(r, decode_angular_distribution)?);
    }
    let has_continuum_inelastic = read_bool(r)?;
    let elastic_angle = read_option(r, decode_angular_distribution)?;
    let fission_energy_dist = read_option(r, decode_energy_distribution)?;
    let inelastic_continuum_edist = read_option(r, decode_energy_distribution)?;
    let n2n_edist = read_option(r, decode_energy_distribution)?;
    let n3n_edist = read_option(r, decode_energy_distribution)?;
    let n_nalpha = read_option(r, decode_reaction_kernel)?;
    let n_nalpha_edist = read_option(r, decode_energy_distribution)?;
    let q_n_nalpha = read_f64(r)?;
    let n_2nalpha = read_option(r, decode_reaction_kernel)?;
    let n_2nalpha_edist = read_option(r, decode_energy_distribution)?;
    let q_n_2nalpha = read_f64(r)?;
    let n_np = read_option(r, decode_reaction_kernel)?;
    let n_np_edist = read_option(r, decode_energy_distribution)?;
    let q_n_np = read_f64(r)?;
    let urr_tables = read_option(r, decode_urr_tables)?;
    let n_photon = read_u64(r)? as usize;
    let mut photon_products = Vec::with_capacity(n_photon);
    for _ in 0..n_photon {
        let mt = read_u32(r)?;
        let pp = decode_photon_product(r)?;
        photon_products.push((mt, pp));
    }
    let n_partial = read_u64(r)? as usize;
    let mut partial_kernels = Vec::with_capacity(n_partial);
    for _ in 0..n_partial {
        let mt = read_u32(r)?;
        let rk = decode_reaction_kernel(r)?;
        partial_kernels.push((mt, rk));
    }
    Ok(NuclideKernels {
        elastic,
        total_table,
        total_xs_raw,
        missing_xs,
        pointwise_xs,
        inelastic,
        n2n,
        n3n,
        n4n,
        n4n_edist,
        fission,
        capture,
        awr,
        nu_bar_const,
        nu_bar_table,
        delayed_nu_bar_table,
        discrete_levels,
        inelastic_cdf,
        discrete_level_angles,
        has_continuum_inelastic,
        elastic_angle,
        fission_energy_dist,
        inelastic_continuum_edist,
        n2n_edist,
        n3n_edist,
        n_nalpha,
        n_nalpha_edist,
        q_n_nalpha,
        n_2nalpha,
        n_2nalpha_edist,
        q_n_2nalpha,
        n_np,
        n_np_edist,
        q_n_np,
        urr_tables,
        photon_products,
        partial_kernels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip_clean() {
        let payload = b"hello world".to_vec();
        let mut buf = Vec::new();
        write_header_and_payload(&mut buf, &payload).unwrap();
        let mut r: &[u8] = &buf;
        let got = read_header_and_payload(&mut r).unwrap();
        assert_eq!(got, payload);
    }

    #[test]
    fn header_corruption_detected() {
        let payload = b"hello".to_vec();
        let mut buf = Vec::new();
        write_header_and_payload(&mut buf, &payload).unwrap();
        let payload_offset = 8 + 4 + 32 + 8;
        buf[payload_offset] ^= 0x01;
        let mut r: &[u8] = &buf;
        match read_header_and_payload(&mut r) {
            Err(DecodeError::PayloadHashMismatch) => {}
            other => panic!("expected PayloadHashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn bad_magic_detected() {
        let mut buf = vec![0_u8; 8 + 4 + 32 + 8];
        let mut r: &[u8] = &buf;
        match read_header_and_payload(&mut r) {
            Err(DecodeError::BadMagic) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
        buf[0] = b'X';
        let mut r: &[u8] = &buf;
        match read_header_and_payload(&mut r) {
            Err(DecodeError::BadMagic) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn vec_f64_roundtrip_zero_copy() {
        let v: Vec<f64> = (0..100).map(|i| (i as f64) * 0.5).collect();
        let mut buf = Vec::new();
        write_vec_f64(&mut buf, &v).unwrap();
        let mut r: &[u8] = &buf;
        let got = read_vec_f64(&mut r).unwrap();
        assert_eq!(v, got);
    }

    #[test]
    fn option_discriminant_roundtrip() {
        let mut buf = Vec::new();
        write_option(&mut buf, Some(&42.0_f64), |w, v| write_f64(w, *v)).unwrap();
        write_option::<_, f64, _>(&mut buf, None, |w, v| write_f64(w, *v)).unwrap();
        let mut r: &[u8] = &buf;
        let a: Option<f64> = read_option(&mut r, |r| Ok(read_f64(r)?)).unwrap();
        let b: Option<f64> = read_option(&mut r, |r| Ok(read_f64(r)?)).unwrap();
        assert_eq!(a, Some(42.0));
        assert_eq!(b, None);
    }

    #[test]
    fn empty_nuclide_kernels_roundtrip() {
        // The simplest possible kernel — all-None fields. Exercises
        // every Option write_0 / read_0 branch without needing real
        // HDF5 data on disk.
        let k = NuclideKernels::empty(238.0289, 2.43);
        let bytes = encode_nuclide_kernels(&k).unwrap();
        let decoded = decode_nuclide_kernels(&bytes).unwrap();
        assert_eq!(decoded.awr, k.awr);
        assert_eq!(decoded.nu_bar_const, k.nu_bar_const);
        assert!(decoded.elastic.is_none());
        assert!(decoded.fission.is_none());
        assert!(decoded.urr_tables.is_none());
        assert!(decoded.discrete_levels.is_empty());
        assert!(decoded.discrete_level_angles.is_empty());
        assert!(decoded.photon_products.is_empty());
        assert!(decoded.partial_kernels.is_empty());
    }

    #[test]
    fn reaction_kernel_table_variant_roundtrip() {
        let rk = ReactionKernel::Table {
            energies: vec![1.0, 2.0, 4.0, 8.0],
            xs: vec![5.0, 4.0, 3.0, 2.5],
        };
        let mut buf = Vec::new();
        encode_reaction_kernel(&mut buf, &rk).unwrap();
        let mut r: &[u8] = &buf;
        let decoded = decode_reaction_kernel(&mut r).unwrap();
        match decoded {
            ReactionKernel::Table { energies, xs } => {
                assert_eq!(energies, vec![1.0, 2.0, 4.0, 8.0]);
                assert_eq!(xs, vec![5.0, 4.0, 3.0, 2.5]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn nu_bar_table_roundtrip() {
        let t = NuBarTable {
            energies: vec![1e-5, 1.0, 1e6],
            values: vec![2.43, 2.45, 2.55],
        };
        let mut buf = Vec::new();
        encode_nu_bar_table(&mut buf, &t).unwrap();
        let mut r: &[u8] = &buf;
        let decoded = decode_nu_bar_table(&mut r).unwrap();
        assert_eq!(decoded.energies, t.energies);
        assert_eq!(decoded.values, t.values);
    }

    #[test]
    fn fission_energy_law_three_variants_roundtrip() {
        let laws = vec![
            FissionEnergyLaw::Watt(WattLaw {
                a_energies: vec![1.0, 2.0],
                a_values: vec![0.988, 1.0],
                b_energies: vec![1.0, 2.0],
                b_values: vec![2.249e-6, 2.5e-6],
                u: 0.0,
            }),
            FissionEnergyLaw::Maxwell(MaxwellLaw {
                theta_energies: vec![1.0, 5e6],
                theta_values: vec![1.3e6, 1.4e6],
                u: 0.0,
            }),
            FissionEnergyLaw::Evaporation(MaxwellLaw {
                theta_energies: vec![0.0],
                theta_values: vec![0.5e6],
                u: 0.0,
            }),
        ];
        for law in &laws {
            let mut buf = Vec::new();
            encode_fission_energy_law(&mut buf, law).unwrap();
            let mut r: &[u8] = &buf;
            let decoded = decode_fission_energy_law(&mut r).unwrap();
            // Match-on-match — variant + payload equality.
            match (law, decoded) {
                (FissionEnergyLaw::Watt(a), FissionEnergyLaw::Watt(b)) => {
                    assert_eq!(a.a_energies, b.a_energies);
                    assert_eq!(a.b_values, b.b_values);
                    assert_eq!(a.u, b.u);
                }
                (FissionEnergyLaw::Maxwell(a), FissionEnergyLaw::Maxwell(b))
                | (FissionEnergyLaw::Evaporation(a), FissionEnergyLaw::Evaporation(b)) => {
                    assert_eq!(a.theta_energies, b.theta_energies);
                    assert_eq!(a.theta_values, b.theta_values);
                    assert_eq!(a.u, b.u);
                }
                _ => panic!("variant mismatch on roundtrip"),
            }
        }
    }

    #[test]
    fn urr_tables_jagged_vec_vec_roundtrip() {
        // 3 energies × variable band count to exercise the
        // length-prefixed inner-vec encoding.
        let urr = UrrProbabilityTables {
            energies: vec![1e3, 1e4, 1e5],
            n_bands: 20,
            cum_prob: vec![
                vec![0.05, 0.1, 0.5, 1.0],
                vec![0.02, 0.08, 0.6, 1.0],
                vec![0.1, 0.3, 0.7, 1.0],
            ],
            total_factor: vec![vec![1.0; 4]; 3],
            elastic_factor: vec![vec![0.9; 4]; 3],
            fission_factor: vec![vec![0.5; 4]; 3],
            capture_factor: vec![vec![0.3; 4]; 3],
            multiply_smooth: true,
            interpolation: 2,
        };
        let mut buf = Vec::new();
        encode_urr_tables(&mut buf, &urr).unwrap();
        let mut r: &[u8] = &buf;
        let decoded = decode_urr_tables(&mut r).unwrap();
        assert_eq!(decoded.energies, urr.energies);
        assert_eq!(decoded.n_bands, urr.n_bands);
        assert_eq!(decoded.cum_prob, urr.cum_prob);
        assert_eq!(decoded.multiply_smooth, urr.multiply_smooth);
        assert_eq!(decoded.interpolation, urr.interpolation);
    }

    #[test]
    fn discrete_level_info_roundtrip() {
        let info = DiscreteLevelInfo {
            mt: 51,
            q_value: -0.078e6,
            threshold: 0.079e6,
        };
        let mut buf = Vec::new();
        encode_discrete_level_info(&mut buf, &info).unwrap();
        let mut r: &[u8] = &buf;
        let decoded = decode_discrete_level_info(&mut r).unwrap();
        assert_eq!(decoded.mt, info.mt);
        assert_eq!(decoded.q_value, info.q_value);
        assert_eq!(decoded.threshold, info.threshold);
    }

    #[test]
    fn inelastic_cdf_roundtrip() {
        let cdf = InelasticCdf {
            n_levels: 3,
            n_temp: 2,
            n_energy: 4,
            log_e_min: -3.0,
            log_e_max: 7.0,
            cdf_flat: (0..24).map(|i| i as f64 * 0.04).collect(),
            level_mts: vec![51, 52, 91],
        };
        let mut buf = Vec::new();
        encode_inelastic_cdf(&mut buf, &cdf).unwrap();
        let mut r: &[u8] = &buf;
        let decoded = decode_inelastic_cdf(&mut r).unwrap();
        assert_eq!(decoded.n_levels, cdf.n_levels);
        assert_eq!(decoded.n_temp, cdf.n_temp);
        assert_eq!(decoded.n_energy, cdf.n_energy);
        assert_eq!(decoded.log_e_min, cdf.log_e_min);
        assert_eq!(decoded.log_e_max, cdf.log_e_max);
        assert_eq!(decoded.cdf_flat, cdf.cdf_flat);
        assert_eq!(decoded.level_mts, cdf.level_mts);
    }
}
