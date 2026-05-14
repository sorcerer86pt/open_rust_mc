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
//! walks the same order. Field-by-field: each leaf type owns its
//! encoder + decoder pair in this module; no field is positional only
//! by accident, every read has a matching write.
//!
//! This skeleton is the infrastructure (reader, writer, helpers,
//! header). The full per-type encoders for `NuclideKernels` land in a
//! follow-up commit — until they exist, [`encode_nuclide_kernels`] and
//! [`decode_nuclide_kernels`] return `Err(EncodeError::Unimplemented)`
//! and the L2 disk store skips writes (logs once). L1 in-memory and
//! the L3 trait slot are unaffected — the in-sweep re-parse is already
//! killed by the L1 hit.

use std::io::{self, Read, Write};

/// Magic bytes identifying our binary cache files. Eight ASCII bytes
/// so simple grep finds them in mixed-content directories.
pub const MAGIC: &[u8; 8] = b"ORM_NK01";

/// Bump on ANY change to the encode/decode layout — including adding a
/// field, changing a sub-type's encoder, or changing the underlying
/// `NuclideKernels` struct. Old entries become unreachable and are
/// transparently rebuilt.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("per-type encoders for NuclideKernels are not implemented yet — \
             L2 disk cache reads / writes will be skipped silently until they land")]
    Unimplemented,
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
    #[error("decoder for this NuclideKernels variant is not implemented yet")]
    Unimplemented,
}

/// Write a u32 in little-endian.
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

// ── Readers ───────────────────────────────────────────────────────────

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

// ── Top-level encode / decode for NuclideKernels ──────────────────────

use crate::transport::xs_provider::NuclideKernels;

/// Encode a `NuclideKernels` into the binary wire / disk format. Returns
/// the full byte sequence (header + payload). Caller writes it to a
/// file (L2) or socket (future L3) atomically.
///
/// **Not yet implemented.** The infrastructure (header, helpers, error
/// type) is here; per-type encoders for the 25-odd sub-types land in a
/// follow-up commit. Until then this returns `Err(Unimplemented)` and
/// the L2 disk store treats every cache write as a no-op (kernel stays
/// in L1 only). The trade-off: L1 already kills the 35× re-parse
/// during one sweep — the immediate ICSBEP win is delivered; L2 cold-
/// start savings come once the per-type encoders are in.
pub fn encode_nuclide_kernels(_kernel: &NuclideKernels) -> Result<Vec<u8>, EncodeError> {
    Err(EncodeError::Unimplemented)
}

/// Decode a `NuclideKernels` from the binary wire / disk format.
/// Companion to [`encode_nuclide_kernels`] — paired implementation lands
/// alongside it.
pub fn decode_nuclide_kernels(_bytes: &[u8]) -> Result<NuclideKernels, DecodeError> {
    Err(DecodeError::Unimplemented)
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
        // Flip one bit inside the payload region (after the 8B magic +
        // 4B version + 32B hash + 8B length header).
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
        // also with garbage but right length
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
}
