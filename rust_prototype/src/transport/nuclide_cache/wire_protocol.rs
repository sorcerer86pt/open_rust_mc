//! TCP framing for the L3 cache daemon — the same `binary_format`
//! payload, wrapped in a request / response envelope.
//!
//! Protocol is intentionally trivial:
//!
//! ```text
//! Request:  [op : u8] [key_blob_len : u32 LE] [key_blob]
//!           [payload_len : u64 LE] [payload]
//! Response: [status : u8] [payload_len : u64 LE] [payload]
//! ```
//!
//! `op` is one of `OP_GET`, `OP_PUT`, `OP_STATS`. For `OP_GET` the
//! request `payload_len` is `0`; for `OP_PUT` the payload is exactly
//! the bytes a fresh L2 disk file would carry (header + binary_format
//! NuclideKernels payload — see `encode_nuclide_kernels`).
//!
//! `status` is `STATUS_HIT` (1) on a GET that produced bytes,
//! `STATUS_MISS` (0) on a GET that didn't, `STATUS_OK` (3) on a
//! successful PUT, `STATUS_ERR` (2) on any error (the response
//! payload may carry a human-readable error string).
//!
//! The protocol is length-prefixed end-to-end so a partially read
//! frame is detectable: any short read at a `read_exact` checkpoint
//! collapses the connection. Re-establishment is the client's
//! responsibility (the `l3_remote` client opens one TCP connection per
//! request; connection pooling can layer on later without a protocol
//! change).
//!
//! Backward compatibility: when `binary_format::FORMAT_VERSION` is
//! bumped, the server-side `STATS` op still works (it doesn't
//! involve a NuclideKernels payload) but `GET`s for keys baked at the
//! old version will mismatch the new client's `FORMAT_VERSION` field
//! and result in cache misses. The misses repopulate the server with
//! the new format. No protocol bump required.

use std::io::{self, Read, Write};

use super::binary_format::{
    DecodeError, read_u32, read_u64, read_vec_u32, write_u32, write_u64, write_vec_u32,
};
use super::key::NuclideKey;

pub const OP_GET: u8 = 0;
pub const OP_PUT: u8 = 1;
pub const OP_STATS: u8 = 2;

pub const STATUS_MISS: u8 = 0;
pub const STATUS_HIT: u8 = 1;
pub const STATUS_ERR: u8 = 2;
pub const STATUS_OK: u8 = 3;

/// Serialise a `NuclideKey` to bytes for transmission. The encoding
/// is fixed-layout — the path is variable-length UTF-8, every other
/// field is fixed-size.
pub fn write_key<W: Write>(w: &mut W, key: &NuclideKey) -> io::Result<()> {
    // path: u32 LE length + UTF-8 bytes (paths are not large; u32 is
    // ample headroom). We use the lossy String conversion because
    // path encoding on disk is OS-specific and not all paths
    // round-trip through UTF-8 — but the path is informational here
    // (the file_hash is the real key), so any lossy fallback is fine.
    let path_str = key.path.to_string_lossy();
    write_u32(w, path_str.len() as u32)?;
    w.write_all(path_str.as_bytes())?;
    w.write_all(&key.file_hash)?;
    w.write_all(&key.policy_hash)?;
    write_u32(w, key.temp_idx)?;
    write_u32(w, key.format_version)?;
    Ok(())
}

pub fn read_key<R: Read>(r: &mut R) -> Result<NuclideKey, DecodeError> {
    let path_len = read_u32(r)? as usize;
    let mut path_bytes = vec![0_u8; path_len];
    r.read_exact(&mut path_bytes)?;
    let path = std::path::PathBuf::from(
        String::from_utf8(path_bytes).map_err(|_| DecodeError::BadUtf8)?,
    );
    let mut file_hash = [0_u8; 32];
    r.read_exact(&mut file_hash)?;
    let mut policy_hash = [0_u8; 32];
    r.read_exact(&mut policy_hash)?;
    let temp_idx = read_u32(r)?;
    let format_version = read_u32(r)?;
    Ok(NuclideKey {
        path,
        file_hash,
        policy_hash,
        temp_idx,
        format_version,
    })
}

/// Frame a request on the wire. Owns no I/O beyond what the writer
/// provides — the caller controls the socket.
pub fn write_request<W: Write>(
    w: &mut W,
    op: u8,
    key: &NuclideKey,
    payload: &[u8],
) -> io::Result<()> {
    // Buffer the key so we can prefix its length.
    let mut key_blob = Vec::with_capacity(192);
    write_key(&mut key_blob, key)?;
    w.write_all(&[op])?;
    write_u32(w, key_blob.len() as u32)?;
    w.write_all(&key_blob)?;
    write_u64(w, payload.len() as u64)?;
    w.write_all(payload)?;
    w.flush()?;
    Ok(())
}

/// Counterpart to `write_request` — for the server side.
pub fn read_request<R: Read>(r: &mut R) -> Result<(u8, NuclideKey, Vec<u8>), DecodeError> {
    let mut op = [0_u8; 1];
    r.read_exact(&mut op)?;
    let key_blob_len = read_u32(r)? as usize;
    let mut key_blob = vec![0_u8; key_blob_len];
    r.read_exact(&mut key_blob)?;
    let mut key_r: &[u8] = &key_blob;
    let key = read_key(&mut key_r)?;
    let payload_len = read_u64(r)? as usize;
    let mut payload = vec![0_u8; payload_len];
    r.read_exact(&mut payload)?;
    Ok((op[0], key, payload))
}

/// Frame a response. `status + payload_len + payload`. The caller
/// holds the socket.
pub fn write_response<W: Write>(w: &mut W, status: u8, payload: &[u8]) -> io::Result<()> {
    w.write_all(&[status])?;
    write_u64(w, payload.len() as u64)?;
    w.write_all(payload)?;
    w.flush()?;
    Ok(())
}

/// Client-side response read.
pub fn read_response<R: Read>(r: &mut R) -> Result<(u8, Vec<u8>), DecodeError> {
    let mut status = [0_u8; 1];
    r.read_exact(&mut status)?;
    let payload_len = read_u64(r)? as usize;
    let mut payload = vec![0_u8; payload_len];
    r.read_exact(&mut payload)?;
    Ok((status[0], payload))
}

// Use these helpers to silence the (intentionally unused) imports
// that exist so this module compiles standalone — they're referenced
// indirectly through the doc-example below.
#[allow(dead_code)]
fn _unused_helpers_anchor() {
    let mut buf: Vec<u8> = Vec::new();
    let _ = write_vec_u32(&mut buf, &[1_u32, 2]);
    let mut r: &[u8] = &buf;
    let _ = read_vec_u32(&mut r);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::xs_provider::RankPolicy;

    #[test]
    fn key_roundtrip_through_wire() {
        let tmp = std::env::temp_dir().join("orm_wire_test.h5");
        std::fs::write(&tmp, b"abc").unwrap();
        let policy = RankPolicy::new(5);
        let k = NuclideKey::from_inputs(&tmp, &policy, 2).unwrap();
        let mut buf = Vec::new();
        write_key(&mut buf, &k).unwrap();
        let mut r: &[u8] = &buf;
        let decoded = read_key(&mut r).unwrap();
        assert_eq!(decoded, k);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn request_frame_roundtrip() {
        let tmp = std::env::temp_dir().join("orm_wire_req_test.h5");
        std::fs::write(&tmp, b"xyz").unwrap();
        let policy = RankPolicy::new(7);
        let key = NuclideKey::from_inputs(&tmp, &policy, 0).unwrap();
        let payload = vec![1_u8, 2, 3, 4];
        let mut socket = Vec::new();
        write_request(&mut socket, OP_PUT, &key, &payload).unwrap();
        let mut r: &[u8] = &socket;
        let (op, k_decoded, p_decoded) = read_request(&mut r).unwrap();
        assert_eq!(op, OP_PUT);
        assert_eq!(k_decoded, key);
        assert_eq!(p_decoded, payload);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn response_frame_roundtrip() {
        let payload = (0..256_u32).map(|i| i as u8).collect::<Vec<_>>();
        let mut socket = Vec::new();
        write_response(&mut socket, STATUS_HIT, &payload).unwrap();
        let mut r: &[u8] = &socket;
        let (status, p_decoded) = read_response(&mut r).unwrap();
        assert_eq!(status, STATUS_HIT);
        assert_eq!(p_decoded, payload);
    }
}
