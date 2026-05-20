// SPDX-License-Identifier: MIT
//! Cache key. File hash makes the cache safe across ENDF library
//! swaps — replacing U235.h5 with the VIII.0 version produces a
//! different blake3 → different key → rebuild.

use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::transport::xs_provider::RankPolicy;

use super::binary_format::FORMAT_VERSION;

/// Every field participates in `Eq + Hash`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct NuclideKey {
    /// Canonicalised; kept human-readable for L2 filenames.
    pub path: PathBuf,
    /// blake3 of file contents.
    pub file_hash: [u8; 32],
    /// blake3 of `(default_rank, sorted_per_mt, sorted_table_mts)`.
    pub policy_hash: [u8; 32],
    /// HDF5 temperature column (0 = 294 K, 1 = 600 K, ...).
    pub temp_idx: u32,
    /// Bump to invalidate cache on layout changes.
    pub format_version: u32,
}

impl NuclideKey {
    /// `Err` only when the file can't be opened — caller skips cache
    /// and routes through the HDF5 loader directly.
    pub fn from_inputs(
        path: &Path,
        policy: &RankPolicy,
        temp_idx: usize,
    ) -> io::Result<Self> {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let file_hash = hash_file(&canonical)?;
        let policy_hash = hash_policy(policy);
        Ok(Self {
            path: canonical,
            file_hash,
            policy_hash,
            temp_idx: temp_idx as u32,
            format_version: FORMAT_VERSION,
        })
    }

    /// Hash-only filename; renaming the .h5 without touching contents
    /// still hits the same cache file.
    pub fn disk_filename(&self) -> String {
        let mut s = String::with_capacity(144 + 4);
        for b in &self.file_hash {
            s.push_str(&format!("{b:02x}"));
        }
        for b in &self.policy_hash {
            s.push_str(&format!("{b:02x}"));
        }
        s.push_str(&format!("_t{}", self.temp_idx));
        s.push_str(&format!("_v{}", self.format_version));
        s.push_str(".nuc");
        s
    }
}

/// 256 KB chunks — actinide .h5 files reach ~200 MB.
fn hash_file(path: &Path) -> io::Result<[u8; 32]> {
    let mut hasher = blake3::Hasher::new();
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0_u8; 256 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Hash the rank policy. Uses a BTreeMap for both `per_mt` and a sorted
/// `Vec<u32>` for `table_mts` so the byte stream is deterministic
/// regardless of HashMap iteration order.
fn hash_policy(policy: &RankPolicy) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(policy.default as u64).to_le_bytes());
    let sorted_per_mt: BTreeMap<u32, usize> =
        policy.per_mt.iter().map(|(m, r)| (*m, *r)).collect();
    hasher.update(&(sorted_per_mt.len() as u32).to_le_bytes());
    for (mt, rank) in &sorted_per_mt {
        hasher.update(&mt.to_le_bytes());
        hasher.update(&(*rank as u64).to_le_bytes());
    }
    let mut sorted_table: Vec<u32> = policy.table_mts.iter().copied().collect();
    sorted_table.sort_unstable();
    hasher.update(&(sorted_table.len() as u32).to_le_bytes());
    for mt in &sorted_table {
        hasher.update(&mt.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_collision_iff_inputs_match() {
        let tmp = std::env::temp_dir().join("orm_key_test_a.h5");
        std::fs::write(&tmp, b"contents_a").unwrap();
        let p1 = RankPolicy::new(5);
        let k1 = NuclideKey::from_inputs(&tmp, &p1, 0).unwrap();
        let k2 = NuclideKey::from_inputs(&tmp, &p1, 0).unwrap();
        assert_eq!(k1, k2, "same inputs must produce identical keys");

        // Different temperature → different key.
        let k3 = NuclideKey::from_inputs(&tmp, &p1, 1).unwrap();
        assert_ne!(k1, k3);

        // Different rank → different policy hash → different key.
        let p2 = RankPolicy::new(7);
        let k4 = NuclideKey::from_inputs(&tmp, &p2, 0).unwrap();
        assert_ne!(k1, k4);

        // Different file contents → different file hash → different key.
        std::fs::write(&tmp, b"contents_b").unwrap();
        let k5 = NuclideKey::from_inputs(&tmp, &p1, 0).unwrap();
        assert_ne!(k1, k5);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn policy_hash_is_deterministic_across_hashmap_insertion_order() {
        let mut a = RankPolicy::new(5);
        a = a.with_mt(2, 1).with_mt(18, 3).with_mt(102, 1);
        let mut b = RankPolicy::new(5);
        b = b.with_mt(102, 1).with_mt(2, 1).with_mt(18, 3);
        assert_eq!(hash_policy(&a), hash_policy(&b));
    }
}
