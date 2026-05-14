//! Cache key — canonical path + blake3 file hash + rank-policy hash +
//! temp index + binary-format version.
//!
//! The file hash is what makes the cache **safe across ENDF library
//! swaps**: replacing `data/endfb-vii.1-hdf5/U235.h5` with the VIII.0
//! version (same path) produces a different blake3 → different key, and
//! the cache transparently rebuilds. There is no time-based eviction.

use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::transport::xs_provider::RankPolicy;

use super::binary_format::FORMAT_VERSION;

/// Process-wide identifier for one cached `Arc<NuclideKernels>`.
///
/// `Eq` + `Hash` come from the field tuple — every key field
/// participates so two keys collide iff every parameter agrees. The 32
/// blake3 bytes dominate the hashing cost but DashMap's xxh3 over them
/// is still O(few hundred ns).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct NuclideKey {
    /// Canonicalised path. Kept in the key (in addition to the hash) so
    /// debug dumps and L2 disk filenames are human-readable.
    pub path: PathBuf,
    /// blake3(file contents). 32 bytes. Differentiates ENDF library
    /// versions and any other on-disk mutation.
    pub file_hash: [u8; 32],
    /// blake3 of the policy fields (default rank + sorted per-MT
    /// overrides + sorted table_mts). One stable hash collapses the
    /// (default, per-MT, table_mts) triple to 32 bytes.
    pub policy_hash: [u8; 32],
    /// HDF5 temperature column index (0 = 294 K, 1 = 600 K, ...). Tied
    /// to the evaluation's per-file temperature grid.
    pub temp_idx: u32,
    /// Binary-format version stamp. Bumped whenever the encode/decode
    /// layout changes — old entries fail key match and are rebuilt.
    pub format_version: u32,
}

impl NuclideKey {
    /// Build a key by canonicalising `path`, hashing its contents, and
    /// hashing the policy. Returns `Err` only when the file cannot be
    /// opened for reading — in that case the caller should skip the
    /// cache and route through the HDF5 loader directly so its existing
    /// error path can run.
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

    /// Hex-encoded compact identifier — used as the L2 disk filename.
    /// Includes only the hash fields so renaming the source `.h5`
    /// without touching its contents still hits the same cache file.
    pub fn disk_filename(&self) -> String {
        // 32B file_hash + 32B policy_hash + 4B temp_idx + 4B format_version
        // → 144 hex chars + ".nuc". Plenty unique across a single
        // user's cache dir; collisions would require a blake3 break.
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

/// Stream-hash the file in 256 KB chunks — large .h5 files (~200 MB
/// for actinides with full temperature ladders) would otherwise pull
/// the whole file into memory just to hash it.
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
