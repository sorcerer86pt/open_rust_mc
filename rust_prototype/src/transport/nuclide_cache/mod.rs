//! Process-wide nuclide kernel cache.
//!
//! HDF5 load + SVD + CDF + URR assembly costs hundreds of ms / file;
//! ICSBEP sweeps re-pay this per case without the cache (U-235
//! re-parsed 35× in one observed run).
//!
//! Tiers:
//! - **L1** — in-process byte-budgeted LRU, see [`l1_memory`].
//! - **L2** — content-addressed files under
//!   `$OPEN_RUST_MC_CACHE_DIR` (default
//!   `%LOCALAPPDATA%\open_rust_mc\cache` Windows,
//!   `~/.cache/open_rust_mc` elsewhere). See [`l2_disk`].
//! - **L3** — reserved trait slot for a remote daemon. Same trait,
//!   same wire format. Not yet implemented.
//!
//! Invalidation is correctness-keyed via `NuclideKey =
//! (file_hash, policy_hash, temp_idx, format_version)`. No
//! time-based eviction.

pub mod binary_format;
pub mod eviction;
pub mod key;
pub mod l1_memory;
pub mod l2_disk;
#[cfg(feature = "cache-remote")]
pub mod l3_remote;
#[cfg(feature = "cache-remote")]
pub mod wire_protocol;

use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::transport::xs_provider::{NuclideKernels, RankPolicy};

pub use key::NuclideKey;

/// Same `NuclideKey → Arc<NuclideKernels>` contract for L1 / L2 / L3.
/// `Send + Sync` for rayon concurrency. `put` may be lossy (e.g.
/// L2 drops the write on I/O error — correctness preserved, next
/// run rebuilds).
pub trait NuclideStore: Send + Sync {
    fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>>;
    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>);
    fn name(&self) -> &str;
}

/// Tiered orchestrator: read = L1 → L2 → L3, populating upstream tiers
/// on every hit. Write = populate all tiers.
pub struct TieredStore {
    l1: l1_memory::L1MemoryStore,
    l2: Option<l2_disk::L2DiskStore>,
    l3: Option<Box<dyn NuclideStore>>,
}

impl TieredStore {
    pub fn new() -> Self {
        let l2 = l2_disk::L2DiskStore::from_env();
        #[cfg(feature = "cache-remote")]
        let l3: Option<Box<dyn NuclideStore>> = l3_remote::L3RemoteStore::from_env()
            .map(|s| Box::new(s) as Box<dyn NuclideStore>);
        #[cfg(not(feature = "cache-remote"))]
        let l3: Option<Box<dyn NuclideStore>> = None;
        Self {
            l1: l1_memory::L1MemoryStore::new(),
            l2,
            l3,
        }
    }

    /// Read L1 → L2 → L3; populate upstream tiers on hit.
    pub fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>> {
        if let Some(v) = self.l1.try_get(key) {
            return Some(v);
        }
        if let Some(l2) = &self.l2 {
            if let Some(v) = l2.try_get(key) {
                self.l1.put(key.clone(), Arc::clone(&v));
                return Some(v);
            }
        }
        if let Some(l3) = &self.l3 {
            if let Some(v) = l3.try_get(key) {
                if let Some(l2) = &self.l2 {
                    l2.put(key.clone(), Arc::clone(&v));
                }
                self.l1.put(key.clone(), Arc::clone(&v));
                return Some(v);
            }
        }
        None
    }

    pub fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        if let Some(l2) = &self.l2 {
            l2.put(key.clone(), Arc::clone(&value));
        }
        if let Some(l3) = &self.l3 {
            l3.put(key.clone(), Arc::clone(&value));
        }
        self.l1.put(key, value);
    }
}

impl Default for TieredStore {
    fn default() -> Self {
        Self::new()
    }
}

fn cache() -> &'static TieredStore {
    static CACHE: OnceLock<TieredStore> = OnceLock::new();
    CACHE.get_or_init(TieredStore::default)
}

/// Seed `preload_weight` on the process-wide L1 cache. Each entry
/// maps `NuclideKey → expected hit count`. Hits + preload are
/// summed in the LFU score (see `eviction`), so a sweep harness
/// can pre-scan its case manifest, count `(file_hash, policy_hash,
/// temp_idx)` appearances across every case, and hand the result
/// here before the first transport call. Pre-marked nuclides
/// (U-235 / O-16 / Fe-56 / U-238 in any HEU sweep) then survive
/// the inevitable rare-nuclide eviction pressure.
///
/// Idempotent — calling again with new weights replaces the old.
/// Keys not yet inserted are stashed for first-insert pickup.
pub fn set_preload_weights(weights: &std::collections::HashMap<NuclideKey, u64>) {
    cache().l1.set_preload_weights(weights);
}

/// Call instead of `xs_provider::load_nuclide_with_policy` to
/// participate in the cache. `path` is canonicalised (Windows `\`
/// vs `/` resolve to the same key). `load_fn` runs only on full miss.
pub fn get_or_load(
    path: &Path,
    policy: &RankPolicy,
    temp_idx: usize,
    awr_fallback: f64,
    nu_bar_fallback: f64,
    load_fn: impl FnOnce() -> NuclideKernels,
) -> Arc<NuclideKernels> {
    let key = match NuclideKey::from_inputs(path, policy, temp_idx) {
        Ok(k) => k,
        Err(_) => {
            // Can't hash the file — bypass cache; loader emits its diagnostic.
            return Arc::new(load_fn());
        }
    };

    if let Some(hit) = cache().try_get(&key) {
        return hit;
    }
    // Fallbacks are baked into load_fn; ignore here.
    let _ = (awr_fallback, nu_bar_fallback);

    let kernel = Arc::new(load_fn());
    cache().put(key, Arc::clone(&kernel));
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiered_store_l1_roundtrip() {
        let store = TieredStore::new();
        let path = std::env::temp_dir().join("open_rust_mc_cache_test_stub.bin");
        std::fs::write(&path, b"hello").unwrap();
        let policy = RankPolicy::new(5);
        let key = NuclideKey::from_inputs(&path, &policy, 0).unwrap();
        let kernel = Arc::new(NuclideKernels::empty(238.0, 2.43));
        store.put(key.clone(), Arc::clone(&kernel));
        let hit = store.try_get(&key).expect("L1 store should return cached kernel");
        assert!(Arc::ptr_eq(&hit, &kernel));
        let _ = std::fs::remove_file(&path);
    }
}
