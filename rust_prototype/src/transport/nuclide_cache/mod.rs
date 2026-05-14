//! Process-wide nuclide kernel cache.
//!
//! Loading a `NuclideKernels` from HDF5 is dominated by SVD decomposition,
//! per-reaction CDF assembly, and URR table normalisation — each h5 file
//! costs hundreds of ms even when the OS page cache is warm. ICSBEP
//! sweeps re-pay this cost per case (90 cases × ~25 nuclides = 2 250 h5
//! reads), and a single sweep showed U-235 being re-parsed 35 times in
//! one run (`outputs/icsbep_full_gpu.log`).
//!
//! This module wraps `xs_provider::load_nuclide_with_policy` behind a
//! tiered store:
//!
//! - **L1** — `DashMap<NuclideKey, Vec<Arc<NuclideKernels>>>`, process-wide.
//!   Kills the in-sweep re-parse. The `Vec<Arc<…>>` slot lets one h5 file
//!   yield kernels at multiple rank policies / temperatures without
//!   evicting each other.
//! - **L2** — content-addressed files under `$OPEN_RUST_MC_CACHE_DIR`
//!   (default `%LOCALAPPDATA%\open_rust_mc\cache` on Windows,
//!   `~/.cache/open_rust_mc` elsewhere). On miss, the parsed kernel is
//!   serialised via `binary_format` and written atomically. See
//!   [`l2_disk`].
//! - **L3** — reserved trait slot for a remote daemon (Unix socket /
//!   named pipe / TCP) that holds the L1 store in a separate process so
//!   multiple PCs on a LAN share one parsed library. Same `NuclideStore`
//!   trait + same binary format on the wire. Not yet implemented; the
//!   `Tiered` orchestrator already accepts an optional L3 backend.
//!
//! ## Invalidation
//!
//! Keys carry the blake3 hash of the h5 file contents, the rank policy
//! hash, the temperature index, and a `format_version` constant. Any of:
//!
//! - swapping ENDF/B-VII.1 for VIII.0 in the same path,
//! - changing the rank policy (per-MT overrides),
//! - bumping `binary_format::FORMAT_VERSION`,
//!
//! produces a different key and the cache transparently rebuilds. There
//! is no time-based eviction; the cache is correctness-keyed, not
//! freshness-keyed.

pub mod binary_format;
pub mod key;
pub mod l1_memory;
pub mod l2_disk;

use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::transport::xs_provider::{NuclideKernels, RankPolicy};

pub use key::NuclideKey;

/// Storage backend behind the tiered cache. Same trait used by L1
/// in-process, L2 on-disk, and (future) L3 remote daemon — all three
/// speak the same `NuclideKey` → `Arc<NuclideKernels>` contract.
///
/// Implementations must be `Send + Sync`; the cache is accessed
/// concurrently from rayon worker threads in `load_or_get_meta` and the
/// per-batch resolution paths.
pub trait NuclideStore: Send + Sync {
    /// Lookup. Returns `None` on miss. Implementations must not block
    /// indefinitely — L2 disk reads are bounded by file I/O, L3 remote
    /// reads should carry a short timeout.
    fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>>;

    /// Insert. Implementations may be lossy (e.g. L2 disk silently drops
    /// the write on I/O error — the next process restart pays the
    /// rebuild cost again, but correctness is preserved).
    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>);

    /// Human-readable name for logging (`"L1 in-memory"`, `"L2 disk
    /// /path"`, `"L3 daemon tcp://host:port"`).
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
        Self {
            l1: l1_memory::L1MemoryStore::new(),
            l2,
            l3: None,
        }
    }

    /// Get a cached kernel, populating upstream tiers on hit. Returns
    /// `None` on a full-miss; callers fall through to the HDF5 loader
    /// and then `put` the result.
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

/// Process-wide cache. First access spins up the L1 DashMap + opens (or
/// creates) the L2 disk directory.
fn cache() -> &'static TieredStore {
    static CACHE: OnceLock<TieredStore> = OnceLock::new();
    CACHE.get_or_init(TieredStore::default)
}

/// High-level entry point — call instead of
/// `xs_provider::load_nuclide_with_policy` to participate in the cache.
///
/// On miss, runs the closure (which is expected to call
/// `xs_provider::load_nuclide_with_policy`) and write-through to every
/// tier. The closure is only invoked on a full cache miss.
///
/// `path` is canonicalised before hashing — relative paths and
/// path-separator differences (Windows `\` vs forward `/`) resolve to
/// the same key.
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
            // Couldn't open the file to hash it — fall back to direct
            // load and skip the cache entirely. The HDF5 loader will
            // emit its own diagnostic.
            return Arc::new(load_fn());
        }
    };

    if let Some(hit) = cache().try_get(&key) {
        return hit;
    }

    // Honour the policy fallback fields by routing through load_fn —
    // keeps `awr_fallback` / `nu_bar_fallback` in one place (the
    // caller's closure already supplies them to `load_nuclide_with_policy`).
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
        // Empty stub kernel — exercises the L1 hashmap insert/get path
        // without needing real HDF5 data.
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
