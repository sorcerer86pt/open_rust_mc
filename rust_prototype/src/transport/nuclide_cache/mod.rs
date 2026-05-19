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
use std::sync::OnceLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::transport::xs_provider::{NuclideKernels, RankPolicy};

pub use key::NuclideKey;

// ── Telemetry ────────────────────────────────────────────────────────
//
// Atomic counters cover the read and write paths of the tiered store
// without a single contended lock. Each tier increments its own slot
// on hit / miss; `put()` increments the put counter. Read with
// `stats()`; Python surfaces them via `cache_stats()`.
//
// The counters are process-wide (same OnceLock pattern as the cache
// itself), so MPI deployments running multiple Python interpreters
// per node each maintain their own counter set — aggregate across
// nodes by sampling each process and summing on the controller.
#[derive(Default, Debug)]
pub struct CacheStats {
    pub l1_hits: AtomicU64,
    pub l1_misses: AtomicU64,
    pub l2_hits: AtomicU64,
    pub l2_misses: AtomicU64,
    pub l3_hits: AtomicU64,
    pub l3_misses: AtomicU64,
    pub puts: AtomicU64,
}

impl CacheStats {
    /// Snapshot to plain `u64`s so callers can serialise without
    /// touching atomics. Uses Relaxed because these are diagnostic
    /// counters, not a synchronisation primitive — readers tolerate
    /// momentary skew between fields.
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            l1_hits: self.l1_hits.load(Ordering::Relaxed),
            l1_misses: self.l1_misses.load(Ordering::Relaxed),
            l2_hits: self.l2_hits.load(Ordering::Relaxed),
            l2_misses: self.l2_misses.load(Ordering::Relaxed),
            l3_hits: self.l3_hits.load(Ordering::Relaxed),
            l3_misses: self.l3_misses.load(Ordering::Relaxed),
            puts: self.puts.load(Ordering::Relaxed),
        }
    }

    /// Reset every counter to zero. Useful between sweep phases when
    /// the caller wants per-phase telemetry rather than process-life
    /// totals.
    pub fn reset(&self) {
        self.l1_hits.store(0, Ordering::Relaxed);
        self.l1_misses.store(0, Ordering::Relaxed);
        self.l2_hits.store(0, Ordering::Relaxed);
        self.l2_misses.store(0, Ordering::Relaxed);
        self.l3_hits.store(0, Ordering::Relaxed);
        self.l3_misses.store(0, Ordering::Relaxed);
        self.puts.store(0, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStatsSnapshot {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub puts: u64,
}

impl CacheStatsSnapshot {
    /// L1 / L2 / L3 hit rate as a fraction in `[0, 1]`. Returns
    /// `None` when the tier has seen no traffic (no hits + no misses).
    pub fn l1_hit_rate(&self) -> Option<f64> { hit_rate(self.l1_hits, self.l1_misses) }
    pub fn l2_hit_rate(&self) -> Option<f64> { hit_rate(self.l2_hits, self.l2_misses) }
    pub fn l3_hit_rate(&self) -> Option<f64> { hit_rate(self.l3_hits, self.l3_misses) }
}

fn hit_rate(hits: u64, misses: u64) -> Option<f64> {
    let total = hits + misses;
    if total == 0 { None } else { Some(hits as f64 / total as f64) }
}

/// Process-wide stats handle. Lives for the life of the process; same
/// `OnceLock` pattern as the cache itself.
pub fn stats() -> &'static CacheStats {
    static S: OnceLock<CacheStats> = OnceLock::new();
    S.get_or_init(CacheStats::default)
}

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

    /// Read L1 → L2 → L3; populate upstream tiers on hit. Telemetry:
    /// every tier increments either its hit or miss counter, so a
    /// `stats()` snapshot lets the harness reason about which tier
    /// is doing the work without per-call printf logging.
    pub fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>> {
        let s = stats();
        if let Some(v) = self.l1.try_get(key) {
            s.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(v);
        }
        s.l1_misses.fetch_add(1, Ordering::Relaxed);
        if let Some(l2) = &self.l2 {
            if let Some(v) = l2.try_get(key) {
                s.l2_hits.fetch_add(1, Ordering::Relaxed);
                self.l1.put(key.clone(), Arc::clone(&v));
                return Some(v);
            }
            s.l2_misses.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(l3) = &self.l3 {
            if let Some(v) = l3.try_get(key) {
                s.l3_hits.fetch_add(1, Ordering::Relaxed);
                if let Some(l2) = &self.l2 {
                    l2.put(key.clone(), Arc::clone(&v));
                }
                self.l1.put(key.clone(), Arc::clone(&v));
                return Some(v);
            }
            s.l3_misses.fetch_add(1, Ordering::Relaxed);
        }
        None
    }

    pub fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        stats().puts.fetch_add(1, Ordering::Relaxed);
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

    /// Counters are process-wide. We can't safely assert absolute
    /// values (other tests in this module touch the cache), so we
    /// pre-snapshot, take known actions, and assert deltas.
    #[test]
    fn stats_counters_increment_on_hit_miss_put() {
        // Unique-per-run path & contents so the L2 disk cache, which
        // persists across test runs, doesn't have a stale entry for
        // this key. The blake3 file hash participates in NuclideKey,
        // so writing fresh bytes produces a fresh key every run.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let store = TieredStore::new();
        let path = std::env::temp_dir().join(format!(
            "open_rust_mc_cache_stats_stub_{nonce}_{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, format!("stats-{nonce}").as_bytes()).unwrap();
        let policy = RankPolicy::new(5);
        let key = NuclideKey::from_inputs(&path, &policy, 0).unwrap();

        let before = stats().snapshot();

        // Miss path — nothing in the store yet for this key.
        assert!(store.try_get(&key).is_none());
        let after_miss = stats().snapshot();
        assert!(
            after_miss.l1_misses > before.l1_misses,
            "L1 miss must increment l1_misses (before={} after={})",
            before.l1_misses, after_miss.l1_misses,
        );

        // Put path.
        let kernel = Arc::new(NuclideKernels::empty(238.0, 2.43));
        store.put(key.clone(), Arc::clone(&kernel));
        let after_put = stats().snapshot();
        assert!(
            after_put.puts > after_miss.puts,
            "put must increment puts counter"
        );

        // Hit path.
        let hit = store.try_get(&key).expect("must hit after put");
        assert!(Arc::ptr_eq(&hit, &kernel));
        let after_hit = stats().snapshot();
        assert!(
            after_hit.l1_hits > after_put.l1_hits,
            "L1 hit must increment l1_hits"
        );

        // Hit-rate helper should return Some(_) once any traffic
        // landed.
        assert!(after_hit.l1_hit_rate().is_some());

        let _ = std::fs::remove_file(&path);
    }
}
