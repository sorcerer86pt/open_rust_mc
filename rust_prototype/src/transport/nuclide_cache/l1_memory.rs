//! L1 in-process byte-budgeted cache with LFU-with-recency
//! eviction (see `super::eviction`). Host-side mirror of
//! `gpu_transport::nuclide_buffer_cache`.
//!
//! Budget: `OPEN_RUST_MC_NUCLIDE_CACHE_BYTES` → explicit byte
//! override → `OPEN_RUST_MC_NUCLIDE_CACHE_FRACTION` × detected
//! system RAM (via `hardware_profile`) → 75 % of RAM default.
//! Falls back to 4 GiB if RAM detection fails.

use std::collections::HashMap;
use std::sync::{Mutex, Arc};

use super::eviction::{
    self, EvictionStats, LfuEntries, LfuEntriesMut, DEFAULT_AGE_DECAY,
};
use super::{NuclideKey, NuclideStore};
use crate::transport::xs_provider::NuclideKernels;

const HARDWARE_QUERY_FALLBACK_BYTES: usize = 4 * crate::hardware_profile::GIB;
const DEFAULT_FRACTION: f64 = 0.75;
const FRACTION_MIN: f64 = 0.05;
const FRACTION_MAX: f64 = 0.95;

struct Inner {
    /// Insertion order preserved for stable iteration; the LFU
    /// policy picks the victim by score.
    entries: HashMap<NuclideKey, (Arc<NuclideKernels>, EvictionStats)>,
    /// Monotonic insert / hit counter. Drives the `age` term in the
    /// LFU score.
    counter: u64,
    total_bytes: usize,
}

pub struct L1MemoryStore {
    inner: Mutex<Inner>,
    budget_bytes: usize,
    name: String,
}

impl L1MemoryStore {
    pub fn new() -> Self {
        Self::with_budget(resolve_budget())
    }

    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                counter: 0,
                total_bytes: 0,
            }),
            budget_bytes: budget_bytes.max(1),
            name: "L1 in-memory".to_string(),
        }
    }

    pub fn n_keys(&self) -> usize {
        self.inner
            .lock()
            .expect("L1MemoryStore mutex poisoned")
            .entries
            .len()
    }

    pub fn approx_bytes_held(&self) -> usize {
        self.inner
            .lock()
            .expect("L1MemoryStore mutex poisoned")
            .total_bytes
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Seed `preload_weight` from a sweep manifest pre-scan. The
    /// ICSBEP harness can count nuclide appearances across every
    /// case JSON, then call this before the first transport run so
    /// U-235 / O-16 / Fe-56 / U-238 stay resident even before any
    /// hits land. Weights are additive with observed hits in
    /// `eviction::score`.
    ///
    /// Entries not yet inserted at call time keep their preload
    /// weight on the side and pick it up at insert (see
    /// `put_with_preload`-aware path).
    pub fn set_preload_weights(&self, weights: &HashMap<NuclideKey, u64>) {
        let mut guard = self.inner.lock().expect("L1MemoryStore mutex poisoned");
        for (k, w) in weights {
            if let Some((_, stats)) = guard.entries.get_mut(k) {
                stats.preload_weight = *w;
            }
        }
        // Stash future-key weights so insert(k) picks them up.
        let mut pending = PENDING_PRELOAD.lock().expect("PENDING_PRELOAD poisoned");
        for (k, w) in weights {
            pending.insert(k.clone(), *w);
        }
    }

    fn evict_with_policy(inner: &mut Inner, new_bytes: usize, budget: usize) {
        let now = inner.counter;
        let mut adapter = InnerAdapter { inner };
        let _ = eviction::evict_to_budget(
            &mut adapter,
            new_bytes,
            budget,
            now,
            DEFAULT_AGE_DECAY,
        );
    }
}

/// Process-wide stash for preload weights set BEFORE an entry is
/// inserted. Picked up at insert time. Pending entries never expire
/// — once an entry inserts they transfer, but if a key is never
/// inserted the stash entry stays harmlessly resident.
static PENDING_PRELOAD: std::sync::Mutex<
    std::sync::LazyLock<HashMap<NuclideKey, u64>>,
> = std::sync::Mutex::new(std::sync::LazyLock::new(HashMap::new));

/// Adapter exposing `Inner` to the policy without leaking its
/// HashMap layout into the policy module.
struct InnerAdapter<'a> {
    inner: &'a mut Inner,
}

impl LfuEntries for InnerAdapter<'_> {
    type Key = NuclideKey;

    fn total_bytes(&self) -> usize {
        self.inner.total_bytes
    }

    fn len(&self) -> usize {
        self.inner.entries.len()
    }

    fn iter_stats(&self) -> Box<dyn Iterator<Item = (&Self::Key, &EvictionStats)> + '_> {
        Box::new(self.inner.entries.iter().map(|(k, (_, s))| (k, s)))
    }

    fn remove(&mut self, key: &Self::Key) {
        if let Some((_, stats)) = self.inner.entries.remove(key) {
            self.inner.total_bytes = self.inner.total_bytes.saturating_sub(stats.bytes);
        }
    }
}

impl LfuEntriesMut for InnerAdapter<'_> {
    fn set_preload_weight(&mut self, key: &Self::Key, weight: u64) {
        if let Some((_, stats)) = self.inner.entries.get_mut(key) {
            stats.preload_weight = weight;
        }
    }
}

impl Default for L1MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl NuclideStore for L1MemoryStore {
    fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>> {
        let mut guard = self.inner.lock().expect("L1MemoryStore mutex poisoned");
        guard.counter = guard.counter.wrapping_add(1);
        let now = guard.counter;
        let (arc, _) = guard.entries.get_mut(key)?;
        let arc = Arc::clone(arc);
        if let Some((_, stats)) = guard.entries.get_mut(key) {
            stats.hits = stats.hits.wrapping_add(1);
            stats.last_touch = now;
        }
        Some(arc)
    }

    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        let bytes = value.approx_host_bytes();
        let mut guard = self.inner.lock().expect("L1MemoryStore mutex poisoned");

        // Concurrent-insert race: replace in place.
        if let Some((_, old_stats)) = guard.entries.remove(&key) {
            guard.total_bytes = guard.total_bytes.saturating_sub(old_stats.bytes);
        }

        Self::evict_with_policy(&mut guard, bytes, self.budget_bytes);

        let counter = guard.counter;
        guard.counter = guard.counter.wrapping_add(1);
        let mut stats = EvictionStats::new(bytes, counter);
        // Pick up any preload weight set before this key first inserted.
        if let Ok(pending) = PENDING_PRELOAD.lock() {
            if let Some(w) = pending.get(&key) {
                stats.preload_weight = *w;
            }
        }
        guard.total_bytes = guard.total_bytes.saturating_add(bytes);
        guard.entries.insert(key, (value, stats));
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn resolve_budget() -> usize {
    if let Some(v) = std::env::var_os("OPEN_RUST_MC_NUCLIDE_CACHE_BYTES") {
        if let Ok(n) = v.to_string_lossy().parse::<usize>() {
            return n.max(1);
        }
    }
    let total_ram = crate::hardware_profile::hardware_profile().total_ram_bytes as usize;
    if total_ram == 0 {
        return HARDWARE_QUERY_FALLBACK_BYTES;
    }
    let fraction = std::env::var("OPEN_RUST_MC_NUCLIDE_CACHE_FRACTION")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_FRACTION)
        .clamp(FRACTION_MIN, FRACTION_MAX);
    ((total_ram as f64) * fraction) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::nuclide_cache::binary_format::FORMAT_VERSION;
    use std::path::PathBuf;

    fn mk_key(seed: u8) -> NuclideKey {
        NuclideKey {
            path: PathBuf::from(format!("/tmp/test{seed}.h5")),
            file_hash: [seed; 32],
            policy_hash: [seed; 32],
            temp_idx: seed as u32,
            format_version: FORMAT_VERSION,
        }
    }

    #[test]
    fn evicts_when_budget_exceeded() {
        let approx_each = NuclideKernels::empty(1.0, 2.43).approx_host_bytes();
        let budget = approx_each * 2 + approx_each / 2;
        let store = L1MemoryStore::with_budget(budget);

        let k1 = mk_key(1);
        let k2 = mk_key(2);
        let k3 = mk_key(3);

        store.put(k1.clone(), Arc::new(NuclideKernels::empty(1.0, 2.43)));
        store.put(k2.clone(), Arc::new(NuclideKernels::empty(16.0, 2.43)));
        assert!(store.try_get(&k1).is_some());
        assert!(store.try_get(&k2).is_some());
        assert_eq!(store.n_keys(), 2);

        store.put(k3.clone(), Arc::new(NuclideKernels::empty(238.0, 2.43)));
        assert_eq!(store.n_keys(), 2, "cap should hold at 2");
        assert!(store.try_get(&k3).is_some(), "newest entry must stay");
    }

    #[test]
    fn single_oversized_entry_still_caches() {
        let store = L1MemoryStore::with_budget(1);
        let k = mk_key(7);
        store.put(k.clone(), Arc::new(NuclideKernels::empty(1.0, 2.43)));
        assert_eq!(store.n_keys(), 1);
        assert!(store.try_get(&k).is_some());
    }

    /// Hot entry (many hits) survives a cold neighbour even when
    /// the neighbour is more recent — the recency-weighted LFU score
    /// keeps it.
    #[test]
    fn hot_entry_survives_cold_pressure() {
        let each = NuclideKernels::empty(1.0, 2.43).approx_host_bytes();
        let store = L1MemoryStore::with_budget(each * 2 + each / 2);

        let k_hot = mk_key(1);
        let k_cold = mk_key(2);
        let k_pressure = mk_key(3);
        store.put(k_hot.clone(), Arc::new(NuclideKernels::empty(1.0, 2.43)));
        store.put(k_cold.clone(), Arc::new(NuclideKernels::empty(16.0, 2.43)));

        // 50 hits on k_hot.
        for _ in 0..50 {
            store.try_get(&k_hot);
        }
        // Insert pressure → evict the cold neighbour, not the hot.
        store.put(
            k_pressure.clone(),
            Arc::new(NuclideKernels::empty(238.0, 2.43)),
        );
        assert!(store.try_get(&k_hot).is_some(), "hot must survive");
        assert!(store.try_get(&k_cold).is_none(), "cold evicts");
        assert!(store.try_get(&k_pressure).is_some(), "newest present");
    }

    /// `set_preload_weights` populated BEFORE the first insert
    /// still drives eviction order. Mirrors sweep warm-start.
    #[test]
    fn preload_weight_seeds_warm_start() {
        let each = NuclideKernels::empty(1.0, 2.43).approx_host_bytes();
        let store = L1MemoryStore::with_budget(each * 2 + each / 2);
        let k_hot = mk_key(10);
        let k_cold = mk_key(11);
        let k_press = mk_key(12);

        let mut weights = HashMap::new();
        weights.insert(k_hot.clone(), 1000);
        store.set_preload_weights(&weights);

        store.put(k_hot.clone(), Arc::new(NuclideKernels::empty(1.0, 2.43)));
        store.put(k_cold.clone(), Arc::new(NuclideKernels::empty(16.0, 2.43)));
        store.put(k_press.clone(), Arc::new(NuclideKernels::empty(238.0, 2.43)));

        assert!(store.try_get(&k_hot).is_some(), "preloaded survives");
        assert!(store.try_get(&k_cold).is_none(), "cold evicts");
    }
}
