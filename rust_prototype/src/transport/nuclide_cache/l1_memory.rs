//! L1 in-process store — bounded byte-budget LRU over
//! `(NuclideKey, Arc<NuclideKernels>, usize_bytes)`.
//!
//! Was a `DashMap<NuclideKey, Vec<Arc<NuclideKernels>>>` (unbounded);
//! a 376-case sweep with distinct nuclide sets accumulates one entry
//! per unique nuclide-temp-policy tuple, monotonically growing host
//! RAM until the OS pages out hot data. The byte-budgeted LRU is the
//! host-side mirror of `gpu_transport::nuclide_buffer_cache` (commit
//! 097c282); same eviction shape, same env knobs, different memory
//! pool.
//!
//! ## Budget
//!
//! - `OPEN_RUST_MC_NUCLIDE_CACHE_BYTES=N` — explicit byte override
//!   (wins).
//! - `OPEN_RUST_MC_NUCLIDE_CACHE_FRACTION=F` × total system RAM
//!   (clamped to [0.05, 0.95]).
//! - Default: 4 GiB. Picks a budget that fits a 16 GB laptop with
//!   Python + browser + IDE running, and scales up via the env knob
//!   on bigger boxes. The lazy 25-line `sysinfo`-equivalent platform
//!   query was deferred to keep the diff small; harness scripts on
//!   servers should set the env var explicitly.
//!
//! ## Eviction
//!
//! Same shape as `gpu_transport::upload_nuclide_data`:
//! - Hit: promote to MRU (back).
//! - Miss: pre-evict from front while `total_bytes + last_inserted_bytes > budget`.
//! - Post-insert: re-trim if the new entry was larger than the
//!   predictor. Always leaves at least one entry — a nuclide bigger
//!   than the budget still caches itself rather than being uploaded
//!   fresh on every call.
//!
//! ## Vec<Arc> slot
//!
//! The previous DashMap stored `Vec<Arc<NuclideKernels>>` per key,
//! "for future bulk-dump APIs". Nothing ever populated the second
//! entry. This refactor drops the Vec and stores a single Arc per
//! key — the semantic the hot path already used.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use super::{NuclideKey, NuclideStore};
use crate::transport::xs_provider::NuclideKernels;

/// Default host-side cache budget when neither env var is set.
/// 4 GiB fits ~80 actinide-heavy nuclides at ~50 MB each, which covers
/// the entire ICSBEP corpus' unique nuclide set with headroom. Smaller
/// machines override via `OPEN_RUST_MC_NUCLIDE_CACHE_BYTES`; large ones
/// can crank up via `OPEN_RUST_MC_NUCLIDE_CACHE_FRACTION`.
const DEFAULT_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;
const FRACTION_MIN: f64 = 0.05;
const FRACTION_MAX: f64 = 0.95;

pub struct L1MemoryStore {
    /// `(key, value, approx_host_bytes)`. Insertion order = LRU
    /// order: front is oldest, back is most-recently-used. Hits move
    /// to the back.
    inner: Mutex<VecDeque<(NuclideKey, Arc<NuclideKernels>, usize)>>,
    /// Byte budget resolved at construction. Memoised — env vars read
    /// once. See module docs for the resolution order.
    budget_bytes: usize,
    name: String,
}

impl L1MemoryStore {
    pub fn new() -> Self {
        Self::with_budget(resolve_budget())
    }

    /// Construct with an explicit byte budget. Intended for tests that
    /// want deterministic eviction without relying on env vars.
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            budget_bytes: budget_bytes.max(1),
            name: "L1 in-memory".to_string(),
        }
    }

    /// Total number of distinct keys held. Useful for diagnostics
    /// ("cache holds 312 kernels").
    pub fn n_keys(&self) -> usize {
        self.inner
            .lock()
            .expect("L1MemoryStore mutex poisoned")
            .len()
    }

    /// Current cache footprint in bytes (sum of every entry's
    /// `approx_host_bytes`). Diagnostic-only — the budget itself
    /// gates inserts.
    pub fn approx_bytes_held(&self) -> usize {
        self.inner
            .lock()
            .expect("L1MemoryStore mutex poisoned")
            .iter()
            .map(|(_, _, b)| *b)
            .sum()
    }

    /// Budget this store was constructed with.
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
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
        let pos = guard.iter().position(|(k, _, _)| k == key)?;
        let entry = guard.remove(pos)?;
        let arc = Arc::clone(&entry.1);
        // Promote to MRU.
        guard.push_back(entry);
        Some(arc)
    }

    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        let bytes = value.approx_host_bytes();
        let mut guard = self.inner.lock().expect("L1MemoryStore mutex poisoned");

        // If the same key is already present (e.g. parallel insert
        // race), replace its entry in place. Don't double-charge the
        // budget.
        if let Some(pos) = guard.iter().position(|(k, _, _)| k == &key) {
            let _ = guard.remove(pos);
        }

        // Evict from the front until adding the new entry fits the
        // budget. Always leaves the new entry as the sole survivor if
        // it exceeds the budget on its own — re-uploading every call
        // would be strictly worse than slightly overshooting.
        let total: usize = guard.iter().map(|(_, _, b)| *b).sum();
        let mut running = total;
        while running.saturating_add(bytes) > self.budget_bytes && !guard.is_empty() {
            if let Some((_, _, popped_bytes)) = guard.pop_front() {
                running = running.saturating_sub(popped_bytes);
            }
        }
        guard.push_back((key, value, bytes));
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Resolve the byte budget from env vars + a sensible default. Reads
/// env at construction time; setting an env var after the process
/// starts has no effect on this store.
fn resolve_budget() -> usize {
    if let Some(v) = std::env::var_os("OPEN_RUST_MC_NUCLIDE_CACHE_BYTES") {
        if let Ok(n) = v.to_string_lossy().parse::<usize>() {
            return n.max(1);
        }
    }
    if let Some(v) = std::env::var_os("OPEN_RUST_MC_NUCLIDE_CACHE_FRACTION") {
        if let Ok(f) = v.to_string_lossy().parse::<f64>() {
            // Total system RAM detection deferred (would add a sysinfo
            // dep). Apply the fraction to a 16 GiB stand-in so the
            // common case (laptop / mid-range workstation) is sensible
            // even without the auto-detect — the user can override
            // explicitly via _BYTES for non-typical machines.
            const STAND_IN_RAM_BYTES: usize = 16 * 1024 * 1024 * 1024;
            let f = f.clamp(FRACTION_MIN, FRACTION_MAX);
            return ((STAND_IN_RAM_BYTES as f64) * f) as usize;
        }
    }
    DEFAULT_BUDGET_BYTES
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

    /// Byte-budgeted LRU evicts the oldest entry when a new one
    /// pushes the running total past the budget.
    #[test]
    fn evicts_when_budget_exceeded() {
        // Each NuclideKernels::empty() has approx_host_bytes() of ~ size_of
        // (no Vec contents). Pad the apparent size by storing them under
        // a budget tight enough that 2 fit but a 3rd evicts.
        // size_of::<NuclideKernels>() is the dominant baseline term for
        // an empty kernel, so we just compute the budget from that.
        let approx_each = NuclideKernels::empty(1.0, 2.43).approx_host_bytes();
        let budget = approx_each * 2 + approx_each / 2; // fits 2, blocks 3
        let store = L1MemoryStore::with_budget(budget);

        let k1 = mk_key(1);
        let k2 = mk_key(2);
        let k3 = mk_key(3);

        store.put(k1.clone(), Arc::new(NuclideKernels::empty(1.0, 2.43)));
        store.put(k2.clone(), Arc::new(NuclideKernels::empty(16.0, 2.43)));
        assert!(store.try_get(&k1).is_some());
        assert!(store.try_get(&k2).is_some());
        assert_eq!(store.n_keys(), 2);

        // k3 forces eviction of the oldest live entry — which, after
        // the two try_get calls above, is k1 (k2 was promoted by its
        // try_get, but k1 was promoted earlier — so the order is k2,
        // k1; eviction pops k2). Either way, len stays at 2.
        store.put(k3.clone(), Arc::new(NuclideKernels::empty(238.0, 2.43)));
        assert_eq!(store.n_keys(), 2, "cap should hold at 2");
        assert!(store.try_get(&k3).is_some(), "newest entry must stay");
    }

    /// A single entry bigger than the entire budget still caches itself.
    /// Better than re-uploading it every call.
    #[test]
    fn single_oversized_entry_still_caches() {
        let store = L1MemoryStore::with_budget(1); // 1-byte budget
        let k = mk_key(7);
        store.put(k.clone(), Arc::new(NuclideKernels::empty(1.0, 2.43)));
        assert_eq!(store.n_keys(), 1);
        assert!(store.try_get(&k).is_some());
    }

    /// LRU semantics: try_get promotes the entry so it's not the next
    /// victim.
    #[test]
    fn try_get_promotes_to_mru() {
        let each = NuclideKernels::empty(1.0, 2.43).approx_host_bytes();
        let store = L1MemoryStore::with_budget(each * 2 + each / 2);

        let k1 = mk_key(1);
        let k2 = mk_key(2);
        let k3 = mk_key(3);
        store.put(k1.clone(), Arc::new(NuclideKernels::empty(1.0, 2.43)));
        store.put(k2.clone(), Arc::new(NuclideKernels::empty(16.0, 2.43)));

        // Promote k1 so k2 becomes oldest.
        assert!(store.try_get(&k1).is_some());
        // Now insert k3 — should evict k2.
        store.put(k3.clone(), Arc::new(NuclideKernels::empty(238.0, 2.43)));
        assert!(store.try_get(&k1).is_some(), "k1 (promoted) survives");
        assert!(store.try_get(&k2).is_none(), "k2 (oldest) evicted");
        assert!(store.try_get(&k3).is_some(), "k3 (newest) present");
    }
}
