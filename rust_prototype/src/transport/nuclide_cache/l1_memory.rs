//! L1 in-process bounded byte-budget LRU. Host-side mirror of
//! `gpu_transport::nuclide_buffer_cache` (commit 097c282) — same
//! eviction shape, different memory pool.
//!
//! Budget knobs: `OPEN_RUST_MC_NUCLIDE_CACHE_BYTES` (explicit) or
//! `OPEN_RUST_MC_NUCLIDE_CACHE_FRACTION` × 16 GiB stand-in.
//! Default 4 GiB (no `sysinfo` dep — set the env var on bigger boxes).
//!
//! Eviction: hit promotes to MRU; insert pre-evicts while
//! `total_bytes + new_bytes > budget`; oversized single entries still
//! cache (re-uploading every call would be strictly worse).
//!
//! Replaces the prior `DashMap<NuclideKey, Vec<Arc<NuclideKernels>>>`
//! (unbounded; the Vec slot's "future bulk-dump API" was never used).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use super::{NuclideKey, NuclideStore};
use crate::transport::xs_provider::NuclideKernels;

const DEFAULT_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;
const FRACTION_MIN: f64 = 0.05;
const FRACTION_MAX: f64 = 0.95;

pub struct L1MemoryStore {
    /// `(key, value, approx_host_bytes)`; insertion order = LRU
    /// order, front oldest. Hits move to the back.
    inner: Mutex<VecDeque<(NuclideKey, Arc<NuclideKernels>, usize)>>,
    budget_bytes: usize,
    name: String,
}

impl L1MemoryStore {
    pub fn new() -> Self {
        Self::with_budget(resolve_budget())
    }

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

    /// Diagnostic-only; budget gates inserts.
    pub fn approx_bytes_held(&self) -> usize {
        self.inner
            .lock()
            .expect("L1MemoryStore mutex poisoned")
            .iter()
            .map(|(_, _, b)| *b)
            .sum()
    }

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
        guard.push_back(entry);
        Some(arc)
    }

    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        let bytes = value.approx_host_bytes();
        let mut guard = self.inner.lock().expect("L1MemoryStore mutex poisoned");
        // Concurrent-insert race: replace in place to avoid
        // double-charging the budget.
        if let Some(pos) = guard.iter().position(|(k, _, _)| k == &key) {
            let _ = guard.remove(pos);
        }
        let mut running: usize = guard.iter().map(|(_, _, b)| *b).sum();
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

/// Env vars are read once at construction; later setenv has no effect.
fn resolve_budget() -> usize {
    if let Some(v) = std::env::var_os("OPEN_RUST_MC_NUCLIDE_CACHE_BYTES") {
        if let Ok(n) = v.to_string_lossy().parse::<usize>() {
            return n.max(1);
        }
    }
    if let Some(v) = std::env::var_os("OPEN_RUST_MC_NUCLIDE_CACHE_FRACTION") {
        if let Ok(f) = v.to_string_lossy().parse::<f64>() {
            // 16 GiB stand-in for total RAM (no sysinfo dep).
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

    #[test]
    fn evicts_when_budget_exceeded() {
        let approx_each = NuclideKernels::empty(1.0, 2.43).approx_host_bytes();
        // Budget fits 2; the 3rd insert evicts.
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
        let store = L1MemoryStore::with_budget(1); // 1-byte budget
        let k = mk_key(7);
        store.put(k.clone(), Arc::new(NuclideKernels::empty(1.0, 2.43)));
        assert_eq!(store.n_keys(), 1);
        assert!(store.try_get(&k).is_some());
    }

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
