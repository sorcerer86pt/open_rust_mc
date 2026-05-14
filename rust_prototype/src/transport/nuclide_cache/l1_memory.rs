//! L1 in-process store — `DashMap<NuclideKey, Vec<Arc<NuclideKernels>>>`.
//!
//! The `Vec<Arc<…>>` slot is intentional. Many callers re-load the same
//! `.h5` at slightly different policies (different SVD rank, different
//! per-MT overrides, ...) — that produces different keys, so they never
//! collide. The Vec is there for callers that ever want to *list* all
//! kernels keyed on the same canonical path (introspection / dump
//! helpers); the normal hot path is single-Arc.

use std::sync::Arc;

use dashmap::DashMap;

use super::{NuclideKey, NuclideStore};
use crate::transport::xs_provider::NuclideKernels;

pub struct L1MemoryStore {
    inner: DashMap<NuclideKey, Vec<Arc<NuclideKernels>>>,
    name: String,
}

impl L1MemoryStore {
    pub fn new() -> Self {
        Self {
            inner: DashMap::new(),
            name: "L1 in-memory".to_string(),
        }
    }

    /// Total number of distinct keys held — counts each kernel, not each
    /// path. Useful for diagnostics ("cache holds 312 kernels").
    pub fn n_keys(&self) -> usize {
        self.inner.len()
    }
}

impl Default for L1MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl NuclideStore for L1MemoryStore {
    fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>> {
        // Take the first Arc — caller-visible behaviour is "did we
        // cache this kernel". The Vec exists for future bulk-dump APIs;
        // identity / equality is by `NuclideKey`, not by Vec index.
        self.inner.get(key).and_then(|v| v.first().map(Arc::clone))
    }

    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        self.inner.entry(key).or_default().push(value);
    }

    fn name(&self) -> &str {
        &self.name
    }
}
