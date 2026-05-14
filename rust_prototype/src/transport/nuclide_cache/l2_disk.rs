//! L2 disk store — content-addressed files under a cross-platform cache dir.
//!
//! Path resolution (in order of precedence):
//!
//! 1. `OPEN_RUST_MC_CACHE_DIR` environment variable. If set to `off`
//!    (case-insensitive), the L2 store is disabled — useful for
//!    benchmarks that need a cold start every run, and for
//!    debugging the HDF5 parse path.
//! 2. `std::env::temp_dir().join("open_rust_mc_cache")` on every
//!    platform. Resolves to e.g. `C:\Users\<user>\AppData\Local\Temp\
//!    open_rust_mc_cache` on Windows, `/tmp/open_rust_mc_cache` on
//!    Linux, `$TMPDIR/open_rust_mc_cache` on macOS. Same API on every
//!    OS, no `#[cfg(target_os)]` gymnastics, no XDG vs LOCALAPPDATA
//!    branching. The trade-off: on systems that wipe `/tmp` at boot
//!    (some Linux distros, most container runtimes) the cache is
//!    lost across reboots — set `OPEN_RUST_MC_CACHE_DIR` explicitly
//!    to a persistent path when that matters.
//!
//! Filenames are derived from the `NuclideKey` via [`NuclideKey::disk_filename`]
//! — a hex-encoded blake3 file hash + policy hash + temp idx +
//! format version. Files are content-addressed, so two processes that
//! cache the same nuclide can race on the write without harm (last
//! writer wins, both contain identical bytes).
//!
//! Writes are atomic: encode to `<filename>.tmp`, then `rename` over
//! the final path. A torn-write from a crash leaves a stray `.tmp`
//! file but never a corrupt cache entry.

use std::path::PathBuf;
use std::sync::Arc;

use super::binary_format::{
    EncodeError, decode_nuclide_kernels, encode_nuclide_kernels,
};
use super::{NuclideKey, NuclideStore};
use crate::transport::xs_provider::NuclideKernels;

pub struct L2DiskStore {
    dir: PathBuf,
    name: String,
}

impl L2DiskStore {
    /// Build an L2 store rooted at `dir`. Creates the directory if it
    /// doesn't exist. Returns `None` if creation fails — L1-only is a
    /// valid runtime state.
    pub fn at(dir: PathBuf) -> Option<Self> {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!(
                "warning: nuclide_cache L2 disk store disabled — could not create {}: {e}",
                dir.display()
            );
            return None;
        }
        let name = format!("L2 disk {}", dir.display());
        Some(Self { dir, name })
    }

    /// Resolve a default cache dir from `OPEN_RUST_MC_CACHE_DIR` →
    /// `std::env::temp_dir() / open_rust_mc_cache`. Returns `None` when
    /// the env override is set to `off` (case-insensitive).
    ///
    /// `std::env::temp_dir()` is documented to never panic and to
    /// always return *some* path, so this method effectively never
    /// returns `None` for the default branch. Set
    /// `OPEN_RUST_MC_CACHE_DIR=off` to disable the L2 store.
    pub fn from_env() -> Option<Self> {
        if let Ok(env) = std::env::var("OPEN_RUST_MC_CACHE_DIR") {
            if env.eq_ignore_ascii_case("off") {
                return None;
            }
            return Self::at(PathBuf::from(env));
        }
        Self::at(std::env::temp_dir().join("open_rust_mc_cache"))
    }

    fn path_for(&self, key: &NuclideKey) -> PathBuf {
        self.dir.join(key.disk_filename())
    }

}

impl NuclideStore for L2DiskStore {
    fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>> {
        let path = self.path_for(key);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return None,
        };
        // `encode_nuclide_kernels` produces a complete header+payload
        // stream — the disk file is exactly those bytes. Decoding goes
        // through one validated path; the header's payload_blake3
        // catches torn writes / on-disk corruption before we touch the
        // body. (Earlier versions of this file double-wrapped: the
        // put path called both `encode_nuclide_kernels` *and*
        // `write_header_and_payload`, and the read path peeled one
        // header before handing the rest to `decode_nuclide_kernels`.
        // The double-wrap was redundant, two integrity hashes for one
        // file, and made the on-disk bytes diverge from the wire
        // bytes the L3 daemon ships. v3 of FORMAT_VERSION drops it.)
        match decode_nuclide_kernels(&bytes) {
            Ok(k) => Some(Arc::new(k)),
            Err(e) => {
                eprintln!(
                    "warning: nuclide_cache L2 decode error on {}: {e}; \
                     removing entry and falling through to HDF5 load",
                    path.display()
                );
                let _ = std::fs::remove_file(&path);
                None
            }
        }
    }

    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        let bytes = match encode_nuclide_kernels(&value) {
            Ok(p) => p,
            Err(EncodeError::Io(e)) => {
                eprintln!("warning: nuclide_cache L2 encode I/O error: {e}");
                return;
            }
        };
        let final_path = self.path_for(&key);
        let tmp_path = final_path.with_extension("tmp");
        let mut f = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!(
                    "warning: nuclide_cache L2 could not create {}: {e}",
                    tmp_path.display()
                );
                return;
            }
        };
        // Write the encoded stream verbatim — header already inside.
        use std::io::Write;
        if let Err(e) = f.write_all(&bytes) {
            eprintln!("warning: nuclide_cache L2 payload write error: {e}");
            return;
        }
        drop(f);
        if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
            eprintln!(
                "warning: nuclide_cache L2 rename {} → {} failed: {e}",
                tmp_path.display(),
                final_path.display()
            );
            let _ = std::fs::remove_file(&tmp_path);
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::nuclide_cache::binary_format::encode_nuclide_kernels;
    use crate::transport::xs_provider::{NuclideKernels, RankPolicy};

    /// Writes a valid `.nuc` to a temp dir via the disk store's
    /// public `put`, then reads it back via the store's `try_get`.
    /// The same write path is what `nuclide_cache_server` uses on
    /// startup eager-preload (with `--cache-dir`), so this roundtrip
    /// also validates that subsequent daemon restarts can pick up
    /// the cached files.
    #[test]
    fn disk_roundtrip_via_put_then_get() {
        let dir = std::env::temp_dir().join("orm_l2_disk_roundtrip_test");
        let _ = std::fs::remove_dir_all(&dir);
        let store = L2DiskStore::at(dir.clone()).expect("at");

        let h5 = std::env::temp_dir().join("orm_l2_disk_test.h5");
        std::fs::write(&h5, b"contents-fixture").unwrap();
        let policy = RankPolicy::new(5);
        let key = NuclideKey::from_inputs(&h5, &policy, 0).unwrap();
        let kernel = Arc::new(NuclideKernels::empty(238.0289, 2.43));

        // put — write through to disk atomically
        store.put(key.clone(), Arc::clone(&kernel));
        // file must exist after the put
        assert!(
            dir.join(key.disk_filename()).exists(),
            "L2DiskStore::put must leave a .nuc file behind"
        );

        // try_get — read back, blake3 must verify, NuclideKernels
        // must reconstruct
        let got = store.try_get(&key).expect("must hit after put");
        assert_eq!(got.awr, 238.0289);
        assert_eq!(got.nu_bar_const, 2.43);

        // Also: encoded payload is byte-identical between
        // disk-roundtrip and a fresh encode — that's the contract
        // the daemon eager-preload relies on.
        let direct = encode_nuclide_kernels(&NuclideKernels::empty(238.0289, 2.43)).unwrap();
        let from_disk = std::fs::read(dir.join(key.disk_filename())).unwrap();
        assert_eq!(direct, from_disk);

        let _ = std::fs::remove_file(&h5);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

