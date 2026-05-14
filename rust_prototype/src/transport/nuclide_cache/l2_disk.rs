//! L2 disk store — content-addressed files under the user cache dir.
//!
//! Path resolution (in order of precedence):
//!
//! 1. `OPEN_RUST_MC_CACHE_DIR` environment variable. If set to `off`
//!    (case-insensitive), the L2 store is disabled — useful for
//!    benchmarks that need a cold start every run, and for
//!    debugging the HDF5 parse path.
//! 2. `%LOCALAPPDATA%\open_rust_mc\cache` on Windows.
//! 3. `$XDG_CACHE_HOME/open_rust_mc` or `~/.cache/open_rust_mc`
//!    elsewhere.
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
use std::sync::{Arc, OnceLock};

use super::binary_format::{
    DecodeError, EncodeError, decode_nuclide_kernels, encode_nuclide_kernels,
    read_header_and_payload, write_header_and_payload,
};
use super::{NuclideKey, NuclideStore};
use crate::transport::xs_provider::NuclideKernels;

pub struct L2DiskStore {
    dir: PathBuf,
    name: String,
    /// Latched once on first encode-Unimplemented so the warning log
    /// fires once per process, not per cache miss.
    warned_unimplemented: OnceLock<()>,
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
        Some(Self {
            dir,
            name,
            warned_unimplemented: OnceLock::new(),
        })
    }

    /// Resolve a default cache dir from `OPEN_RUST_MC_CACHE_DIR` →
    /// platform default. Returns `None` when explicitly disabled or
    /// when no directory can be derived (e.g. no `HOME` on a degraded
    /// environment).
    pub fn from_env() -> Option<Self> {
        if let Ok(env) = std::env::var("OPEN_RUST_MC_CACHE_DIR") {
            if env.eq_ignore_ascii_case("off") {
                return None;
            }
            return Self::at(PathBuf::from(env));
        }
        let base = default_cache_dir()?;
        Self::at(base.join("open_rust_mc").join("cache"))
    }

    fn path_for(&self, key: &NuclideKey) -> PathBuf {
        self.dir.join(key.disk_filename())
    }

    fn warn_unimplemented_once(&self) {
        let _ = self.warned_unimplemented.get_or_init(|| {
            eprintln!(
                "note: nuclide_cache L2 disk store is staged — \
                 binary format per-type encoders not yet implemented; \
                 cache writes are no-ops until they land. L1 in-process \
                 cache is unaffected."
            );
        });
    }
}

impl NuclideStore for L2DiskStore {
    fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>> {
        let path = self.path_for(key);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return None,
        };
        let mut r: &[u8] = &bytes;
        let payload = match read_header_and_payload(&mut r) {
            Ok(p) => p,
            Err(e) => {
                if !matches!(e, DecodeError::Io(_)) {
                    eprintln!(
                        "warning: nuclide_cache L2 entry {} unreadable ({e}); \
                         removing and falling through to HDF5 load",
                        path.display()
                    );
                    let _ = std::fs::remove_file(&path);
                }
                return None;
            }
        };
        match decode_nuclide_kernels(&payload) {
            Ok(k) => Some(Arc::new(k)),
            Err(DecodeError::Unimplemented) => None,
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
        let payload = match encode_nuclide_kernels(&value) {
            Ok(p) => p,
            Err(EncodeError::Unimplemented) => {
                self.warn_unimplemented_once();
                return;
            }
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
        // The header carries the blake3 over `payload`; tearing the
        // write here leaves a malformed file, but read_header_and_payload
        // will detect it and drop the entry next time.
        let mut buf = Vec::with_capacity(payload.len() + 64);
        if let Err(e) = write_header_and_payload(&mut buf, &payload) {
            eprintln!("warning: nuclide_cache L2 header write error: {e}");
            return;
        }
        use std::io::Write;
        if let Err(e) = f.write_all(&buf) {
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

#[cfg(target_os = "windows")]
fn default_cache_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(PathBuf::from).or_else(|| {
        std::env::var_os("USERPROFILE")
            .map(|h| PathBuf::from(h).join("AppData").join("Local"))
    })
}

#[cfg(not(target_os = "windows"))]
fn default_cache_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(xdg));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache"))
}
