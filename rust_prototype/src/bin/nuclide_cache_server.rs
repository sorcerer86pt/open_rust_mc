//! L3 nuclide-kernel cache daemon.
//!
//! Single-process TCP server holding an in-memory map from
//! `NuclideKey` to the binary-format payload (raw bytes, not decoded
//! `NuclideKernels`). Slaves on a LAN connect over TCP, send GET /
//! PUT requests using the wire protocol, and skip their local HDF5
//! parse on a server hit.
//!
//! Storage: `HashMap<NuclideKey, Vec<u8>>`. The server never decodes —
//! it just hands back whatever bytes a client previously PUT or the
//! eager-load step injected. The binary-format `payload_blake3`
//! header guards integrity end-to-end; a corrupted cache file on
//! disk or a torn TCP frame both surface as
//! `DecodeError::PayloadHashMismatch` on the client and trigger a
//! fall-through to the lower tiers (L2 disk → HDF5 reparse).
//!
//! Concurrency: thread-per-connection. Suitable for the small number
//! of clients we expect (single-digit LAN PCs). Switch to tokio /
//! async only if the client population grows past a few dozen.
//!
//! ## Warm-up paths
//!
//! 1. **Eager pre-load (`--data <HDF5_DIR>`).** At startup the daemon
//!    walks `<HDF5_DIR>/*.h5`, parses each via
//!    `xs_provider::load_nuclide_with_policy` at the configured rank
//!    + `temp_idx=0`, encodes via `binary_format::encode_nuclide_kernels`,
//!    and inserts into the in-memory map. With `--cache-dir` also set,
//!    a matching `.nuc` on disk short-circuits the parse — second
//!    daemon launch on the same library is ~50 ms / nuclide rather
//!    than the full 1+ s HDF5 + SVD pass.
//! 2. **JIT (no `--data`).** The daemon starts cold and warms via
//!    client `OP_PUT`s on first miss. Clients run the full HDF5
//!    parse the first time; the next client (or next case on the
//!    same client) hits the daemon and skips it. Each client's
//!    first miss costs one full parse; once that PUT reaches the
//!    daemon, every subsequent client gets a hit.
//!
//! Both paths coexist. Eager pre-load fills the *common* policies
//! (one rank, `temp_idx=0`); off-policy requests (different rank,
//! per-MT overrides, other temperatures) fall through to JIT and
//! warm via PUT, just like the cold-start path.
//!
//! Listening port: `--listen <host:port>`, default `0.0.0.0:53700`.
//!
//! ## Operational notes
//!
//! - **No authentication.** Run on a trusted LAN. A public-internet
//!   deployment needs WireGuard / mTLS / a reverse proxy out of band.
//! - **No eviction.** The cache grows monotonically. ENDF/B-VII.1 in
//!   full carries ~5 GB parsed; the daemon will use ~10-15 GB RAM
//!   when fully populated. Acceptable for a build-server-class
//!   machine; restart to evict.

use std::collections::HashMap;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc as StdArc, Mutex};
use std::time::Instant;

use clap::Parser;

use open_rust_mc::transport::nuclide_cache::binary_format::{
    DecodeError, encode_nuclide_kernels, read_header_and_payload,
};
use open_rust_mc::transport::nuclide_cache::key::NuclideKey;
use open_rust_mc::transport::nuclide_cache::wire_protocol::{
    OP_GET, OP_PUT, OP_STATS, STATUS_ERR, STATUS_HIT, STATUS_MISS, STATUS_OK, read_request,
    write_response,
};
use open_rust_mc::transport::xs_provider::{RankPolicy, load_nuclide_with_policy};

#[derive(Parser, Debug)]
#[command(about = "Nuclide kernel cache daemon (L3 tier).")]
struct Args {
    /// TCP address to bind. Default `0.0.0.0:53700`.
    #[arg(long, default_value = "0.0.0.0:53700")]
    listen: String,
    /// Optional disk persistence directory. PUTs write through here
    /// before acknowledging, and the eager pre-load step (`--data`)
    /// reads a `.nuc` file in preference to re-parsing the HDF5
    /// source when both match by `NuclideKey::disk_filename`.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Optional HDF5 data directory. When set, every `*.h5` file in
    /// the directory is parsed at startup (or loaded from
    /// `--cache-dir` if a matching `.nuc` exists) and inserted into
    /// the in-memory map. Clients then GET-hit on first request.
    ///
    /// Without `--data`, the daemon starts cold and warms via client
    /// PUTs (the JIT path).
    #[arg(long)]
    data: Option<PathBuf>,
    /// SVD rank for the eager pre-load step. Default `5`. Only used
    /// with `--data`. Clients requesting a different rank produce a
    /// different policy hash → different `NuclideKey` → cache miss
    /// → JIT path (full HDF5 parse on the client, then PUT).
    #[arg(long, default_value_t = 5)]
    rank: usize,
    /// Temperature index for the eager pre-load step (HDF5 column
    /// ordinal in the temperature ladder of each `.h5`). Default `0`
    /// (typically 294 K). Same off-policy semantics as `--rank`.
    #[arg(long, default_value_t = 0)]
    temp_idx: usize,
}

type Cache = HashMap<NuclideKey, Vec<u8>>;

struct State {
    map: Mutex<Cache>,
    cache_dir: Option<PathBuf>,
}

impl State {
    /// Eager pre-load. Walks `<data_dir>/*.h5`, computes the
    /// `NuclideKey` for each at the configured `policy` + `temp_idx`,
    /// and inserts into the in-memory map. Disk hits (via
    /// `cache_dir`) avoid the HDF5 parse; misses re-parse and write
    /// through.
    ///
    /// Returns `(loaded_from_disk, parsed_from_hdf5)`.
    fn eager_preload(
        &self,
        data_dir: &std::path::Path,
        policy: &RankPolicy,
        temp_idx: usize,
    ) -> std::io::Result<(usize, usize)> {
        let mut from_disk = 0_usize;
        let mut from_hdf5 = 0_usize;
        let entries = std::fs::read_dir(data_dir)?;
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                (p.extension().is_some_and(|x| x == "h5")).then_some(p)
            })
            .collect();
        paths.sort();

        for path in paths {
            let key = match NuclideKey::from_inputs(&path, policy, temp_idx) {
                Ok(k) => k,
                Err(e) => {
                    eprintln!(
                        "  skip {}: cannot hash file ({e})",
                        path.display()
                    );
                    continue;
                }
            };

            // Disk-cache hit?
            if let Some(dir) = &self.cache_dir {
                let nuc_path = dir.join(key.disk_filename());
                if let Ok(bytes) = std::fs::read(&nuc_path) {
                    // Validate header before we keep it — corrupt files
                    // get re-parsed.
                    let mut r: &[u8] = &bytes;
                    if read_header_and_payload(&mut r).is_ok() {
                        self.map.lock().unwrap().insert(key, bytes);
                        from_disk += 1;
                        continue;
                    }
                    eprintln!(
                        "  corrupt {} — re-parsing source",
                        nuc_path.display()
                    );
                }
            }

            // HDF5 parse + encode + insert + write-through.
            let t0 = Instant::now();
            let kernel = load_nuclide_with_policy(&path, policy, temp_idx, 0.0, 2.43);
            let bytes = match encode_nuclide_kernels(&kernel) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  skip {}: encode failed ({e})", path.display());
                    continue;
                }
            };
            let elapsed_ms = t0.elapsed().as_millis();
            println!(
                "  parsed {} ({} KB, {elapsed_ms} ms)",
                path.display(),
                bytes.len() / 1024,
            );
            self.persist_put(&key, &bytes);
            self.map.lock().unwrap().insert(key, bytes);
            from_hdf5 += 1;
        }
        Ok((from_disk, from_hdf5))
    }

    fn persist_put(&self, key: &NuclideKey, bytes: &[u8]) {
        let Some(dir) = &self.cache_dir else { return };
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        let final_path = dir.join(key.disk_filename());
        let tmp_path = final_path.with_extension("tmp");
        if let Ok(mut f) = std::fs::File::create(&tmp_path) {
            if f.write_all(bytes).is_ok() {
                drop(f);
                let _ = std::fs::rename(&tmp_path, &final_path);
            } else {
                let _ = std::fs::remove_file(&tmp_path);
            }
        }
    }
}

fn handle_connection(state: &State, mut sock: TcpStream) {
    let peer = sock.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    loop {
        let req = match read_request(&mut sock) {
            Ok(r) => r,
            Err(DecodeError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                eprintln!("{peer}: bad request: {e}");
                let _ = write_response(&mut sock, STATUS_ERR, e.to_string().as_bytes());
                break;
            }
        };
        let (op, key, payload) = req;
        match op {
            OP_GET => {
                let bytes = state.map.lock().unwrap().get(&key).cloned();
                match bytes {
                    Some(b) => {
                        let _ = write_response(&mut sock, STATUS_HIT, &b);
                    }
                    None => {
                        let _ = write_response(&mut sock, STATUS_MISS, &[]);
                    }
                }
            }
            OP_PUT => {
                state.persist_put(&key, &payload);
                state.map.lock().unwrap().insert(key, payload);
                let _ = write_response(&mut sock, STATUS_OK, &[]);
            }
            OP_STATS => {
                let map = state.map.lock().unwrap();
                let n_entries = map.len();
                let total_bytes: usize = map.values().map(|v| v.len()).sum();
                let body = format!("{n_entries} entries, {total_bytes} bytes");
                let _ = write_response(&mut sock, STATUS_OK, body.as_bytes());
            }
            other => {
                let msg = format!("unknown op {other}");
                let _ = write_response(&mut sock, STATUS_ERR, msg.as_bytes());
            }
        }
    }
}

fn run(args: Args) -> std::io::Result<()> {
    let state = State {
        map: Mutex::new(HashMap::new()),
        cache_dir: args.cache_dir.clone(),
    };
    if let Some(data_dir) = &args.data {
        let policy = RankPolicy::new(args.rank);
        let t0 = Instant::now();
        println!(
            "eager pre-load from {} (rank={}, temp_idx={}, cache_dir={:?})",
            data_dir.display(),
            args.rank,
            args.temp_idx,
            args.cache_dir,
        );
        match state.eager_preload(data_dir, &policy, args.temp_idx) {
            Ok((from_disk, from_hdf5)) => {
                println!(
                    "  warm: {from_disk} from disk-cache, {from_hdf5} parsed from HDF5 \
                     ({} ms)",
                    t0.elapsed().as_millis()
                );
            }
            Err(e) => {
                eprintln!("warning: eager pre-load failed: {e}; starting cold (JIT path).");
            }
        }
    }
    println!(
        "nuclide_cache_server listening on {} (cache_dir={:?})",
        args.listen, args.cache_dir,
    );
    let listener = TcpListener::bind(&args.listen)?;
    let state = StdArc::new(state);
    for accepted in listener.incoming() {
        let sock = match accepted {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let state = StdArc::clone(&state);
        std::thread::spawn(move || handle_connection(&state, sock));
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    run(Args::parse())
}
