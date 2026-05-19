//! L3 remote `NuclideStore` — TCP client(s) that talk to one or more
//! `nuclide_cache_server` daemons over the [wire_protocol].
//!
//! ## Multi-peer model (MPI-friendly)
//!
//! `OPEN_RUST_MC_CACHE_SERVER` accepts a single `tcp://host:port` or
//! a comma-separated list:
//!
//!   `OPEN_RUST_MC_CACHE_SERVER=tcp://node01:53700,tcp://node02:53700`
//!
//! Read semantics: walk peers in order, **first HIT wins**. A peer
//! that's down (connect timeout, transport error) counts as MISS and
//! the walker moves to the next peer. The aggregate result is HIT iff
//! at least one peer holds the key.
//!
//! Write semantics: replicate to **every reachable peer**. Each peer
//! that's down at write time silently misses the replication — the
//! cache is eventually-consistent, not transactional. A peer that
//! comes back online and missed a previous PUT will simply MISS on
//! the next GET and the cache will re-warm via JIT (client parses,
//! PUTs, replication fans out again).
//!
//! Why this is fine for MPI / HPC: nuclide-kernel data is immutable
//! given a `NuclideKey` (file_hash + policy_hash + temp_idx pin the
//! content). Any peer that holds key K holds the same bytes as any
//! other peer that holds K. There's no consistency model to worry
//! about beyond "is the data there or not".
//!
//! ## Lifecycle
//!
//! Connection-per-request (no pooling). The cost amortises against
//! a HDF5 parse (~1 s+ per nuclide) trivially — even a few ms of
//! TCP setup is two orders of magnitude faster than re-parsing on
//! a miss-elsewhere. Pooling can layer on later without a protocol
//! change; the daemon already serves one thread per accepted socket.
//!
//! ## Failure semantics
//!
//! Every socket error is a *miss*. A daemon going down is benign;
//! the client falls through to L2 disk and then HDF5. No retries.
//! Per-peer timeout is bounded by `CONNECT_TIMEOUT + IO_TIMEOUT`.
//! Worst-case L3 latency with N peers all unreachable is
//! `N × CONNECT_TIMEOUT`, so keep the peer list short (≤ ~20) or
//! tune the timeouts down.

use std::io::Write;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

use super::binary_format::{decode_nuclide_kernels, encode_nuclide_kernels};
use super::wire_protocol::{
    OP_GET, OP_PUT, STATUS_HIT, STATUS_OK, read_response, write_request,
};
use super::{NuclideKey, NuclideStore};
use crate::transport::xs_provider::NuclideKernels;

const CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
const IO_TIMEOUT: Duration = Duration::from_secs(5);

/// One remote daemon endpoint. Stored as a resolved `SocketAddr` so
/// hostname lookup happens once at construction.
#[derive(Clone, Debug)]
struct Peer {
    addr: SocketAddr,
    label: String,
}

impl Peer {
    fn open(&self) -> Option<TcpStream> {
        let sock = TcpStream::connect_timeout(&self.addr, CONNECT_TIMEOUT).ok()?;
        let _ = sock.set_read_timeout(Some(IO_TIMEOUT));
        let _ = sock.set_write_timeout(Some(IO_TIMEOUT));
        let _ = sock.set_nodelay(true);
        Some(sock)
    }
}

pub struct L3RemoteStore {
    peers: Vec<Peer>,
    name: String,
}

impl L3RemoteStore {
    /// Single-peer constructor. Kept for back-compat with callers that
    /// already had a single-URL handle. Multi-peer callers use
    /// [`L3RemoteStore::with_peers`] or [`L3RemoteStore::from_env`].
    pub fn new(server_url: &str) -> Option<Self> {
        Self::with_peers(&[server_url])
    }

    /// Build from a slice of `tcp://host:port` strings. Each entry is
    /// resolved; entries that don't parse / don't resolve are dropped
    /// with a warning. Returns `None` if **no** entries resolve.
    pub fn with_peers(server_urls: &[&str]) -> Option<Self> {
        let mut peers = Vec::with_capacity(server_urls.len());
        for url in server_urls {
            let trimmed = url.strip_prefix("tcp://").unwrap_or(url).trim();
            if trimmed.is_empty() {
                continue;
            }
            match trimmed.to_socket_addrs().ok().and_then(|mut it| it.next()) {
                Some(addr) => peers.push(Peer {
                    addr,
                    label: format!("tcp://{addr}"),
                }),
                None => eprintln!(
                    "warning: nuclide-cache peer {url:?} could not be resolved; dropped."
                ),
            }
        }
        if peers.is_empty() {
            return None;
        }
        let name = if peers.len() == 1 {
            format!("L3 daemon {}", peers[0].label)
        } else {
            let labels: Vec<&str> = peers.iter().map(|p| p.label.as_str()).collect();
            format!("L3 daemon swarm [{}]", labels.join(", "))
        };
        Some(Self { peers, name })
    }

    /// Build from `OPEN_RUST_MC_CACHE_SERVER`. Accepts:
    ///   - a single URL:  `tcp://host:port`
    ///   - a comma-separated list:
    ///       `tcp://a:53700,tcp://b:53700,tcp://c:53700`
    ///   - the literal `off` (case-insensitive) → disabled
    ///   - empty / unset → disabled (silent)
    ///
    /// Returns `None` when no peer resolves.
    pub fn from_env() -> Option<Self> {
        let env = std::env::var("OPEN_RUST_MC_CACHE_SERVER").ok()?;
        let trimmed = env.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("off") {
            return None;
        }
        let parts: Vec<&str> = trimmed
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let store = Self::with_peers(&parts);
        if store.is_none() {
            eprintln!(
                "warning: OPEN_RUST_MC_CACHE_SERVER={env:?} produced no resolvable peers; \
                 L3 cache tier disabled."
            );
        }
        store
    }

    /// Number of peers configured. Diagnostic only.
    pub fn peer_count(&self) -> usize { self.peers.len() }
}

impl NuclideStore for L3RemoteStore {
    /// First-hit-wins read. Walks peers in insertion order; each peer
    /// failing or returning MISS advances to the next. Stops at the
    /// first HIT.
    fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>> {
        for peer in &self.peers {
            let Some(mut sock) = peer.open() else { continue; };
            if write_request(&mut sock, OP_GET, key, &[]).is_err() {
                continue;
            }
            let _ = sock.flush();
            let Ok((status, payload)) = read_response(&mut sock) else { continue; };
            if status != STATUS_HIT || payload.is_empty() {
                continue;
            }
            match decode_nuclide_kernels(&payload) {
                Ok(k) => return Some(Arc::new(k)),
                Err(e) => {
                    eprintln!(
                        "warning: L3 peer {} returned payload that failed decode ({e}); \
                         trying next peer.",
                        peer.label
                    );
                }
            }
        }
        None
    }

    /// Replicate to every reachable peer. Per-peer failures are logged
    /// at most as a single line and otherwise silent — the L3 PUT is
    /// advisory, the canonical record lives in L1 and (when enabled)
    /// L2 on the writer side.
    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        let bytes = match encode_nuclide_kernels(&value) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("warning: L3 PUT — encode failed: {e}");
                return;
            }
        };
        for peer in &self.peers {
            let Some(mut sock) = peer.open() else { continue; };
            if write_request(&mut sock, OP_PUT, &key, &bytes).is_err() {
                continue;
            }
            let _ = sock.flush();
            // Best-effort: read the status byte so the daemon's
            // `write_response` doesn't sit blocked. Drop any error —
            // each peer write is advisory.
            let _ = read_response(&mut sock);
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tcp_url() {
        // `tcp://` prefix optional. IPv4 + port resolves cleanly via
        // ToSocketAddrs (`127.0.0.1:9999` is a valid hostname:port).
        let s = L3RemoteStore::new("tcp://127.0.0.1:9999").expect("parse");
        assert!(s.name.contains("127.0.0.1:9999"));
        assert_eq!(s.peer_count(), 1);
        let s = L3RemoteStore::new("127.0.0.1:9999").expect("parse");
        assert!(s.name.contains("127.0.0.1:9999"));
    }

    #[test]
    fn parse_comma_separated_peer_list() {
        let s = L3RemoteStore::with_peers(&[
            "tcp://127.0.0.1:9999",
            "tcp://127.0.0.1:10000",
            "tcp://127.0.0.1:10001",
        ])
        .expect("at least one peer must resolve");
        assert_eq!(s.peer_count(), 3);
        assert!(s.name.contains("swarm"));
    }

    #[test]
    fn from_env_with_commas_parses_each_peer() {
        let prev = std::env::var("OPEN_RUST_MC_CACHE_SERVER").ok();
        // SAFETY: see from_env_with_no_server_is_silent for the
        // env-test caveat.
        unsafe {
            std::env::set_var(
                "OPEN_RUST_MC_CACHE_SERVER",
                "tcp://127.0.0.1:9001,tcp://127.0.0.1:9002",
            )
        };
        let s = L3RemoteStore::from_env().expect("two peers must resolve");
        assert_eq!(s.peer_count(), 2);
        // Restore the prior value (or unset).
        match prev {
            Some(v) => unsafe { std::env::set_var("OPEN_RUST_MC_CACHE_SERVER", v) },
            None => unsafe { std::env::remove_var("OPEN_RUST_MC_CACHE_SERVER") },
        }
    }

    #[test]
    fn from_env_with_no_server_is_silent() {
        let prev = std::env::var("OPEN_RUST_MC_CACHE_SERVER").ok();
        // SAFETY: tests in this crate are not run concurrently with
        // anything that reads this env (no `cargo test` parallel
        // group depends on it). The remove + restore is a best-effort
        // hygiene step.
        unsafe { std::env::remove_var("OPEN_RUST_MC_CACHE_SERVER") };
        assert!(L3RemoteStore::from_env().is_none());
        if let Some(v) = prev {
            unsafe { std::env::set_var("OPEN_RUST_MC_CACHE_SERVER", v) };
        }
    }

    /// Multi-peer GET: first peer is dead (port 1, unreachable), second
    /// peer hosts the key. The client must skip the dead peer and HIT
    /// on the second. This is the MPI-deployment semantic — drop a
    /// node, the swarm keeps serving.
    #[test]
    fn multipeer_first_hit_wins_skips_dead_peer() {
        use super::super::wire_protocol::{
            OP_GET, OP_PUT, STATUS_HIT, STATUS_MISS, STATUS_OK, read_request, write_response,
        };
        use std::collections::HashMap;
        use std::net::TcpListener;
        use std::sync::{Arc as StdArc, Mutex};
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let alive_addr = listener.local_addr().unwrap();
        let store_inner: StdArc<Mutex<HashMap<NuclideKey, Vec<u8>>>> =
            StdArc::new(Mutex::new(HashMap::new()));
        let server_store = StdArc::clone(&store_inner);

        // Accept up to 4 connections: (1) PUT to seed the live peer,
        // (2) the eventual successful GET that hits the live peer,
        // and 2 extra slots in case timing differs.
        let _srv = thread::spawn(move || {
            for _ in 0..4 {
                let (mut sock, _) = match listener.accept() {
                    Ok(s) => s,
                    Err(_) => return,
                };
                let (op, key, payload) = match read_request(&mut sock) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                match op {
                    OP_GET => {
                        let hit = server_store.lock().unwrap().get(&key).cloned();
                        if let Some(bytes) = hit {
                            let _ = write_response(&mut sock, STATUS_HIT, &bytes);
                        } else {
                            let _ = write_response(&mut sock, STATUS_MISS, &[]);
                        }
                    }
                    OP_PUT => {
                        server_store.lock().unwrap().insert(key, payload);
                        let _ = write_response(&mut sock, STATUS_OK, &[]);
                    }
                    _ => {}
                }
            }
        });

        let tmp = std::env::temp_dir().join("orm_l3_multipeer_test.h5");
        std::fs::write(&tmp, b"multi-peer-bytes").unwrap();
        let policy = crate::transport::xs_provider::RankPolicy::new(5);
        let key = NuclideKey::from_inputs(&tmp, &policy, 0).unwrap();
        let kernel = NuclideKernels::empty(56.0, 0.0);

        // Seed the live peer with a single-peer store first.
        let seed = L3RemoteStore::new(&alive_addr.to_string()).expect("seed parse");
        seed.put(key.clone(), Arc::new(kernel));

        // Multi-peer store: dead peer first, then live. First-hit-wins
        // must skip the dead peer and HIT on the live one.
        let dead_first = format!("127.0.0.1:1,{alive_addr}");
        let parts: Vec<&str> = dead_first.split(',').collect();
        let client = L3RemoteStore::with_peers(&parts).expect("at least one peer resolves");
        assert_eq!(client.peer_count(), 2);
        let hit = client.try_get(&key).expect("live peer must HIT after dead peer skipped");
        assert_eq!(hit.awr, 56.0);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn try_get_on_dead_address_returns_none() {
        // No server is listening on 127.0.0.1:1 (port 1 / tcpmux on
        // most systems is restricted) — the connect_timeout fires and
        // try_get returns None, which is the expected miss behaviour
        // when the daemon is down.
        let tmp = std::env::temp_dir().join("orm_l3_dead_test.h5");
        std::fs::write(&tmp, b"abc").unwrap();
        let policy = crate::transport::xs_provider::RankPolicy::new(5);
        let key = NuclideKey::from_inputs(&tmp, &policy, 0).unwrap();
        let store = L3RemoteStore::new("127.0.0.1:1").expect("parse");
        assert!(store.try_get(&key).is_none());
        let _ = std::fs::remove_file(&tmp);
    }

    /// End-to-end: spawn a minimal in-process server thread on an
    /// ephemeral port, run a client GET (miss) → PUT → GET (hit)
    /// sequence, verify the bytes round-trip cleanly through the
    /// real socket. Validates the wire framing + status codes +
    /// connection lifecycle end-to-end without involving the
    /// `nuclide_cache_server` binary.
    #[test]
    fn live_daemon_get_put_get_roundtrip() {
        use super::super::wire_protocol::{
            OP_GET, OP_PUT, STATUS_HIT, STATUS_MISS, STATUS_OK, read_request, write_response,
        };
        use std::collections::HashMap;
        use std::net::TcpListener;
        use std::sync::{Arc as StdArc, Mutex};
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let addr = listener.local_addr().unwrap();
        let store_inner: StdArc<Mutex<HashMap<NuclideKey, Vec<u8>>>> =
            StdArc::new(Mutex::new(HashMap::new()));
        let server_store = StdArc::clone(&store_inner);

        // Server loop — accept up to 3 connections (GET miss, PUT, GET
        // hit) then exit.
        let _srv = thread::spawn(move || {
            for _ in 0..3 {
                let (mut sock, _) = match listener.accept() {
                    Ok(s) => s,
                    Err(_) => return,
                };
                let (op, key, payload) = match read_request(&mut sock) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                match op {
                    OP_GET => {
                        let hit = server_store.lock().unwrap().get(&key).cloned();
                        if let Some(bytes) = hit {
                            let _ = write_response(&mut sock, STATUS_HIT, &bytes);
                        } else {
                            let _ = write_response(&mut sock, STATUS_MISS, &[]);
                        }
                    }
                    OP_PUT => {
                        server_store.lock().unwrap().insert(key, payload);
                        let _ = write_response(&mut sock, STATUS_OK, &[]);
                    }
                    _ => {}
                }
            }
        });

        // Build a real NuclideKey + a real binary-format payload.
        let tmp = std::env::temp_dir().join("orm_l3_live_test.h5");
        std::fs::write(&tmp, b"hello-world").unwrap();
        let policy = crate::transport::xs_provider::RankPolicy::new(5);
        let key = NuclideKey::from_inputs(&tmp, &policy, 0).unwrap();
        let kernel = NuclideKernels::empty(238.0289, 2.43);

        let client = L3RemoteStore::new(&addr.to_string()).expect("client parse");

        // GET miss
        assert!(client.try_get(&key).is_none(), "fresh store must miss");

        // PUT — encode + ship over the wire.
        client.put(key.clone(), Arc::new(kernel));

        // GET hit — decode back, verify the empty kernel reconstructs
        // the same AWR / nu_bar_const we put in.
        let hit = client.try_get(&key).expect("must hit after PUT");
        assert_eq!(hit.awr, 238.0289);
        assert_eq!(hit.nu_bar_const, 2.43);

        let _ = std::fs::remove_file(&tmp);
    }
}
