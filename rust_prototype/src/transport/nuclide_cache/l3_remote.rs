//! L3 remote `NuclideStore` — a TCP client that talks to
//! `nuclide_cache_server` over the [wire_protocol].
//!
//! Lifecycle is intentionally connection-per-request. The cost
//! amortises against a HDF5 parse (~1 s+ per nuclide) trivially — even
//! a few ms of TCP setup is two orders of magnitude faster than
//! re-parsing on a miss-elsewhere. Connection pooling can layer on
//! later without a protocol change; the daemon already serves one
//! thread per accepted socket so a future pool is purely a client-side
//! concern.
//!
//! Discovery: `OPEN_RUST_MC_CACHE_SERVER=tcp://host:port`. Unset
//! means no L3 — `TieredStore` skips this tier silently.
//!
//! Failure semantics: every socket error is a *miss*. The daemon
//! going down is benign; the client falls through to L2 disk and
//! then HDF5. We do not retry. We do not block longer than
//! `CONNECT_TIMEOUT` and `IO_TIMEOUT` on any single request.

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

pub struct L3RemoteStore {
    addr: SocketAddr,
    name: String,
}

impl L3RemoteStore {
    /// Connect-on-demand store backed by a daemon at `tcp://host:port`.
    /// Resolves the address once at construction. Returns `None` if
    /// the input cannot be parsed.
    pub fn new(server_url: &str) -> Option<Self> {
        let trimmed = server_url
            .strip_prefix("tcp://")
            .unwrap_or(server_url);
        let addr = trimmed.to_socket_addrs().ok()?.next()?;
        let name = format!("L3 daemon tcp://{addr}");
        Some(Self { addr, name })
    }

    /// Build from `OPEN_RUST_MC_CACHE_SERVER` env. Returns `None` if
    /// the env is unset, empty, or set to a string that fails to
    /// resolve. No error is logged in the unset case (the L3 store
    /// being inert is the default).
    pub fn from_env() -> Option<Self> {
        let env = std::env::var("OPEN_RUST_MC_CACHE_SERVER").ok()?;
        if env.trim().is_empty() {
            return None;
        }
        if env.eq_ignore_ascii_case("off") {
            return None;
        }
        let store = Self::new(&env);
        if store.is_none() {
            eprintln!(
                "warning: OPEN_RUST_MC_CACHE_SERVER={env:?} could not be \
                 resolved; L3 cache tier disabled."
            );
        }
        store
    }

    fn open(&self) -> Option<TcpStream> {
        let sock = TcpStream::connect_timeout(&self.addr, CONNECT_TIMEOUT).ok()?;
        let _ = sock.set_read_timeout(Some(IO_TIMEOUT));
        let _ = sock.set_write_timeout(Some(IO_TIMEOUT));
        let _ = sock.set_nodelay(true);
        Some(sock)
    }
}

impl NuclideStore for L3RemoteStore {
    fn try_get(&self, key: &NuclideKey) -> Option<Arc<NuclideKernels>> {
        let mut sock = self.open()?;
        write_request(&mut sock, OP_GET, key, &[]).ok()?;
        let _ = sock.flush();
        let (status, payload) = read_response(&mut sock).ok()?;
        if status != STATUS_HIT || payload.is_empty() {
            return None;
        }
        match decode_nuclide_kernels(&payload) {
            Ok(k) => Some(Arc::new(k)),
            Err(e) => {
                eprintln!(
                    "warning: L3 daemon returned payload that failed decode ({e}); \
                     falling through to lower tiers."
                );
                None
            }
        }
    }

    fn put(&self, key: NuclideKey, value: Arc<NuclideKernels>) {
        let bytes = match encode_nuclide_kernels(&value) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("warning: L3 PUT — encode failed: {e}");
                return;
            }
        };
        let Some(mut sock) = self.open() else {
            return;
        };
        if write_request(&mut sock, OP_PUT, &key, &bytes).is_err() {
            return;
        }
        let _ = sock.flush();
        // Best-effort: read the status byte so the daemon's `write_response`
        // doesn't sit blocked. Drop any error — the put is advisory.
        let _ = read_response(&mut sock);
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
        let s = L3RemoteStore::new("127.0.0.1:9999").expect("parse");
        assert!(s.name.contains("127.0.0.1:9999"));
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
