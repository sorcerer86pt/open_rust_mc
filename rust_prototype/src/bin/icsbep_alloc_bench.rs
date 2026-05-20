// SPDX-License-Identifier: MIT
//! Transport-time allocator pressure bench.
//!
//! Runs one ICSBEP CPU eigenvalue loop (Godiva, 80×5k×1 seed) with a
//! counting `#[global_allocator]` that wraps `System` and counts every
//! `alloc` / `dealloc` call. Reports total allocations, total bytes
//! requested, and the delta from a warm-up baseline so the static
//! library / HDF5 loading allocations are excluded — the number is
//! the transport hot loop's allocator pressure for one batch loop.
//!
//! Used to quantify the impact of the per-particle Vec elimination
//! (TransportCtx, recycled ParticleTallies, etc.). Run baseline vs
//! optimised by `git stash`-ing the CPU transport changes between
//! invocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use open_rust_mc::geometry::scene_io;
use open_rust_mc::transport::dispatch::{CpuRunner, EigenvalueRunner};
use open_rust_mc::transport::material_resolve;
use open_rust_mc::transport::nuclides::NuclideLibrary;
use open_rust_mc::transport::simulate::SimConfig;

// ── Counting allocator ────────────────────────────────────────────────

struct Counting;

static N_ALLOC: AtomicUsize = AtomicUsize::new(0);
static N_FREE: AtomicUsize = AtomicUsize::new(0);
static B_ALLOC: AtomicUsize = AtomicUsize::new(0);
static B_FREE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        N_ALLOC.fetch_add(1, Ordering::Relaxed);
        B_ALLOC.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        N_FREE.fetch_add(1, Ordering::Relaxed);
        B_FREE.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[derive(Copy, Clone)]
struct AllocSnap {
    n_alloc: usize,
    n_free: usize,
    b_alloc: usize,
    b_free: usize,
}

fn snap() -> AllocSnap {
    AllocSnap {
        n_alloc: N_ALLOC.load(Ordering::Relaxed),
        n_free: N_FREE.load(Ordering::Relaxed),
        b_alloc: B_ALLOC.load(Ordering::Relaxed),
        b_free: B_FREE.load(Ordering::Relaxed),
    }
}

fn delta(prev: AllocSnap, now: AllocSnap) -> (usize, usize, usize, usize) {
    (
        now.n_alloc - prev.n_alloc,
        now.n_free - prev.n_free,
        now.b_alloc - prev.b_alloc,
        now.b_free - prev.b_free,
    )
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.2} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.2} KB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

fn bench_dir() -> PathBuf {
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("bench/icsbep").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p.join("bench/icsbep")
}

fn data_dir() -> PathBuf {
    if let Ok(v) = std::env::var("ICSBEP_DATA_DIR") {
        return PathBuf::from(v);
    }
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("data/endfb-vii.1-hdf5/neutron").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p.join("data/endfb-vii.1-hdf5/neutron")
}

fn main() {
    let case_file = bench_dir().join("heu-met-fast-001_case-1.json");
    let text = std::fs::read_to_string(&case_file).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let scene = &value["scene"];

    let loaded = scene_io::load_scene_from_json(&scene.to_string()).unwrap();
    let lib = NuclideLibrary::from_data_dir(&data_dir());
    let resolved =
        material_resolve::resolve_materials(&loaded.materials, &lib, 15).unwrap();

    let mut cfg = SimConfig::default();
    cfg.batches = 80;
    cfg.inactive = 20;
    cfg.particles_per_batch = 5_000;
    cfg.seed = 42;
    cfg.verbose = false;

    let runner = CpuRunner {
        geometry: &loaded.geometry,
        materials: &resolved.materials,
        xs_provider: &resolved.provider,
    };

    println!("=== ICSBEP alloc bench — Godiva, 80 batches × 5 000 particles ===");
    println!("(Active histories: 60 × 5 000 = 300 000)");
    println!();

    // Snapshot AFTER library / scene load — those allocations are static
    // setup, not transport hot path.
    let pre = snap();
    let t0 = std::time::Instant::now();
    let outcome = runner.run(&cfg);
    let wall = t0.elapsed();
    let post = snap();

    let (na, nf, ba, bf) = delta(pre, post);
    let net_n = na as i64 - nf as i64;
    let net_b = ba as i64 - bf as i64;

    println!("transport_eigenvalue allocator activity:");
    println!(
        "  allocations  : {na:>12}   (live at end: {net_n:+})"
    );
    println!(
        "  bytes alloc'd: {:>12}   (live at end: {} net)",
        fmt_bytes(ba),
        if net_b >= 0 {
            fmt_bytes(net_b as usize)
        } else {
            format!("-{}", fmt_bytes((-net_b) as usize))
        }
    );
    println!(
        "  bytes freed  : {:>12}",
        fmt_bytes(bf)
    );
    println!(
        "  avg alloc sz : {:>12}",
        fmt_bytes(if na > 0 { ba / na } else { 0 })
    );
    println!();

    let n_particles_active = 60 * 5_000;
    println!("per-history averages (active batches × 5 000):");
    println!(
        "  allocs / history: {:>10.2}",
        na as f64 / n_particles_active as f64
    );
    println!(
        "  bytes / history : {:>10.2}",
        ba as f64 / n_particles_active as f64
    );
    println!();

    // Final sanity: print one k_eff so the optimiser can't dead-code-
    // eliminate the entire transport call.
    let active: Vec<f64> = outcome
        .batches
        .iter()
        .skip(20)
        .map(|b| b.k_eff)
        .collect();
    let mean = active.iter().sum::<f64>() / active.len() as f64;
    println!("k_eff sanity: ⟨k⟩ = {mean:.5} (active {} batches)", active.len());
    println!("wall time   : {:.3} s", wall.as_secs_f64());
    println!(
        "throughput  : {:.0} histories / s",
        300_000.0 / wall.as_secs_f64()
    );
}
