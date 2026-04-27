//! Batch-size scaling for the GPU Compton kernels.
//!
//! Sweeps N = 10k → 100M particles for both the light kernel (free-KN +
//! S(x,Z)/Z) and the heavy kernel (free-KN + Doppler) on uranium at 1
//! MeV. Reports per-event nanoseconds *including* H2D + launch + D2H
//! at each N, plus the asymptote (kernel-only at large N).
//!
//! What the curve answers:
//!   - If ns/ev falls sharply with N and plateaus → launch overhead
//!     dominates at small N; the asymptote is the real per-particle
//!     compute cost.
//!   - If ns/ev is roughly flat across N → the kernel is compute-or
//!     memory-bound, not launch-bound; bigger batches won't help.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use open_rust_mc::photon::PhotonElement;
use open_rust_mc::photon::gpu::{GpuComptonContext, GpuComptonDopplerCtx};

const SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];
const REPS: u32 = 5;
const E_IN: f64 = 1.0e6;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let dir = match args.next() {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!("usage: gpu_compton_scaling <photon_data_dir>");
            return ExitCode::from(1);
        }
    };

    let elem = PhotonElement::from_hdf5(&dir.join("U.h5")).expect("load U.h5");

    let ctx_free = GpuComptonContext::new(&elem).expect("free ctx");
    let ctx_dop = GpuComptonDopplerCtx::new(&elem).expect("doppler ctx");

    // Warm up both kernels.
    let _ = ctx_free.sample_batch(E_IN, 0, 4096);
    let _ = ctx_dop.sample_batch(E_IN, 0, 4096);

    println!();
    println!("# GPU Compton batch-size scaling on U @ 1 MeV");
    println!("# Each row: best of {} reps; ns/ev = total wall / N", REPS);
    println!();
    println!(
        "{:>12} {:>12} {:>12} {:>10} {:>10}",
        "N", "free ms", "doppler ms", "free ns/ev", "dop ns/ev"
    );

    for &n in SIZES {
        let mut best_free = f64::INFINITY;
        let mut best_dop = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            let _ = ctx_free.sample_batch(E_IN, 0, n).expect("free");
            let dt = t.elapsed().as_secs_f64();
            if dt < best_free {
                best_free = dt;
            }

            let t = Instant::now();
            let _ = ctx_dop.sample_batch(E_IN, 0, n).expect("dop");
            let dt = t.elapsed().as_secs_f64();
            if dt < best_dop {
                best_dop = dt;
            }
        }
        let ns_free = best_free * 1e9 / n as f64;
        let ns_dop = best_dop * 1e9 / n as f64;
        println!(
            "{:>12} {:>12.2} {:>12.2} {:>10.2} {:>10.2}",
            n,
            best_free * 1e3,
            best_dop * 1e3,
            ns_free,
            ns_dop
        );
    }

    ExitCode::SUCCESS
}
