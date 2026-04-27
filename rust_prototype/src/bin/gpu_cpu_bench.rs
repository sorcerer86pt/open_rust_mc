//! Full CPU-vs-GPU benchmark across the photon kernels in two modes.
//!
//! Mode A — Per-kernel batched (apples-to-apples per call)
//!   Each row is a single sampling kernel (Compton, Rayleigh, Pair,
//!   Photoelectric Phase 1) at fixed E_in, batch N=1M particles.
//!   CPU side runs the production CPU sampler in parallel via rayon.
//!   GPU side runs the corresponding kernel one launch including
//!   H2D + D2H. This is the "small unit of work" comparison — what
//!   you pay if every collision incurs a kernel boundary.
//!
//! Mode B — Per-history full transport (apples-to-apples wall time)
//!   1M Compton-only histories, E_in = 1 MeV, E_cut = 1 keV, max 64
//!   collisions/history, free-KN sampling (no S(x,Z) — pure compute).
//!   CPU side: rayon-parallel loop, one history per task. GPU side:
//!   single persistent-kernel launch from gpu_photon_features. This
//!   is the right unit of work for transport — the launch tax is paid
//!   ONCE per N histories, not per collision.
//!
//! What this measures:
//!   - The size of the per-kernel-call CPU win on small kernels (Mode A)
//!   - The size of the per-history GPU win when the launch tax is
//!     amortized over many collisions (Mode B)
//!   - The crossover point: which "unit of work" actually matters

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use rayon::prelude::*;

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc;

use open_rust_mc::photon::PhotonElement;
use open_rust_mc::photon::coherent::coherent_scatter;
use open_rust_mc::photon::compton::{compton_scatter, compton_scatter_free};
use open_rust_mc::photon::gpu::{
    GpuComptonContext, GpuComptonDopplerCtx, GpuPairContext, GpuPhotoelectricCtx,
    GpuRayleighContext,
};
use open_rust_mc::photon::pair::pair_produce;
use open_rust_mc::photon::photoelectric::{DEFAULT_PHOTON_CUTOFF_EV, photoelectric_absorb};
use open_rust_mc::transport::rng::Rng;

const N: usize = 1_000_000;
const BATCH_ID: u64 = 0;

const TEST_CASES: &[(&str, &str, u32)] = &[
    ("H", "H.h5", 1),
    ("O", "O.h5", 8),
    ("Zr", "Zr.h5", 40),
    ("U", "U.h5", 92),
];

fn time<F: FnOnce()>(f: F) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64()
}

// ============================================================================
// Mode A: per-kernel batched
// ============================================================================

fn mode_a_compton(elem: &PhotonElement, sym: &str, e_in: f64) -> (f64, f64) {
    let ctx = GpuComptonContext::new(elem).expect("ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096); // warm

    let t_cpu = time(|| {
        let _: Vec<(f64, f64)> = (0..N)
            .into_par_iter()
            .map(|tid| {
                let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
                let o = compton_scatter_free(elem, e_in, &mut rng);
                (o.energy_out / e_in, o.mu)
            })
            .collect();
    });
    let mut t_gpu = 0.0;
    let _ = ctx.sample_batch(e_in, BATCH_ID, N).map(|_| {
        let t = Instant::now();
        let _ = ctx.sample_batch(e_in, BATCH_ID, N);
        t_gpu = t.elapsed().as_secs_f64();
    });
    let _ = sym;
    (t_cpu * 1e9 / N as f64, t_gpu * 1e9 / N as f64)
}

fn mode_a_compton_doppler(elem: &PhotonElement, e_in: f64) -> (f64, f64) {
    let ctx = GpuComptonDopplerCtx::new(elem).expect("dop ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096);

    let t_cpu = time(|| {
        let _: Vec<(f64, f64)> = (0..N)
            .into_par_iter()
            .map(|tid| {
                let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
                let o = compton_scatter(elem, e_in, &mut rng);
                (o.energy_out, o.mu)
            })
            .collect();
    });
    let t = Instant::now();
    let _ = ctx.sample_batch(e_in, BATCH_ID, N).expect("dop launch");
    let t_gpu = t.elapsed().as_secs_f64();
    (t_cpu * 1e9 / N as f64, t_gpu * 1e9 / N as f64)
}

fn mode_a_rayleigh(elem: &PhotonElement, e_in: f64) -> (f64, f64) {
    let ctx = GpuRayleighContext::new(elem).expect("ray ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096);

    let t_cpu = time(|| {
        let _: Vec<f64> = (0..N)
            .into_par_iter()
            .map(|tid| {
                let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
                coherent_scatter(elem, e_in, &mut rng).mu
            })
            .collect();
    });
    let t = Instant::now();
    let _ = ctx.sample_batch(e_in, BATCH_ID, N).expect("ray launch");
    let t_gpu = t.elapsed().as_secs_f64();
    (t_cpu * 1e9 / N as f64, t_gpu * 1e9 / N as f64)
}

fn mode_a_pair(e_in: f64) -> (f64, f64) {
    let ctx = GpuPairContext::new().expect("pair ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096);

    let t_cpu = time(|| {
        let _: Vec<(f64, f64)> = (0..N)
            .into_par_iter()
            .map(|tid| {
                let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
                match pair_produce(e_in, &mut rng) {
                    Some(o) => (o.electron_kinetic, o.positron_kinetic),
                    None => (0.0, 0.0),
                }
            })
            .collect();
    });
    let t = Instant::now();
    let _ = ctx.sample_batch(e_in, BATCH_ID, N).expect("pair launch");
    let t_gpu = t.elapsed().as_secs_f64();
    (t_cpu * 1e9 / N as f64, t_gpu * 1e9 / N as f64)
}

fn mode_a_photoelectric(elem: &PhotonElement, e_in: f64) -> (f64, f64) {
    let ctx = GpuPhotoelectricCtx::new(elem).expect("pe ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096);

    let t_cpu = time(|| {
        let _: Vec<(f64, i32)> = (0..N)
            .into_par_iter()
            .map(|tid| {
                let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
                let o = photoelectric_absorb(elem, e_in, DEFAULT_PHOTON_CUTOFF_EV, &mut rng);
                let struck_idx = (o.struck_subshell_designator as usize).saturating_sub(1);
                let b = elem.subshells[struck_idx].binding_energy;
                ((e_in - b).max(0.0), o.struck_subshell_designator as i32)
            })
            .collect();
    });
    let t = Instant::now();
    let _ = ctx.sample_batch(e_in, BATCH_ID, N).expect("pe launch");
    let t_gpu = t.elapsed().as_secs_f64();
    (t_cpu * 1e9 / N as f64, t_gpu * 1e9 / N as f64)
}

// ============================================================================
// Mode B: per-history full Compton transport
// ============================================================================

const PERSISTENT_KERNEL: &str = r#"
typedef unsigned long long u64;
typedef unsigned int u32;
__device__ __forceinline__ u32 rotr32(u32 x, u32 r) {
    u32 rm = r & 31u;
    return (x >> rm) | (x << ((32u - rm) & 31u));
}
struct PCG { u64 state; u64 inc; };
__device__ __forceinline__ u32 pcg_next_u32(PCG* r) {
    u64 old_state = r->state;
    r->state = old_state * 6364136223846793005ULL + r->inc;
    u32 xorshifted = (u32)(((old_state >> 18) ^ old_state) >> 27);
    u32 rot = (u32)(old_state >> 59);
    return rotr32(xorshifted, rot);
}
__device__ __forceinline__ double pcg_uniform(PCG* r) {
    u64 a = (u64)(pcg_next_u32(r) >> 5);
    u64 b = (u64)(pcg_next_u32(r) >> 6);
    return (double)(a * 67108864ULL + b) * (1.0 / 9007199254740992.0);
}
__device__ void pcg_for_particle(PCG* r, u64 batch, u64 pid) {
    u64 seed = batch * 6364136223846793005ULL + pid;
    r->inc = (pid << 1) | 1ULL;
    r->state = 0ULL;
    (void)pcg_next_u32(r);
    r->state += seed;
    (void)pcg_next_u32(r);
}

extern "C" __global__ void persistent_compton(
    double e_in, double e_cut, int max_collisions,
    u64 batch_id,
    double* __restrict__ e_dep_out,
    int* __restrict__ n_coll_out,
    int n_histories)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_histories) return;
    PCG rng; pcg_for_particle(&rng, batch_id, (u64)tid);
    const double M_E_C2 = 510998.95;
    double e = e_in;
    double e_dep = 0.0;
    int n_coll = 0;
    while (e > e_cut && n_coll < max_collisions) {
        double alpha = e / M_E_C2;
        double kappa = 1.0 + 2.0 * alpha;
        double kappa_inv = 1.0 / kappa;
        double kappa_inv_sq = kappa_inv * kappa_inv;
        double a1 = log(kappa);
        double a2 = 0.5 * (1.0 - kappa_inv_sq);
        double p1 = a1 / (a1 + a2);
        double k = 0.0, mu = 0.0;
        for (int it = 0; it < 256; ++it) {
            double xi_b = pcg_uniform(&rng);
            double xi_s = pcg_uniform(&rng);
            double xi_r = pcg_uniform(&rng);
            if (xi_b < p1) k = exp(-xi_s * a1);
            else           k = sqrt(kappa_inv_sq + xi_s * (1.0 - kappa_inv_sq));
            mu = 1.0 - (1.0 - k) / (alpha * k);
            double kn_acc = 1.0 - (1.0 - mu * mu) / (k + 1.0 / k);
            if (xi_r < kn_acc) break;
        }
        double e_out_p = e * k;
        e_dep += (e - e_out_p);
        e = e_out_p;
        n_coll++;
    }
    e_dep += e;
    e_dep_out[tid] = e_dep;
    n_coll_out[tid] = n_coll;
}
"#;

const E_IN_HIST: f64 = 1.0e6;
const E_CUT_HIST: f64 = 1.0e3;
const MAX_COLL: i32 = 64;
const M_E_C2: f64 = 510_998.95;

/// Single CPU history: Compton free-KN loop, mirrors persistent kernel.
fn cpu_one_history(rng: &mut Rng) -> (f64, i32) {
    let mut e = E_IN_HIST;
    let mut e_dep = 0.0;
    let mut n_coll = 0;
    while e > E_CUT_HIST && n_coll < MAX_COLL {
        let alpha = e / M_E_C2;
        let kappa = 1.0 + 2.0 * alpha;
        let kappa_inv = 1.0 / kappa;
        let kappa_inv_sq = kappa_inv * kappa_inv;
        let a1 = kappa.ln();
        let a2 = 0.5 * (1.0 - kappa_inv_sq);
        let p1 = a1 / (a1 + a2);
        let mut k = 0.0;
        for _ in 0..256 {
            let xi_b = rng.uniform();
            let xi_s = rng.uniform();
            let xi_r = rng.uniform();
            k = if xi_b < p1 {
                (-xi_s * a1).exp()
            } else {
                (kappa_inv_sq + xi_s * (1.0 - kappa_inv_sq)).sqrt()
            };
            let mu = 1.0 - (1.0 - k) / (alpha * k);
            let kn_acc = 1.0 - (1.0 - mu * mu) / (k + 1.0 / k);
            if xi_r < kn_acc {
                break;
            }
        }
        let e_out = e * k;
        e_dep += e - e_out;
        e = e_out;
        n_coll += 1;
    }
    e_dep += e;
    (e_dep, n_coll)
}

fn run_mode_b() {
    println!("\n=========================================================================");
    println!("Mode B — Per-history full Compton transport (apples-to-apples wall time)");
    println!("=========================================================================");
    println!(
        "N = {} histories, E_in = {} keV, E_cut = {} eV, max coll = {}",
        N,
        E_IN_HIST as u64 / 1000,
        E_CUT_HIST as u64,
        MAX_COLL
    );
    println!("Free-KN sampling on CPU and GPU; no element data, pure compute.");
    println!(
        "CPU baseline: rayon ({} threads).",
        rayon::current_num_threads()
    );

    // CPU
    let t_cpu = time(|| {
        let _: Vec<(f64, i32)> = (0..N)
            .into_par_iter()
            .map(|tid| {
                let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
                cpu_one_history(&mut rng)
            })
            .collect();
    });

    // GPU
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let ptx = nvrtc::compile_ptx(PERSISTENT_KERNEL).expect("ptx");
    let module = ctx.load_module(ptx).expect("load");
    let func = module.load_function("persistent_compton").expect("fn");

    let mut d_dep: CudaSlice<f64> = stream.alloc_zeros(N).expect("dep");
    let mut d_n: CudaSlice<i32> = stream.alloc_zeros(N).expect("n");
    let block: u32 = 256;
    let grid = (N as u32 + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i32 = N as i32;

    // Warm
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&E_IN_HIST)
            .arg(&E_CUT_HIST)
            .arg(&MAX_COLL)
            .arg(&BATCH_ID)
            .arg(&mut d_dep)
            .arg(&mut d_n)
            .arg(&n_i32)
            .launch(cfg)
            .expect("warm");
    }
    stream.synchronize().expect("sync");

    let t = Instant::now();
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&E_IN_HIST)
            .arg(&E_CUT_HIST)
            .arg(&MAX_COLL)
            .arg(&BATCH_ID)
            .arg(&mut d_dep)
            .arg(&mut d_n)
            .arg(&n_i32)
            .launch(cfg)
            .expect("launch");
    }
    let dep = stream.clone_dtoh(&d_dep).expect("dep h");
    let n_coll = stream.clone_dtoh(&d_n).expect("n h");
    let t_gpu = t.elapsed().as_secs_f64();

    let mean_n = n_coll.iter().map(|&v| v as f64).sum::<f64>() / N as f64;
    let mean_dep = dep.iter().sum::<f64>() / N as f64;
    let total_collisions = n_coll.iter().map(|&v| v as u64).sum::<u64>();

    println!();
    println!("                          CPU rayon          GPU persistent     GPU/CPU");
    println!(
        "  wall time            : {:>10.2} ms     {:>10.2} ms       {:>5.2}x",
        t_cpu * 1e3,
        t_gpu * 1e3,
        t_cpu / t_gpu
    );
    println!(
        "  µs / history         : {:>10.2}        {:>10.2}",
        t_cpu * 1e6 / N as f64,
        t_gpu * 1e6 / N as f64
    );
    println!(
        "  ns / collision       : {:>10.2}        {:>10.2}",
        t_cpu * 1e9 / total_collisions as f64,
        t_gpu * 1e9 / total_collisions as f64
    );
    println!("  total collisions     : {:>10}", total_collisions);
    println!("  mean coll / history  : {:>10.2}", mean_n);
    println!(
        "  mean E_dep / history : {:>10.0} eV (= E_in by construction)",
        mean_dep
    );
}

// ============================================================================
// main
// ============================================================================

fn main() -> ExitCode {
    let dir = match std::env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: gpu_cpu_bench <photon_data_dir>");
            return ExitCode::from(1);
        }
    };

    println!(
        "# Full CPU-vs-GPU benchmark (CPU = rayon {} threads, GPU = RTX A1000)",
        rayon::current_num_threads()
    );
    println!(
        "# N = {} per kernel-call, photon data: {}",
        N,
        dir.display()
    );

    println!("\n=========================================================================");
    println!("Mode A — Per-kernel batched (apples-to-apples per call, ns/event)");
    println!("=========================================================================");
    println!(
        "{:<32} {:>10} {:>10} {:>10}",
        "kernel", "CPU ns/ev", "GPU ns/ev", "GPU/CPU"
    );

    for &(sym, file, _) in TEST_CASES {
        let p = dir.join(file);
        let elem = match PhotonElement::from_hdf5(&p) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skip {}: {}", file, e);
                continue;
            }
        };
        for &e_in in &[1.0e6_f64, 5.0e6_f64] {
            let (c, g) = mode_a_compton(&elem, sym, e_in);
            println!(
                "{:<32} {:>10.2} {:>10.2} {:>10.2}x",
                format!("Compton[{} {:.0}MeV]", sym, e_in / 1e6),
                c,
                g,
                c / g
            );
        }
        for &e_in in &[1.0e6_f64, 5.0e6_f64] {
            let (c, g) = mode_a_compton_doppler(&elem, e_in);
            println!(
                "{:<32} {:>10.2} {:>10.2} {:>10.2}x",
                format!("Compton+Doppler[{} {:.0}MeV]", sym, e_in / 1e6),
                c,
                g,
                c / g
            );
        }
        for &e_in in &[100_000.0_f64, 1.0e6_f64] {
            let (c, g) = mode_a_rayleigh(&elem, e_in);
            println!(
                "{:<32} {:>10.2} {:>10.2} {:>10.2}x",
                format!("Rayleigh[{} {:.0}keV]", sym, e_in / 1e3),
                c,
                g,
                c / g
            );
        }
        for &e_in in &[100_000.0_f64, 1.0e6_f64] {
            let (c, g) = mode_a_photoelectric(&elem, e_in);
            println!(
                "{:<32} {:>10.2} {:>10.2} {:>10.2}x",
                format!("Photoelec[{} {:.0}keV]", sym, e_in / 1e3),
                c,
                g,
                c / g
            );
        }
    }
    for &e_in in &[2.0e6_f64, 5.0e6_f64, 20.0e6_f64] {
        let (c, g) = mode_a_pair(e_in);
        println!(
            "{:<32} {:>10.2} {:>10.2} {:>10.2}x",
            format!("Pair[{:.0}MeV]", e_in / 1e6),
            c,
            g,
            c / g
        );
    }

    run_mode_b();

    ExitCode::SUCCESS
}
