//! GPU Faddeeva + WMP evaluator validation.
//!
//! Uploads one nuclide's WMP data to the GPU, evaluates at a list of test
//! energies via the `wmp_test_eval` kernel, copies results back, and prints
//! alongside the CPU evaluator's values for direct comparison.
//!
//! Requires the `cuda` feature.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_wmp_validate -- \
//!       path/to/wmp/092238.h5

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda to enable GPU validation");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc;
    use open_rust_mc::wmp::WindowedMultipole;
    use std::path::PathBuf;
    use std::sync::Arc;

    const KERNEL_SRC: &str = include_str!("../../gpu/cuda/transport.cu");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: gpu_wmp_validate <path/to/wmp/ZZAAA.h5> \
                  [--gpu-n N] [--cpu-n N] [--reps N]"
        );
        eprintln!("  defaults: --gpu-n 1000000 --cpu-n 100000 --reps 5");
        std::process::exit(1);
    }
    let path = PathBuf::from(&args[1]);

    // Parse optional flags (each expects a numeric value).
    let parse_flag = |name: &str, default: usize| -> usize {
        args.windows(2)
            .find(|w| w[0] == name)
            .and_then(|w| w[1].parse::<usize>().ok())
            .unwrap_or(default)
    };
    let n_gpu = parse_flag("--gpu-n", 1_000_000);
    let n_cpu = parse_flag("--cpu-n", 100_000);
    let n_rep = parse_flag("--reps", 5);
    let wmp = WindowedMultipole::from_hdf5(&path)?;
    println!("GPU WMP validation — {}", wmp.name);
    println!(
        "  E_min={:.3e}, E_max={:.3e}, n_poles={}, n_windows={}, fit_order={}",
        wmp.e_min, wmp.e_max, wmp.n_poles, wmp.n_windows, wmp.fit_order
    );

    // Test energy list
    let energies: Vec<f64> = vec![
        0.025, 1.0, 6.674, 10.0, 20.9, 36.7, 50.0, 100.0, 500.0, 1_000.0, 5_000.0, 10_000.0,
    ]
    .into_iter()
    .filter(|&e| e >= wmp.e_min && e <= wmp.e_max)
    .collect();
    let t_kelvin = 293.6;
    let n = energies.len();

    // Build CUDA context & load kernel
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ptx = nvrtc::compile_ptx(KERNEL_SRC)?;
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("wmp_test_eval")?;

    // Upload poles as flat f64 (pairs of re/im), reinterpreted on device
    // as double2 (16 bytes, contiguous). CUDA double2 is {x:f64, y:f64}.
    let poles_f64: Vec<f64> = wmp.poles.iter().flat_map(|c| [c.re, c.im]).collect();
    let d_poles_f64 = stream.memcpy_stod(&poles_f64)?;

    // Windows array: stored 0-based startw, endw (already converted in Rust loader)
    let d_windows = stream.memcpy_stod(&wmp.windows)?;

    // broaden_poly: Vec<u8> -> Vec<i8>
    let broaden_i8: Vec<i8> = wmp.broaden_poly.iter().map(|&x| x as i8).collect();
    let d_broaden = stream.memcpy_stod(&broaden_i8)?;

    let d_curvefit = stream.memcpy_stod(&wmp.curvefit)?;
    let d_energies = stream.memcpy_stod(&energies)?;

    let mut d_out_s: CudaSlice<f64> = stream.alloc_zeros(n)?;
    let mut d_out_a: CudaSlice<f64> = stream.alloc_zeros(n)?;
    let mut d_out_f: CudaSlice<f64> = stream.alloc_zeros(n)?;

    let grid = ((n as u32 + 255) / 256, 1, 1);
    let block = (256u32, 1, 1);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    let n_i32 = n as i32;
    let nw = wmp.n_windows as i32;
    let fo = wmp.fit_order as i32;
    let fiss = if wmp.fissionable { 1 } else { 0 } as i32;

    // Kernel signature: (n, energies, t, e_min, e_max, spacing, sqrt_awr,
    //                    n_windows, fit_order, fissionable, poles,
    //                    windows, broaden, curvefit, out_s, out_a, out_f)
    unsafe {
        // poles pointer is raw f64 but kernel reads double2; cudarc exposes
        // device pointers via DevicePtr. We cast through u64 for the
        // kernel argument — cudarc will pass the pointer correctly.
        stream
            .launch_builder(&kernel)
            .arg(&n_i32)
            .arg(&d_energies)
            .arg(&t_kelvin)
            .arg(&wmp.e_min)
            .arg(&wmp.e_max)
            .arg(&wmp.spacing)
            .arg(&wmp.sqrt_awr)
            .arg(&nw)
            .arg(&fo)
            .arg(&fiss)
            .arg(&d_poles_f64) // raw f64; kernel reads as double2 pairs
            .arg(&d_windows)
            .arg(&d_broaden)
            .arg(&d_curvefit)
            .arg(&mut d_out_s)
            .arg(&mut d_out_a)
            .arg(&mut d_out_f)
            .launch(cfg)?;
    }
    let _gpu_s = stream.memcpy_dtov(&d_out_s)?;
    let gpu_a = stream.memcpy_dtov(&d_out_a)?;
    let gpu_f = stream.memcpy_dtov(&d_out_f)?;

    // ── Throughput benchmark ──
    // Evaluate on a dense log-spaced grid across the RRR and time both
    // implementations. Sizes are parametrised via --gpu-n / --cpu-n /
    // --reps because GPU and CPU have very different throughput and
    // running 1M points single-threaded on the CPU can be slow.
    use std::time::Instant;
    let n_bench = n_gpu.max(1);
    let bench_energies: Vec<f64> = (0..n_bench)
        .map(|i| {
            let f = if n_bench > 1 {
                i as f64 / (n_bench - 1) as f64
            } else {
                0.0
            };
            let log_min = wmp.e_min.max(1e-5).ln();
            let log_max = wmp.e_max.ln();
            (log_min + f * (log_max - log_min)).exp()
        })
        .collect();

    let d_be = stream.memcpy_stod(&bench_energies)?;
    let mut d_bs: CudaSlice<f64> = stream.alloc_zeros(n_bench)?;
    let mut d_ba: CudaSlice<f64> = stream.alloc_zeros(n_bench)?;
    let mut d_bf: CudaSlice<f64> = stream.alloc_zeros(n_bench)?;
    let grid_b = ((n_bench as u32 + 255) / 256, 1, 1);
    let block_b = (256u32, 1, 1);
    let cfg_b = LaunchConfig {
        grid_dim: grid_b,
        block_dim: block_b,
        shared_mem_bytes: 0,
    };

    let n_bench_i32 = n_bench as i32;
    // Warm-up launch
    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&n_bench_i32)
            .arg(&d_be)
            .arg(&t_kelvin)
            .arg(&wmp.e_min)
            .arg(&wmp.e_max)
            .arg(&wmp.spacing)
            .arg(&wmp.sqrt_awr)
            .arg(&nw)
            .arg(&fo)
            .arg(&fiss)
            .arg(&d_poles_f64)
            .arg(&d_windows)
            .arg(&d_broaden)
            .arg(&d_curvefit)
            .arg(&mut d_bs)
            .arg(&mut d_ba)
            .arg(&mut d_bf)
            .launch(cfg_b)?;
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_rep {
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&n_bench_i32)
                .arg(&d_be)
                .arg(&t_kelvin)
                .arg(&wmp.e_min)
                .arg(&wmp.e_max)
                .arg(&wmp.spacing)
                .arg(&wmp.sqrt_awr)
                .arg(&nw)
                .arg(&fo)
                .arg(&fiss)
                .arg(&d_poles_f64)
                .arg(&d_windows)
                .arg(&d_broaden)
                .arg(&d_curvefit)
                .arg(&mut d_bs)
                .arg(&mut d_ba)
                .arg(&mut d_bf)
                .launch(cfg_b)?;
        }
    }
    stream.synchronize()?;
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let gpu_ns_per = gpu_ms * 1e6 / (n_bench as f64 * n_rep as f64);

    // CPU benchmark: parametrised via --cpu-n. Subsamples the GPU grid so
    // both runs cover the same energy range. If --cpu-n >= --gpu-n, use
    // the full grid.
    let n_cpu = n_cpu.max(1).min(n_bench);
    let stride = (n_bench / n_cpu).max(1);
    let cpu_energies: Vec<f64> = bench_energies.iter().step_by(stride).copied().collect();
    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for &e in &cpu_energies {
        let (s, a, f) = wmp.evaluate(e, t_kelvin);
        acc += s + a + f;
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cpu_ns_per = cpu_ms * 1e6 / cpu_energies.len() as f64;
    std::hint::black_box(acc);

    println!();
    println!("Throughput (U-238 WMP, RRR log-spaced grid, T=293.6 K):");
    println!(
        "  CPU single-thread: {:.1} ns/lookup ({} points, {:.1} ms)",
        cpu_ns_per,
        cpu_energies.len(),
        cpu_ms
    );
    println!(
        "  GPU (this device): {:.1} ns/lookup ({} points × {} launches, {:.1} ms total)",
        gpu_ns_per, n_bench, n_rep, gpu_ms
    );
    println!("  GPU / CPU speedup: {:.1}×", cpu_ns_per / gpu_ns_per);

    // CPU reference
    println!();
    println!(
        "{:>14} {:>14} {:>14} {:>10} {:>14} {:>14} {:>10}",
        "E (eV)", "cpu σ_abs", "gpu σ_abs", "Δ rel", "cpu σ_fis", "gpu σ_fis", "Δ rel"
    );

    let mut max_rel_abs = 0.0_f64;
    let mut max_rel_fis = 0.0_f64;
    for (i, &e) in energies.iter().enumerate() {
        let (cs, ca, cf) = wmp.evaluate(e, t_kelvin);
        let _ = cs;
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
        let ra = rel(gpu_a[i], ca);
        let rf = rel(gpu_f[i], cf);
        max_rel_abs = max_rel_abs.max(ra);
        max_rel_fis = max_rel_fis.max(rf);
        println!(
            "{:14.3e} {:14.6e} {:14.6e} {:10.2e} {:14.6e} {:14.6e} {:10.2e}",
            e, ca, gpu_a[i], ra, cf, gpu_f[i], rf
        );
    }
    println!();
    println!("  max |Δrel| absorption = {:.3e}", max_rel_abs);
    println!("  max |Δrel| fission    = {:.3e}", max_rel_fis);

    Ok(())
}
