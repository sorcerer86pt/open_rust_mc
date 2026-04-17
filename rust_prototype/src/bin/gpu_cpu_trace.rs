//! GPU vs CPU transport trace — step-by-step comparison.
//!
//! Traces N particles for M steps on both GPU and CPU with identical initial
//! conditions and RNG seeds, writes per-step data to CSV for diff analysis.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_cpu_trace -- <data_dir> \
//!     --particles 100 --steps 500 --geometry pwr

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() { cuda_main::run(); }

#[cfg(feature = "cuda")]
mod cuda_main {
    use std::io::Write;
    use std::path::PathBuf;
    use clap::Parser;
    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::transport::xs_provider;
    use open_rust_mc::transport::material::Material;

    #[derive(Parser)]
    struct Args {
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 6)]
        rank: usize,
        #[arg(short, long, default_value_t = 100)]
        particles: usize,
        #[arg(short, long, default_value_t = 500)]
        steps: usize,
        #[arg(short, long, default_value = "pwr")]
        geometry: String,
    }

    const PWR_NUCLIDES: &[(&str, f64, f64, usize)] = &[
        ("U235.h5", 233.025, 2.43, 3),
        ("U238.h5", 236.006, 2.49, 3),
        ("O16.h5",  15.858,  0.0,  3),
        ("H1.h5",    0.999,  0.0,  2),
        ("Zr90.h5", 89.132,  0.0,  2),
        ("Zr91.h5", 90.130,  0.0,  2),
        ("Zr92.h5", 91.126,  0.0,  2),
        ("Zr94.h5", 93.120,  0.0,  2),
        ("O16.h5",  15.858,  0.0,  2),
    ];

    fn setup_materials() -> Vec<Material> {
        let mut fuel = Material::new("UO2", 900.0);
        fuel.add_nuclide(0.000719, 0);
        fuel.add_nuclide(0.022482, 1);
        fuel.add_nuclide(0.046402, 2);
        let mut clad = Material::new("Zircaloy", 600.0);
        clad.add_nuclide(0.022932, 4);
        clad.add_nuclide(0.004996, 5);
        clad.add_nuclide(0.007636, 6);
        clad.add_nuclide(0.007740, 7);
        let mut water = Material::new("H2O", 600.0);
        water.add_nuclide(0.049486, 3);
        water.add_nuclide(0.024743, 8);
        vec![fuel, clad, water]
    }

    fn initial_source(n: usize) -> Vec<(f64, f64, f64, f64)> {
        use open_rust_mc::transport::rng::Rng;
        let fuel_or = 0.4096_f64;
        let half = 0.63_f64;
        let mut rng = Rng::new(42, 0);
        let mut sites = Vec::with_capacity(n);
        while sites.len() < n {
            let x = -fuel_or + rng.uniform() * 2.0 * fuel_or;
            let y = -fuel_or + rng.uniform() * 2.0 * fuel_or;
            let z = -half + rng.uniform() * 2.0 * half;
            if x * x + y * y < fuel_or * fuel_or {
                sites.push((x, y, z, 1.0e6));
            }
        }
        sites
    }

    pub fn run() {
        let args = Args::parse();
        let n = args.particles;
        let max_steps = args.steps;
        let geom_type: i32 = if args.geometry == "godiva" { 1 } else { 0 };

        println!("=== GPU vs CPU Transport Trace ===");
        println!("Particles: {n}, Steps: {max_steps}, Geometry: {}", args.geometry);

        // Load nuclear data
        println!("\n── Loading nuclear data ──");
        let mut kernels = Vec::new();
        for &(filename, awr, nu_bar, nuc_temp_idx) in PWR_NUCLIDES {
            let path = args.data_dir.join(filename);
            kernels.push(xs_provider::load_nuclide(&path, args.rank, nuc_temp_idx, awr, nu_bar));
        }

        // Init GPU
        let gpu = GpuTransportContext::new().expect("GPU init");
        let nuc_data = gpu.upload_nuclide_data(&kernels, args.rank).expect("upload nuc");
        let materials = setup_materials();
        let awrs: Vec<f64> = PWR_NUCLIDES.iter().map(|s| s.1).collect();
        let nu_bars: Vec<f64> = PWR_NUCLIDES.iter().map(|s| s.2).collect();
        let mat_data = gpu.upload_material_data(&materials, &awrs, &nu_bars).expect("upload mat");

        let h2o_path = args.data_dir.join("c_H_in_H2O.h5");
        let sab_data = if h2o_path.exists() {
            match open_rust_mc::hdf5_reader::load_thermal_scattering(&h2o_path) {
                Ok(tsl) => {
                    let t_idx = tsl.select_temperature(600.0, 0.5);
                    gpu.upload_sab_data(&tsl, t_idx).expect("upload sab")
                }
                Err(_) => gpu.upload_sab_data_empty().expect("empty sab"),
            }
        } else {
            gpu.upload_sab_data_empty().expect("empty sab")
        };

        let source = initial_source(n);

        // Run GPU trace
        println!("\n── Running GPU trace ({n} particles, {max_steps} steps) ──");
        let gpu_trace = gpu.run_debug_trace(
            &source, &nuc_data, &mat_data, &sab_data,
            max_steps as u32, geom_type,
        ).expect("GPU trace");

        // Write GPU trace to CSV
        let gpu_file = "gpu_trace.csv";
        {
            let mut f = std::fs::File::create(gpu_file).expect("create gpu csv");
            writeln!(f, "particle,step,energy,pos_x,pos_y,pos_z,cell,material,macro_total,d_coll,d_surf,event,hit_nuc,micro_el,micro_inel,micro_fis,micro_cap,out_energy,rng_xi").unwrap();
            let cols = 17;
            for pid in 0..n {
                let steps = gpu_trace.step_counts[pid] as usize;
                for s in 0..steps {
                    let base = pid * max_steps * cols + s * cols;
                    write!(f, "{pid},{s}").unwrap();
                    for c in 0..cols {
                        write!(f, ",{:.8e}", gpu_trace.data[base + c]).unwrap();
                    }
                    writeln!(f).unwrap();
                }
            }
        }
        println!("  Wrote {gpu_file}");

        // Run CPU trace with same source
        println!("\n── Running CPU trace ({n} particles, {max_steps} steps) ──");
        let cpu_file = "cpu_trace.csv";
        // TODO: implement CPU step-by-step trace with same output format
        // For now, just write a summary
        println!("  CPU trace: TODO (use existing CPU transport with logging)");
        println!("\n── Done. Compare {gpu_file} with CPU output. ──");

        // Quick summary stats
        let mut event_counts = [0u64; 10];
        let cols = 17;
        for pid in 0..n {
            let steps = gpu_trace.step_counts[pid] as usize;
            for s in 0..steps {
                let base = pid * max_steps * cols + s * cols;
                let ev = gpu_trace.data[base + 9] as usize;
                if ev < 10 { event_counts[ev] += 1; }
            }
        }
        println!("\nGPU event summary:");
        let names = ["elastic","inelastic","n2n","n3n","fission","capture","reflect","transmit","leak","void"];
        for (i, name) in names.iter().enumerate() {
            if event_counts[i] > 0 {
                println!("  {name:>12}: {}", event_counts[i]);
            }
        }
    }
}
