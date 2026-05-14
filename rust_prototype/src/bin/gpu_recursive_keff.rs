//! GPU recursive transport with real XS — smoke + parity test (task #22, stage 3b).
//!
//! Runs `transport_recursive_persistent` on a recursive Godiva geometry
//! (single-sphere universe) and compares aggregate counts against the
//! existing `transport_persistent` kernel running on the *hardcoded*
//! Godiva path. Both consume the same SVD/Pointwise/WMP/URR/SAB device
//! buffers uploaded once; the only difference is which geometry hook
//! the kernel walks.
//!
//! Bit-exact agreement is **not** expected — the recursive kernel
//! advances RNG via `pcg_uniform` in a different order (initial cell
//! find, surface re-resolution after every transmission) — but
//! aggregate collisions / fissions / leakage / bank size should agree
//! within a few-σ MC envelope, and k_eff should agree within MC noise.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_recursive_keff -- \
//!     <data_dir> --particles 50000 --rank 5

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: this binary requires the 'cuda' feature.");
    eprintln!("Build with: cargo run --release --features cuda --bin gpu_recursive_keff -- ...");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    cuda_main::run();
}

#[cfg(feature = "cuda")]
mod cuda_main {
    use std::path::PathBuf;
    use std::time::Instant;

    use clap::Parser;

    use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
    use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
    use open_rust_mc::geometry::universe::{Universe, UniverseId};
    use open_rust_mc::geometry::{Geometry, Vec3};
    use open_rust_mc::gpu_recursive::GpuRecursiveContext;
    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::transport::material::Material;
    use open_rust_mc::transport::xs_provider;
    use rust_mc_sim::Pcg64;

    #[derive(Parser)]
    #[command(name = "gpu_recursive_keff")]
    struct Args {
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 5)]
        rank: usize,
        #[arg(short, long, default_value_t = 50_000)]
        particles: usize,
        #[arg(short, long, default_value_t = 5_000)]
        max_events: i32,
    }

    /// Recursive Godiva: one universe with two cells (inside / outside
    /// of an 8.7407 cm sphere with vacuum BC). Single material → mat 0.
    fn build_recursive_godiva() -> Geometry {
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 8.7407,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];
        let universes = vec![Universe::new(UniverseId(0), vec![0, 1])];
        Geometry::new(surfaces, cells, universes, vec![], UniverseId(0))
            .expect("build recursive Godiva")
    }

    /// Uniform isotropic source inside the Godiva sphere.
    fn make_source(n: usize, seed: u64) -> Vec<(f64, f64, f64, f64)> {
        let r = 8.7407_f64;
        let mut rng = Pcg64::new(seed, 0);
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let x = -r + 2.0 * r * rng.uniform();
            let y = -r + 2.0 * r * rng.uniform();
            let z = -r + 2.0 * r * rng.uniform();
            if x * x + y * y + z * z < r * r {
                out.push((x, y, z, 1.0e6));
            }
        }
        out
    }

    pub fn run() {
        let args = Args::parse();
        let n = args.particles;

        println!("=== GPU recursive transport with real XS ===");
        println!("  data dir   : {}", args.data_dir.display());
        println!("  particles  : {n}");
        println!("  SVD rank   : {}", args.rank);
        println!("  max events : {}", args.max_events);

        // ── Load Godiva nuclides (U-234 / U-235 / U-238 at 294 K) ──
        println!("\n── Loading nuclear data ──");
        let nuclide_specs: &[(&str, f64, f64, usize)] = &[
            ("U234.h5", 232.029, 2.49, 1),
            ("U235.h5", 233.025, 2.43, 1),
            ("U238.h5", 236.006, 2.49, 1),
        ];
        let t0 = Instant::now();
        let mut kernels = Vec::new();
        for &(filename, awr, nu_bar, t_idx) in nuclide_specs {
            let path = args.data_dir.join(filename);
            println!("  loading {filename}...");
            kernels.push(std::sync::Arc::new(xs_provider::load_nuclide(
                &path, args.rank, t_idx, awr, nu_bar,
            )));
        }
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  loaded in {load_ms:.0} ms");

        // ── Initialise GPU ──
        let t1 = Instant::now();
        let gpu = GpuTransportContext::new().expect("GpuTransportContext::new");
        let nuc_data = gpu
            .upload_nuclide_data(&kernels, args.rank)
            .expect("upload nuclide data");
        let mut heu = Material::new("HEU", 294.0);
        heu.add_nuclide(0.000_483, 0); // U-234
        heu.add_nuclide(0.045_09, 1); // U-235
        heu.add_nuclide(0.002_65, 2); // U-238
        let materials = vec![heu];
        let awrs: Vec<f64> = nuclide_specs.iter().map(|s| s.1).collect();
        let nu_bars: Vec<f64> = nuclide_specs.iter().map(|s| s.2).collect();
        let mat_data = gpu
            .upload_material_data(&materials, &awrs, &nu_bars)
            .expect("upload material data");
        let sab_data = gpu
            .upload_sab_data_empty(nuclide_specs.len())
            .expect("upload empty S(α,β)");
        let wmp_data = gpu
            .upload_wmp_data_empty(nuclide_specs.len())
            .expect("upload empty WMP");
        let gpu_init_ms = t1.elapsed().as_secs_f64() * 1000.0;
        println!("  GPU ready in {gpu_init_ms:.0} ms");

        // ── Hardcoded-Godiva reference run (existing kernel) ──
        println!("\n── Reference: transport_persistent (geom_type=GODIVA) ──");
        let source = make_source(n, 0xDEAD_BEEF);
        let t2 = Instant::now();
        let ref_result = gpu
            .run_batch(
                &source,
                1,
                &nuc_data,
                &mat_data,
                &sab_data,
                &wmp_data,
                args.max_events as u32,
                /* GEOM_GODIVA */ 1,
            )
            .expect("run_batch");
        let ref_ms = t2.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  k_eff       = {:.5}\n  collisions  = {}\n  fissions    = {}\n  leakage     = {}",
            ref_result.k_eff, ref_result.collisions, ref_result.fissions, ref_result.leakage
        );
        println!(
            "  bank size   = {} (sites)\n  time        = {:.0} ms",
            ref_result.fission_bank.len(),
            ref_ms
        );

        // ── Build recursive geometry + recursive context ──
        println!("\n── Recursive: transport_recursive_persistent ──");
        let geom = build_recursive_godiva();
        let rec = GpuRecursiveContext::build(&geom, n).expect("GpuRecursiveContext::build");

        // Per-particle RNG seeds.
        let rng_seeds: Vec<(u64, u64)> = (0..n)
            .map(|i| {
                let p = Pcg64::for_particle(0, i as u64);
                (p.state(), p.stream())
            })
            .collect();

        // Single material at 294 K → kT = 294 × 8.617e-5 ≈ 2.534e-2 eV.
        let mat_k_t: Vec<f64> = vec![294.0 * 8.617_333_262e-5];
        // No S(α,β) on Godiva.
        let sab_nuc_idx: i32 = -1;

        let t3 = Instant::now();
        let rec_result = rec
            .transport_recursive(
                &gpu,
                &nuc_data,
                &mat_data,
                &sab_data,
                &wmp_data,
                &source,
                &rng_seeds,
                &mat_k_t,
                sab_nuc_idx,
                args.max_events,
                n * 4,
            )
            .expect("transport_recursive");
        let rec_ms = t3.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  k_eff       = {:.5}\n  collisions  = {}\n  fissions    = {}\n  leakage     = {}",
            rec_result.k_eff, rec_result.n_collisions, rec_result.n_fissions, rec_result.n_leakage
        );
        println!(
            "  bank size   = {} (sites)\n  time        = {:.0} ms",
            rec_result.fission_bank.len(),
            rec_ms
        );

        // ── Comparison ──
        println!("\n── Comparison ──");
        let cmp = |label: &str, c: u64, g: u64| {
            let diff = (c as f64 - g as f64).abs();
            let scale = (c as f64).max(g as f64).max(1.0);
            let pct = diff / scale * 100.0;
            println!(
                "  {label:<14} hardcoded {c:>10}   recursive {g:>10}   Δ {diff:>9.0} ({pct:.3}%)"
            );
        };
        cmp(
            "collisions",
            ref_result.collisions as u64,
            rec_result.n_collisions,
        );
        cmp(
            "fissions",
            ref_result.fissions as u64,
            rec_result.n_fissions,
        );
        cmp("leakage", ref_result.leakage as u64, rec_result.n_leakage);
        cmp(
            "bank size",
            ref_result.fission_bank.len() as u64,
            rec_result.fission_bank.len() as u64,
        );
        let k_diff_pcm = (ref_result.k_eff - rec_result.k_eff).abs() * 1e5;
        println!(
            "\n  k(hardcoded) = {:.5}, k(recursive) = {:.5}, |Δ| = {:.0} pcm",
            ref_result.k_eff, rec_result.k_eff, k_diff_pcm
        );
        println!(
            "  speedup recursive / hardcoded = {:.2}x",
            ref_ms / rec_ms.max(1e-3)
        );
    }
}
