//! Per-discrete-level XS A/B between CPU and GPU.
//!
//! `metal_stats_diag` localised the metal hot bias to discrete-level
//! inelastic — CPU ⟨|Q|⟩ = 926 keV vs GPU ⟨|Q|⟩ = 659 keV on Godiva.
//! Both backends should sample levels proportional to per-level XS at
//! the incident energy. This binary verifies the per-level XS
//! computation itself: for a fixed nuclide and a sweep of E_in, it
//! prints
//!   - CPU `discrete_level_xs(nuc, E)` array (41 values for U-235)
//!   - GPU-formula evaluation on the *same* uploaded basis/coeffs
//!     (round-tripped via `clone_dtoh` from device memory)
//!   - Per-level Q values
//!   - The XS-weighted ⟨|Q|⟩ at this E (= expected mean Q per
//!     inelastic event in an analog sampling)
//!
//! A zero Δ between CPU and GPU rows + an XS-weighted ⟨|Q|⟩ close to
//! the measured 926 keV rules out per-level XS bias. A non-zero Δ
//! locates the bug in upload / basis layout / kernel reconstruction.

use std::path::PathBuf;

use open_rust_mc::transport::xs_provider::{load_nuclide, NuclideKernels, ReactionKernel};

// Energies span the bulk of the Godiva fission spectrum.
const TEST_E_EV: &[f64] = &[5.0e5, 1.0e6, 1.5e6, 2.0e6, 3.0e6, 5.0e6];

const NUCLIDE: &str = "U235.h5";

fn data_dir() -> PathBuf {
    if let Some(v) = std::env::args().nth(1) {
        let p = PathBuf::from(v);
        if p.exists() {
            return p;
        }
    }
    if let Ok(v) = std::env::var("ICSBEP_DATA_DIR") {
        return PathBuf::from(v);
    }
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("data/endfb-vii.1-hdf5/neutron").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p.join("data/endfb-vii.1-hdf5/neutron")
}

/// Single-point SVD reconstruction — bit-for-bit port of
/// `transport.cu::svd_reconstruct`. Used to check whether the
/// GPU-style XS at e_idx differs from the CPU's `kernel.lookup`.
#[cfg(feature = "cuda")]
fn gpu_svd_reconstruct(basis: &[f64], coeffs: &[f64], e_idx: usize, rank: usize) -> f64 {
    let row = &basis[e_idx * rank..e_idx * rank + rank];
    let log_val: f64 = row.iter().zip(coeffs.iter()).map(|(a, b)| a * b).sum();
    f64::exp2(log_val * std::f64::consts::LOG2_10)
}

/// Mirror of the GPU's `energy_index` — binary search returning the
/// largest index with `grid[idx] <= energy`.
#[cfg(feature = "cuda")]
fn gpu_energy_index(grid: &[f64], energy: f64) -> usize {
    let n = grid.len();
    if energy <= grid[0] {
        return 0;
    }
    if energy >= grid[n - 1] {
        return n - 1;
    }
    let (mut lo, mut hi) = (0_usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if grid[mid] <= energy {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn cpu_level_xs_and_q(kern: &NuclideKernels, energy: f64) -> Vec<(usize, u32, f64, f64, f64)> {
    // Returns Vec<(level_idx, mt, threshold, q_value, xs)>
    let mut out = Vec::with_capacity(kern.discrete_levels.len());
    for (idx, lvl) in kern.discrete_levels.iter().enumerate() {
        let xs = if energy < lvl.info.threshold {
            0.0
        } else {
            lvl.kernel.as_ref().map_or(0.0, |k| k.lookup(energy))
        };
        out.push((idx, lvl.info.mt, lvl.info.threshold, lvl.info.q_value, xs));
    }
    out
}

fn xs_weighted_mean_q(rows: &[(usize, u32, f64, f64, f64)]) -> (f64, f64) {
    // Returns (xs_sum, ⟨|Q|⟩). Excludes MT=91 (continuum) since its
    // Q is sampled per-event, not from level table.
    let mut xs_sum = 0.0;
    let mut q_sum = 0.0;
    for &(_, mt, _, q, xs) in rows {
        if mt == 91 || xs <= 0.0 {
            continue;
        }
        xs_sum += xs;
        q_sum += xs * q.abs();
    }
    if xs_sum > 0.0 {
        (xs_sum, q_sum / xs_sum)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(feature = "cuda")]
fn gpu_round_trip_check(kern: &NuclideKernels) -> Result<(), Box<dyn std::error::Error>> {
    use open_rust_mc::gpu_transport::GpuTransportContext;

    let gpu = GpuTransportContext::new()?;
    let nuc_data = gpu.upload_nuclide_data(std::slice::from_ref(kern), 15)?;

    // Pull back the level basis/coeffs/offsets and the grid.
    let energies_h = gpu.stream().clone_dtoh(&nuc_data.all_energy_grids)?;
    let lev_basis_h = gpu.stream().clone_dtoh(&nuc_data.level_basis)?;
    let lev_coeffs_h = gpu.stream().clone_dtoh(&nuc_data.level_coeffs)?;
    let lev_boff_h = gpu.stream().clone_dtoh(&nuc_data.level_basis_offsets)?;
    let lev_coff_h = gpu.stream().clone_dtoh(&nuc_data.level_coeffs_offsets)?;
    let lev_thr_h = gpu.stream().clone_dtoh(&nuc_data.level_thresholds)?;
    let lev_mt_h = gpu.stream().clone_dtoh(&nuc_data.level_mt)?;
    let lev_q_h = gpu.stream().clone_dtoh(&nuc_data.level_q_values)?;
    let lev_has_k_h = gpu.stream().clone_dtoh(&nuc_data.level_has_kernel)?;
    let rank = nuc_data.rank as usize;
    let grid_offset = gpu
        .stream()
        .clone_dtoh(&nuc_data.grid_offsets)?[0] as usize;
    let n_e = gpu.stream().clone_dtoh(&nuc_data.n_energies)?[0] as usize;
    let grid = &energies_h[grid_offset..grid_offset + n_e];

    println!(
        "\n  uploaded: rank={rank}  grid_n={}  level_basis_pts={}  level_coeffs_pts={}",
        n_e,
        lev_basis_h.len(),
        lev_coeffs_h.len()
    );

    for &e in TEST_E_EV {
        let e_idx = gpu_energy_index(grid, e);
        println!(
            "\n  ─── E = {:.3e} eV   (e_idx = {}, grid_lo = {:.3e}, grid_hi = {:.3e}) ───",
            e,
            e_idx,
            grid[e_idx],
            if e_idx + 1 < grid.len() {
                grid[e_idx + 1]
            } else {
                grid[e_idx]
            }
        );
        let cpu_rows = cpu_level_xs_and_q(kern, e);

        let mut worst_rel = 0.0_f64;
        let mut worst_label = String::new();
        let mut xs_sum_cpu = 0.0;
        let mut xs_sum_gpu = 0.0;
        let mut q_cpu = 0.0;
        let mut q_gpu = 0.0;
        let mut header_done = false;
        for (li, &(_, mt, thr, q, xs_cpu)) in cpu_rows.iter().enumerate() {
            let has_k = lev_has_k_h[li] != 0;
            let xs_gpu = if e >= lev_thr_h[li] && has_k {
                let boff = lev_boff_h[li] as usize;
                let coff = lev_coff_h[li] as usize;
                gpu_svd_reconstruct(
                    &lev_basis_h[boff..],
                    &lev_coeffs_h[coff..coff + rank],
                    e_idx,
                    rank,
                )
            } else {
                0.0
            };
            // Sanity check: GPU-uploaded Q vs CPU's Q.
            let q_dev = lev_q_h[li];
            let mt_dev = lev_mt_h[li] as u32;
            if mt != 91 && (xs_cpu > 0.0 || xs_gpu > 0.0) {
                let rel = if xs_cpu > 0.0 {
                    ((xs_gpu - xs_cpu) / xs_cpu).abs()
                } else {
                    1.0
                };
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_label = format!(
                        "level {li} (MT={mt}, Q={:.3e}): cpu={:.3e}, gpu={:.3e}, Δ={:+.2e} ({:+.2}%)",
                        q, xs_cpu, xs_gpu, xs_gpu - xs_cpu, (xs_gpu - xs_cpu) / xs_cpu * 100.0
                    );
                }
                if !header_done && (rel > 0.01 || li < 6) {
                    println!(
                        "    {:>3} {:>4} {:>10} {:>10}  {:>12} {:>12}  {:>9}",
                        "lev", "MT", "thr", "Q", "cpu_xs", "gpu_xs", "Δ%"
                    );
                    header_done = true;
                }
                if rel > 0.01 || li < 6 {
                    println!(
                        "    {:>3} {:>4} {:>10.3e} {:>10.3e}  {:>12.4e} {:>12.4e}  {:>+9.3}%",
                        li, mt_dev, thr, q_dev, xs_cpu, xs_gpu,
                        if xs_cpu > 0.0 { (xs_gpu - xs_cpu) / xs_cpu * 100.0 } else { 0.0 }
                    );
                }
                if mt != 91 {
                    xs_sum_cpu += xs_cpu;
                    xs_sum_gpu += xs_gpu;
                    q_cpu += xs_cpu * q.abs();
                    q_gpu += xs_gpu * q.abs();
                }
            }
        }
        let (_, mean_q_cpu_from_rows) = xs_weighted_mean_q(&cpu_rows);
        println!(
            "    Σxs    cpu={:.4e}  gpu={:.4e}   Δ={:+.2}%",
            xs_sum_cpu,
            xs_sum_gpu,
            if xs_sum_cpu > 0.0 {
                (xs_sum_gpu - xs_sum_cpu) / xs_sum_cpu * 100.0
            } else {
                0.0
            }
        );
        let mq_cpu = if xs_sum_cpu > 0.0 { q_cpu / xs_sum_cpu } else { 0.0 };
        let mq_gpu = if xs_sum_gpu > 0.0 { q_gpu / xs_sum_gpu } else { 0.0 };
        println!(
            "    ⟨|Q|⟩  cpu={:.4e} eV  gpu={:.4e} eV  Δ={:+.2}%   (cross-check via rows: {:.4e})",
            mq_cpu,
            mq_gpu,
            if mq_cpu > 0.0 {
                (mq_gpu - mq_cpu) / mq_cpu * 100.0
            } else {
                0.0
            },
            mean_q_cpu_from_rows
        );
        println!("    Worst per-level Δ: {worst_rel:.3e}");
        if !worst_label.is_empty() {
            println!("        {worst_label}");
        }
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn gpu_round_trip_check(_kern: &NuclideKernels) -> Result<(), Box<dyn std::error::Error>> {
    println!("(CUDA feature disabled; build with --features cuda)");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = data_dir();
    let path = dir.join(NUCLIDE);
    println!("Loading {}", path.display());
    let kern = load_nuclide(&path, 15, 0, 233.025, 2.43);

    println!("\nLevel inventory ({} discrete levels):", kern.discrete_levels.len());
    for (li, lvl) in kern.discrete_levels.iter().enumerate().take(12) {
        let kernel_kind = lvl.kernel.as_ref().map_or("none", |k| match k {
            ReactionKernel::Svd { .. } => "Svd",
            ReactionKernel::Table { .. } => "Table",
        });
        println!(
            "  [{li:>3}] MT={:>2} thr={:.3e} eV  Q={:.3e} eV  kernel={kernel_kind}",
            lvl.info.mt, lvl.info.threshold, lvl.info.q_value
        );
    }
    if kern.discrete_levels.len() > 12 {
        println!("  ... ({} more)", kern.discrete_levels.len() - 12);
    }

    println!("\n=== CPU XS-weighted ⟨|Q|⟩ over discrete (non-MT=91) levels ===");
    for &e in TEST_E_EV {
        let rows = cpu_level_xs_and_q(&kern, e);
        let (xs_sum, mean_q) = xs_weighted_mean_q(&rows);
        let n_acc = rows
            .iter()
            .filter(|(_, mt, _, _, xs)| *xs > 0.0 && *mt != 91)
            .count();
        println!(
            "  E = {:.3e} eV: Σxs = {:.4e} barn, ⟨|Q|⟩ = {:.4e} eV  ({} accessible non-91 levels)",
            e, xs_sum, mean_q, n_acc
        );
    }

    gpu_round_trip_check(&kern)?;
    Ok(())
}
