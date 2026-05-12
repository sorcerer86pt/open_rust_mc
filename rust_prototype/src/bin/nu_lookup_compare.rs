//! ν̄(E) CPU↔GPU A/B diagnostic.
//!
//! Investigates whether the metal-hot bias on Godiva/Jezebel is a ν
//! table problem. `metal_stats_diag` showed GPU ⟨E_in fission⟩ =
//! 1.611 MeV vs OpenMC 1.826 MeV — GPU spectrum is *softer* — yet
//! GPU ⟨ν⟩/fission = 2.624 vs OpenMC 2.598 — GPU is *higher*. For a
//! shared ν(E) curve this is Jensen-impossible: a softer spectrum on
//! a monotone-increasing ν(E) must yield a *lower* ⟨ν⟩. So either
//! the GPU sees a different ν(E) (table upload bug, slot misalignment,
//! prompt+delayed double-count, …) or the E_in distribution at fission
//! has a tail that compensates.
//!
//! This binary rules out the first half. For each of U-234/235/238:
//!   1. Load `NuclideKernels` exactly as the main pipeline does.
//!   2. Compute `ν̄_cpu(E) = nuc.nu_bar_at(E)` at a dense E grid.
//!   3. With `--features cuda`: upload the nuclides via
//!      `GpuTransportContext::upload_nuclide_data`, copy back the
//!      packed `nu_bar_{energies,values,offsets,sizes}` buffers, and
//!      evaluate the GPU `nu_bar_lookup` formula in Rust on the
//!      *round-tripped* bytes.
//!   4. Print the table per nuclide and flag the worst |Δ|.
//!
//! A zero (or sub-ppm) Δ rules out table upload corruption AND slot
//! misalignment as causes of the +500-700 pcm gap. A non-zero Δ is
//! the bug.

use std::path::PathBuf;

use open_rust_mc::transport::xs_provider::{load_nuclide, NuclideKernels};

const TEST_E_EV: &[f64] = &[
    2.53e-2, 1.0e3, 1.0e4, 1.0e5, 5.0e5, 1.0e6, 1.611e6, 1.826e6, 2.0e6, 3.0e6, 5.0e6, 1.0e7,
    1.5e7, 2.0e7,
];

const NUCLIDES: &[(&str, f64, f64)] = &[
    // (h5_filename, awr_fallback, nu_bar_fallback)
    ("U234.h5", 232.030, 2.40),
    ("U235.h5", 233.025, 2.43),
    ("U238.h5", 236.006, 2.50),
];

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

/// Bit-for-bit Rust port of `transport.cu::nu_bar_lookup`. Operates on
/// the packed device buffers (energies/values + per-nuclide offset and
/// size). The formula is plain linear interpolation, identical to
/// `NuBarTable::lookup` — but applying it here on the round-tripped
/// device bytes catches packing and stride bugs that a CPU-side
/// formula check cannot.
#[cfg(feature = "cuda")]
fn gpu_nu_bar_lookup(e: f64, energies: &[f64], values: &[f64], offset: i32, n: i32) -> f64 {
    if n <= 0 {
        return 0.0;
    }
    let off = offset as usize;
    let len = n as usize;
    let es = &energies[off..off + len];
    let vs = &values[off..off + len];
    if e <= es[0] {
        return vs[0];
    }
    if e >= es[len - 1] {
        return vs[len - 1];
    }
    let (mut lo, mut hi) = (0_usize, len - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) >> 1;
        if es[mid] <= e {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let f = (e - es[lo]) / (es[hi] - es[lo]);
    vs[lo] + f * (vs[hi] - vs[lo])
}

fn summarize_table(kern: &NuclideKernels, name: &str) {
    match &kern.nu_bar_table {
        Some(t) if !t.energies.is_empty() => {
            let n = t.energies.len();
            println!(
                "  {name:>10}: {n:>4} pts, E ∈ [{:.3e}, {:.3e}] eV, ν ∈ [{:.4}, {:.4}], nu_bar_const = {:.4}",
                t.energies[0],
                t.energies[n - 1],
                t.values[0],
                t.values[n - 1],
                kern.nu_bar_const,
            );
        }
        _ => {
            println!(
                "  {name:>10}: NO nu_bar table — falls back to nu_bar_const = {:.4}",
                kern.nu_bar_const
            );
        }
    }
}

fn cpu_only_report(kerns: &[NuclideKernels], names: &[String]) {
    println!("\n=== CPU-only ν̄(E) sweep ===");
    print!("{:>12}", "E (eV)");
    for n in names {
        print!("{:>14}", n);
    }
    println!();
    for &e in TEST_E_EV {
        print!("{:>12.3e}", e);
        for kern in kerns {
            print!("{:>14.6}", kern.nu_bar_at(e));
        }
        println!();
    }
}

#[cfg(feature = "cuda")]
fn gpu_round_trip_report(
    kerns: &[NuclideKernels],
    names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    use open_rust_mc::gpu_transport::GpuTransportContext;

    println!("\n=== GPU round-trip ν̄(E) A/B ===");
    let gpu = GpuTransportContext::new()?;
    let nuc_data = gpu.upload_nuclide_data(kerns, 15)?;

    let energies_h = gpu.stream().clone_dtoh(&nuc_data.nu_bar_energies)?;
    let values_h = gpu.stream().clone_dtoh(&nuc_data.nu_bar_values)?;
    let offsets_h = gpu.stream().clone_dtoh(&nuc_data.nu_bar_offsets)?;
    let sizes_h = gpu.stream().clone_dtoh(&nuc_data.nu_bar_sizes)?;

    println!(
        "  packed buffer: {} energies, {} values, {} offsets, {} sizes",
        energies_h.len(),
        values_h.len(),
        offsets_h.len(),
        sizes_h.len(),
    );

    for (i, (kern, name)) in kerns.iter().zip(names).enumerate() {
        let off = offsets_h[i];
        let sz = sizes_h[i];
        if sz <= 0 {
            println!(
                "  [{i}] {name:>10}: size=0 (no table — GPU uses nu_bar_const = {:.4})",
                kern.nu_bar_const
            );
            continue;
        }
        let first_e = energies_h[off as usize];
        let last_e = energies_h[off as usize + sz as usize - 1];
        let first_v = values_h[off as usize];
        let last_v = values_h[off as usize + sz as usize - 1];
        let (cpu_first, cpu_last) = match &kern.nu_bar_table {
            Some(t) if !t.energies.is_empty() => {
                let n = t.energies.len();
                (
                    (t.energies[0], t.values[0]),
                    (t.energies[n - 1], t.values[n - 1]),
                )
            }
            _ => ((0.0, 0.0), (0.0, 0.0)),
        };
        let edge_ok = (first_e == cpu_first.0)
            && (last_e == cpu_last.0)
            && (first_v == cpu_first.1)
            && (last_v == cpu_last.1);
        println!(
            "  [{i}] {name:>10}: off={off} sz={sz}   gpu=({:.3e},{:.4})→({:.3e},{:.4})  cpu=({:.3e},{:.4})→({:.3e},{:.4})  {}",
            first_e,
            first_v,
            last_e,
            last_v,
            cpu_first.0,
            cpu_first.1,
            cpu_last.0,
            cpu_last.1,
            if edge_ok { "OK" } else { "MISMATCH" },
        );
    }

    // Track two failure modes separately:
    //   - `worst_table_delta`: Δ when the GPU is reading from a packed
    //     table (sz > 0). A non-zero value here means the data on the
    //     device disagrees with the CPU's `NuBarTable::lookup`.
    //   - `worst_fallback_delta`: Δ when the GPU has no table (sz = 0)
    //     and falls through to `nu_bar_const`. A non-zero value here
    //     comes from `NuBarTable::lookup` returning its hardcoded
    //     2.43 fallback for an empty Some(NuBarTable) on the CPU while
    //     the GPU uses the per-nuclide constant.
    let mut worst_table_delta = 0.0_f64;
    let mut worst_table_label = String::new();
    let mut worst_fallback_delta = 0.0_f64;
    let mut worst_fallback_label = String::new();
    print!("\n{:>12}", "E (eV)");
    for n in names {
        print!("  {:>30}", format!("{} (cpu / gpu / Δ)", n));
    }
    println!();
    for &e in TEST_E_EV {
        print!("{:>12.3e}", e);
        for (i, (kern, name)) in kerns.iter().zip(names).enumerate() {
            let cpu = kern.nu_bar_at(e);
            let off = offsets_h[i];
            let sz = sizes_h[i];
            let gpu = if sz > 0 {
                gpu_nu_bar_lookup(e, &energies_h, &values_h, off, sz)
            } else {
                kern.nu_bar_const
            };
            let d = gpu - cpu;
            if sz > 0 {
                if d.abs() > worst_table_delta {
                    worst_table_delta = d.abs();
                    worst_table_label =
                        format!("{name} at E = {e:.3e} eV  cpu = {cpu:.6}  gpu = {gpu:.6}");
                }
            } else if d.abs() > worst_fallback_delta {
                worst_fallback_delta = d.abs();
                worst_fallback_label =
                    format!("{name} at E = {e:.3e} eV  cpu = {cpu:.6}  gpu = {gpu:.6}");
            }
            print!("  {:>10.6} / {:>10.6} / {:+.2e}", cpu, gpu, d);
        }
        println!();
    }

    println!(
        "\nWorst |Δ| over actively-tabulated nuclides (sz > 0): {:.3e}",
        worst_table_delta
    );
    if !worst_table_label.is_empty() {
        println!("    at: {worst_table_label}");
    }
    if worst_table_delta < 1e-12 {
        println!("    → ν table data is BIT-IDENTICAL CPU↔GPU for every");
        println!("       nuclide that carries a real ν(E) table. The");
        println!("       +500-700 pcm fast-metal hot bias is NOT a ν-table");
        println!("       upload bug. Next suspect: the E_in distribution at");
        println!("       fission (shape, not mean).");
    } else {
        println!("    → NON-ZERO. ν table data on device disagrees with CPU.");
        println!("       Investigate `upload_nuclide_data` packing and");
        println!("       P_NB_{{OFFSETS,SIZES,ENERGIES,VALUES}} slot wiring.");
    }
    println!(
        "\nWorst |Δ| over fallback-only nuclides (sz = 0): {:.3e}",
        worst_fallback_delta
    );
    if !worst_fallback_label.is_empty() {
        println!("    at: {worst_fallback_label}");
    }
    if worst_fallback_delta > 1e-9 {
        println!("    → CPU `NuBarTable::lookup` falls back to a hardcoded 2.43");
        println!("       when the table is `Some(empty)`, while the GPU uses");
        println!("       the per-nuclide `nu_bar_const`. Real but small impact");
        println!("       on Godiva — bounded by (fission share of affected");
        println!("       nuclide) × Δν. Not the source of the +500-700 pcm bias");
        println!("       on its own.");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn gpu_round_trip_report(
    _kerns: &[NuclideKernels],
    _names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n(CUDA feature disabled; rebuild with `--features cuda` for the GPU A/B)");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = data_dir();
    println!("Data dir: {}", dir.display());

    let mut kerns = Vec::new();
    let mut names = Vec::new();
    for &(file, awr_fb, nu_fb) in NUCLIDES {
        let path = dir.join(file);
        if !path.exists() {
            eprintln!("  skipping {} — not found", path.display());
            continue;
        }
        let kern = load_nuclide(&path, 5, 0, awr_fb, nu_fb);
        let name = file.trim_end_matches(".h5").to_string();
        summarize_table(&kern, &name);
        kerns.push(kern);
        names.push(name);
    }

    if kerns.is_empty() {
        return Err("no nuclides loaded".into());
    }

    cpu_only_report(&kerns, &names);
    gpu_round_trip_report(&kerns, &names)?;
    Ok(())
}
