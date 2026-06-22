// SPDX-License-Identifier: MIT
//! Inspector for U-235 MT=91 distribution_0 layout in ENDF/B-VIII.1.
//!
//! Dumps the shape and a few samples of `energy_out` and `mu` datasets
//! so we can confirm whether the Law 61 KalbachMann mu-coupling is
//! actually present in this evaluation (and not all-zero / pure
//! marginal).

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        // Try common locations
        for p in [
            "data/endfb-viii.1-hdf5/neutron/U235.h5",
            "../data/endfb-viii.1-hdf5/neutron/U235.h5",
            "../../data/endfb-viii.1-hdf5/neutron/U235.h5",
        ] {
            if PathBuf::from(p).exists() {
                return p.to_string();
            }
        }
        "data/endfb-viii.1-hdf5/neutron/U235.h5".to_string()
    });
    let nuclide_name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "U235".to_string());
    let mt: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(91);

    eprintln!("Opening {}", path);
    let file = hdf5_pure::File::open(&path).expect("open hdf5");
    let root = file.root();
    let nuc = root.group(&nuclide_name).expect("nuclide group");
    let rxn_name = format!("reaction_{:03}", mt);
    let rxn = nuc
        .group("reactions")
        .expect("reactions")
        .group(&rxn_name)
        .expect("reaction");
    let product = rxn.group("product_0").expect("product_0");
    let dist0 = product.group("distribution_0").expect("distribution_0");

    println!("=== {} {} distribution_0 ===", nuclide_name, rxn_name);
    let groups = dist0.groups().unwrap_or_default();
    let datasets = dist0.datasets().unwrap_or_default();
    println!("groups: {:?}", groups);
    println!("datasets: {:?}", datasets);

    let attrs = dist0.attrs().unwrap_or_default();
    for (k, v) in &attrs {
        println!("attr  {} = {:?}", k, v);
    }

    // Now print energy_out shape + first rows summary.
    if datasets.iter().any(|n| n == "energy_out") {
        let ds = dist0.dataset("energy_out")?;
        let shape = ds.shape()?;
        let raw = ds.read_f64()?;
        println!("\nenergy_out shape: {:?}, total_len: {}", shape, raw.len());
        let dattrs = ds.attrs().unwrap_or_default();
        for (k, v) in &dattrs {
            println!("  attr {} = {:?}", k, v);
        }
        let n_rows = shape[0] as usize;
        let n_cols = shape[1] as usize;
        // Rows are typical layout: 0=E_out, 1=PDF, 2=CDF, 3=mu_lo or mu, 4=PDF_mu, 5=CDF_mu.
        // We'll print the per-row min/max/sum (over all bins) to see if rows 3..6 are non-zero.
        println!(
            "\nPer-row min / mean / max / nonzero_count over {} bins:",
            n_cols
        );
        for r in 0..n_rows {
            let row = &raw[r * n_cols..(r + 1) * n_cols];
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            let mut sum = 0.0;
            let mut nz = 0usize;
            for &v in row {
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
                sum += v;
                if v.abs() > 1e-30 {
                    nz += 1;
                }
            }
            println!(
                "  row {}: min={:>13.5e} mean={:>13.5e} max={:>13.5e} nonzero={}/{}",
                r,
                mn,
                sum / n_cols as f64,
                mx,
                nz,
                n_cols
            );
        }

        // Print a sample slice at a chosen E_in offset (use offsets attr).
        let off_attr = dattrs.get("offsets");
        if let Some(hdf5_pure::AttrValue::I64Array(offs)) = off_attr {
            // Pick offset at incident energy ~5 MeV.
            let energies = dist0.dataset("energy")?.read_f64()?;
            // Find bracketing index for 5 MeV.
            let mut idx = 0usize;
            for (i, &e) in energies.iter().enumerate() {
                if e >= 5.0e6 {
                    idx = i;
                    break;
                }
            }
            let start = offs[idx] as usize;
            let end = offs.get(idx + 1).copied().unwrap_or(n_cols as i64) as usize;
            println!(
                "\nSample at E_in[{}]={:.3e} eV, bin range [{}, {}), len={}",
                idx,
                energies[idx],
                start,
                end,
                end - start
            );
            let show_n = (end - start).min(8);
            for k in 0..show_n {
                let mut vs = vec![];
                for r in 0..n_rows {
                    vs.push(raw[r * n_cols + start + k]);
                }
                println!(
                    "  bin {}: {}",
                    k,
                    vs.iter()
                        .map(|v| format!("{:>13.5e}", v))
                        .collect::<Vec<_>>()
                        .join("  ")
                );
            }

            // Now dump the mu slice for first 3 outgoing E_out bins. row 4
            // of energy_out is mu-table offsets per (E_in_bin, E_out_bin).
            println!(
                "\nmu slices for first 3 outgoing-E_out bins at E_in[{}]={:.3e} eV:",
                idx, energies[idx]
            );
            let mu_ds = dist0.dataset("mu")?;
            let mu_shape = mu_ds.shape()?;
            let mu_raw = mu_ds.read_f64()?;
            let mu_ncols = mu_shape[1] as usize;
            for k in 0..3.min(end - start) {
                let mu_off_lo = raw[4 * n_cols + start + k] as usize;
                let mu_off_hi = if start + k + 1 < end {
                    raw[4 * n_cols + start + k + 1] as usize
                } else {
                    mu_ncols
                };
                println!(
                    "  E_out bin {} (E_out={:.3e} eV) mu rows [{}..{}) (len {}):",
                    k,
                    raw[start + k],
                    mu_off_lo,
                    mu_off_hi,
                    mu_off_hi.saturating_sub(mu_off_lo)
                );
                let m = (mu_off_hi - mu_off_lo).min(8);
                for j in 0..m {
                    let mu_v = mu_raw[mu_off_lo + j];
                    let pdf_v = mu_raw[mu_ncols + mu_off_lo + j];
                    let cdf_v = mu_raw[2 * mu_ncols + mu_off_lo + j];
                    println!(
                        "    mu={:>10.5}  pdf={:>10.5e}  cdf={:>10.5e}",
                        mu_v, pdf_v, cdf_v
                    );
                }
            }
        }
    } else {
        println!("No energy_out dataset — different layout");
    }

    if datasets.iter().any(|n| n == "mu") {
        let ds = dist0.dataset("mu")?;
        let shape = ds.shape()?;
        let raw = ds.read_f64()?;
        println!("\nmu dataset shape: {:?}, total_len: {}", shape, raw.len());
        let dattrs = ds.attrs().unwrap_or_default();
        for (k, v) in &dattrs {
            match v {
                hdf5_pure::AttrValue::I64Array(arr) => println!(
                    "  attr {} = I64Array(len={}, first 12: {:?})",
                    k,
                    arr.len(),
                    &arr[..arr.len().min(12)]
                ),
                _ => println!("  attr {} = {:?}", k, v),
            }
        }
        // Print per-row summary
        if shape.len() == 2 {
            let n_rows = shape[0] as usize;
            let n_cols = shape[1] as usize;
            println!(
                "\nmu per-row min / mean / max / nonzero_count over {} bins:",
                n_cols
            );
            for r in 0..n_rows {
                let row = &raw[r * n_cols..(r + 1) * n_cols];
                let mut mn = f64::INFINITY;
                let mut mx = f64::NEG_INFINITY;
                let mut sum = 0.0;
                let mut nz = 0usize;
                for &v in row {
                    if v < mn {
                        mn = v;
                    }
                    if v > mx {
                        mx = v;
                    }
                    sum += v;
                    if v.abs() > 1e-30 {
                        nz += 1;
                    }
                }
                println!(
                    "  row {}: min={:>13.5e} mean={:>13.5e} max={:>13.5e} nonzero={}/{}",
                    r,
                    mn,
                    sum / n_cols as f64,
                    mx,
                    nz,
                    n_cols
                );
            }
        }
    }
    Ok(())
}
