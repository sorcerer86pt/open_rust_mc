//! Per-nuclide fission spectrum + ν̄(E) diagnostic dump.
//!
//! Built to investigate the U-233 Jezebel-23 −2876 pcm bias (resume.md
//! 2026-05-11). The deep-dive ruled out residual-routing, MT=22/24/28/37
//! threshold effects, and SVD-rank — leaving two prime suspects:
//!
//!   1. ν̄(E) sparse-interpolation: U-233 ships only ~13 points on
//!      [thermal, 20 MeV], vs U-235's 79. Linear interpolation across
//!      the 1–5 MeV range that dominates Jezebel-23 may be inadequate.
//!   2. Fission spectrum χ(E_in, E_out): prompt-neutron outgoing
//!      distribution that drives the fast Watt mean. A wrong χ shifts
//!      the in-system spectrum and biases k_eff.
//!
//! Output is plain text for human reading + diffing against OpenMC's
//! Python API output (`openmc.data.IncidentNeutron.from_hdf5(...)`
//! → `.reactions[18].products[0]`). With `--json`, dumps a structured
//! payload for scripted comparison.
//!
//! Usage:
//!   u233_diag <data_dir>
//!   u233_diag <data_dir> --nuclides U233.h5,U235.h5,Pu239.h5
//!   u233_diag <data_dir> --json
//!   u233_diag <data_dir> --sample-energies 1e6,5e6 --samples 1000000

use open_rust_mc::hdf5_reader::{
    EnergyDistribution, FissionEnergyLaw, NuBarTable, NuclideFileReader,
};
use open_rust_mc::transport::rng::Rng;
use std::path::PathBuf;

const JEZEBEL_BAND_LO: f64 = 1.0e5; // 100 keV
const JEZEBEL_BAND_HI: f64 = 1.0e7; // 10 MeV
const DEFAULT_NUCLIDES: &[&str] = &["U233.h5", "U235.h5"];
const DEFAULT_SAMPLE_ENERGIES: &[f64] = &[1.0e6, 2.5e6, 5.0e6];
const DEFAULT_SAMPLES: usize = 200_000;
const HIST_BINS: usize = 40;

struct Args {
    data_dir: PathBuf,
    nuclides: Vec<String>,
    json: bool,
    sample_energies: Vec<f64>,
    samples: usize,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() < 2 || raw.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "usage: u233_diag <data_dir> [--nuclides A.h5,B.h5] \
             [--sample-energies eV,eV] [--samples N] [--json]"
        );
        std::process::exit(if raw.len() < 2 { 1 } else { 0 });
    }
    let data_dir = PathBuf::from(&raw[1]);
    let flag = |name: &str| {
        raw.iter()
            .position(|a| a == name)
            .and_then(|i| raw.get(i + 1).cloned())
    };
    let nuclides = flag("--nuclides")
        .map(|s| s.split(',').map(str::trim).map(String::from).collect())
        .unwrap_or_else(|| DEFAULT_NUCLIDES.iter().map(|s| (*s).into()).collect());
    let sample_energies = flag("--sample-energies")
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<f64>().ok())
                .collect()
        })
        .unwrap_or_else(|| DEFAULT_SAMPLE_ENERGIES.to_vec());
    let samples = flag("--samples")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SAMPLES);
    let json = raw.iter().any(|a| a == "--json");
    Args {
        data_dir,
        nuclides,
        json,
        sample_energies,
        samples,
    }
}

fn nu_bar_stats(nb: &NuBarTable) -> (f64, f64, f64, f64) {
    // Average ν̄ over [JEZEBEL_BAND_LO, JEZEBEL_BAND_HI] via trapezoid on a
    // 200-point log-uniform mesh — same numerical recipe as OpenMC's
    // postprocessing scripts so the value is directly comparable.
    let n = 200;
    let log_lo = JEZEBEL_BAND_LO.ln();
    let log_hi = JEZEBEL_BAND_HI.ln();
    let mut sum = 0.0;
    let mut weight = 0.0;
    let mut last_e = 0.0;
    let mut last_v = 0.0;
    for i in 0..n {
        let e = (log_lo + (log_hi - log_lo) * (i as f64) / ((n - 1) as f64)).exp();
        let v = nb.lookup(e);
        if i > 0 {
            let de = e - last_e;
            sum += 0.5 * (v + last_v) * de;
            weight += de;
        }
        last_e = e;
        last_v = v;
    }
    let mean_band = if weight > 0.0 { sum / weight } else { 0.0 };
    let v_thermal = nb.lookup(2.53e-2);
    let v_1mev = nb.lookup(1.0e6);
    let v_5mev = nb.lookup(5.0e6);
    (v_thermal, v_1mev, v_5mev, mean_band)
}

fn sample_histogram(
    edist: &EnergyDistribution,
    e_in: f64,
    n_samples: usize,
    seed: u64,
) -> (Vec<f64>, Vec<u64>, f64, f64, f64) {
    let mut rng = Rng::new(seed, 0xC0DE_F133);
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        samples.push(edist.sample(e_in, &mut rng));
    }
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max = samples.iter().copied().fold(0.0_f64, f64::max);
    let mean: f64 = samples.iter().sum::<f64>() / (n_samples as f64);
    // Log-spaced histogram covering [max(1 eV, min), max].
    let lo = min.max(1.0).ln();
    let hi = max.ln();
    let mut edges = Vec::with_capacity(HIST_BINS + 1);
    for i in 0..=HIST_BINS {
        edges.push((lo + (hi - lo) * (i as f64) / (HIST_BINS as f64)).exp());
    }
    let mut counts = vec![0u64; HIST_BINS];
    for &e in &samples {
        if e < edges[0] {
            continue;
        }
        let idx = ((e.ln() - lo) / (hi - lo) * (HIST_BINS as f64)) as isize;
        let idx = idx.clamp(0, (HIST_BINS - 1) as isize) as usize;
        counts[idx] += 1;
    }
    (edges, counts, mean, min, max)
}

fn dump_nu_bar(label: &str, nb: &NuBarTable, json: bool) {
    let (v_th, v_1, v_5, v_band) = nu_bar_stats(nb);
    if json {
        print!("    \"nu_bar_{label}\": {{");
        print!("\"n_points\": {},", nb.energies.len());
        print!(" \"thermal\": {v_th:.6}, \"e_1MeV\": {v_1:.6}, \"e_5MeV\": {v_5:.6},");
        print!(" \"mean_100keV_10MeV\": {v_band:.6},");
        print!(" \"energies_eV\": [");
        for (i, e) in nb.energies.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{e:.6e}");
        }
        print!("], \"values\": [");
        for (i, v) in nb.values.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{v:.6}");
        }
        println!("]}},");
        return;
    }
    println!("  ν̄_{label} table: {} points", nb.energies.len());
    println!("    ν̄(thermal=0.0253 eV) = {v_th:.5}");
    println!("    ν̄(1 MeV)            = {v_1:.5}");
    println!("    ν̄(5 MeV)            = {v_5:.5}");
    println!("    ⟨ν̄⟩ over 100 keV–10 MeV (log-uniform trapezoid) = {v_band:.5}");
    println!("    points (E_eV, ν̄):");
    for (e, v) in nb.energies.iter().zip(nb.values.iter()) {
        println!("      {e:>13.6e}  {v:.5}");
    }
}

fn dump_fission_spectrum(edist: &EnergyDistribution, samples: usize, sample_es: &[f64], json: bool) {
    let inc = &edist.energies;
    if json {
        print!("    \"fission_spectrum\": {{");
        print!("\"n_incident\": {},", inc.len());
        print!(" \"incident_energies_eV\": [");
        for (i, e) in inc.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{e:.6e}");
        }
        print!("],");
        print!(" \"per_incident_summary\": [");
        for (i, dist) in edist.distributions.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            let n = dist.e_out.len();
            let lo = dist.e_out.first().copied().unwrap_or(0.0);
            let hi = dist.e_out.last().copied().unwrap_or(0.0);
            let mean_out = if !dist.pdf.is_empty() && dist.pdf.len() == n {
                let mut num = 0.0;
                let mut den = 0.0;
                for k in 0..n.saturating_sub(1) {
                    let de = dist.e_out[k + 1] - dist.e_out[k];
                    let pe_lo = dist.pdf[k] * dist.e_out[k];
                    let pe_hi = dist.pdf[k + 1] * dist.e_out[k + 1];
                    let p_lo = dist.pdf[k];
                    let p_hi = dist.pdf[k + 1];
                    num += 0.5 * (pe_lo + pe_hi) * de;
                    den += 0.5 * (p_lo + p_hi) * de;
                }
                if den > 0.0 {
                    num / den
                } else {
                    0.5 * (lo + hi)
                }
            } else {
                0.5 * (lo + hi)
            };
            print!(
                "{{\"e_in_eV\":{:.6e},\"n_e_out\":{},\"e_out_min\":{:.6e},\"e_out_max\":{:.6e},\"mean_e_out\":{:.6e}}}",
                inc[i], n, lo, hi, mean_out
            );
        }
        println!("],");
        // Sampled histograms
        print!(" \"sampled_histograms\": [");
        for (j, &e_in) in sample_es.iter().enumerate() {
            if j > 0 {
                print!(",");
            }
            let (edges, counts, mean, mn, mx) = sample_histogram(edist, e_in, samples, 0xD1A6_0000 + j as u64);
            print!(
                "{{\"e_in_eV\":{e_in:.6e},\"n_samples\":{samples},\"mean_e_out\":{mean:.6e},\"min\":{mn:.6e},\"max\":{mx:.6e},\"bin_edges\":["
            );
            for (k, x) in edges.iter().enumerate() {
                if k > 0 {
                    print!(",");
                }
                print!("{x:.6e}");
            }
            print!("],\"counts\":[");
            for (k, c) in counts.iter().enumerate() {
                if k > 0 {
                    print!(",");
                }
                print!("{c}");
            }
            print!("]}}");
        }
        println!("]}}");
        return;
    }
    let law_name = match &edist.closed_form {
        Some(FissionEnergyLaw::Watt(_)) => "Watt (ENDF Law 11)",
        Some(FissionEnergyLaw::Maxwell(_)) => "Maxwell (ENDF Law 7)",
        Some(FissionEnergyLaw::Evaporation(_)) => "Evaporation (ENDF Law 9)",
        None => "Tabular (ENDF Law 4 / Law 61)",
    };
    println!("  fission χ(E_in, E_out): {} incident-energy nodes  [law: {law_name}]", inc.len());
    match &edist.closed_form {
        Some(FissionEnergyLaw::Watt(w)) => {
            println!("    Watt parameters a(E_in), b(E_in), u = {:.4e} eV:", w.u);
            let n = w.a_energies.len().max(w.b_energies.len());
            for i in 0..n {
                let ae = w.a_energies.get(i).copied();
                let av = w.a_values.get(i).copied();
                let be = w.b_energies.get(i).copied();
                let bv = w.b_values.get(i).copied();
                println!(
                    "      [{i:>2}] a@{:>13.6e}={:>10.4e}   b@{:>13.6e}={:>10.4e}",
                    ae.unwrap_or(f64::NAN),
                    av.unwrap_or(f64::NAN),
                    be.unwrap_or(f64::NAN),
                    bv.unwrap_or(f64::NAN)
                );
            }
        }
        Some(FissionEnergyLaw::Maxwell(m)) | Some(FissionEnergyLaw::Evaporation(m)) => {
            println!("    θ(E_in), u = {:.4e} eV:", m.u);
            for (i, (e, t)) in m.theta_energies.iter().zip(m.theta_values.iter()).enumerate() {
                println!("      [{i:>2}] θ@{e:>13.6e} = {t:>10.4e}");
            }
        }
        None => {
            println!("    incident energy grid (eV):");
            for (i, e) in inc.iter().enumerate() {
                let dist = &edist.distributions[i];
                let n = dist.e_out.len();
                let lo = dist.e_out.first().copied().unwrap_or(0.0);
                let hi = dist.e_out.last().copied().unwrap_or(0.0);
                println!(
                    "      [{i:>2}] E_in = {e:>13.6e}   n_e_out = {n:>4}   E_out ∈ [{lo:.3e}, {hi:.3e}]"
                );
            }
        }
    }
    println!("  sampled-output histograms (deterministic seed):");
    for (j, &e_in) in sample_es.iter().enumerate() {
        let (edges, counts, mean, mn, mx) = sample_histogram(edist, e_in, samples, 0xD1A6_0000 + j as u64);
        println!(
            "    E_in = {e_in:.3e} eV  N = {samples}  ⟨E_out⟩ = {mean:.6e}  [{mn:.3e}, {mx:.3e}]"
        );
        let max_count = counts.iter().copied().max().unwrap_or(1).max(1);
        for (k, &c) in counts.iter().enumerate() {
            let bar_len = (40 * c / max_count) as usize;
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            println!(
                "      {:>10.3e} – {:<10.3e}  {:>8}  {}",
                edges[k],
                edges[k + 1],
                c,
                bar
            );
        }
    }
}

fn dump_nuclide(data_dir: &std::path::Path, file_name: &str, args: &Args, last: bool) {
    let path = data_dir.join(file_name);
    let reader = match NuclideFileReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error opening {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let nu_total = reader.nu_bar().unwrap_or(NuBarTable {
        energies: vec![],
        values: vec![],
    });
    let nu_delayed = reader.delayed_nu_bar();
    let edist = reader.fission_energy_dist();

    if args.json {
        println!("  \"{}\": {{", reader.nuclide_name);
        println!("    \"file\": \"{file_name}\",");
        println!("    \"n_temperatures\": {},", reader.temperatures.len());
        dump_nu_bar("total", &nu_total, true);
        match nu_delayed {
            Some(nd) => dump_nu_bar("delayed", &nd, true),
            None => println!("    \"nu_bar_delayed\": null,"),
        }
        match edist {
            Some(ed) => dump_fission_spectrum(&ed, args.samples, &args.sample_energies, true),
            None => println!("    \"fission_spectrum\": null"),
        }
        if last {
            println!("  }}");
        } else {
            println!("  }},");
        }
        return;
    }

    println!("══════════════════════════════════════════════════════════════════════");
    println!(" Nuclide: {} (file: {file_name})", reader.nuclide_name);
    println!("   temperatures present: {:?} K", reader.temperatures);
    println!("──────────────────────────────────────────────────────────────────────");
    dump_nu_bar("total", &nu_total, false);
    match nu_delayed {
        Some(nd) => {
            println!();
            dump_nu_bar("delayed", &nd, false);
        }
        None => println!("  ν̄_delayed: (not in file)"),
    }
    println!();
    match edist {
        Some(ed) => dump_fission_spectrum(&ed, args.samples, &args.sample_energies, false),
        None => println!("  fission_energy_dist: (not in file)"),
    }
    println!();
}

fn main() {
    let args = parse_args();
    if args.json {
        println!("{{");
        println!("  \"_metadata\": {{");
        println!("    \"data_dir\": \"{}\",", args.data_dir.display());
        println!("    \"samples_per_e_in\": {},", args.samples);
        print!("    \"sample_energies_eV\": [");
        for (i, e) in args.sample_energies.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{e:.6e}");
        }
        println!("]");
        println!("  }},");
    } else {
        println!(
            "u233_diag — fission χ(E_in, E_out) and ν̄(E) dump across {} nuclide(s)",
            args.nuclides.len()
        );
        println!("data_dir: {}", args.data_dir.display());
        println!(
            "samples per E_in: {}   sample E_in (eV): {:?}",
            args.samples, args.sample_energies
        );
        println!();
    }
    for (i, name) in args.nuclides.iter().enumerate() {
        let last = i + 1 == args.nuclides.len();
        dump_nuclide(&args.data_dir, name, &args, last);
    }
    if args.json {
        println!("}}");
    }
}
