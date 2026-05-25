#![allow(
    // SPDX-License-Identifier: MIT
    clippy::unwrap_used,
    clippy::expect_used
)]
//! U-235 σ_inelastic per-MT audit: rank-k SVD vs pointwise table.
//!
//! Investigation context — `docs/engine-vs-openmc-bias-investigation.md`,
//! Task #20. The engine over-predicts k_eff by +200 pcm on fast-metal
//! ICSBEP cases (Godiva, HMF-027). Reaction-rate tallies showed MT=91
//! (continuum inelastic) ~4.5% low and MT=51..90 (discrete levels) ~3.9%
//! high vs OpenMC on Godiva, with total MT=4 ~2% low — a channel-routing
//! imbalance, not a total-inelastic miss. Bumping SVD rank from 15 to 30
//! did NOT close it; the residual lives in the SVD parametrisation itself.
//!
//! This binary loads U-235 from one HDF5 file twice — once through the
//! rank-k SVD path, once through the pointwise table — and dumps σ_MT(E)
//! for every MT in {4, 51..91} on a fine log-spaced E grid over [0.1, 10]
//! MeV. CSV columns: `energy_eV, mt, threshold_eV, svd_xs_b, table_xs_b,
//! abs_diff_b, rel_diff`. Console summary prints worst |rel_diff| per MT
//! and a sum-of-discretes vs MT=4 sanity check.
//!
//! Usage (defaults shown):
//!   cargo run --release --bin u235_inelastic_audit -- \
//!       [data_dir]              auto-discover via data_paths::discover_neutron_dir
//!       [--rank 15]             SVD rank
//!       [--temp-idx 0]          temperature index in the HDF5 file
//!       [--nuclide U235.h5]     filename inside data_dir
//!       [--awr 233.025]         AWR fallback
//!       [--points 400]          log-spaced grid points in [E_lo, E_hi]
//!       [--e-lo 1.0e5]          low edge in eV (default 0.1 MeV)
//!       [--e-hi 1.0e7]          high edge in eV (default 10 MeV)
//!       [--out outputs/u235_inelastic_audit.csv]

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use open_rust_mc::data_paths;
use open_rust_mc::transport::xs_provider::{
    load_nuclide_table, load_nuclide_with_policy, RankPolicy, ReactionKernel,
};

const DEFAULT_RANK: usize = 15;
const DEFAULT_TEMP_IDX: usize = 0;
const DEFAULT_NUCLIDE: &str = "U235.h5";
const DEFAULT_AWR: f64 = 233.025;
const DEFAULT_NU_BAR: f64 = 2.43;
const DEFAULT_POINTS: usize = 400;
const DEFAULT_E_LO: f64 = 1.0e5;
const DEFAULT_E_HI: f64 = 1.0e7;
const DEFAULT_OUT: &str = "outputs/u235_inelastic_audit.csv";

struct Args {
    data_dir: PathBuf,
    rank: usize,
    temp_idx: usize,
    nuclide: String,
    awr: f64,
    points: usize,
    e_lo: f64,
    e_hi: f64,
    out: PathBuf,
    mt91_table: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut data_dir: Option<PathBuf> = None;
    let mut rank = DEFAULT_RANK;
    let mut temp_idx = DEFAULT_TEMP_IDX;
    let mut nuclide = DEFAULT_NUCLIDE.to_string();
    let mut awr = DEFAULT_AWR;
    let mut points = DEFAULT_POINTS;
    let mut e_lo = DEFAULT_E_LO;
    let mut e_hi = DEFAULT_E_HI;
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut mt91_table = false;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let need_val = |i: usize, name: &str| -> Result<&String, String> {
            raw.get(i + 1)
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match raw[i].as_str() {
            "--rank" => {
                rank = need_val(i, "--rank")?
                    .parse()
                    .map_err(|e| format!("--rank: {e}"))?;
                i += 2;
            }
            "--temp-idx" => {
                temp_idx = need_val(i, "--temp-idx")?
                    .parse()
                    .map_err(|e| format!("--temp-idx: {e}"))?;
                i += 2;
            }
            "--nuclide" => {
                nuclide = need_val(i, "--nuclide")?.clone();
                i += 2;
            }
            "--awr" => {
                awr = need_val(i, "--awr")?
                    .parse()
                    .map_err(|e| format!("--awr: {e}"))?;
                i += 2;
            }
            "--points" => {
                points = need_val(i, "--points")?
                    .parse()
                    .map_err(|e| format!("--points: {e}"))?;
                i += 2;
            }
            "--e-lo" => {
                e_lo = need_val(i, "--e-lo")?
                    .parse()
                    .map_err(|e| format!("--e-lo: {e}"))?;
                i += 2;
            }
            "--e-hi" => {
                e_hi = need_val(i, "--e-hi")?
                    .parse()
                    .map_err(|e| format!("--e-hi: {e}"))?;
                i += 2;
            }
            "--out" => {
                out = PathBuf::from(need_val(i, "--out")?);
                i += 2;
            }
            "--mt91-table" => {
                mt91_table = true;
                i += 1;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            arg if !arg.starts_with("--") => {
                if data_dir.is_some() {
                    return Err(format!("unexpected positional arg: {arg}"));
                }
                data_dir = Some(PathBuf::from(arg));
                i += 1;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    if points < 2 {
        return Err("--points must be >= 2".into());
    }
    if !(e_lo > 0.0 && e_hi > e_lo) {
        return Err("must have 0 < e-lo < e-hi".into());
    }

    let data_dir = match data_dir {
        Some(p) => p,
        None => {
            let start: PathBuf = env!("CARGO_MANIFEST_DIR").into();
            data_paths::discover_neutron_dir(&start).ok_or_else(|| {
                "could not auto-discover neutron data dir; pass it as the first positional arg"
                    .to_string()
            })?
        }
    };

    Ok(Args {
        data_dir,
        rank,
        temp_idx,
        nuclide,
        awr,
        points,
        e_lo,
        e_hi,
        out,
        mt91_table,
    })
}

fn print_usage() {
    eprintln!(
        "usage: u235_inelastic_audit [data_dir] \
[--rank N] [--temp-idx N] [--nuclide FILE] [--awr F] \
[--points N] [--e-lo eV] [--e-hi eV] [--out PATH] [--mt91-table]"
    );
}

fn log_grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let lo_log = lo.log10();
    let hi_log = hi.log10();
    let span = hi_log - lo_log;
    (0..n)
        .map(|i| {
            let f = i as f64 / (n - 1) as f64;
            10f64.powf(lo_log + span * f)
        })
        .collect()
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let path = args.data_dir.join(&args.nuclide);
    if !path.exists() {
        eprintln!("nuclide file not found: {}", path.display());
        return ExitCode::from(2);
    }

    println!("u235 inelastic audit");
    println!("  data:      {}", path.display());
    println!("  rank:      {}", args.rank);
    println!("  temp_idx:  {}", args.temp_idx);
    println!("  AWR:       {}", args.awr);
    println!(
        "  grid:      {} log-spaced pts in [{:.2e}, {:.2e}] eV",
        args.points, args.e_lo, args.e_hi
    );
    println!("  out:       {}", args.out.display());
    println!("  MT=91 fix: {}", if args.mt91_table { "Table (bypass SVD)" } else { "SVD (default)" });

    let mut policy = RankPolicy::new(args.rank);
    if args.mt91_table {
        policy = policy.with_table(91);
    }

    println!("\nloading SVD provider...");
    let svd = load_nuclide_with_policy(&path, &policy, args.temp_idx, args.awr, DEFAULT_NU_BAR);

    println!("loading Table provider...");
    let tab = load_nuclide_table(&path, args.temp_idx, args.awr, DEFAULT_NU_BAR);

    if svd.discrete_levels.len() != tab.discrete_levels.len() {
        eprintln!(
            "WARNING: discrete-level count mismatch — SVD={} Table={}",
            svd.discrete_levels.len(),
            tab.discrete_levels.len(),
        );
    }

    println!("\ndiscrete levels (SVD view):");
    println!("  idx   MT     threshold (eV)         Q (eV)");
    for (i, lvl) in svd.discrete_levels.iter().enumerate() {
        println!(
            "  {:>3}   {:>3}    {:>15.4e}    {:>13.4e}",
            i, lvl.info.mt, lvl.info.threshold, lvl.info.q_value,
        );
    }
    println!(
        "  has_continuum_inelastic (MT=91 present as last level): {}",
        svd.has_continuum_inelastic
    );
    println!(
        "  MT=4 total inelastic kernel: svd={} tab={}",
        svd.inelastic.is_some(),
        tab.inelastic.is_some(),
    );
    // Diagnostic: confirm the SVD provider's MT=91 kernel variant
    // tracks the --mt91-table flag (Table-variant when the flag is on).
    for lvl in &svd.discrete_levels {
        if lvl.info.mt == 91 {
            let variant = match &lvl.kernel {
                Some(ReactionKernel::Svd { coeffs, .. }) => {
                    format!("Svd (coeffs={})", coeffs.len())
                }
                Some(ReactionKernel::Table { xs, .. }) => {
                    format!("Table ({} pts)", xs.len())
                }
                None => "None".to_string(),
            };
            println!("  SVD-provider MT=91 kernel variant: {variant}");
        }
    }

    let mut energies = log_grid(args.e_lo, args.e_hi, args.points);
    for lvl in &svd.discrete_levels {
        let t = lvl.info.threshold;
        if t >= args.e_lo && t <= args.e_hi {
            for f in [1.001_f64, 1.01, 1.05, 1.1] {
                let e = t * f;
                if e <= args.e_hi {
                    energies.push(e);
                }
            }
        }
    }
    energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    energies.dedup_by(|a, b| (*a - *b).abs() / b.abs().max(1e-30) < 1e-12);

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    let f = match std::fs::File::create(&args.out) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not create {}: {e}", args.out.display());
            return ExitCode::from(2);
        }
    };
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "energy_eV,mt,threshold_eV,svd_xs_b,table_xs_b,abs_diff_b,rel_diff"
    )
    .unwrap();

    // (max_abs_rel, signed_rel_at_max, E, svd, tab)
    let mut worst_per_mt: std::collections::BTreeMap<u32, (f64, f64, f64, f64, f64)> =
        Default::default();

    let n_levels = svd.discrete_levels.len().min(tab.discrete_levels.len());

    // Hot-path-matched lookups: ReactionKernel::reconstruct_interp for
    // the SVD provider, StochTempTable::lookup_at_idx for the Table
    // provider. Both interpolate log-log between grid points exactly
    // as SvdXsProvider::lookup and TableXsProvider::lookup do during
    // transport. Earlier audit revisions called the non-interpolating
    // `lookup()` on the SVD side and the interpolating `lookup()` on
    // the Table side — that mismatch was the bulk of the apparent
    // SVD bias.
    let svd_grid = svd
        .elastic
        .as_ref()
        .map(|k| k.energies().to_vec())
        .expect("U-235 always has elastic kernel");

    let grid_idx_and_frac = |energy: f64| -> (usize, f64) {
        let n = svd_grid.len();
        if energy <= svd_grid[0] {
            return (0, 0.0);
        }
        if energy >= svd_grid[n - 1] {
            return (n - 1, 0.0);
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if svd_grid[mid] <= energy {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let log_e = energy.ln();
        let log_lo = svd_grid[lo].ln();
        let log_hi = svd_grid[lo + 1].ln();
        let frac = ((log_e - log_lo) / (log_hi - log_lo)).clamp(0.0, 1.0);
        (lo, frac)
    };

    for &e in &energies {
        let (idx, log_frac) = grid_idx_and_frac(e);

        for i in 0..n_levels {
            let info = &svd.discrete_levels[i].info;
            let s = svd.discrete_levels[i]
                .kernel
                .as_ref()
                .map_or(0.0, |k| {
                    if e < info.threshold {
                        0.0
                    } else {
                        k.reconstruct_interp(idx, log_frac)
                    }
                });
            let t = tab.discrete_levels[i]
                .table
                .as_ref()
                .map_or(0.0, |t| {
                    if e < info.threshold {
                        0.0
                    } else {
                        t.lookup_at_idx(e, idx)
                    }
                });
            let abs_d = s - t;
            let denom = t.abs().max(s.abs());
            let rel_d = if denom > 1e-30 { abs_d / denom } else { 0.0 };
            writeln!(
                w,
                "{:.6e},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                e, info.mt, info.threshold, s, t, abs_d, rel_d
            )
            .unwrap();
            let entry = worst_per_mt
                .entry(info.mt)
                .or_insert((0.0, 0.0, e, s, t));
            if rel_d.abs() > entry.0 {
                *entry = (rel_d.abs(), rel_d, e, s, t);
            }
        }

        let s_mt4 = svd
            .inelastic
            .as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let t_mt4 = tab
            .inelastic
            .as_ref()
            .map_or(0.0, |t| t.lookup_at_idx(e, idx));
        let abs4 = s_mt4 - t_mt4;
        let denom4 = t_mt4.abs().max(s_mt4.abs());
        let rel4 = if denom4 > 1e-30 { abs4 / denom4 } else { 0.0 };
        writeln!(
            w,
            "{:.6e},4,0.000000e0,{:.6e},{:.6e},{:.6e},{:.6e}",
            e, s_mt4, t_mt4, abs4, rel4
        )
        .unwrap();
        let entry = worst_per_mt.entry(4).or_insert((0.0, 0.0, e, s_mt4, t_mt4));
        if rel4.abs() > entry.0 {
            *entry = (rel4.abs(), rel4, e, s_mt4, t_mt4);
        }
    }

    w.flush().unwrap();

    println!("\nMT=4 vs Σ(MT=51..91) sanity (one path at a time):");
    println!(
        "  E_eV         SVD: MT4         Σlevels        ratio    | TAB: MT4         Σlevels        ratio"
    );
    for &e in &[5.0e5_f64, 1.0e6, 2.0e6, 5.0e6, 1.0e7] {
        if e < args.e_lo || e > args.e_hi {
            continue;
        }
        let (idx, log_frac) = grid_idx_and_frac(e);
        let s_mt4 = svd
            .inelastic
            .as_ref()
            .map_or(0.0, |k| k.reconstruct_interp(idx, log_frac));
        let s_sum: f64 = svd
            .discrete_levels
            .iter()
            .filter(|lvl| e >= lvl.info.threshold)
            .filter_map(|lvl| lvl.kernel.as_ref())
            .map(|k| k.reconstruct_interp(idx, log_frac).max(0.0))
            .sum();
        let s_r = if s_mt4 > 0.0 { s_sum / s_mt4 } else { 0.0 };
        let t_mt4 = tab
            .inelastic
            .as_ref()
            .map_or(0.0, |t| t.lookup_at_idx(e, idx));
        let t_sum: f64 = tab
            .discrete_levels
            .iter()
            .filter(|lvl| e >= lvl.info.threshold)
            .filter_map(|lvl| lvl.table.as_ref())
            .map(|t| t.lookup_at_idx(e, idx).max(0.0))
            .sum();
        let t_r = if t_mt4 > 0.0 { t_sum / t_mt4 } else { 0.0 };
        println!(
            "  {:.2e}    {:.5e}    {:.5e}    {:.5}  |    {:.5e}    {:.5e}    {:.5}",
            e, s_mt4, s_sum, s_r, t_mt4, t_sum, t_r,
        );
    }

    println!("\nworst |rel_diff| per MT  (SVD vs Table; sign of rel_diff = SVD high(+)/low(-)):");
    println!("  MT     rel_diff      E_eV          svd_xs_b      tab_xs_b");
    for (mt, &(_, rel, e, s, t)) in &worst_per_mt {
        println!(
            "  {:>3}    {:+.4e}   {:.3e}    {:.4e}    {:.4e}",
            mt, rel, e, s, t
        );
    }

    println!("\nwrote {}", args.out.display());
    ExitCode::SUCCESS
}
