use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use open_rust_mc::kernel;
use open_rust_mc::loader::SvdFactors;
use open_rust_mc::table::PointwiseTable;
use std::hint::black_box;

fn output_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .expect("HOME or USERPROFILE must be set");
    PathBuf::from(home)
        .join("madman_svd_experiment")
        .join("outputs")
}

fn bench_reconstruct(c: &mut Criterion) {
    let dir = output_dir();
    let prefix = std::env::var("BENCH_PREFIX").unwrap_or_else(|_| "jeff33_".into());

    let factors = SvdFactors::load(&dir, &prefix).expect("load SVD factors");
    let n_e = factors.energies.len();
    let full_rank = factors.s.len();

    let mut group = c.benchmark_group("full_spectrum");

    for k in [3, 4, 5, 6] {
        if k > full_rank {
            continue;
        }

        let kern = factors.clone().into_kernel(k);
        let coeffs = kern.temp_coeffs(0);
        let mut buf = vec![0.0_f64; n_e];

        group.bench_with_input(BenchmarkId::new("manual_fma", k), &k, |b, _| {
            b.iter(|| kern.reconstruct_log(black_box(&coeffs), black_box(&mut buf)));
        });

        group.bench_with_input(BenchmarkId::new("faer_simd", k), &k, |b, _| {
            b.iter(|| {
                kernel::reconstruct_log_faer(
                    black_box(&kern),
                    black_box(&coeffs),
                    black_box(&mut buf),
                );
            });
        });
    }

    group.finish();

    // Table lookup baseline
    let raw_path = dir.join(format!("{prefix}A_raw_u235_mt18.npy"));
    if raw_path.exists() {
        let file = std::fs::File::open(&raw_path).expect("open A_raw");
        let a_raw: ndarray::Array2<f64> =
            ndarray_npy::ReadNpyExt::read_npy(file).expect("parse A_raw");
        let xs_col: Vec<f64> = a_raw.column(0).to_vec();
        let energies: Vec<f64> = factors.energies.clone();
        let tbl = PointwiseTable::from_vecs(energies.clone(), xs_col);
        let mut tbl_buf = vec![0.0_f64; n_e];

        let mut tbl_group = c.benchmark_group("table_lookup");
        tbl_group.bench_function("binary_search_loglog", |b| {
            b.iter(|| tbl.batch_lookup(black_box(&energies), black_box(&mut tbl_buf)));
        });
        tbl_group.finish();
    }
}

criterion_group!(benches, bench_reconstruct);
criterion_main!(benches);
