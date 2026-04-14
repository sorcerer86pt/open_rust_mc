# open_rust_mc — Cache-Resident Cross-Section Reconstruction via SVD

A pure-Rust Monte Carlo neutronics engine prototype that replaces multi-gigabyte
pointwise cross-section tables with cache-resident SVD basis vectors,
achieving 8–13× CPU speedup and 2.6–2.8× GPU speedup over traditional
table lookup.

## Key Results

- **Accuracy**: Δk_eff < 10 pcm on Godiva benchmark (indistinguishable from OpenMC)
- **Speed**: 3–5 ns/point reconstruction via FMA, vs 40 ns/point table lookup
- **Memory**: Hybrid SVD+WMP reduces 11 GB of nuclear data to ~20 MB (fits in L3 cache)
- **47 of 52** U-235 reaction channels are effectively rank-1 (trivially compressible)

## Structure

```
scripts/           Python analysis pipeline (Phases 1–5)
rust_prototype/    Pure-Rust engine (reads HDF5 natively via hdf5-pure)
cuda_bench/        CUDA GPU benchmark kernel
paper/             LaTeX manuscript
```

## Quick Start

### Rust (no Python needed)
```bash
cd rust_prototype
cargo run --release -- hdf5 path/to/U235.h5 --mt 18
```

### Python analysis
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install openmc h5py numpy scipy matplotlib
python scripts/phase1_extraction.py path/to/U235.h5
python scripts/phase2_svd_analysis.py
python scripts/phase3_error_analysis.py
```

### Nuclear data
Download ENDF/B-VII.1 HDF5 from https://openmc.org/data/ and extract to `data/`.

## Paper

See `paper/svd_cross_section_compression.pdf` — "Cache-Resident Cross-Section
Reconstruction via Singular Value Decomposition for Monte Carlo Neutron Transport"

## License

MIT
