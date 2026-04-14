# How to Resume This Project

## Quick Start (new Claude Code session)

Paste this as your first message:

```
Read C:\Users\fog\madman_svd_experiment\CLAUDE.md then continue working on open_rust_mc. 

Current state: pure Rust Monte Carlo engine gets k_eff=1.00016 +/- 0.00080 on Godiva
(OpenMC gets 0.99857). Gap = 16 pcm from experiment. Repo: https://github.com/sorcerer86pt/open_rust_mc

Completed physics:
- Energy-dependent nu-bar (total = prompt + delayed) from HDF5
- Discrete inelastic levels (MT=51-91) with real Q-values
- Continuum inelastic (MT=91) evaporation spectrum
- (n,3n) MT=17
- Anisotropic scattering angular distributions (tabular CDF from HDF5)
- Data-driven fission energy spectrum (continuous tabulated from HDF5)
- Proper two-body kinematics for all inelastic channels
- URR probability tables (20-band sampling, multiply_smooth + absolute)
- Stochastic interpolation between energy bins for all distributions
- Free gas thermal scattering (Maxwell-Boltzmann target velocity)
- Rayon parallel transport (8.7x speedup)

Remaining work:
- S(alpha,beta) thermal scattering (critical for PWR pin cell)
- Single-pass HDF5 loading (currently re-reads per reaction)
- Event-based transport + BVH integration
- Photon transport, depletion

Working directory: C:\Users\fog\madman_svd_experiment
Rust project: C:\Users\fog\madman_svd_experiment\rust_prototype
Nuclear data: C:\Users\fog\madman_svd_experiment\data\endfb-vii.1-hdf5\neutron
Paper: C:\Users\fog\madman_svd_experiment\paper\svd_cross_section_compression.tex

Git is configured for sorcerer86pt with GPG signing (gpg.program=/usr/bin/gpg).
OpenMC is available in WSL: wsl -d Ubuntu-24.04, conda activate openmc.
```

## Environment Setup (already done, just verify)

```bash
# Rust
cd ~/madman_svd_experiment/rust_prototype && cargo test --lib
# Should show 26 passing tests

# Python venv (for analysis scripts)
source ~/madman_svd_experiment/.venv/Scripts/activate
python -c "import openmc; print(openmc.__version__)"

# OpenMC in WSL (for k_eff validation)
wsl -d Ubuntu-24.04 -- bash -c 'source ~/miniforge3/bin/activate openmc && openmc --version'

# Nuclear data
ls ~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron/U235.h5

# Git
cd ~/madman_svd_experiment && git status && git log --oneline -3
```

## If starting completely fresh (new machine)

```bash
# Clone
git clone https://github.com/sorcerer86pt/open_rust_mc.git ~/madman_svd_experiment

# Rust deps
cd ~/madman_svd_experiment/rust_prototype && cargo build --release

# Python deps
python -m venv ~/madman_svd_experiment/.venv
source ~/madman_svd_experiment/.venv/Scripts/activate
pip install openmc h5py numpy scipy matplotlib

# Nuclear data (1.6 GB download)
# Get ENDF/B-VII.1 HDF5 from https://openmc.org/data/
# Extract to ~/madman_svd_experiment/data/endfb-vii.1-hdf5/

# Run Godiva to verify
cargo run --release --bin godiva -- ~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron \
  --rank 5 --batches 50 --inactive 10 --particles 5000
# Should get k_eff ≈ 0.994
```

## Key commands

```bash
# Run all Rust tests
cd ~/madman_svd_experiment/rust_prototype && cargo test --lib

# Run Godiva eigenvalue (pure Rust)
cargo run --release --bin godiva -- ~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron --rank 5

# Explore HDF5 structure of a nuclide file
cargo run --release -- explore ~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron/U235.h5

# Validate against OpenMC reference
cargo run --release --bin validate_vs_openmc -- \
  ~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron/U235.h5 \
  ~/madman_svd_experiment/outputs/openmc_ref_u235_mt18.npy \
  ~/madman_svd_experiment/outputs/openmc_ref_energies.npy

# Run OpenMC Godiva benchmark (WSL)
wsl -d Ubuntu-24.04 -- bash -c 'source ~/miniforge3/bin/activate openmc && cd /mnt/c/Users/fog/madman_svd_experiment/scripts && python phase4_keff_benchmark.py'

# Compile paper
cd ~/madman_svd_experiment/paper && pdflatex svd_cross_section_compression.tex

# GPU benchmark (WSL with CUDA)
wsl -d Ubuntu-24.04 -- bash -c 'source ~/miniforge3/bin/activate openmc && cd /mnt/c/Users/fog/madman_svd_experiment/cuda_bench && nvcc -O3 -arch=sm_86 svd_gpu_bench.cu -o svd_gpu_bench && ./svd_gpu_bench'
```

## What NOT to do

- Don't bypass GPG signing (configured for sorcerer86pt)
- Don't delete `data/endfb-vii.1-hdf5/` (5.8 GB, slow to re-download)
- Don't amend commits (create new ones)
- Don't push to main without tests passing
