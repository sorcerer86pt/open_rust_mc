#!/bin/bash
# ============================================================================
# Setup OpenMC in WSL for reference benchmarking
#
# Run from Windows: wsl -d Ubuntu-24.04 bash scripts/setup_openmc_wsl.sh
# Or from within WSL: bash scripts/setup_openmc_wsl.sh
#
# Creates a conda environment with OpenMC + Python for:
#   - Running reference k_eff benchmarks
#   - Generating ENDF/B-VII.1 HDF5 cross-section data
#   - Validating our SVD reconstruction against OpenMC's native lookup
# ============================================================================

set -e

echo "========================================"
echo "  OpenMC Setup for WSL"
echo "========================================"

# ── Install miniconda if not present ──
if ! command -v conda &> /dev/null; then
    echo ""
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    rm /tmp/miniconda.sh
    echo "Miniconda installed. Restart shell and rerun this script."
    exit 0
fi

eval "$(conda shell.bash hook)"

# ── Create openmc environment ──
ENV_NAME="openmc"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo ""
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" -y python=3.11
fi

conda activate "$ENV_NAME"

# ── Install OpenMC ──
echo ""
echo "Installing OpenMC..."
conda install -y -c conda-forge openmc

# ── Install Python dependencies for scripts ──
pip install numpy matplotlib h5py

# ── Verify installation ──
echo ""
echo "========================================"
echo "  Verification"
echo "========================================"
python -c "import openmc; print(f'OpenMC version: {openmc.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# ── Set cross-section data path ──
# The Windows data directory is accessible from WSL via /mnt/c/...
WINDATA="/mnt/c/Users/$(whoami)/madman_svd_experiment/data"
if [ -d "$WINDATA/endfb-vii.1-hdf5" ]; then
    export OPENMC_CROSS_SECTIONS="$WINDATA/endfb-vii.1-hdf5/cross_sections.xml"
    echo ""
    echo "Cross-section data: $OPENMC_CROSS_SECTIONS"
    echo ""
    echo "Add to your .bashrc:"
    echo "  export OPENMC_CROSS_SECTIONS=\"$OPENMC_CROSS_SECTIONS\""
else
    echo ""
    echo "WARNING: Nuclear data not found at $WINDATA/endfb-vii.1-hdf5"
    echo "Run setup_nuclear_data.ps1 on Windows first, or set OPENMC_CROSS_SECTIONS manually."
fi

echo ""
echo "========================================"
echo "  Setup Complete"
echo "========================================"
echo ""
echo "Usage:"
echo "  conda activate openmc"
echo "  python scripts/paper_openmc_benchmark.py     # Run OpenMC reference"
echo "  python scripts/honesty_test.py               # Compare SVD vs OpenMC"
