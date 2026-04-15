# run_tests.ps1 — Full test suite for open_rust_mc
# Run from repo root: .\run_tests.ps1
# Use -Quick for fast smoke test, -Full for benchmark

param(
    [switch]$Quick,
    [switch]$Full,
    [switch]$Cuda
)

$ErrorActionPreference = "Continue"
$RustDir = "$PSScriptRoot\rust_prototype"
$DataDir = "$PSScriptRoot\data\endfb-vii.1-hdf5\neutron"

Set-Location $RustDir

Write-Host "`n=== open_rust_mc Test Suite ===" -ForegroundColor Cyan
Write-Host "Rust dir: $RustDir"
Write-Host "Data dir: $DataDir`n"

# ── 1. Cargo test (unit tests) ──────────────────────────────────────
Write-Host "── 1. Unit tests ──" -ForegroundColor Yellow
cargo test --lib 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: unit tests" -ForegroundColor Red; exit 1 }
Write-Host "PASS: all unit tests`n" -ForegroundColor Green

# ── 2. Godiva eigenvalue (fast Godiva sphere) ───────────────────────
Write-Host "── 2. Godiva eigenvalue (quick) ──" -ForegroundColor Yellow
if ($Full) {
    cargo run --release --bin godiva -- $DataDir --mode both --rank 5 --batches 150 --inactive 20 --particles 20000 --seeds 3
} else {
    cargo run --release --bin godiva -- $DataDir --mode both --rank 5 --batches 30 --inactive 5 --particles 5000
}
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: godiva" -ForegroundColor Red; exit 1 }
Write-Host "`nPASS: Godiva`n" -ForegroundColor Green

# ── 3. PWR pin cell (with S(a,b) thermal scattering) ────────────────
Write-Host "── 3. PWR pin cell ──" -ForegroundColor Yellow
if ($Full) {
    cargo run --release --bin pwr_pincell -- $DataDir --mode both --rank 5 --batches 100 --inactive 20 --particles 50000 --seeds 3
} else {
    cargo run --release --bin pwr_pincell -- $DataDir --mode both --rank 5 --batches 30 --inactive 5 --particles 5000
}
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: pwr_pincell" -ForegroundColor Red; exit 1 }
Write-Host "`nPASS: PWR pin cell`n" -ForegroundColor Green

# ── 4. PWR pin cell — SVD only (for timing) ─────────────────────────
Write-Host "── 4. PWR pin cell SVD-only (timing) ──" -ForegroundColor Yellow
if ($Full) {
    cargo run --release --bin pwr_pincell -- $DataDir --mode svd --rank 5 --batches 50 --inactive 10 --particles 20000 --seeds 3
} else {
    cargo run --release --bin pwr_pincell -- $DataDir --mode svd --rank 5 --batches 20 --inactive 5 --particles 5000
}
Write-Host ""

# ── 5. JEFF-33 SVD benchmark (if .npy files exist) ──────────────────
$JeffPrefix = "$RustDir\..\data\jeff33_"
if (Test-Path "${JeffPrefix}fission_energies.npy") {
    Write-Host "── 5. JEFF-33 SVD validation ──" -ForegroundColor Yellow
    cargo run --release -- npy --prefix jeff33_
    Write-Host ""
} else {
    Write-Host "── 5. JEFF-33: skipped (no .npy files) ──" -ForegroundColor DarkGray
}

# ── 6. GPU/CUDA benchmark (optional, requires --features cuda) ──────
if ($Cuda) {
    Write-Host "── 6. GPU benchmark ──" -ForegroundColor Yellow
    cargo build --release --features cuda --bin gpu_bench 2>&1
    if ($LASTEXITCODE -eq 0) {
        cargo run --release --features cuda --bin gpu_bench -- $DataDir --rank 5 --particles 1000000
    } else {
        Write-Host "SKIP: CUDA build failed (check CUDA toolkit)" -ForegroundColor DarkGray
    }
    Write-Host ""
} else {
    Write-Host "── 6. GPU: skipped (use -Cuda flag) ──" -ForegroundColor DarkGray
}

# ── 7. Memory benchmark ─────────────────────────────────────────────
Write-Host "`n── 7. Memory comparison ──" -ForegroundColor Yellow
cargo run --release --bin bench_mem -- $DataDir --rank 5
Write-Host ""

Write-Host "=== All tests complete ===" -ForegroundColor Cyan
