param(
    [switch]$Quick,
    [switch]$Full,
    [switch]$Cuda,        # Runs the GPU benchmark
    [switch]$Unit,        # Runs only unit tests
    [switch]$Godiva,      # Runs only Godiva tests
    [switch]$Pwr,         # Runs only PWR pin cell tests
    [switch]$Jeff,        # Runs only Jeff validation
    [switch]$Mem,         # Runs only Memory benchmark
    [string]$Filter = ""  # Custom filter for cargo test (e.g., -Filter "tests::my_test")
)

$ErrorActionPreference = "Stop"
$RustDir = "$PSScriptRoot\rust_prototype"
$DataDir = "$PSScriptRoot\data\endfb-vii.1-hdf5\neutron"

# Determine if we run everything or specific parts
$RunAll = -not ($Unit -or $Godiva -or $Pwr -or $Jeff -or $Cuda -or $Mem)


Write-Host "`n=== open_rust_mc Test Suite ===" -ForegroundColor Cyan
Write-Host "Mode: $(if ($Full) {'Full'} else {'Quick'})"

Set-Location $RustDir

# ── 1. Unit tests ──────────────────────────────────────
if ($RunAll -or $Unit) {
    Write-Host "`n── 1. Unit tests ──" -ForegroundColor Yellow
    cargo test --lib $Filter
    Write-Host "PASS: unit tests" -ForegroundColor Green
}

# ── 2. Godiva eigenvalue ───────────────────────
if ($RunAll -or $Godiva) {
    Write-Host "`n── 2. Godiva eigenvalue ──" -ForegroundColor Yellow
    $GArgs = if ($Full) { "--mode both --rank 5 --batches 150 --inactive 20 --particles 20000 --seeds 3" } 
             else { "--mode both --rank 5 --batches 30 --inactive 5 --particles 5000" }
    cargo run --release --bin godiva -- $DataDir $GArgs
}

# ── 3 & 4. PWR pin cell ────────────────
if ($RunAll -or $Pwr) {
    Write-Host "`n── 3. PWR pin cell ──" -ForegroundColor Yellow
    $PwrArgs = if ($Full) { "--mode both --rank 5 --batches 30 --inactive 5 --particles 5000" } 
               else { "--mode both --rank 5 --batches 15 --inactive 2 --particles 2000" }
    cargo run --release --bin pwr_pincell -- $DataDir $PwrArgs
}

# ── 5. JEFF-33 SVD benchmark ──────────────────
if ($RunAll -or $Jeff) {
    $JeffPrefix = "$RustDir\..\data\jeff33_"
    if (Test-Path "${JeffPrefix}fission_energies.npy") {
        Write-Host "`n── 5. JEFF-33 SVD validation ──" -ForegroundColor Yellow
        cargo run --release -- npy --prefix jeff33_
    }
}

# ── 6. GPU/CUDA benchmark ──────
if ($Cuda) {
    Write-Host "`n── 6. GPU benchmark ──" -ForegroundColor Yellow
    # Note: We keep the feature flag here to ensure it compiles with CUDA support
    cargo run --release --features cuda --bin gpu_bench -- $DataDir --rank 5 --particles 1000000
}

# ── 7. Memory benchmark ─────────────────────────────────────────────
if ($RunAll -or $Mem) {
    Write-Host "`n── 7. Memory comparison ──" -ForegroundColor Yellow
    cargo run --release --bin bench_mem -- $DataDir --rank 5
}

Write-Host "`n=== All Selected Tests Passed ===`n" -ForegroundColor Green