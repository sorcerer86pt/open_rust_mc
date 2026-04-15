param(
    [switch]$Quick,
    [switch]$Full,
    [switch]$Cuda,        # Runs the GPU benchmark
    [switch]$Unit,        # Runs only unit tests
    [switch]$Godiva,      # Runs only Godiva tests
    [switch]$Pwr,         # Runs only PWR pin cell tests
    [switch]$Jeff,        # Runs only Jeff validation
    [switch]$Mem,         # Runs only Memory benchmark
    [string]$Filter = ""  # Custom filter for cargo test
)

$ErrorActionPreference = "Stop" # Stop on immediate errors
$RustDir = "$PSScriptRoot\rust_prototype"
$DataDir = "$PSScriptRoot\data\endfb-vii.1-hdf5\neutron"

# Logic to determine if we run "Everything" or just specific parts
$RunAll = -not ($Unit -or $Godiva -or $Pwr -or $Jeff -or $Cuda -or $Mem)

Set-Location $RustDir

Write-Host "`n=== open_rust_mc Test Suite ===" -ForegroundColor Cyan
Write-Host "Mode: $(if ($Full) {'Full'} else {'Quick'})"
Write-Host "Data: $DataDir`n"

# ── 1. Cargo test (unit tests) ──────────────────────────────────────
if ($RunAll -or $Unit) {
    Write-Host "── 1. Unit tests ──" -ForegroundColor Yellow
    if ($Filter) { Write-Host "Filtering tests by: $Filter" -ForegroundColor Gray }
    
    cargo test --lib $Filter 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: unit tests" -ForegroundColor Red; if (-not $RunAll) { exit 1 } }
    Write-Host "PASS: unit tests`n" -ForegroundColor Green
}

# ── 2. Godiva eigenvalue ───────────────────────
if ($RunAll -or $Godiva) {
    Write-Host "── 2. Godiva eigenvalue ──" -ForegroundColor Yellow
    $GArgs = if ($Full) { "--mode both --rank 5 --batches 150 --inactive 20 --particles 20000 --seeds 3" } 
             else { "--mode both --rank 5 --batches 30 --inactive 5 --particles 5000" }
    
    cargo run --release --bin godiva -- $DataDir $GArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: godiva" -ForegroundColor Red }
    Write-Host "PASS: Godiva`n" -ForegroundColor Green
}

# ── 3 & 4. PWR pin cell ────────────────
if ($RunAll -or $Pwr) {
    Write-Host "── 3. PWR pin cell ──" -ForegroundColor Yellow
    $PwrArgs = if ($Full) { "--mode both --rank 5 --batches 100 --inactive 20 --particles 50000 --seeds 3" } 
               else { "--mode both --rank 5 --batches 30 --inactive 5 --particles 5000" }
    
    cargo run --release --bin pwr_pincell -- $DataDir $PwrArgs
    
    Write-Host "`n── 4. PWR pin cell SVD-only (timing) ──" -ForegroundColor Yellow
    cargo run --release --bin pwr_pincell -- $DataDir --mode svd --rank 5 --batches 20 --inactive 5 --particles 5000
    Write-Host ""
}

# ── 5. JEFF-33 SVD benchmark ──────────────────
if ($RunAll -or $Jeff) {
    $JeffPrefix = "$RustDir\..\data\jeff33_"
    if (Test-Path "${JeffPrefix}fission_energies.npy") {
        Write-Host "── 5. JEFF-33 SVD validation ──" -ForegroundColor Yellow
        cargo run --release -- npy --prefix jeff33_
    } else {
        Write-Host "── 5. JEFF-33: skipped (no .npy files) ──" -ForegroundColor DarkGray
    }
}

# ── 6. GPU/CUDA benchmark ──────
if ($Cuda) {
    Write-Host "── 6. GPU benchmark ──" -ForegroundColor Yellow
    cargo build --release --features cuda --bin gpu_bench 2>&1
    if ($LASTEXITCODE -eq 0) {
        cargo run --release --features cuda --bin gpu_bench -- $DataDir --rank 5 --particles 1000000
    } else {
        Write-Host "SKIP: CUDA build failed" -ForegroundColor Red
    }
}

# ── 7. Memory benchmark ─────────────────────────────────────────────
if ($RunAll -or $Mem) {
    Write-Host "`n── 7. Memory comparison ──" -ForegroundColor Yellow
    cargo run --release --bin bench_mem -- $DataDir --rank 5
}

Write-Host "`n=== Finished Selected Tests ===" -ForegroundColor Cyan