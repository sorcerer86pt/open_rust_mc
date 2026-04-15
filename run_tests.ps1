param(
    [switch]$Quick,
    [switch]$Full,
    [switch]$Cuda,
    [switch]$Unit,
    [switch]$Godiva,
    [switch]$Pwr,
    [switch]$Jeff,
    [switch]$Mem,
    [string]$Filter = ""
)

$ErrorActionPreference = "Stop"
$RustDir = "$PSScriptRoot\rust_prototype"
$DataDir = "$PSScriptRoot\data\endfb-vii.1-hdf5\neutron"

# Determine if we run everything or specific modules
$RunAll = -not ($Unit -or $Godiva -or $Pwr -or $Jeff -or $Cuda -or $Mem)

Write-Host "`n=== open_rust_mc Test Suite ===" -ForegroundColor Cyan
Write-Host "Mode: $(if ($Full) {'Full'} else {'Quick'})"

Set-Location $RustDir

# ── 1. Unit tests ──────────────────────────────────────
if ($RunAll -or $Unit) {
    Write-Host "`n── 1. Unit tests ──" -ForegroundColor Yellow
    # If filter is empty, we just pass --lib
    $testArgs = if ($Filter) { @("--lib", $Filter) } else { @("--lib") }
    cargo test @testArgs
    Write-Host "PASS: unit tests" -ForegroundColor Green
}

# ── 2. Godiva eigenvalue ───────────────────────
if ($RunAll -or $Godiva) {
    Write-Host "`n── 2. Godiva eigenvalue ──" -ForegroundColor Yellow
    # Use an Array @() so PowerShell passes arguments separately
    $GArgs = if ($Full) { @($DataDir, "--mode", "both", "--rank", "5", "--batches", "150", "--inactive", "20", "--particles", "20000", "--seeds", "3") } 
             else { @($DataDir, "--mode", "both", "--rank", "5", "--batches", "30", "--inactive", "5", "--particles", "5000") }
    
    cargo run --release --bin godiva -- @GArgs
}

# ── 3. PWR pin cell ────────────────
if ($RunAll -or $Pwr) {
    Write-Host "`n── 3. PWR pin cell ──" -ForegroundColor Yellow
    $PwrArgs = if ($Full) { @($DataDir, "--mode", "both", "--rank", "5", "--batches", "30", "--inactive", "5", "--particles", "5000") } 
               else { @($DataDir, "--mode", "both", "--rank", "5", "--batches", "15", "--inactive", "2", "--particles", "2000") }
    
    cargo run --release --bin pwr_pincell -- @PwrArgs
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
    cargo run --release --features cuda --bin gpu_bench -- $DataDir --rank 5 --particles 1000000
}

# ── 7. Memory benchmark ─────────────────────────────────────────────
if ($RunAll -or $Mem) {
    Write-Host "`n── 7. Memory comparison ──" -ForegroundColor Yellow
    # Wrapped $DataDir to ensure it's treated as a single path string
    cargo run --release --bin bench_mem -- "$DataDir" --rank 5
}

Write-Host "`n=== All Selected Tests Passed ===`n" -ForegroundColor Green