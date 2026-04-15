<#
.SYNOPSIS
    Test suite for open-rust-mc. Handles Unit tests, Physics benchmarks, and GPU acceleration.
#>
param(
    [switch]$Quick, 
    [switch]$Full, 
    [switch]$Unit, 
    [switch]$Godiva,
    [switch]$Pwr, 
    [switch]$Jeff, 
    [switch]$Cuda, 
    [switch]$Mem,
    [string]$Filter = "", 
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$OriginalLocation = $PSScriptRoot
$RustDir = "$PSScriptRoot\rust_prototype"
# Updated to the specific neutron data directory
$DataDir = "$PSScriptRoot\data\endfb-vii.1-hdf5\neutron"

# Default suite: Unit, Godiva, Jeff, and Mem
$AnyTestPicked = $Unit -or $Godiva -or $Jeff -or $Cuda -or $Mem -or $Pwr
if (-not $AnyTestPicked) { $Unit = $Godiva = $Jeff = $Mem = $true }

if ($Output) {
    Write-Host "Logging output to: $Output" -ForegroundColor Cyan
    Start-Transcript -Path $Output -Append -Force | Out-Null
}

Push-Location $OriginalLocation

try {
    Write-Host "`n=== open_rust_mc Test Suite ===" -ForegroundColor Cyan
    Set-Location $RustDir

    # 1. Unit Tests
    if ($Unit) {
        Write-Host "`n── 1. Unit tests ──" -ForegroundColor Yellow
        cargo test --lib $Filter 2>&1 | Out-String -Stream
    }

    # 2. Godiva Eigenvalue
    if ($Godiva) {
        Write-Host "`n── 2. Godiva eigenvalue ──" -ForegroundColor Yellow
        $GArgs = if ($Full) { 
            @("$DataDir", "--mode", "both", "--rank", "5", "--batches", "150", "--inactive", "20", "--particles", "20000", "--seeds", "3") 
        } else { 
            @("$DataDir", "--mode", "both", "--rank", "5", "--batches", "30", "--inactive", "5", "--particles", "5000") 
        }
        cargo run --release --bin godiva -- $GArgs 2>&1 | Out-String -Stream
    }

    # 3. PWR Pin Cell
    if ($Pwr) {
        Write-Host "`n── 3. PWR pin cell ──" -ForegroundColor Yellow
        $PArgs = if ($Full) { 
            @("$DataDir", "--mode", "both", "--rank", "5", "--batches", "100", "--inactive", "20", "--particles", "50000", "--seeds", "5") 
        } else { 
            @("$DataDir", "--mode", "both", "--rank", "5", "--batches", "30", "--inactive", "5", "--particles", "5000") 
        }
        cargo run --release --bin pwr_pincell -- $PArgs 2>&1 | Out-String -Stream
    }

    # 5. JEFF-3.3 SVD Validation
    if ($Jeff) {
        # Fix: Look in the DataDir where the .npy files were extracted
        $JeffFile = Join-Path $DataDir "jeff33_fission_energies.npy"
        
        if (Test-Path $JeffFile) {
            Write-Host "`n── 5. JEFF-33 SVD validation ──" -ForegroundColor Yellow
            cargo run --release -- npy --prefix jeff33_ 2>&1 | Out-String -Stream
        } else {
            Write-Host "`n── 5. JEFF-33 SVD validation (Skipped: $JeffFile not found) ──" -ForegroundColor Gray
        }
    }

    # 7. Memory Benchmark
    if ($Mem) {
        Write-Host "`n── 7. Memory comparison ──" -ForegroundColor Yellow
        # Passing the directory directly; bench_mem now correctly filters internal files
        cargo run --release --bin bench_mem -- "$DataDir" --rank 5 2>&1 | Out-String -Stream
    }

    Write-Host "`n=== All Selected Tests Passed ===`n" -ForegroundColor Green
}
catch {
    Write-Host "`nTEST SUITE FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
    if ($Output) { Stop-Transcript }
}