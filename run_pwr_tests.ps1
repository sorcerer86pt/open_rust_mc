# PWR Pin Cell + S(a,b) validation suite
#
# Usage:
#   .\run_pwr_tests.ps1                    # run all tests
#   .\run_pwr_tests.ps1 -Download          # download ENDF data first
#   .\run_pwr_tests.ps1 -Output results    # log to results_YYYYMMDD_HHMM.txt
#   .\run_pwr_tests.ps1 -Debug             # tiny run for debugging crashes

param(
    [switch]$Download,
    [string]$Output,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
$DATA_ROOT = "data\endfb-vii.1-hdf5"
$DATA_NEUTRON = "$DATA_ROOT\neutron"
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"

# ── Tee helper: log to file AND console ─────────────────────────────────

$logFile = $null
if ($Output) {
    $logFile = "${Output}_${timestamp}.txt"
    Write-Host "Logging output to: $logFile"
}

function Log {
    param([string]$msg)
    Write-Host $msg
    if ($logFile) { $msg | Out-File -Append -FilePath $logFile }
}

function RunAndLog {
    param([string]$label, [scriptblock]$cmd)
    Log "`n$('=' * 60)"
    Log "  $label"
    Log "$('=' * 60)`n"
    if ($logFile) {
        & $cmd 2>&1 | Tee-Object -Append -FilePath $logFile
    } else {
        & $cmd 2>&1
    }
    if ($LASTEXITCODE -ne 0) {
        Log "  FAILED (exit code $LASTEXITCODE)"
    }
}

# ── Data download ───────────────────────────────────────────────────────

if ($Download) {
    Log "`nDownloading ENDF/B-VII.1 HDF5 nuclear data (~5.8 GB)..."
    $url = "https://anl.box.com/shared/static/9igk353zpy8fn9ttvtrqgzvw1vtejoz6.xz"
    $archive = "endfb-vii.1-hdf5.tar.xz"

    if (-not (Test-Path $DATA_ROOT)) {
        Log "  Downloading $archive ..."
        curl.exe -L -o $archive $url
        Log "  Extracting..."
        if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" | Out-Null }
        tar -xf $archive -C data
        Remove-Item $archive -ErrorAction SilentlyContinue
        Log "  Done. Data at $DATA_ROOT"
    } else {
        Log "  Data already exists at $DATA_ROOT"
    }
}

# ── Verify data ─────────────────────────────────────────────────────────

if (-not (Test-Path "$DATA_NEUTRON\U235.h5")) {
    Log "ERROR: Nuclear data not found at $DATA_NEUTRON\U235.h5"
    Log "Run with -Download to fetch: .\run_pwr_tests.ps1 -Download"
    exit 1
}

$hasThermal = Test-Path "$DATA_NEUTRON\c_H_in_H2O.h5"
Log "S(a,b) data: $(if ($hasThermal) {'FOUND'} else {'NOT FOUND'}) (c_H_in_H2O.h5)"

# ── Pick particle counts ────────────────────────────────────────────────

if ($Debug) {
    $smokeP = 100;  $smokeB = 5;   $smokeI = 1
    $modP   = 500;  $modB   = 10;  $modI   = 2
    $fullP  = 1000; $fullB  = 15;  $fullI  = 3;  $seeds = 2
    $godP   = 500;  $godB   = 10;  $godI   = 2
    Log "DEBUG MODE: tiny particle counts for crash diagnosis"
} else {
    $smokeP = 2000;  $smokeB = 20;   $smokeI = 5
    $modP   = 20000; $modB   = 100;  $modI   = 20
    $fullP  = 50000; $fullB  = 150;  $fullI  = 20;  $seeds = 5
    $godP   = 10000; $godB   = 80;   $godI   = 15
}

# ── Build ───────────────────────────────────────────────────────────────

Set-Location rust_prototype
$DATA = "..\$DATA_NEUTRON"

RunAndLog "Building release binaries" {
    cargo build --release --bin pwr_pincell --bin godiva
}

# ── Unit tests ──────────────────────────────────────────────────────────

RunAndLog "TEST 0: Unit tests (36 tests)" {
    cargo test --lib
}

# ── PWR smoke test ──────────────────────────────────────────────────────

RunAndLog "TEST 1: PWR pin cell smoke test" {
    cargo run --release --bin pwr_pincell -- $DATA `
      --mode both --rank 5 --batches $smokeB --inactive $smokeI --particles $smokeP
}

# ── PWR moderate ────────────────────────────────────────────────────────

RunAndLog "TEST 2: PWR pin cell moderate stats" {
    cargo run --release --bin pwr_pincell -- $DATA `
      --mode both --rank 5 --batches $modB --inactive $modI --particles $modP
}

# ── PWR full benchmark ──────────────────────────────────────────────────

if (-not $Debug) {
    RunAndLog "TEST 3: PWR pin cell full benchmark ($seeds seeds)" {
        cargo run --release --bin pwr_pincell -- $DATA `
          --mode both --rank 5 --batches $fullB --inactive $fullI `
          --particles $fullP --seeds $seeds
    }
}

# ── Godiva regression ───────────────────────────────────────────────────

RunAndLog "TEST 4: Godiva regression" {
    cargo run --release --bin godiva -- $DATA `
      --mode both --rank 5 --batches $godB --inactive $godI --particles $godP
}

# ── JEFF-33 ─────────────────────────────────────────────────────────────

$JEFF_PREFIX = "..\data\jeff33_"
if (Test-Path "${JEFF_PREFIX}energies.npy") {
    RunAndLog "TEST 5: JEFF-33 SVD validation" {
        cargo run --release -- npy --prefix $JEFF_PREFIX
    }
} else {
    Log "`n  SKIP TEST 5: JEFF-33 NPY data not found"
}

# ── GPU ─────────────────────────────────────────────────────────────────

RunAndLog "TEST 6a: GPU benchmark — U-235 fission (single nuclide)" {
    $env:RUST_BACKTRACE = "0"
    cargo build --release --features cuda --bin gpu_bench 2>&1
    if ($LASTEXITCODE -eq 0) {
        cargo run --release --features cuda --bin gpu_bench -- $DATA `
          --rank 5 --particles 1000000
    } else {
        Write-Host "  SKIP: CUDA build failed"
    }
}

RunAndLog "TEST 6b: GPU benchmark — PWR pin cell (8 nuclides, CPU vs CUDA)" {
    if ($LASTEXITCODE -eq 0) {
        cargo run --release --features cuda --bin gpu_bench -- $DATA `
          --rank 5 --particles 1000000 --pwr
    } else {
        Write-Host "  SKIP: CUDA build not available"
    }
}

# ── Done ────────────────────────────────────────────────────────────────

Log "`n$('=' * 60)"
Log "  ALL TESTS COMPLETE — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log "$('=' * 60)`n"

if ($logFile) {
    Log "Results saved to: $logFile"
}

Set-Location ..
