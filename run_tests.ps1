<#
.SYNOPSIS
    Developer test runner for open-rust-mc.

.DESCRIPTION
    Runs unit tests and short integration benchmarks. All paths are resolved
    relative to this script's location, so the script can be invoked from
    anywhere.

    Use -Tests to select which suites to run. Valid values:
        unit      Cargo unit tests (src/lib.rs).
        godiva    Godiva (HEU-MET-FAST-001) smoke/integration run.
        pwr       PWR pin cell smoke/integration run.
        jeff      JEFF-3.3 SVD validation (skipped if data missing).
        gpu       GPU PWR + Godiva smoke runs (requires CUDA feature).
        mem       Memory comparison (bench_mem binary).
        xs        XS reconstruction Pareto (pareto_bench binary).
        all       Every suite above.

    -Quick and -Full are presets that scale particle counts and seeds.

.EXAMPLE
    ./run_tests.ps1
    Run the default suite (unit, godiva, jeff, mem).

.EXAMPLE
    ./run_tests.ps1 -Tests pwr,gpu -Full
    Run the PWR and GPU suites at full statistics.

.EXAMPLE
    ./run_tests.ps1 -Tests unit -Filter kernel
    Run only unit tests whose name matches "kernel".
#>
[CmdletBinding()]
param(
    [ValidateSet("unit", "godiva", "pwr", "jeff", "gpu", "mem", "xs", "all")]
    [string[]]$Tests = @("unit", "godiva", "jeff", "mem"),

    [switch]$Quick,
    [switch]$Full,

    [int]$Rank = 5,
    [int]$Seeds,
    [int]$Particles,
    [int]$Batches,
    [int]$Inactive,

    [string]$Filter = "",
    [string]$Output = "",

    [string]$DataDir = (Join-Path $PSScriptRoot "data\endfb-vii.1-hdf5\neutron")
)

$ErrorActionPreference = "Stop"

# ── Resolve paths relative to script location ─────────────────────────────
$RepoRoot = $PSScriptRoot
$RustDir  = Join-Path $RepoRoot "rust_prototype"
$Manifest = Join-Path $RustDir  "Cargo.toml"
$DataDir  = (Resolve-Path -LiteralPath $DataDir -ErrorAction SilentlyContinue) ?? $DataDir

# ── Presets ───────────────────────────────────────────────────────────────
if ($Quick -and $Full) {
    throw "Choose at most one of -Quick and -Full."
}

# Default, -Quick, -Full presets for each knob; explicit CLI overrides win.
$defaults = if ($Full) {
    @{ Seeds = 5; Particles = 20000; Batches = 100; Inactive = 20 }
} elseif ($Quick) {
    @{ Seeds = 1; Particles = 2000;  Batches = 20;  Inactive = 5  }
} else {
    @{ Seeds = 2; Particles = 5000;  Batches = 30;  Inactive = 5  }
}
if (-not $PSBoundParameters.ContainsKey('Seeds'))     { $Seeds     = $defaults.Seeds }
if (-not $PSBoundParameters.ContainsKey('Particles')) { $Particles = $defaults.Particles }
if (-not $PSBoundParameters.ContainsKey('Batches'))   { $Batches   = $defaults.Batches }
if (-not $PSBoundParameters.ContainsKey('Inactive'))  { $Inactive  = $defaults.Inactive }

# Expand "all".
if ($Tests -contains "all") {
    $Tests = @("unit", "godiva", "pwr", "jeff", "gpu", "mem", "xs")
}

# ── Logging ───────────────────────────────────────────────────────────────
if ($Output) {
    $stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
    $logPath = if ($Output.EndsWith(".txt") -or $Output.EndsWith(".log")) {
        $Output
    } else {
        "${Output}_${stamp}.txt"
    }
    Write-Host "Logging to: $logPath" -ForegroundColor Cyan
    Start-Transcript -Path $logPath -Append -Force | Out-Null
}

# ── Helpers ───────────────────────────────────────────────────────────────
function Write-Banner {
    param([string]$Text, [ConsoleColor]$Color = "Yellow")
    Write-Host ""
    Write-Host ("── {0} ──" -f $Text) -ForegroundColor $Color
}

function Invoke-Cargo {
    param([string[]]$CargoArgs)
    $allArgs = @("--manifest-path", $Manifest) + $CargoArgs
    Write-Verbose ("cargo " + ($allArgs -join " "))
    & cargo @allArgs
    if ($LASTEXITCODE -ne 0) {
        throw "cargo exited with code $LASTEXITCODE"
    }
}

# Verify nuclear data except when only running unit/xs/mem.
$needData = @("godiva", "pwr", "gpu") | Where-Object { $Tests -contains $_ }
if ($needData -and -not (Test-Path (Join-Path $DataDir "U235.h5"))) {
    throw "Nuclear data not found at ${DataDir}. Pass -DataDir or place ENDF/B-VII.1 HDF5 there."
}

Write-Host ""
Write-Host "=== open-rust-mc test runner ===" -ForegroundColor Cyan
Write-Host ("  Suites    : {0}" -f ($Tests -join ", "))
Write-Host ("  Rank      : {0}" -f $Rank)
Write-Host ("  Seeds     : {0}" -f $Seeds)
Write-Host ("  Particles : {0}" -f $Particles)
Write-Host ("  Batches   : {0} ({1} inactive)" -f $Batches, $Inactive)
Write-Host ("  Data      : {0}" -f $DataDir)
Write-Host ""

try {
    # ── 1. Unit tests ─────────────────────────────────────────────────────
    if ($Tests -contains "unit") {
        Write-Banner "Unit tests"
        $testArgs = @("test", "--lib")
        if ($Filter) { $testArgs += $Filter }
        Invoke-Cargo $testArgs
    }

    # ── 2. Godiva ─────────────────────────────────────────────────────────
    if ($Tests -contains "godiva") {
        Write-Banner "Godiva eigenvalue"
        Invoke-Cargo @(
            "run", "--release", "--bin", "godiva", "--",
            $DataDir,
            "--mode", "both", "--rank", $Rank,
            "--batches", $Batches, "--inactive", $Inactive,
            "--particles", $Particles, "--seeds", $Seeds
        )
    }

    # ── 3. PWR pin cell ───────────────────────────────────────────────────
    if ($Tests -contains "pwr") {
        Write-Banner "PWR pin cell"
        Invoke-Cargo @(
            "run", "--release", "--bin", "pwr_pincell", "--",
            $DataDir,
            "--mode", "both", "--rank", $Rank,
            "--batches", $Batches, "--inactive", $Inactive,
            "--particles", $Particles, "--seeds", $Seeds
        )
    }

    # ── 4. JEFF-3.3 ───────────────────────────────────────────────────────
    if ($Tests -contains "jeff") {
        $jeffFile = Join-Path $RepoRoot "data\jeff33_fission_energies.npy"
        if (Test-Path $jeffFile) {
            Write-Banner "JEFF-3.3 SVD validation"
            Invoke-Cargo @(
                "run", "--release", "--",
                "npy", "--prefix", (Join-Path $RepoRoot "data\jeff33_")
            )
        } else {
            Write-Banner "JEFF-3.3 SVD validation (skipped: data not found)" "DarkGray"
        }
    }

    # ── 5. GPU (CUDA) ─────────────────────────────────────────────────────
    if ($Tests -contains "gpu") {
        Write-Banner "GPU PWR"
        Invoke-Cargo @(
            "run", "--release", "--features", "cuda",
            "--bin", "gpu_pwr_bench", "--",
            $DataDir,
            "--rank", $Rank, "-B", $Batches, "--inactive", $Inactive,
            "--particles", $Particles, "--seeds", $Seeds,
            "--geometry", "pwr"
        )
        Write-Banner "GPU Godiva"
        Invoke-Cargo @(
            "run", "--release", "--features", "cuda",
            "--bin", "gpu_pwr_bench", "--",
            $DataDir,
            "--rank", $Rank, "-B", $Batches, "--inactive", $Inactive,
            "--particles", $Particles, "--seeds", $Seeds,
            "--geometry", "godiva"
        )
    }

    # ── 6. Memory comparison ──────────────────────────────────────────────
    if ($Tests -contains "mem") {
        Write-Banner "Memory comparison"
        Invoke-Cargo @(
            "run", "--release", "--bin", "bench_mem", "--",
            $DataDir, "--rank", $Rank
        )
    }

    # ── 7. XS Pareto ──────────────────────────────────────────────────────
    if ($Tests -contains "xs") {
        Write-Banner "XS reconstruction Pareto (pareto_bench)"
        $outDir = Join-Path $RepoRoot "outputs\pareto"
        if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
        $csv = Join-Path $outDir "xs_accuracy.csv"
        Write-Host "  writing $csv"
        # pareto_bench prints CSV to stdout; redirect to the canonical path.
        Invoke-Cargo @("run", "--release", "--bin", "pareto_bench", "--", $DataDir) `
            *> $csv
    }

    Write-Host ""
    Write-Host "=== All selected suites passed ===" -ForegroundColor Green
}
finally {
    if ($Output) { Stop-Transcript | Out-Null }
}
