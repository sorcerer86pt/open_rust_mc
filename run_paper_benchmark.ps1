<#
.SYNOPSIS
    Paper-grade benchmark runner: CPU table, CPU SVD, GPU pointwise, GPU SVD.

.DESCRIPTION
    Runs the CPU and GPU transport benchmarks that appear in the paper.
    All paths are resolved relative to this script's location.

    -Modes selects which providers to run:
        cpu-table       CPU pointwise table (pwr_pincell or godiva, --mode table).
        cpu-svd         CPU SVD reconstruction (--mode svd).
        gpu-pointwise   GPU event-based solver, default pointwise path.
        gpu-svd         GPU event-based solver, --force-svd path.
        all             Every mode above.

    -Geometry selects which benchmark:
        pwr      PWR pin cell (9 nuclides, S(a,b) thermal).
        godiva   Godiva HEU-MET-FAST-001 (3 nuclides, fast).
        both     Run every selected mode on both geometries.

    -Quick and -Full are presets that scale particle counts and seeds.

.EXAMPLE
    ./run_paper_benchmark.ps1
    Default: every mode on both geometries, 5 seeds, 50 batches, 10k particles.

.EXAMPLE
    ./run_paper_benchmark.ps1 -Modes gpu-svd -Geometry pwr -Full
    Just the GPU force-SVD path on PWR, at full production statistics.

.EXAMPLE
    ./run_paper_benchmark.ps1 -Modes cpu-table,cpu-svd -Rank 6 -Output bench
    CPU table and CPU SVD at rank 6, log to bench_<timestamp>.txt.
#>
[CmdletBinding()]
param(
    [ValidateSet("cpu-table", "cpu-svd", "gpu-pointwise", "gpu-svd", "all")]
    [string[]]$Modes = @("all"),

    [ValidateSet("pwr", "godiva", "both")]
    [string]$Geometry = "both",

    [switch]$Quick,
    [switch]$Full,

    [int]$Rank = 5,
    [int]$Seeds,
    [int]$Particles,
    [int]$Batches,
    [int]$Inactive,

    [string]$Output = "",

    [string]$DataDir = (Join-Path $PSScriptRoot "data\endfb-vii.1-hdf5\neutron")
)

$ErrorActionPreference = "Stop"

# ── Resolve paths ─────────────────────────────────────────────────────────
$RepoRoot = $PSScriptRoot
$RustDir  = Join-Path $RepoRoot "rust_prototype"
$Manifest = Join-Path $RustDir  "Cargo.toml"
$DataDir  = (Resolve-Path -LiteralPath $DataDir -ErrorAction SilentlyContinue) ?? $DataDir

# ── Presets ───────────────────────────────────────────────────────────────
if ($Quick -and $Full) {
    throw "Choose at most one of -Quick and -Full."
}

$defaults = if ($Full) {
    @{ Seeds = 10; Particles = 50000; Batches = 150; Inactive = 20 }
} elseif ($Quick) {
    @{ Seeds = 1;  Particles = 5000;  Batches = 30;  Inactive = 5  }
} else {
    @{ Seeds = 5;  Particles = 10000; Batches = 50;  Inactive = 20 }
}
if (-not $PSBoundParameters.ContainsKey('Seeds'))     { $Seeds     = $defaults.Seeds }
if (-not $PSBoundParameters.ContainsKey('Particles')) { $Particles = $defaults.Particles }
if (-not $PSBoundParameters.ContainsKey('Batches'))   { $Batches   = $defaults.Batches }
if (-not $PSBoundParameters.ContainsKey('Inactive'))  { $Inactive  = $defaults.Inactive }

# Expand "all" and "both".
if ($Modes -contains "all") {
    $Modes = @("cpu-table", "cpu-svd", "gpu-pointwise", "gpu-svd")
}
$geomList = switch ($Geometry) {
    "both"   { @("pwr", "godiva") }
    default  { @($Geometry) }
}

# ── Data check ────────────────────────────────────────────────────────────
if (-not (Test-Path (Join-Path $DataDir "U235.h5"))) {
    throw "Nuclear data not found at ${DataDir}. Pass -DataDir or place ENDF/B-VII.1 HDF5 there."
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
function Write-Section {
    param([string]$Title)
    $rule = "=" * 70
    Write-Host ""
    Write-Host $rule -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host $rule -ForegroundColor Cyan
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

function Invoke-CpuRun {
    param(
        [string]$Geom,      # "pwr" or "godiva"
        [string]$XsMode     # "table" or "svd"
    )
    $bin = if ($Geom -eq "godiva") { "godiva" } else { "pwr_pincell" }
    Write-Section "$bin --mode $XsMode (rank=$Rank, $Seeds seeds)"
    Invoke-Cargo @(
        "run", "--release", "--bin", $bin, "--",
        $DataDir,
        "--mode", $XsMode, "--rank", $Rank,
        "--batches", $Batches, "--inactive", $Inactive,
        "--particles", $Particles, "--seeds", $Seeds
    )
}

function Invoke-GpuRun {
    param(
        [string]$Geom,      # "pwr" or "godiva"
        [bool]$ForceSvd
    )
    $label = if ($ForceSvd) { "GPU SVD (--force-svd)" } else { "GPU pointwise" }
    Write-Section "$label on $Geom (rank=$Rank, $Seeds seeds)"
    $baseArgs = @(
        "run", "--release", "--features", "cuda",
        "--bin", "gpu_pwr_bench", "--",
        $DataDir,
        "--rank", $Rank, "-B", $Batches, "--inactive", $Inactive,
        "--particles", $Particles, "--seeds", $Seeds,
        "--geometry", $Geom
    )
    if ($ForceSvd) { $baseArgs += "--force-svd" }
    Invoke-Cargo $baseArgs
}

# ── Summary banner ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== open-rust-mc paper benchmark ===" -ForegroundColor Cyan
Write-Host ("  Modes      : {0}" -f ($Modes -join ", "))
Write-Host ("  Geometries : {0}" -f ($geomList -join ", "))
Write-Host ("  Rank       : {0}" -f $Rank)
Write-Host ("  Seeds      : {0}" -f $Seeds)
Write-Host ("  Batches    : {0} ({1} inactive, {2} active)" -f $Batches, $Inactive, ($Batches - $Inactive))
Write-Host ("  Particles  : {0}/batch" -f $Particles)
Write-Host ("  Data       : {0}" -f $DataDir)
Write-Host ""

$tStart = Get-Date
try {
    foreach ($g in $geomList) {
        if ($Modes -contains "cpu-table")     { Invoke-CpuRun -Geom $g -XsMode "table" }
        if ($Modes -contains "cpu-svd")       { Invoke-CpuRun -Geom $g -XsMode "svd"   }
        if ($Modes -contains "gpu-pointwise") { Invoke-GpuRun -Geom $g -ForceSvd $false }
        if ($Modes -contains "gpu-svd")       { Invoke-GpuRun -Geom $g -ForceSvd $true  }
    }

    $elapsed = (Get-Date) - $tStart
    Write-Host ""
    Write-Host ("=== Benchmarks complete in {0:mm\:ss} ===" -f $elapsed) -ForegroundColor Green
}
finally {
    if ($Output) { Stop-Transcript | Out-Null }
}
