# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    SVD rank sweep for Godiva and PWR pin cell (CPU and GPU).

.DESCRIPTION
    Iterates the transport benchmarks across multiple SVD ranks and writes
    one CSV row per (geometry, platform, rank) to the results directory.
    Paths are resolved relative to this script's location, so the script
    works regardless of the shell's current directory.

    -Geometries   Subset of {godiva, pwr} to sweep. Default: both.
    -Platforms    Subset of {cpu, gpu} to sweep. Default: cpu.
    -Ranks        SVD ranks to test. Default: 2..6.

    For the CPU path, the same engine also runs the pointwise-table
    baseline (once, rank-independent).

.EXAMPLE
    ./scripts/run_rank_sweep.ps1
    CPU sweep, ranks 2..6, both geometries, 5 seeds.

.EXAMPLE
    ./scripts/run_rank_sweep.ps1 -Platforms cpu,gpu -Geometries pwr -Ranks 3,5,6
    CPU and GPU, PWR only, ranks 3/5/6.
#>
[CmdletBinding()]
param(
    [ValidateSet("godiva", "pwr")]
    [string[]]$Geometries = @("godiva", "pwr"),

    [ValidateSet("cpu", "gpu")]
    [string[]]$Platforms = @("cpu"),

    [int[]]$Ranks = @(2, 3, 4, 5, 6),

    [switch]$Quick,
    [switch]$Full,

    [int]$Seeds,
    [int]$Particles,
    [int]$Batches,
    [int]$Inactive,

    [string]$DataDir     = (Join-Path $PSScriptRoot "..\data\endfb-vii.1-hdf5\neutron"),
    [string]$ResultsDir  = (Join-Path $PSScriptRoot "..\outputs\rank_sweep")
)

$ErrorActionPreference = "Stop"

# ── Resolve paths ─────────────────────────────────────────────────────────
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $RepoRoot "rust_prototype\Cargo.toml"
$DataDir  = (Resolve-Path -LiteralPath $DataDir -ErrorAction SilentlyContinue) ?? $DataDir

if (-not (Test-Path (Join-Path $DataDir "U235.h5"))) {
    throw "Nuclear data not found at ${DataDir}. Pass -DataDir."
}

if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

# ── Presets ───────────────────────────────────────────────────────────────
if ($Quick -and $Full) {
    throw "Choose at most one of -Quick and -Full."
}
$defaults = if ($Full) {
    @{ Seeds = 10; Particles = 50000; Batches = 150; Inactive = 20 }
} elseif ($Quick) {
    @{ Seeds = 2;  Particles = 3000;  Batches = 30;  Inactive = 5  }
} else {
    @{ Seeds = 5;  Particles = 10000; Batches = 50;  Inactive = 20 }
}
if (-not $PSBoundParameters.ContainsKey('Seeds'))     { $Seeds     = $defaults.Seeds }
if (-not $PSBoundParameters.ContainsKey('Particles')) { $Particles = $defaults.Particles }
if (-not $PSBoundParameters.ContainsKey('Batches'))   { $Batches   = $defaults.Batches }
if (-not $PSBoundParameters.ContainsKey('Inactive'))  { $Inactive  = $defaults.Inactive }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Invoke-Cargo {
    param([string[]]$CargoArgs)
    $allArgs = @("--manifest-path", $Manifest) + $CargoArgs
    Write-Verbose ("cargo " + ($allArgs -join " "))
    & cargo @allArgs 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "cargo exited with code $LASTEXITCODE"
    }
}

# Parse k_eff mean and sigma from a standalone-engine stdout blob.
function Parse-CpuResult {
    param([string]$Text, [string]$Mode)
    $patterns = @{
        "svd"   = '(?s)SVD.*?k_inf\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+).*?ns/particle\s*=\s*([\d.]+)'
        "table" = '(?s)Pointwise Table.*?k_inf\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+).*?ns/particle\s*=\s*([\d.]+)'
    }
    if ($Text -match $patterns[$Mode]) {
        return [pscustomobject]@{
            k     = [double]$Matches[1]
            sigma = [double]$Matches[2]
            nsp   = [double]$Matches[3]
        }
    }
    # Godiva binary uses "k_eff" in place of "k_inf".
    $altPatterns = @{
        "svd"   = '(?s)SVD.*?k_eff\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+).*?ns/particle\s*=\s*([\d.]+)'
        "table" = '(?s)Pointwise Table.*?k_eff\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+).*?ns/particle\s*=\s*([\d.]+)'
    }
    if ($Text -match $altPatterns[$Mode]) {
        return [pscustomobject]@{
            k     = [double]$Matches[1]
            sigma = [double]$Matches[2]
            nsp   = [double]$Matches[3]
        }
    }
    return $null
}

function Parse-GpuResult {
    param([string]$Text)
    if ($Text -match '(?s)k_inf\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+).*?ns/particle\s*=\s*([\d.]+)') {
        return [pscustomobject]@{
            k     = [double]$Matches[1]
            sigma = [double]$Matches[2]
            nsp   = [double]$Matches[3]
        }
    }
    return $null
}

function Run-CpuSweep {
    param([string]$Geom)
    $bin = if ($Geom -eq "godiva") { "godiva" } else { "pwr_pincell" }
    $csv = Join-Path $ResultsDir "sweep_cpu_${Geom}_${stamp}.csv"
    "rank,mode,k,sigma_seed,ns_per_particle" | Out-File $csv -Encoding utf8

    Write-Host ""
    Write-Host ("── CPU sweep on {0} ({1}) ──" -f $Geom, $bin) -ForegroundColor Yellow
    Write-Host ("   ranks={0}  seeds={1}  particles={2}  batches={3}" `
        -f ($Ranks -join ","), $Seeds, $Particles, $Batches)

    foreach ($rank in $Ranks) {
        Write-Host ("   rank={0}..." -f $rank) -ForegroundColor DarkYellow
        $out = Invoke-Cargo @(
            "run", "--release", "--bin", $bin, "--",
            $DataDir,
            "--mode", "both", "--rank", $rank,
            "--batches", $Batches, "--inactive", $Inactive,
            "--particles", $Particles, "--seeds", $Seeds
        )
        $svd = Parse-CpuResult -Text $out -Mode "svd"
        $tbl = Parse-CpuResult -Text $out -Mode "table"
        if ($svd) { "${rank},svd,$($svd.k),$($svd.sigma),$($svd.nsp)" | Out-File $csv -Append -Encoding utf8 }
        # Write table only on the first rank (it's rank-independent).
        if ($tbl -and $rank -eq $Ranks[0]) {
            "-,table,$($tbl.k),$($tbl.sigma),$($tbl.nsp)" | Out-File $csv -Append -Encoding utf8
        }
    }
    Write-Host ("   -> {0}" -f $csv)
}

function Run-GpuSweep {
    param([string]$Geom)
    $csv = Join-Path $ResultsDir "sweep_gpu_${Geom}_${stamp}.csv"
    "rank,mode,k,sigma_seed,ns_per_particle" | Out-File $csv -Encoding utf8

    Write-Host ""
    Write-Host ("── GPU sweep on {0} ──" -f $Geom) -ForegroundColor Magenta
    Write-Host ("   ranks={0}  seeds={1}  particles={2}  batches={3}" `
        -f ($Ranks -join ","), $Seeds, $Particles, $Batches)

    # Pointwise baseline (rank-independent): one run.
    Write-Host "   pointwise..." -ForegroundColor DarkMagenta
    $outPw = Invoke-Cargo @(
        "run", "--release", "--features", "cuda",
        "--bin", "gpu_pwr_bench", "--",
        $DataDir,
        "--rank", $Ranks[0], "-B", $Batches, "--inactive", $Inactive,
        "--particles", $Particles, "--seeds", $Seeds,
        "--geometry", $Geom
    )
    $pw = Parse-GpuResult -Text $outPw
    if ($pw) { "-,pointwise,$($pw.k),$($pw.sigma),$($pw.nsp)" | Out-File $csv -Append -Encoding utf8 }

    foreach ($rank in $Ranks) {
        Write-Host ("   force-svd rank={0}..." -f $rank) -ForegroundColor DarkMagenta
        $out = Invoke-Cargo @(
            "run", "--release", "--features", "cuda",
            "--bin", "gpu_pwr_bench", "--",
            $DataDir,
            "--rank", $rank, "-B", $Batches, "--inactive", $Inactive,
            "--particles", $Particles, "--seeds", $Seeds,
            "--geometry", $Geom,
            "--force-svd"
        )
        $r = Parse-GpuResult -Text $out
        if ($r) { "${rank},force-svd,$($r.k),$($r.sigma),$($r.nsp)" | Out-File $csv -Append -Encoding utf8 }
    }
    Write-Host ("   -> {0}" -f $csv)
}

# ── Banner ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== open-rust-mc rank sweep ===" -ForegroundColor Cyan
Write-Host ("  Geometries : {0}" -f ($Geometries -join ", "))
Write-Host ("  Platforms  : {0}" -f ($Platforms -join ", "))
Write-Host ("  Ranks      : {0}" -f ($Ranks -join ", "))
Write-Host ("  Seeds      : {0}" -f $Seeds)
Write-Host ("  Particles  : {0}/batch" -f $Particles)
Write-Host ("  Batches    : {0} ({1} inactive)" -f $Batches, $Inactive)
Write-Host ("  Results    : {0}" -f $ResultsDir)
Write-Host ""

foreach ($g in $Geometries) {
    if ($Platforms -contains "cpu") { Run-CpuSweep -Geom $g }
    if ($Platforms -contains "gpu") { Run-GpuSweep -Geom $g }
}

Write-Host ""
Write-Host "=== Rank sweep complete ===" -ForegroundColor Green
