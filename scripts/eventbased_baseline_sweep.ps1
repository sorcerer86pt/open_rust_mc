# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    Strong-scaling baseline sweep of the event-based GPU pipeline.

.DESCRIPTION
    Runs gpu_pwr_bench at five particle counts (50k, 100k, 250k, 500k,
    1M) on the current branch (event-based) and logs k_inf, ns/p, and
    total sim wall time to a CSV. Captured BEFORE the 3-D bucketing
    (class, hit_nuc, ebin) sort lands so we can read the delta after.

    One seed, 50 batches (10 inactive), PWR pin-cell geometry, SVD
    mode. Single seed is sufficient: we measure throughput, not k_inf
    statistics. The same seed ensures bit-equivalent particle counts
    across rows.

.OUTPUTS
    outputs/eventbased_baseline_3080.csv
    outputs/eventbased_baseline_3080.log
#>

param(
    [string]$DataDir = "data\endfb-vii.1-hdf5\neutron",
    [string]$Bin    = "gpu_pwr_bench",
    [int]$Batches   = 50,
    [int]$Inactive  = 10,
    [int]$Rank      = 5,
    [int]$Seeds     = 1
)

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot"

$OutputsDir = Join-Path $RepoRoot "outputs"
if (-not (Test-Path $OutputsDir)) { New-Item -ItemType Directory $OutputsDir | Out-Null }
$Csv = Join-Path $OutputsDir "eventbased_baseline_a1000.csv"
$LogPath = Join-Path $OutputsDir "eventbased_baseline_a1000.log"

# Header
"particles,k_inf,ns_per_particle,total_sim_ms,active_histories" | Out-File -Encoding ascii $Csv
"# eventbased baseline sweep - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Encoding ascii $LogPath

# Build once with cuda
Write-Host "`n-- Building $Bin (cuda) --"
$buildLog = & cargo build --release --features cuda --bin $Bin --manifest-path rust_prototype/Cargo.toml 2>&1 | Out-String
$buildLog | Add-Content $LogPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed. See $LogPath"
    exit 1
}
Write-Host "Build OK."

$exe = Join-Path $RepoRoot "rust_prototype\target\release\$Bin.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Binary not found at $exe"
    exit 1
}

$particleCounts = @(10000, 25000, 50000)

foreach ($p in $particleCounts) {
    Write-Host ("`n-- Particles = {0:N0} --" -f $p)
    $args = @(
        $DataDir,
        "--rank", $Rank,
        "--batches", $Batches,
        "--inactive", $Inactive,
        "--particles", $p,
        "--seeds", $Seeds,
        "--mode", "svd",
        "--geometry", "pwr"
    )
    $t0 = Get-Date
    $stdout = & $exe @args 2>&1 | Out-String
    $wall = (Get-Date) - $t0
    Write-Host ("  Wall: {0:N1} s" -f $wall.TotalSeconds)

    "" | Add-Content $LogPath
    "============================================================" | Add-Content $LogPath
    "particles=$p   wall=$($wall.TotalSeconds.ToString('F1'))s" | Add-Content $LogPath
    "============================================================" | Add-Content $LogPath
    $stdout | Add-Content $LogPath

    # Parse: "k_inf            = X.XXXXX"
    #        "ns/particle      = XX.XX"
    #        "Total sim time   = XXXX ms"
    $kMatch  = [regex]::Match($stdout, "k_inf\s+=\s+([0-9.]+)")
    $nsMatch = [regex]::Match($stdout, "ns/particle\s+=\s+([0-9.]+)")
    $msMatch = [regex]::Match($stdout, "Total sim time\s+=\s+([0-9.]+)\s*ms")

    $kInf   = if ($kMatch.Success)  { $kMatch.Groups[1].Value }  else { "NaN" }
    $nsPp   = if ($nsMatch.Success) { $nsMatch.Groups[1].Value } else { "NaN" }
    $simMs  = if ($msMatch.Success) { $msMatch.Groups[1].Value } else { "NaN" }
    $active = $p * ($Batches - $Inactive)

    "$p,$kInf,$nsPp,$simMs,$active" | Add-Content $Csv
    Write-Host "  k_inf=$kInf  ns/p=$nsPp  sim=${simMs}ms"
}

Write-Host "`nDone. CSV: $Csv"
Write-Host "Log: $LogPath"
Get-Content $Csv | Write-Host
