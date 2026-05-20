<#
.SYNOPSIS
    Post-bucketing sweep — same shape as eventbased_baseline_sweep.ps1
    but writes to eventbased_bucketed_a1000.{csv,log}. Run AFTER the
    3-D (class, hit_nuc_local, ebin) partition lands in
    transport_event_based.cu.
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
$Csv = Join-Path $OutputsDir "eventbased_bucketed_a1000.csv"
$LogPath = Join-Path $OutputsDir "eventbased_bucketed_a1000.log"

"particles,k_inf,ns_per_particle,total_sim_ms,active_histories" | Out-File -Encoding ascii $Csv
"# eventbased 3-D bucketing sweep - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Encoding ascii $LogPath

$exe = Join-Path $RepoRoot "rust_prototype\target\release\$Bin.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Binary not found at $exe -- build first: cargo build --release --features cuda --bin $Bin"
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
Get-Content $Csv | Write-Host
