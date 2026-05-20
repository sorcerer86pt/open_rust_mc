<#
.SYNOPSIS
    Quick CPU throughput probe on HMF-001 — three particle counts to
    estimate the CPU's per-batch sweet spot vs the 3080's measured
    500k-1M saturation point. Used to motivate the cpu/gpu split in
    recommended_settings JSON.
#>

param(
    [string]$Case = 'heu-met-fast-001_case-1',
    [int[]]$ParticleCounts = @(5000, 20000, 100000),
    [int]$Batches = 40,
    [int]$Inactive = 10
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $RepoRoot
$Csv = Join-Path $RepoRoot 'outputs\cpu_saturation_probe.csv'
$LogPath = Join-Path $RepoRoot 'outputs\cpu_saturation_probe.log'

"particles,sim_s,collisions,us_per_collision,us_per_particle,hist_per_sec,k_calc,delta_pcm" | Out-File -Encoding ascii $Csv
"# CPU saturation probe - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Encoding ascii $LogPath
"# Case: $Case  Batches: $Batches  Inactive: $Inactive" | Add-Content $LogPath

$pythonExe = (Get-Command python).Source
$caseRoot = Join-Path $RepoRoot 'bench\icsbep'
$dataDir  = Join-Path $RepoRoot 'data\endfb-vii.1-hdf5\neutron'

$tempHarness = Join-Path ([System.IO.Path]::GetTempPath()) 'cpu_probe.py'
$pyHarness = @'
import sys
from pathlib import Path
from open_rust_mc import Runner, Settings, run_icsbep_case

case_stem, batches, inactive, particles, seed, rank = sys.argv[1:7]
case_root, data_dir = Path(sys.argv[7]), Path(sys.argv[8])

settings = Settings(batches=int(batches), inactive=int(inactive), particles=int(particles), seed=int(seed))
result = run_icsbep_case(
    case_json=case_root / f"{case_stem}.json",
    data_dir=data_dir,
    settings=settings,
    runner=Runner.Cpu,
    rank=int(rank),
)
print(f"  k_calc        : k = {result.k_eff:.5f} +/- {result.k_sigma:.5f}")
print(f"  delta         : {result.delta_pcm:+.0f} pcm")
print(f"  timing        : load = {result.load_time_seconds:.2f} s, sim = {result.sim_time_seconds:.2f} s")
print(f"  tallies       : coll = {result.total_collisions:,}, fis = {result.total_fissions:,}, leak = {result.total_leakage:,}")
'@
Set-Content -Path $tempHarness -Value $pyHarness -Encoding ascii

$inv = [System.Globalization.CultureInfo]::InvariantCulture

foreach ($p in $ParticleCounts) {
    Write-Host ("`n-- CPU @ Particles = {0:N0} --" -f $p)
    $t0 = Get-Date
    $raw = & $pythonExe $tempHarness $Case $Batches $Inactive $p 1 15 $caseRoot $dataDir 2>&1 | Out-String
    $wall = (Get-Date) - $t0
    Write-Host ("  Wall: {0:N1} s" -f $wall.TotalSeconds)

    "" | Add-Content $LogPath
    "============================================================" | Add-Content $LogPath
    "particles=$p   wall=$($wall.TotalSeconds.ToString('F1'))s" | Add-Content $LogPath
    "============================================================" | Add-Content $LogPath
    $raw | Add-Content $LogPath

    $sim   = [regex]::Match($raw, "sim\s+=\s+([0-9.]+)\s*s")
    $coll  = [regex]::Match($raw, "coll\s+=\s+([0-9,]+)")
    $kcalc = [regex]::Match($raw, "k\s+=\s+([0-9.]+)")
    $delta = [regex]::Match($raw, "delta\s+:\s+([+-]?[0-9.]+)\s*pcm")

    $simS = if ($sim.Success) { [double]::Parse($sim.Groups[1].Value, $inv) } else { [double]::NaN }
    $collN = if ($coll.Success) { [int64]($coll.Groups[1].Value -replace ',','') } else { 0 }
    $k = if ($kcalc.Success) { $kcalc.Groups[1].Value } else { 'NaN' }
    $d = if ($delta.Success) { $delta.Groups[1].Value } else { 'NaN' }
    $active = $p * ($Batches - $Inactive)
    $usPerColl = if ($collN -gt 0) { (($simS * 1e6 / $collN)).ToString('F3', $inv) } else { 'NaN' }
    $usPerP = if ($active -gt 0) { (($simS * 1e6 / $active)).ToString('F2', $inv) } else { 'NaN' }
    $histPerSec = if ($simS -gt 0) { [int]($active / $simS) } else { 'NaN' }

    "$p,$($simS.ToString('F3', $inv)),$collN,$usPerColl,$usPerP,$histPerSec,$k,$d" | Add-Content $Csv
    Write-Host ("  sim={0}s coll={1}  us/coll={2}  us/p={3}  hist/s={4}  k={5}  delta={6} pcm" -f $simS, $collN, $usPerColl, $usPerP, $histPerSec, $k, $d)
}

Write-Host "`nDone."
Get-Content $Csv | Write-Host
