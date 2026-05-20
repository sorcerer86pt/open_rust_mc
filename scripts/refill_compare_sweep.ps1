# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    Run a control + refill-on pair through nnuc_scaling_sweep.ps1 and
    print a side-by-side comparison of ns/p, throughput, and k_eff.

.DESCRIPTION
    Two-row comparison for the PHYSOR 2022 refill feature. Use on a
    big card (3080+) at saturated batch size — refill costs slightly
    on under-occupied hardware (per the saturation curve in
    outputs/saturation_*.csv) and only pays off when the kernel grid
    has SMs left idle in the batch tail.

    Outputs:
      outputs/<OutputBase>_refill_off.csv  + .log
      outputs/<OutputBase>_refill_2x.csv   + .log

.PARAMETER Case
    Single ICSBEP case stem (no .json). Default: heu-met-fast-001_case-1
    (Godiva, fastest fast-metal — easy to drive the SMs at saturation).

.PARAMETER Particles
    Per-batch active-slot count. Use the saturation knee for the
    target hardware (3080: ~500 k; A100: ~2-5 M; H100: higher still).

.PARAMETER RefillFactor
    Refill multiplier on the source-bank size. Default 2.0 doubles
    the histories per batch via overflow-into-dead-slots.

.PARAMETER Batches / Inactive / Seed / Rank
    Passed through to the underlying sweep.

.PARAMETER OutputBase
    File-name prefix for the two CSV/log pairs. The script appends
    `_refill_off` and `_refill_2x`.

.EXAMPLE
    pwsh scripts\refill_compare_sweep.ps1 -Particles 500000

.EXAMPLE
    pwsh scripts\refill_compare_sweep.ps1 -Case heu-met-fast-008 -Particles 1000000 -RefillFactor 4.0
#>

param(
    [string]$Case = 'heu-met-fast-001_case-1',
    [int]$Particles = 500000,
    [double]$RefillFactor = 2.0,
    [int]$Batches = 80,
    [int]$Inactive = 20,
    [int]$Seed = 1,
    [int]$Rank = 15,
    [string]$OutputBase = 'refill_compare',
    # When set, appends an `auto` variant to the comparison (engine
    # picks the factor based on SM count + kernel reg count via
    # recommend_refill_factor). Adds a third row to the summary so
    # the auto-pick can be compared against the explicit factor.
    [switch]$IncludeAuto
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $RepoRoot

$inner = Join-Path $PSScriptRoot 'nnuc_scaling_sweep.ps1'
if (-not (Test-Path $inner)) {
    Write-Error "Inner sweep script not found: $inner"
    exit 1
}

# Variant table. Always: off + explicit factor. -IncludeAuto adds a
# third row where the engine picks the factor itself.
$variants = @(
    [PSCustomObject]@{ Tag = 'refill_off';                       Factor = 0.0;          Auto = $false },
    [PSCustomObject]@{ Tag = ("refill_{0}x" -f $RefillFactor);   Factor = $RefillFactor; Auto = $false }
)
if ($IncludeAuto.IsPresent) {
    $variants += [PSCustomObject]@{ Tag = 'refill_auto'; Factor = 0.0; Auto = $true }
}

$outputs = @{}
foreach ($v in $variants) {
    $base = "${OutputBase}_$($v.Tag)"
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host ("Running variant: {0}  (RefillFactor={1}, Auto={2})" -f $v.Tag, $v.Factor, $v.Auto)
    Write-Host ("=" * 70)
    $innerArgs = @(
        '-Cases', @($Case),
        '-Particles', $Particles,
        '-Batches', $Batches,
        '-Inactive', $Inactive,
        '-Seed', $Seed,
        '-Rank', $Rank,
        '-RefillFactor', $v.Factor,
        '-OutputBase', $base
    )
    if ($v.Auto) { $innerArgs += '-AutoRefill' }
    & pwsh -ExecutionPolicy Bypass -File $inner @innerArgs
    $outputs[$v.Tag] = Join-Path $RepoRoot "outputs\$base.csv"
}

# Side-by-side comparison
Write-Host ""
Write-Host ("=" * 70)
Write-Host "Summary"
Write-Host ("=" * 70)
$rows = @{}
foreach ($v in $variants) {
    $csv = $outputs[$v.Tag]
    if (-not (Test-Path $csv)) {
        Write-Warning "CSV missing for $($v.Tag): $csv"
        continue
    }
    $row = Import-Csv $csv | Select-Object -First 1
    $rows[$v.Tag] = $row
}

if ($rows.Count -eq 2) {
    $off = $rows[$variants[0].Tag]
    $on  = $rows[$variants[1].Tag]
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $parse = { param($s) [double]::Parse($s, $inv) }

    $offNs = & $parse $off.us_per_particle
    $onNs  = & $parse $on.us_per_particle
    $offSim = & $parse $off.sim_s
    $onSim  = & $parse $on.sim_s
    $offColl = [int64]$off.collisions
    $onColl  = [int64]$on.collisions
    $offK = & $parse $off.k_calc
    $onK  = & $parse $on.k_calc

    $speedup = if ($onNs -gt 0) { $offNs / $onNs } else { 0 }
    $deltaK_pcm = ($onK - $offK) * 1e5

    "{0,-22} {1,15} {2,15}" -f "metric", $variants[0].Tag, $variants[1].Tag
    "{0,-22} {1,15} {2,15}" -f ("-" * 22), ("-" * 15), ("-" * 15)
    "{0,-22} {1,15:F2} {2,15:F2}"      -f "us_per_particle", $offNs, $onNs
    "{0,-22} {1,15:F2} {2,15:F2}"      -f "sim_s", $offSim, $onSim
    "{0,-22} {1,15:N0} {2,15:N0}"      -f "collisions", $offColl, $onColl
    "{0,-22} {1,15:F5} {2,15:F5}"      -f "k_calc", $offK, $onK
    "{0,-22} {1,15}   {2,15}"          -f "status", $off.status, $on.status
    ""
    "ns/p speedup (off / on) = {0:F3}x" -f $speedup
    "k_eff delta (on - off)   = {0:+0;-0;0} pcm" -f $deltaK_pcm
} else {
    Write-Warning "Comparison skipped — expected 2 rows, got $($rows.Count)."
}
