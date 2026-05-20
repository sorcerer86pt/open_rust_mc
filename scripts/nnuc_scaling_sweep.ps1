<#
.SYNOPSIS
    Sweep ICSBEP cases across the n_nuc spectrum and measure how
    per-collision kernel cost scales with material nuclide count.

.DESCRIPTION
    For each case, the script:
      1. Parses max(n_nuc_per_material) and total_nuclides from the
         JSON case file.
      2. Invokes the Python ICSBEP harness against the GPU runner.
      3. Captures sim_time, total collisions, ns/particle, k_calc,
         and pass/fail status.
      4. Computes us/collision = sim_time / collisions and writes
         one CSV row.

    The CSV is the union-grid hypothesis test data. If the
    us/collision column is roughly flat across n_nuc rows, the
    per-nuclide loop is NOT dominating — union-grid implementation
    is a small win. If it scales noticeably with n_nuc, union-grid
    is the right next attack.

    Pass `-Ncu` to wrap each run in `ncu --kernel-name gr_trace_and_sample`
    (Nsight Compute CLI must be on PATH). NCU output goes to
    outputs/<case>_ncu.txt; runtime numbers in the CSV become
    unusable in that mode (NCU serializes kernel launches), so run
    without -Ncu first for the scaling CSV, then with -Ncu on a few
    target cases for the per-kernel counters.

.PARAMETER Cases
    Override the default case list. Bare names without `.json`.

.PARAMETER Particles
    Particles per batch (passed via env var; icsbep_run.py defaults to
    5000 if the override isn't honoured). Default: 5000.

.PARAMETER Ncu
    Wrap each run in Nsight Compute targeting gr_trace_and_sample.

.PARAMETER NcuKernel
    Override the ncu kernel filter. Default: gr_trace_and_sample.

.PARAMETER OutputBase
    Base name (no extension) for the CSV / log under outputs/.
    Default: nnuc_scaling.

.EXAMPLE
    # Plain scaling sweep, no profiler:
    pwsh scripts\nnuc_scaling_sweep.ps1

.EXAMPLE
    # Same sweep but capture ncu metrics for gr_trace_and_sample:
    pwsh scripts\nnuc_scaling_sweep.ps1 -Ncu

.EXAMPLE
    # Custom case list:
    pwsh scripts\nnuc_scaling_sweep.ps1 -Cases @('heu-met-fast-001_case-1','heu-met-fast-069_case-1')
#>

param(
    [string[]]$Cases = @(
        'heu-met-fast-001_case-1',   # ~3 nuclides   (Godiva-class fast metal)
        'heu-met-fast-011',          # ~14 nuclides  (HEU + W isotopes + poly + Fe)
        'heu-met-fast-008',          # ~10 nuclides
        'leu-comp-therm-008_case-1', # ~28 nuclides  (LEU thermal)
        'heu-sol-therm-001_case-1',  # ~30 nuclides  (HEU solution thermal)
        'mix-met-fast-008_8h',       # ~37 nuclides
        'pu-sol-therm-009_case-1',   # ~67 nuclides  (may OOM on <8 GB)
        'heu-met-fast-069_case-1'    # ~69 nuclides  (needs ~8 GB+ VRAM)
    ),
    # Per-batch particle count. icsbep_run.py hardcodes 5000 which on
    # a 3080 only fills 40 thread blocks (~7.5% achieved occupancy per
    # ncu). Bumping this to ~35 000 is what actually saturates the SMs
    # on a 10 GB card. Per-case benchmark.recommended_settings is NOT
    # consulted — `run_icsbep_case` (the PyO3 binding) uses whatever
    # `Settings(...)` it's handed verbatim; only `icsbep_sweep.py`
    # (a different harness) reads recommended_settings. So this flag
    # is the single source of truth for what gets uploaded.
    [int]$Particles = 5000,
    [int]$Batches = 80,
    [int]$Inactive = 20,
    [int]$Seed = 1,
    [int]$Rank = 15,
    # PHYSOR 2022 Optimization F — continuous particle refill. When > 1.0,
    # the GPU runner builds a source bank of `Particles * RefillFactor`
    # per batch and uses the overflow to refill dead slots between event
    # steps. Default 0 means "use the engine default" (None on the Rust
    # side -> disabled). Hardware-dependent: A1000 sees no benefit
    # (saturates at 5k particles); 3080 is at saturation around 1M but
    # the batch-tail still under-occupies; A100/H100 likely benefit
    # most. CPU runner ignores this entirely.
    [double]$RefillFactor = 0.0,
    # Device-attribute-driven auto-refill. When set, queries the active
    # GPU's SM count + the kernel's compiled register count and picks
    # the refill factor automatically using the saturation-knee heuristic
    # in gpu_recursive::recommend_refill_factor. Logged to stdout so the
    # auto-pick is always visible. Explicit -RefillFactor > 0 wins.
    [switch]$AutoRefill,
    [switch]$Ncu,
    [string]$NcuKernel = 'gr_trace_and_sample',
    [string]$OutputBase = 'nnuc_scaling'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $RepoRoot

$OutputsDir = Join-Path $RepoRoot 'outputs'
if (-not (Test-Path $OutputsDir)) { New-Item -ItemType Directory $OutputsDir | Out-Null }

$Suffix = if ($Ncu) { '_ncu' } else { '' }
$Csv = Join-Path $OutputsDir ("{0}{1}.csv" -f $OutputBase, $Suffix)
$LogPath = Join-Path $OutputsDir ("{0}{1}.log" -f $OutputBase, $Suffix)

'case,max_n_nuc,total_n_nuc,n_mats,sim_s,load_s,collisions,fissions,leaks,us_per_collision,us_per_particle,k_calc,k_sigma,delta_pcm,sigma_ratio,status' | Out-File -Encoding ascii $Csv
"# n_nuc scaling sweep - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Encoding ascii $LogPath
"# Particles/batch: $Particles  Batches: $Batches  Inactive: $Inactive  Seed: $Seed  Rank: $Rank" | Add-Content $LogPath
"# Ncu mode: $Ncu" | Add-Content $LogPath
if ($Ncu) { "# Ncu kernel filter: $NcuKernel" | Add-Content $LogPath }

# Locate Python
$python = (Get-Command python -ErrorAction SilentlyContinue) ?? (Get-Command python3 -ErrorAction SilentlyContinue)
if (-not $python) {
    Write-Error "Could not find 'python' on PATH. Ensure the Python with open_rust_mc installed is on PATH."
    exit 1
}
$pythonExe = $python.Source
Write-Host "Using Python: $pythonExe"

# Verify Ncu is on PATH if requested
if ($Ncu) {
    $ncuCmd = Get-Command ncu -ErrorAction SilentlyContinue
    if (-not $ncuCmd) {
        Write-Error "ncu not on PATH. Add the Nsight Compute install dir (e.g. 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.x\') to PATH."
        exit 1
    }
    Write-Host "Using ncu: $($ncuCmd.Source)"
}

# Verify the GPU runner is available
$probe = & $pythonExe -c "from open_rust_mc import Runner; print(Runner.GpuCuda.name())" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not import Runner.GpuCuda. Did you 'maturin develop --release --features cuda'?"
    Write-Warning "Probe output: $probe"
}

# Inline Python harness that calls run_icsbep_case directly with our
# overridden Settings. Replaces icsbep_run.py so we control particles /
# batches / inactive / seed. The PyO3 binding does NOT honour
# `benchmark.recommended_settings` in the JSON; it uses Settings(...)
# verbatim. Output format mirrors icsbep_run.py line-for-line so the
# regex parsers in Invoke-Case work unchanged.
$pyHarness = @'
import sys
from pathlib import Path
from open_rust_mc import Runner, Settings, run_icsbep_case

case_stem = sys.argv[1]
batches = int(sys.argv[2])
inactive = int(sys.argv[3])
particles = int(sys.argv[4])
seed = int(sys.argv[5])
rank = int(sys.argv[6])
case_root = Path(sys.argv[7])
data_dir = Path(sys.argv[8])
refill_factor_raw = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
refill_factor = refill_factor_raw if refill_factor_raw > 1.0 else None
# argv[10]: gpu_auto_refill flag ("1" / "0"). Explicit refill_factor
# always wins over auto (engine policy in CudaRunner::run).
auto_refill = (len(sys.argv) > 10 and sys.argv[10] == "1")

case_json = case_root / f"{case_stem}.json"
if not case_json.exists():
    print(f"case not found: {case_json}", file=sys.stderr)
    sys.exit(2)
if not data_dir.exists():
    print(f"data dir not found: {data_dir}", file=sys.stderr)
    sys.exit(2)

settings = Settings(
    batches=batches,
    inactive=inactive,
    particles=particles,
    seed=seed,
    gpu_refill_pool_factor=refill_factor,
    gpu_auto_refill=auto_refill,
)
print(f"  settings      : {settings!r}  gpu_auto_refill={auto_refill}")
result = run_icsbep_case(
    case_json=case_json,
    data_dir=data_dir,
    settings=settings,
    runner=Runner.GpuCuda,
    rank=rank,
)

verdict = "PASS" if result.passed else "FAIL"
print(f"  case          : {result.case}")
print(f"  runner        : gpu_cuda")
print(f"  reference     : {result.ref_source}")
print(f"  handbook      : k = {result.handbook_k:.5f} +/- {result.handbook_sigma:.5f}")
print(f"  acceptance    : k = {result.k_ref:.5f} +/- {result.sigma_exp:.5f}   sigma_combined = {result.sigma_combined * 1e5:.0f} pcm")
print(f"  k_calc        : {result.k_eff:.5f} +/- {result.k_sigma:.5f}")
print(f"  delta         : {result.delta_pcm:+.0f} pcm   {result.sigma_ratio:.2f}-sigma   bound = +/-{result.bound_pcm:.0f} pcm   [{verdict}]")
print(f"  timing        : load = {result.load_time_seconds:.2f} s, sim = {result.sim_time_seconds:.2f} s, total = {result.runtime_seconds:.2f} s")
print(f"  active batches: {batches - inactive} (total {(batches - inactive) * particles:,} histories)")
print(f"  tallies       : coll = {result.total_collisions:,}, fis = {result.total_fissions:,}, leak = {result.total_leakage:,}")
'@

$caseRoot = Join-Path $RepoRoot 'bench\icsbep'
$dataDir  = Join-Path $RepoRoot 'data\endfb-vii.1-hdf5\neutron'

# Write the harness to a temp file so ncu can launch python <file>
# instead of python -c "..." (ncu's -c parsing with the quoting
# required for the inline string is fragile across pwsh versions).
$tempHarness = Join-Path ([System.IO.Path]::GetTempPath()) 'nnuc_harness.py'
Set-Content -Path $tempHarness -Value $pyHarness -Encoding ascii

function Get-CaseNNuc {
    param([string]$JsonPath)
    $json = Get-Content $JsonPath -Raw | ConvertFrom-Json
    $mats = $json.scene.materials
    if (-not $mats) {
        return @{ max = 0; total = 0; n_mats = 0 }
    }
    $sizes = @($mats | ForEach-Object { @($_.nuclides).Count })
    return @{
        max    = ($sizes | Measure-Object -Maximum).Maximum
        total  = ($sizes | Measure-Object -Sum).Sum
        n_mats = $sizes.Count
    }
}

function Invoke-Case {
    param(
        [string]$Case,
        [bool]$UseNcu
    )
    $caseJson = Join-Path $RepoRoot "bench\icsbep\$Case.json"
    if (-not (Test-Path $caseJson)) {
        Write-Warning "Case JSON not found, skipping: $caseJson"
        return @{ status = 'MISSING'; raw = '' }
    }
    $nnuc = Get-CaseNNuc -JsonPath $caseJson

    Write-Host ""
    Write-Host ("== {0} (max_n_nuc={1}, total={2}, n_mats={3}) ==" -f $Case, $nnuc.max, $nnuc.total, $nnuc.n_mats)

    $autoFlag = if ($AutoRefill.IsPresent) { 1 } else { 0 }
    $pyArgs = @($tempHarness, $Case, $Batches, $Inactive, $Particles, $Seed, $Rank, $caseRoot, $dataDir, $RefillFactor, $autoFlag)

    $t0 = Get-Date
    if ($UseNcu) {
        # ncu wraps the python process. --target-processes all so child
        # processes spawned by maturin/PyO3 are also profiled.
        $ncuRep = Join-Path $OutputsDir "$Case.ncu-rep"
        $ncuTxt = Join-Path $OutputsDir "$Case`_ncu.txt"
        $raw = & ncu `
            --target-processes all `
            --kernel-name $NcuKernel `
            --launch-count 5 `
            --section MemoryWorkloadAnalysis `
            --section ComputeWorkloadAnalysis `
            --section LaunchStats `
            --section Occupancy `
            --section WarpStateStats `
            -f -o $ncuRep `
            $pythonExe @pyArgs 2>&1 | Out-String
        & ncu --import $ncuRep --csv 2>&1 | Out-File -Encoding ascii $ncuTxt
        Write-Host "  ncu report: $ncuRep"
        Write-Host "  ncu text:   $ncuTxt"
    } else {
        $raw = & $pythonExe @pyArgs 2>&1 | Out-String
    }
    $wall = (Get-Date) - $t0
    Write-Host ("  Wall: {0:N1} s" -f $wall.TotalSeconds)

    "" | Add-Content $LogPath
    "============================================================" | Add-Content $LogPath
    "case=$Case  max_n_nuc=$($nnuc.max)  total_nuc=$($nnuc.total)  n_mats=$($nnuc.n_mats)  wall=$($wall.TotalSeconds.ToString('F1'))s" | Add-Content $LogPath
    "============================================================" | Add-Content $LogPath
    $raw | Add-Content $LogPath

    # Parse metrics from icsbep_run.py output
    $sim   = [regex]::Match($raw, "sim\s+=\s+([0-9.]+)\s*s")
    $load  = [regex]::Match($raw, "load\s+=\s+([0-9.]+)\s*s")
    $coll  = [regex]::Match($raw, "coll\s+=\s+([0-9,]+)")
    $fis   = [regex]::Match($raw, "fis\s+=\s+([0-9,]+)")
    $leak  = [regex]::Match($raw, "leak\s+=\s+([0-9,]+)")
    $kcalc = [regex]::Match($raw, "k_calc\s+:\s+k\s+=\s+([0-9.]+)\s+\+/-\s+([0-9.]+)")
    if (-not $kcalc.Success) {
        $kcalc = [regex]::Match($raw, "k_calc\s+:\s+([0-9.]+)\s+\+/-\s+([0-9.]+)")
    }
    if (-not $kcalc.Success) {
        $kcalc = [regex]::Match($raw, "k_calc\s+=\s+([0-9.]+)\s+\+/-\s+([0-9.]+)")
    }
    if (-not $kcalc.Success) {
        $kcalc = [regex]::Match($raw, "k_calc\s+[:=]?\s+([0-9.]+)\s*\+/-\s*([0-9.]+)")
    }
    $delta = [regex]::Match($raw, "delta\s+:\s+([+-]?[0-9.]+)\s*pcm")
    if (-not $delta.Success) {
        $delta = [regex]::Match($raw, "delta\s+[:=]\s+([+-]?[0-9.]+)\s*pcm")
    }
    $sratio = [regex]::Match($raw, "([0-9.]+)-sigma")
    $verdict = if ($raw -match '\[PASS\]') { 'PASS' }
               elseif ($raw -match '\[FAIL\]') { 'FAIL' }
               elseif ($raw -match 'CUDA_ERROR_OUT_OF_MEMORY') { 'OOM' }
               elseif ($raw -match 'error|Traceback') { 'ERROR' }
               else { 'UNKNOWN' }

    # Pin to invariant culture so the CSV uses "." for the decimal
    # separator (avoids collision with the field separator on systems
    # whose current culture is e.g. pt-PT / de-DE that default to ",").
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $parseDouble = { param($s) [double]::Parse($s, $inv) }

    $simS  = if ($sim.Success)   { & $parseDouble $sim.Groups[1].Value }   else { [double]::NaN }
    $loadS = if ($load.Success)  { & $parseDouble $load.Groups[1].Value }  else { [double]::NaN }
    $collN = if ($coll.Success)  { [int64]($coll.Groups[1].Value -replace ',','') } else { 0 }
    $fisN  = if ($fis.Success)   { [int64]($fis.Groups[1].Value -replace ',','') }  else { 0 }
    $leakN = if ($leak.Success)  { [int64]($leak.Groups[1].Value -replace ',','') } else { 0 }
    $k     = if ($kcalc.Success) { $kcalc.Groups[1].Value } else { 'NaN' }
    $kSig  = if ($kcalc.Success) { $kcalc.Groups[2].Value } else { 'NaN' }
    $dPcm  = if ($delta.Success) { $delta.Groups[1].Value } else { 'NaN' }
    $sigR  = if ($sratio.Success){ $sratio.Groups[1].Value } else { 'NaN' }

    $fmt = { param($v, $f) ([double]$v).ToString($f, $inv) }

    $simStr = if ([double]::IsNaN($simS))  { 'NaN' } else { & $fmt $simS 'F3' }
    $loadStr = if ([double]::IsNaN($loadS)) { 'NaN' } else { & $fmt $loadS 'F3' }

    $usPerColl = if ($collN -gt 0 -and -not [double]::IsNaN($simS)) {
        & $fmt ($simS * 1e6 / $collN) 'F3'
    } else { 'NaN' }
    # Total active histories = $Particles * (Batches - Inactive).
    $active = $Particles * ($Batches - $Inactive)
    $usPerP = if ($active -gt 0 -and -not [double]::IsNaN($simS)) {
        & $fmt ($simS * 1e6 / $active) 'F2'
    } else { 'NaN' }

    Write-Host ("  n_nuc max={0}  sim={1}s  coll={2}  us/coll={3}  us/p={4}  k={5}  delta={6} pcm  [{7}]" -f `
        $nnuc.max, $simStr, $collN, $usPerColl, $usPerP, $k, $dPcm, $verdict)

    "$Case,$($nnuc.max),$($nnuc.total),$($nnuc.n_mats),$simStr,$loadStr,$collN,$fisN,$leakN,$usPerColl,$usPerP,$k,$kSig,$dPcm,$sigR,$verdict" | Add-Content $Csv
}

foreach ($c in $Cases) {
    try {
        Invoke-Case -Case $c -UseNcu $Ncu.IsPresent
    } catch {
        Write-Warning "Case $c failed: $_"
        "$c,,,,,,,,,,,,,,,EXCEPTION" | Add-Content $Csv
    }
}

Write-Host ""
Write-Host "Done."
Write-Host "CSV: $Csv"
Write-Host "Log: $LogPath"
Write-Host ""
Get-Content $Csv | Write-Host
