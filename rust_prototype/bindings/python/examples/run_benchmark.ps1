<#
.SYNOPSIS
    One-shot ICSBEP / engine-internal benchmark sweep.

.DESCRIPTION
    1. Locates the repository root (the directory containing
       bench/icsbep) by walking upward from this script's location.
    2. Detects whether the Python extension was built with the CUDA
       feature by importing `open_rust_mc` and inspecting
       `Runner.recommended()`. Falls back to CPU automatically.
    3. Launches `icsbep_sweep.py` with paper-quality CLI defaults
       (the JSON per-case `benchmark.recommended_settings` overrides
       still apply per case). Writes per-case rows to
       outputs/icsbep_full_<runner>.csv as they complete.
    4. Watches outputs/STOP — `New-Item outputs\STOP -ItemType File`
       from another shell terminates the sweep cleanly between cases.
       Ctrl-C also flushes the partial CSV and exits 0.
    5. Prints the list of files to git-add when finished.

.PARAMETER Runner
    Override the auto-detected runner. "auto" (default), "cpu", "gpu".

.PARAMETER Batches
    Active + inactive batches per seed. JSON per-case override wins.

.PARAMETER Inactive
    Inactive batches per seed.

.PARAMETER Particles
    Particles per batch.

.PARAMETER Seeds
    Seeds averaged per case.

.PARAMETER Resume
    Re-use an existing CSV; skip cases already in it.

.PARAMETER Filter
    Regex over case stems; only matching cases run. Useful for
    smoke-testing one family before committing to the full corpus.

.PARAMETER Limit
    Cap on cases after filtering. 0 means no cap.

.EXAMPLE
    .\run_benchmark.ps1
    # Full corpus, auto-detect runner, paper-quality settings.

.EXAMPLE
    .\run_benchmark.ps1 -Filter "heu-met-fast" -Seeds 3
    # Just the HMF family, 3 seeds (engine regression).

.EXAMPLE
    .\run_benchmark.ps1 -Runner cpu -Resume
    # Resume the previous outputs/icsbep_full_cpu.csv run.

.NOTES
    Built for Windows PowerShell 7+. Sequential commands chained with
    `&&` / `||`. Background processes are NOT used so Ctrl-C interrupts
    the sweep directly.
#>

[CmdletBinding()]
param(
    [ValidateSet("auto", "cpu", "gpu")]
    [string]$Runner   = "auto",
    [int]   $Batches  = 150,
    [int]   $Inactive = 30,
    # Particles-per-batch ceiling on GPU MC scales linearly with VRAM
    # up to ~8 M per Tramm et al. PHYSOR 2022. Practical recommendations:
    #   4 GB VRAM (RTX A1000 laptop):  ≤   50_000
    #   10 GB VRAM (RTX 3080):          200_000 – 500_000
    #   40 GB VRAM (A100 SXM4):       1_000_000+
    # The default below is laptop-safe; override with `-Particles` on
    # larger cards.
    [int]   $Particles = 50000,
    [int]   $Seeds    = 5,
    [int]   $BaseSeed = 42,
    [int]   $Rank     = 15,
    [string]$Filter   = "",
    [int]   $Limit    = 0,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

# ── 1. Locate the repo root ───────────────────────────────────────────
$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot  = $scriptDir
while ($repoRoot -and -not (Test-Path (Join-Path $repoRoot "bench\icsbep"))) {
    $parent = Split-Path -Parent $repoRoot
    if ($parent -eq $repoRoot) { break }   # reached drive root
    $repoRoot = $parent
}
if (-not (Test-Path (Join-Path $repoRoot "bench\icsbep"))) {
    Write-Error "Could not locate bench/icsbep starting from $scriptDir"
    exit 2
}
Set-Location $repoRoot
Write-Host "Repo root: $repoRoot"

# ── 2. Auto-detect the runner ─────────────────────────────────────────
$resolvedRunner = $Runner
if ($Runner -eq "auto") {
    $probe = & python -c @"
try:
    from open_rust_mc import Runner
    print(Runner.recommended().name())
except Exception as e:
    print(f'error:{e}')
"@ 2>&1
    $probe = ($probe | Out-String).Trim()
    if ($probe -match "^gpu_cuda$") {
        $resolvedRunner = "gpu"
    } elseif ($probe -match "^cpu$") {
        $resolvedRunner = "cpu"
    } else {
        Write-Error "Could not import open_rust_mc. Build the Python extension first:"
        Write-Error "  cd rust_prototype/bindings/python"
        Write-Error "  maturin develop --release --features cuda    # or --release for CPU-only"
        Write-Error "Probe output: $probe"
        exit 3
    }
    Write-Host "Auto-detected runner: $resolvedRunner"
} else {
    Write-Host "Runner (override): $resolvedRunner"
}

# ── 3. Output paths ───────────────────────────────────────────────────
$outDir = Join-Path $repoRoot "outputs"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}
$csvPath  = Join-Path $outDir "icsbep_full_$resolvedRunner.csv"
$logPath  = Join-Path $outDir "icsbep_full_$resolvedRunner.log"
$stopPath = Join-Path $outDir "STOP"

if (Test-Path $stopPath) {
    Write-Warning "Stale stop-file $stopPath found; removing before launch."
    Remove-Item $stopPath -Force
}

Write-Host ""
Write-Host "── Sweep configuration ────────────────────────────────"
Write-Host "  runner       : $resolvedRunner"
Write-Host "  CLI defaults : batches=$Batches, inactive=$Inactive, particles=$Particles, seeds=$Seeds, base_seed=$BaseSeed, rank=$Rank"
Write-Host "  CSV (append) : $csvPath"
Write-Host "  Log          : $logPath"
Write-Host "  Stop file    : $stopPath  (create to terminate gracefully)"
if ($Resume)   { Write-Host "  Mode         : RESUME (skip cases already in CSV)" }
if ($Filter)   { Write-Host "  Filter regex : $Filter" }
if ($Limit -gt 0) { Write-Host "  Case cap     : $Limit" }
Write-Host "───────────────────────────────────────────────────────"
Write-Host ""

# ── 4. Assemble the argument list ─────────────────────────────────────
$sweepArgs = @(
    Join-Path "rust_prototype/bindings/python/examples" "icsbep_sweep.py"
    "--runner",   $resolvedRunner
    "--batches",  $Batches
    "--inactive", $Inactive
    "--particles", $Particles
    "--seeds",    $Seeds
    "--base-seed", $BaseSeed
    "--rank",     $Rank
    "--csv",      $csvPath
    "--stop-file", $stopPath
)
if ($Resume)    { $sweepArgs += "--resume" }
if ($Filter)    { $sweepArgs += @("--filter", $Filter) }
if ($Limit -gt 0) { $sweepArgs += @("--limit",  $Limit) }

# ── 5. Run, tee output to log ─────────────────────────────────────────
$started = Get-Date
Write-Host "Launching sweep at $started`n"
& python @sweepArgs 2>&1 | Tee-Object -FilePath $logPath
$exit = $LASTEXITCODE

$finished = Get-Date
$elapsed  = $finished - $started

Write-Host ""
Write-Host "── Sweep finished ────────────────────────────────────"
Write-Host "  started       : $started"
Write-Host "  finished      : $finished"
Write-Host "  wall time     : $([math]::Round($elapsed.TotalMinutes, 1)) min"
Write-Host "  python exit   : $exit"
Write-Host "  CSV           : $csvPath"
Write-Host "  Log           : $logPath"
Write-Host "──────────────────────────────────────────────────────"

# ── 6. Hand off to the user with commit instructions ──────────────────
if (Test-Path $csvPath) {
    $rowCount = (Get-Content $csvPath | Select-Object -Skip 1 | Measure-Object).Count
    Write-Host ""
    Write-Host "Result file: $rowCount case row(s) in CSV."
    Write-Host ""
    Write-Host "To commit the results back to the repo (outputs/* is gitignored, so use -f):"
    Write-Host ""
    Write-Host "  git add -f $csvPath $logPath"
    Write-Host "  git commit -m `"icsbep: $resolvedRunner sweep, $rowCount cases, $([math]::Round($elapsed.TotalMinutes, 1)) min on `$(hostname)`""
    Write-Host ""
    Write-Host "Or, to keep the run local-only, no action needed — outputs/ stays untracked."
} else {
    Write-Warning "CSV not found at $csvPath — something went wrong before the first case wrote a row."
}

exit $exit
