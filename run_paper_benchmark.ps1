# Paper benchmark: CPU SVD vs CPU Table vs GPU SVD
#
# Runs all three modes with multi-seed statistics for paper-quality results.
# Each mode runs independently to avoid thermal throttling contamination.
#
# Usage:
#   .\run_paper_benchmark.ps1                     # default (20k particles, 3 seeds)
#   .\run_paper_benchmark.ps1 -Particles 50000 -Seeds 5 -Batches 150
#   .\run_paper_benchmark.ps1 -Output results      # log to results_YYYYMMDD_HHMM.txt
#   .\run_paper_benchmark.ps1 -Quick               # fast sanity check (5k particles, 1 seed)

param(
    [int]$Particles = 20000,
    [int]$Batches = 100,
    [int]$Inactive = 20,
    [int]$Seeds = 3,
    [int]$Rank = 5,
    [string]$Output = "",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"

if ($Quick) {
    $Particles = 5000; $Batches = 30; $Inactive = 5; $Seeds = 1
}

if ($Output) {
    $logFile = "${Output}_${timestamp}.txt"
    Start-Transcript -Path $logFile -Append -Force | Out-Null
    Write-Host "Logging to: $logFile" -ForegroundColor Cyan
}

$DATA = "data\endfb-vii.1-hdf5\neutron"
$active = $Batches - $Inactive
$histories = $active * $Particles

Write-Host "`n$('=' * 70)" -ForegroundColor Cyan
Write-Host "  PAPER BENCHMARK — PWR Pin Cell (SVD vs Table vs GPU)" -ForegroundColor Cyan
Write-Host "$('=' * 70)" -ForegroundColor Cyan
Write-Host "  Particles:    $Particles/batch"
Write-Host "  Batches:      $Batches ($Inactive inactive + $active active)"
Write-Host "  Seeds:        $Seeds"
Write-Host "  Histories:    $histories per seed"
Write-Host "  SVD rank:     $Rank"
Write-Host "  Data:         $DATA"
Write-Host "  Timestamp:    $timestamp"
Write-Host "$('=' * 70)`n"

Set-Location rust_prototype

# Build all binaries
Write-Host "── Building release binaries ──" -ForegroundColor Yellow
cargo build --release --bin pwr_pincell --bin gpu_pwr_bench --features cuda 2>&1 | Out-String -Stream
Write-Host ""

# ── 1. CPU Table baseline ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 1: CPU Pointwise Table (baseline)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
cargo run --release --bin pwr_pincell -- "..\$DATA" `
    --mode table --rank $Rank --batches $Batches --inactive $Inactive `
    --particles $Particles --seeds $Seeds 2>&1 | Out-String -Stream

# ── 2. CPU SVD ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 2: CPU SVD (rank=$Rank)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
cargo run --release --bin pwr_pincell -- "..\$DATA" `
    --mode svd --rank $Rank --batches $Batches --inactive $Inactive `
    --particles $Particles --seeds $Seeds 2>&1 | Out-String -Stream

# ── 3. GPU SVD ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 3: GPU SVD (fused+sort, rank=$Rank)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
cargo run --release --features cuda --bin gpu_pwr_bench -- "..\$DATA" `
    --rank $Rank -B $Batches --inactive $Inactive `
    --particles $Particles --seeds $Seeds --mode fused 2>&1 | Out-String -Stream

# ── 4. GPU Scaling (fused only, 1 seed) ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 4: GPU Scaling (particle count sweep)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
foreach ($N in @(5000, 10000, 20000, 50000, 100000)) {
    Write-Host "  --- $N particles ---" -ForegroundColor Yellow
    cargo run --release --features cuda --bin gpu_pwr_bench -- "..\$DATA" `
        --rank $Rank -B 30 --inactive 5 `
        --particles $N --seeds 1 --mode fused 2>&1 |
        Select-String -Pattern "k_inf|ns/particle|sim time" | Out-String -Stream
    Write-Host ""
}

Write-Host "`n$('=' * 70)" -ForegroundColor Cyan
Write-Host "  ALL BENCHMARKS COMPLETE — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "$('=' * 70)`n"

Set-Location ..

if ($Output) {
    Stop-Transcript
    Write-Host "Results saved to: $logFile" -ForegroundColor Cyan
}
