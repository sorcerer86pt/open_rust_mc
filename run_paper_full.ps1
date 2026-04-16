# Full paper benchmark: 10-seed runs for all modes
#
# Runs sequentially to avoid thermal contamination:
#   1. OpenMC Godiva + PWR (WSL)
#   2. CPU Table Godiva (10 seeds)
#   3. CPU SVD Godiva (10 seeds)
#   4. CPU Table PWR (10 seeds)
#   5. CPU SVD PWR (10 seeds)
#   6. GPU SVD PWR (10 seeds)
#   7. GPU scaling sweep
#
# Usage:
#   .\run_paper_full.ps1
#   .\run_paper_full.ps1 -Seeds 5 -Particles 20000    # shorter run
#   .\run_paper_full.ps1 -SkipOpenMC                   # skip WSL/OpenMC

param(
    [int]$Seeds = 10,
    [int]$Particles = 50000,
    [int]$GodivaBatches = 150,
    [int]$PwrBatches = 150,
    [int]$Inactive = 20,
    [int]$Rank = 5,
    [switch]$SkipOpenMC
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$logFile = "paper_results_${timestamp}.txt"

Start-Transcript -Path $logFile -Append -Force | Out-Null
Write-Host "Logging to: $logFile" -ForegroundColor Cyan

$DATA = "data\endfb-vii.1-hdf5\neutron"

Write-Host "`n$('=' * 70)" -ForegroundColor Cyan
Write-Host "  FULL PAPER BENCHMARK — $Seeds seeds, $Particles particles" -ForegroundColor Cyan
Write-Host "  Godiva: $GodivaBatches batches | PWR: $PwrBatches batches" -ForegroundColor Cyan
Write-Host "  Inactive: $Inactive | SVD rank: $Rank" -ForegroundColor Cyan
Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "$('=' * 70)`n"

# ── 0. Build ──
Write-Host "── Building release binaries ──" -ForegroundColor Yellow
Set-Location rust_prototype
cargo build --release --bin godiva --bin pwr_pincell --bin gpu_pwr_bench --features cuda 2>&1 | Out-String -Stream
Set-Location ..
Write-Host ""

# ── 1. OpenMC reference (WSL) ──
if (-not $SkipOpenMC) {
    Write-Host "`n$('=' * 70)" -ForegroundColor Green
    Write-Host "  TEST 1: OpenMC reference (Godiva + PWR, $Seeds seeds)" -ForegroundColor Green
    Write-Host "$('=' * 70)`n"
    wsl -d Ubuntu-24.04 -- bash -c "source ~/miniforge3/bin/activate openmc && cd /mnt/c/Users/fog/madman_svd_experiment/scripts && python paper_openmc_benchmark.py --seeds $Seeds --particles $Particles --batches $GodivaBatches --inactive $Inactive" 2>&1 | Out-String -Stream
} else {
    Write-Host "`n  SKIPPING OpenMC (--SkipOpenMC)`n" -ForegroundColor Gray
}

# ── 2. Godiva CPU Table ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 2: Godiva — CPU Table ($Seeds seeds)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
Set-Location rust_prototype
cargo run --release --bin godiva -- "..\$DATA" `
    --mode table --batches $GodivaBatches --inactive $Inactive `
    --particles $Particles --seeds $Seeds 2>&1 | Out-String -Stream
Set-Location ..

# ── 3. Godiva CPU SVD ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 3: Godiva — CPU SVD rank=$Rank ($Seeds seeds)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
Set-Location rust_prototype
cargo run --release --bin godiva -- "..\$DATA" `
    --mode svd --rank $Rank --batches $GodivaBatches --inactive $Inactive `
    --particles $Particles --seeds $Seeds 2>&1 | Out-String -Stream
Set-Location ..

# ── 4. PWR CPU Table ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 4: PWR pin cell — CPU Table ($Seeds seeds)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
Set-Location rust_prototype
cargo run --release --bin pwr_pincell -- "..\$DATA" `
    --mode table --batches $PwrBatches --inactive $Inactive `
    --particles $Particles --seeds $Seeds 2>&1 | Out-String -Stream
Set-Location ..

# ── 5. PWR CPU SVD ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 5: PWR pin cell — CPU SVD rank=$Rank ($Seeds seeds)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
Set-Location rust_prototype
cargo run --release --bin pwr_pincell -- "..\$DATA" `
    --mode svd --rank $Rank --batches $PwrBatches --inactive $Inactive `
    --particles $Particles --seeds $Seeds 2>&1 | Out-String -Stream
Set-Location ..

# ── 6. PWR GPU SVD ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 6: PWR pin cell — GPU SVD rank=$Rank ($Seeds seeds)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
Set-Location rust_prototype
cargo run --release --features cuda --bin gpu_pwr_bench -- "..\$DATA" `
    --rank $Rank -B $PwrBatches --inactive $Inactive `
    --particles $Particles --seeds $Seeds --mode fused 2>&1 | Out-String -Stream
Set-Location ..

# ── 7. GPU Scaling sweep ──
Write-Host "`n$('=' * 70)" -ForegroundColor Green
Write-Host "  TEST 7: GPU Scaling (particle count sweep, 3 seeds each)" -ForegroundColor Green
Write-Host "$('=' * 70)`n"
Set-Location rust_prototype
foreach ($N in @(5000, 10000, 20000, 50000, 100000, 200000)) {
    Write-Host "  --- $N particles ---" -ForegroundColor Yellow
    cargo run --release --features cuda --bin gpu_pwr_bench -- "..\$DATA" `
        --rank $Rank -B 50 --inactive 10 `
        --particles $N --seeds 3 --mode fused 2>&1 |
        Select-String -Pattern "k_inf|ns/particle|Total sim" | Out-String -Stream
    Write-Host ""
}
Set-Location ..

# ── Done ──
Write-Host "`n$('=' * 70)" -ForegroundColor Cyan
Write-Host "  ALL BENCHMARKS COMPLETE" -ForegroundColor Cyan
Write-Host "  Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "  Results:  $logFile" -ForegroundColor Cyan
Write-Host "$('=' * 70)`n"

Stop-Transcript
