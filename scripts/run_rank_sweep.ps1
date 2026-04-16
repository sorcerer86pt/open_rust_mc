# ============================================================================
# Rank sweep: SVD accuracy vs speed tradeoff for paper
#
# Runs both Godiva and PWR pin cell at ranks 1-6, measuring:
#   - SVD vs Table k_eff gap (pcm) — the SVD compression error
#   - SVD speedup vs Table
#   - Absolute delta from experiment
#
# Output: results/rank_sweep_godiva.csv, results/rank_sweep_pwr.csv
#
# Usage:
#   .\scripts\run_rank_sweep.ps1                    # default: 5 seeds, 50k Godiva / 20k PWR
#   .\scripts\run_rank_sweep.ps1 -Seeds 10 -Quick   # 10 seeds, reduced particles
# ============================================================================

param(
    [int]$Seeds = 5,
    [int]$GodivaParticles = 50000,
    [int]$PwrParticles = 20000,
    [int]$Batches = 150,
    [int]$Inactive = 20,
    [switch]$Quick,       # Reduced stats for quick testing
    [switch]$GpuGodiva,   # Also run GPU Godiva at each rank
    [switch]$GpuPwr,      # Also run GPU PWR at each rank
    [string]$DataDir = "../data/endfb-vii.1-hdf5/neutron"
)

if ($Quick) {
    $Seeds = 3
    $GodivaParticles = 20000
    $PwrParticles = 10000
    $Batches = 80
    $Inactive = 15
}

$ErrorActionPreference = "Stop"
$ResultsDir = "results"
if (-not (Test-Path $ResultsDir)) { New-Item -ItemType Directory -Path $ResultsDir | Out-Null }

$Ranks = 1..6
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# ── Godiva rank sweep ──
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Godiva Rank Sweep (ranks 1-6)" -ForegroundColor Cyan
Write-Host "  Seeds=$Seeds, Particles=$GodivaParticles, Batches=$Batches" -ForegroundColor Cyan
Write-Host "========================================`n"

$GodivaFile = "$ResultsDir/rank_sweep_godiva_${Timestamp}.csv"
"rank,svd_k,svd_sigma_pcm,table_k,table_sigma_pcm,svd_table_gap_pcm,svd_delta_exp_pcm,table_delta_exp_pcm,svd_nsp,table_nsp,speedup" | Out-File $GodivaFile -Encoding utf8

foreach ($rank in $Ranks) {
    Write-Host "--- Godiva rank=$rank ---" -ForegroundColor Yellow
    $output = & cargo run --release --bin godiva -- $DataDir `
        --mode both --rank $rank `
        --batches $Batches --inactive $Inactive `
        --particles $GodivaParticles --seeds $Seeds 2>&1 | Out-String

    # Parse output
    $svdK = if ($output -match "SVD \(rank=\d+\):\s*\n\s*k_eff\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
    $svdSigma = if ($output -match "SVD \(rank=\d+\):\s*\n\s*k_eff\s*=\s*[\d.]+\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
    $tblK = if ($output -match "Pointwise Table:\s*\n\s*k_eff\s*=\s*([\d.]+)") { $Matches[1] } else { "?" }
    $tblSigma = if ($output -match "Pointwise Table:\s*\n\s*k_eff\s*=\s*[\d.]+\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
    $gap = if ($output -match "k_eff gap.*=\s*(\d+)\s*pcm") { $Matches[1] } else { "?" }
    $svdDelta = if ($output -match "SVD delta\(exp\)\s*=\s*(\d+)\s*pcm") { $Matches[1] } else { "?" }
    $tblDelta = if ($output -match "Table delta\(exp\)\s*=\s*(\d+)\s*pcm") { $Matches[1] } else { "?" }
    $speedup = if ($output -match "SVD speedup\s*=\s*([\d.]+)x\s*\(([\d.]+)\s*vs\s*([\d.]+)") {
        "$($Matches[1]),$($Matches[2]),$($Matches[3])"
    } else { "?,?,?" }

    $line = "$rank,$svdK,$svdSigma,$tblK,$tblSigma,$gap,$svdDelta,$tblDelta,$speedup"
    $line | Out-File $GodivaFile -Append -Encoding utf8
    Write-Host "  rank=$rank  SVD-Table gap=${gap} pcm  SVD delta=${svdDelta} pcm  speedup=$($speedup.Split(',')[0])x"
}

Write-Host "`nGodiva results written to $GodivaFile"

# ── PWR pin cell rank sweep ──
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  PWR Pin Cell Rank Sweep (ranks 1-6)" -ForegroundColor Cyan
Write-Host "  Seeds=$Seeds, Particles=$PwrParticles, Batches=$Batches" -ForegroundColor Cyan
Write-Host "========================================`n"

$PwrFile = "$ResultsDir/rank_sweep_pwr_${Timestamp}.csv"
"rank,svd_k,svd_sigma_pcm,table_k,table_sigma_pcm,svd_table_gap_pcm,svd_nsp,table_nsp,speedup" | Out-File $PwrFile -Encoding utf8

foreach ($rank in $Ranks) {
    Write-Host "--- PWR rank=$rank ---" -ForegroundColor Yellow
    $output = & cargo run --release --bin pwr_pincell -- $DataDir `
        --mode both --rank $rank `
        --batches $Batches --inactive $Inactive `
        --particles $PwrParticles --seeds $Seeds 2>&1 | Out-String

    $svdK = if ($output -match "SVD \(rank=\d+\):\s*\n\s*k_eff\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
    $svdSigma = if ($output -match "SVD \(rank=\d+\):\s*\n\s*k_eff\s*=\s*[\d.]+\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
    $tblK = if ($output -match "Pointwise Table:\s*\n\s*k_eff\s*=\s*([\d.]+)") { $Matches[1] } else { "?" }
    $tblSigma = if ($output -match "Pointwise Table:\s*\n\s*k_eff\s*=\s*[\d.]+\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
    $gap = if ($output -match "k_eff gap.*=\s*(\d+)\s*pcm") { $Matches[1] } else { "?" }
    $speedup = if ($output -match "SVD speedup\s*=\s*([\d.]+)x\s*\(([\d.]+)\s*vs\s*([\d.]+)") {
        "$($Matches[1]),$($Matches[2]),$($Matches[3])"
    } else { "?,?,?" }

    $line = "$rank,$svdK,$svdSigma,$tblK,$tblSigma,$gap,$speedup"
    $line | Out-File $PwrFile -Append -Encoding utf8
    Write-Host "  rank=$rank  SVD-Table gap=${gap} pcm  speedup=$($speedup.Split(',')[0])x"
}

Write-Host "`nPWR results written to $PwrFile"

# ── GPU benchmarks (optional) ──
if ($GpuGodiva) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  GPU Godiva Rank Sweep" -ForegroundColor Green
    Write-Host "========================================`n"

    $GpuGodivaFile = "$ResultsDir/rank_sweep_gpu_godiva_${Timestamp}.csv"
    "rank,gpu_k,gpu_sigma_pcm,gpu_nsp" | Out-File $GpuGodivaFile -Encoding utf8

    foreach ($rank in $Ranks) {
        Write-Host "--- GPU Godiva rank=$rank ---" -ForegroundColor Yellow
        $output = & cargo run --release --features cuda --bin gpu_pwr_bench -- $DataDir `
            --geometry godiva --rank $rank `
            --batches $Batches --inactive $Inactive `
            -B $Batches `
            --particles $GodivaParticles --seeds $Seeds 2>&1 | Out-String

        $gpuK = if ($output -match "k_inf\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
        $gpuSigma = if ($output -match "k_inf\s*=\s*[\d.]+\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
        $gpuNsp = if ($output -match "ns/particle\s*=\s*([\d.]+)") { $Matches[1] } else { "?" }

        "$rank,$gpuK,$gpuSigma,$gpuNsp" | Out-File $GpuGodivaFile -Append -Encoding utf8
        Write-Host "  rank=$rank  GPU k=$gpuK  ns/p=$gpuNsp"
    }
    Write-Host "`nGPU Godiva results written to $GpuGodivaFile"
}

if ($GpuPwr) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  GPU PWR Pin Cell Rank Sweep" -ForegroundColor Green
    Write-Host "========================================`n"

    $GpuPwrFile = "$ResultsDir/rank_sweep_gpu_pwr_${Timestamp}.csv"
    "rank,gpu_k,gpu_sigma_pcm,gpu_nsp" | Out-File $GpuPwrFile -Encoding utf8

    foreach ($rank in $Ranks) {
        Write-Host "--- GPU PWR rank=$rank ---" -ForegroundColor Yellow
        $output = & cargo run --release --features cuda --bin gpu_pwr_bench -- $DataDir `
            --geometry pwr --rank $rank `
            --batches $Batches --inactive $Inactive `
            -B $Batches `
            --particles $PwrParticles --seeds $Seeds 2>&1 | Out-String

        $gpuK = if ($output -match "k_inf\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
        $gpuSigma = if ($output -match "k_inf\s*=\s*[\d.]+\s*\+/-\s*([\d.]+)") { $Matches[1] } else { "?" }
        $gpuNsp = if ($output -match "ns/particle\s*=\s*([\d.]+)") { $Matches[1] } else { "?" }

        "$rank,$gpuK,$gpuSigma,$gpuNsp" | Out-File $GpuPwrFile -Append -Encoding utf8
        Write-Host "  rank=$rank  GPU k=$gpuK  ns/p=$gpuNsp"
    }
    Write-Host "`nGPU PWR results written to $GpuPwrFile"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  DONE — all results in $ResultsDir/" -ForegroundColor Cyan
Write-Host "========================================`n"

# Print Godiva summary table
Write-Host "Godiva SVD Rank vs Accuracy (from $GodivaFile):"
Write-Host "Rank | SVD-Table Gap | Δ(exp) | Speedup"
Write-Host "-----|---------------|--------|--------"
Get-Content $GodivaFile | Select-Object -Skip 1 | ForEach-Object {
    $cols = $_ -split ","
    Write-Host ("{0,4} | {1,13} | {2,6} | {3}" -f $cols[0], "$($cols[5]) pcm", "$($cols[6]) pcm", "$($cols[8])x")
}
