# SPDX-License-Identifier: MIT
# ============================================================================
# Download and organize nuclear data libraries for benchmarking
#
# Downloads from https://openmc.org/official-data-libraries/:
#   - ENDF/B-VII.1 HDF5 (required, ~5.8 GB)
#   - ENDF/B-VIII.0 HDF5 (optional, ~6.4 GB)
#   - JEFF-3.3 HDF5 (optional, ~5.2 GB)
#
# Also sets up S(alpha,beta) thermal scattering data.
#
# Usage:
#   .\scripts\setup_nuclear_data.ps1                    # ENDF/B-VII.1 only
#   .\scripts\setup_nuclear_data.ps1 -All               # All three libraries
#   .\scripts\setup_nuclear_data.ps1 -Jeff -Endf8       # JEFF + ENDF/B-VIII.0
# ============================================================================

param(
    [switch]$All,
    [switch]$Jeff,
    [switch]$Endf8,
    [string]$DataDir = "data"
)

$ErrorActionPreference = "Stop"

function Download-And-Extract {
    param([string]$Url, [string]$OutDir, [string]$Label)

    if (Test-Path "$OutDir/neutron") {
        Write-Host "  $Label already exists at $OutDir — skipping" -ForegroundColor Green
        $count = (Get-ChildItem "$OutDir/neutron/*.h5" -ErrorAction SilentlyContinue).Count
        Write-Host "  ($count nuclide files found)"
        return
    }

    $archive = "$env:TEMP/$Label.tar.xz"
    Write-Host "  Downloading $Label..." -ForegroundColor Yellow
    Write-Host "  URL: $Url"

    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L -o $archive $Url --progress-bar
    } elseif (Get-Command wget -ErrorAction SilentlyContinue) {
        & wget -O $archive $Url
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $archive
    }

    if (-not (Test-Path $archive)) {
        Write-Host "  ERROR: Download failed for $Label" -ForegroundColor Red
        return
    }

    Write-Host "  Extracting $Label to $OutDir..."
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

    # Try 7z first (handles tar.xz natively), then tar
    if (Get-Command 7z -ErrorAction SilentlyContinue) {
        & 7z x $archive -o"$env:TEMP" -y | Out-Null
        $tarFile = $archive -replace '\.xz$', ''
        & 7z x $tarFile -o"$OutDir" -y | Out-Null
        Remove-Item $tarFile -ErrorAction SilentlyContinue
    } elseif (Get-Command tar -ErrorAction SilentlyContinue) {
        & tar -xf $archive -C $OutDir
    } else {
        Write-Host "  ERROR: No tar.xz extractor found. Install 7-Zip or use WSL." -ForegroundColor Red
        Write-Host "  Manual: extract $archive to $OutDir"
        return
    }

    Remove-Item $archive -ErrorAction SilentlyContinue
    $count = (Get-ChildItem "$OutDir/neutron/*.h5" -ErrorAction SilentlyContinue).Count
    Write-Host "  ${Label}: $count nuclide files extracted" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Nuclear Data Setup" -ForegroundColor Cyan
Write-Host "========================================`n"

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

# ── ENDF/B-VII.1 (always required) ──
Write-Host "`n[1/3] ENDF/B-VII.1 HDF5 (primary library)" -ForegroundColor Cyan
Download-And-Extract `
    -Url "https://anl.box.com/shared/static/9igk353lmfgbpvhq3556nb4h6fheanzb.xz" `
    -OutDir "$DataDir/endfb-vii.1-hdf5" `
    -Label "endfb-vii.1-hdf5"

# ── ENDF/B-VIII.0 (optional) ──
if ($All -or $Endf8) {
    Write-Host "`n[2/3] ENDF/B-VIII.0 HDF5" -ForegroundColor Cyan
    Download-And-Extract `
        -Url "https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz" `
        -OutDir "$DataDir/endfb-viii.0-hdf5" `
        -Label "endfb-viii.0-hdf5"
} else {
    Write-Host "`n[2/3] ENDF/B-VIII.0 — skipped (use -Endf8 or -All)" -ForegroundColor DarkGray
}

# ── JEFF-3.3 (optional) ──
if ($All -or $Jeff) {
    Write-Host "`n[3/3] JEFF-3.3 HDF5" -ForegroundColor Cyan
    Download-And-Extract `
        -Url "https://anl.box.com/shared/static/3v7pru88pgm6f67sh6vcsod97m52asof.xz" `
        -OutDir "$DataDir/jeff-3.3-hdf5" `
        -Label "jeff-3.3-hdf5"
} else {
    Write-Host "`n[3/3] JEFF-3.3 — skipped (use -Jeff or -All)" -ForegroundColor DarkGray
}

# ── Verify ──
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Data Directory Summary" -ForegroundColor Cyan
Write-Host "========================================`n"

Get-ChildItem $DataDir -Directory | ForEach-Object {
    $neutronDir = Join-Path $_.FullName "neutron"
    if (Test-Path $neutronDir) {
        $count = (Get-ChildItem "$neutronDir/*.h5").Count
        $sizeMB = [math]::Round((Get-ChildItem "$neutronDir/*.h5" | Measure-Object -Property Length -Sum).Sum / 1MB)
        Write-Host ("  {0,-25} {1,4} nuclides  {2,6} MB" -f $_.Name, $count, $sizeMB)
    } else {
        Write-Host ("  {0,-25} (no neutron/ folder)" -f $_.Name) -ForegroundColor DarkGray
    }
}

Write-Host "`nKey files for benchmarks:"
foreach ($f in @("U234.h5","U235.h5","U238.h5","O16.h5","H1.h5","Zr90.h5","c_H_in_H2O.h5")) {
    $path = "$DataDir/endfb-vii.1-hdf5/neutron/$f"
    if (Test-Path $path) {
        $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Write-Host "  [OK] $f ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $f" -ForegroundColor Red
    }
}
