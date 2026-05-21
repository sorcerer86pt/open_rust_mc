# SPDX-License-Identifier: MIT
# ============================================================================
# Download and organize nuclear data libraries for benchmarking
#
# Downloads from https://openmc.org/data/:
#   - ENDF/B-VIII.1 HDF5 (default,  ~6.5 GB — latest, Oct 2024 release, NJOY 2016.78)
#   - ENDF/B-VIII.0 HDF5 (optional, ~6.4 GB)
#   - ENDF/B-VII.1 HDF5  (optional, ~5.8 GB — legacy, kept for reproducibility)
#   - JEFF-3.3 HDF5      (optional, ~5.2 GB)
#
# Also sets up S(alpha,beta) thermal scattering data.
#
# Usage:
#   .\scripts\setup_nuclear_data.ps1                    # ENDF/B-VIII.1 (default)
#   .\scripts\setup_nuclear_data.ps1 -All               # All four libraries
#   .\scripts\setup_nuclear_data.ps1 -Vii1              # only the legacy VII.1
#   .\scripts\setup_nuclear_data.ps1 -Jeff -Endf8       # JEFF + ENDF/B-VIII.0
# ============================================================================

param(
    [switch]$All,
    [switch]$Jeff,
    [switch]$Endf8,
    [switch]$Endf81,
    [switch]$Vii1,
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

    # Some archives (e.g. ENDF/B-VIII.1) contain a top-level wrapper folder
    # matching the label — extraction lands at $OutDir/$Label/neutron/... instead
    # of $OutDir/neutron/... Flatten if we detect that shape.
    $wrapper = Join-Path $OutDir $Label
    if (-not (Test-Path "$OutDir/neutron") -and (Test-Path "$wrapper/neutron")) {
        Write-Host "  Flattening wrapper directory ($Label/ → .)..." -ForegroundColor DarkYellow
        Get-ChildItem $wrapper -Force | ForEach-Object {
            Move-Item -Path $_.FullName -Destination $OutDir -Force
        }
        Remove-Item $wrapper -Recurse -Force
    }

    $count = (Get-ChildItem "$OutDir/neutron/*.h5" -ErrorAction SilentlyContinue).Count
    Write-Host "  ${Label}: $count nuclide files extracted" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Nuclear Data Setup" -ForegroundColor Cyan
Write-Host "========================================`n"

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

# ENDF/B-VIII.1 is the default library. It only gets skipped when the caller asks
# for a single specific other library (e.g. `-Vii1`, `-Endf8`, or `-Jeff` alone).
$onlyOtherRequested = ($Vii1 -or $Endf8 -or $Jeff) -and -not $All -and -not $Endf81
$skipEndf81 = $onlyOtherRequested

# ── ENDF/B-VIII.1 (default unless caller asked for a single specific other library) ──
if (-not $skipEndf81) {
    Write-Host "`n[1/4] ENDF/B-VIII.1 HDF5 (default, NJOY 2016.78)" -ForegroundColor Cyan
    Download-And-Extract `
        -Url "https://anl.box.com/shared/static/6qr7jezzihkj9p9esl5jn19qgpujyjyz.xz" `
        -OutDir "$DataDir/endfb-viii.1-hdf5" `
        -Label "endfb-viii.1-hdf5"
} else {
    Write-Host "`n[1/4] ENDF/B-VIII.1 — skipped (only a legacy library was requested)" -ForegroundColor DarkGray
}

# ── ENDF/B-VIII.0 (optional) ──
if ($All -or $Endf8) {
    Write-Host "`n[2/4] ENDF/B-VIII.0 HDF5" -ForegroundColor Cyan
    Download-And-Extract `
        -Url "https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz" `
        -OutDir "$DataDir/endfb-viii.0-hdf5" `
        -Label "endfb-viii.0-hdf5"
} else {
    Write-Host "`n[2/4] ENDF/B-VIII.0 — skipped (use -Endf8 or -All)" -ForegroundColor DarkGray
}

# ── ENDF/B-VII.1 (legacy, opt-in) ──
if ($All -or $Vii1) {
    Write-Host "`n[3/4] ENDF/B-VII.1 HDF5 (legacy)" -ForegroundColor Cyan
    Download-And-Extract `
        -Url "https://anl.box.com/shared/static/9igk353lmfgbpvhq3556nb4h6fheanzb.xz" `
        -OutDir "$DataDir/endfb-vii.1-hdf5" `
        -Label "endfb-vii.1-hdf5"
} else {
    Write-Host "`n[3/4] ENDF/B-VII.1 — skipped (use -Vii1 or -All)" -ForegroundColor DarkGray
}

# ── JEFF-3.3 (optional) ──
if ($All -or $Jeff) {
    Write-Host "`n[4/4] JEFF-3.3 HDF5" -ForegroundColor Cyan
    Download-And-Extract `
        -Url "https://anl.box.com/shared/static/3v7pru88pgm6f67sh6vcsod97m52asof.xz" `
        -OutDir "$DataDir/jeff-3.3-hdf5" `
        -Label "jeff-3.3-hdf5"
} else {
    Write-Host "`n[4/4] JEFF-3.3 — skipped (use -Jeff or -All)" -ForegroundColor DarkGray
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

# Verify each library that was requested in this invocation. Always check VII.1
# unless the caller explicitly skipped it; check VIII.0 / VIII.1 / JEFF only when
# they were part of this run so a single-library invocation doesn't false-fail.
$librariesToCheck = @()
if (-not $skipEndf81) { $librariesToCheck += "endfb-viii.1-hdf5" }
if ($All -or $Endf8)  { $librariesToCheck += "endfb-viii.0-hdf5" }
if ($All -or $Vii1)   { $librariesToCheck += "endfb-vii.1-hdf5" }
if ($All -or $Jeff)   { $librariesToCheck += "jeff-3.3-hdf5" }

foreach ($lib in $librariesToCheck) {
    Write-Host "`nKey files in $lib :"
    foreach ($f in @("U234.h5","U235.h5","U238.h5","O16.h5","H1.h5","Zr90.h5","c_H_in_H2O.h5")) {
        $path = "$DataDir/$lib/neutron/$f"
        if (Test-Path $path) {
            $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
            Write-Host "  [OK] $f ($sizeMB MB)" -ForegroundColor Green
        } else {
            Write-Host "  [MISSING] $f" -ForegroundColor Red
        }
    }
}
