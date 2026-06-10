#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# ============================================================================
# Download and organize nuclear data libraries for benchmarking (Ubuntu/Linux).
#
# Bash port of scripts/setup_nuclear_data.ps1, designed for headless Linux pods
# (e.g. RunPod) where the persistent volume is the only disk with room for the
# ~30 GB uncompressed library. Unlike the PowerShell version it STREAM-EXTRACTS
# (curl | tar) so the ~10 GB .tar.xz never lands on disk.
#
# Downloads from the OpenMC / ANL box.com mirrors:
#   - ENDF/B-VIII.1 HDF5 (default,  ~6.5 GB — Oct 2024 release, NJOY 2016.78)
#   - ENDF/B-VIII.0 HDF5 (optional, ~6.4 GB)
#   - ENDF/B-VII.1 HDF5  (optional, ~5.8 GB — legacy, kept for reproducibility)
#   - JEFF-3.3 HDF5      (optional, ~5.2 GB)
#
# Usage:
#   bash scripts/setup_nuclear_data.sh                  # ENDF/B-VIII.1 (default)
#   bash scripts/setup_nuclear_data.sh --all            # all four libraries
#   bash scripts/setup_nuclear_data.sh --vii1           # only the legacy VII.1
#   bash scripts/setup_nuclear_data.sh --jeff --endf8   # JEFF + ENDF/B-VIII.0
#   bash scripts/setup_nuclear_data.sh --data-dir /workspace/open_rust_mc/data
#
# Survive SSH disconnects on a remote pod:
#   nohup bash scripts/setup_nuclear_data.sh > setup_data.log 2>&1 &
#   disown
# ============================================================================
set -euo pipefail

# ── Mirror URLs (kept in lockstep with setup_nuclear_data.ps1) ──────────────
URL_ENDF81="https://anl.box.com/shared/static/6qr7jezzihkj9p9esl5jn19qgpujyjyz.xz"
URL_ENDF80="https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz"
URL_VII1="https://anl.box.com/shared/static/9igk353lmfgbpvhq3556nb4h6fheanzb.xz"
URL_JEFF="https://anl.box.com/shared/static/3v7pru88pgm6f67sh6vcsod97m52asof.xz"

# Default data dir = <repo-root>/data, resolved from this script's location so
# it works regardless of the caller's cwd. Overridable with --data-dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

want_all=0 want_endf8=0 want_vii1=0 want_jeff=0 want_endf81=0
while [ $# -gt 0 ]; do
  case "$1" in
    --all)      want_all=1 ;;
    --endf8)    want_endf8=1 ;;
    --endf81)   want_endf81=1 ;;
    --vii1)     want_vii1=1 ;;
    --jeff)     want_jeff=1 ;;
    --data-dir) DATA_DIR="$2"; shift ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# VIII.1 is the default unless the caller asked for ONLY a specific other lib.
only_other=0
if { [ $want_vii1 -eq 1 ] || [ $want_endf8 -eq 1 ] || [ $want_jeff -eq 1 ]; } \
   && [ $want_all -eq 0 ] && [ $want_endf81 -eq 0 ]; then
  only_other=1
fi

mkdir -p "$DATA_DIR"

# ── Stream-extract one library ──────────────────────────────────────────────
# The box.com tarballs are packaged as `<label>/neutron`, `<label>/thermal`,
# ... so `--strip-components=1` drops the wrapper, landing files at
# `$out/neutron/...`. `--no-same-owner` avoids uid-mapping warnings on MooseFS.
fetch_lib() {
  local url="$1" out="$2" label="$3"
  if [ -d "$out/neutron" ]; then
    local n; n=$(find "$out/neutron" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l)
    echo "  $label already present at $out ($n nuclide files) — skipping"
    return 0
  fi
  echo "[$(date '+%F %T')] stream-extract $label"
  echo "  $url -> $out"
  mkdir -p "$out"
  curl --fail -L --progress-bar "$url" \
    | tar -xJ -C "$out" --no-same-owner --strip-components=1
  if [ ! -d "$out/neutron" ]; then
    echo "  ERROR: $label extracted but no neutron/ dir found at $out" >&2
    return 1
  fi
  local n; n=$(find "$out/neutron" -maxdepth 1 -name '*.h5' | wc -l)
  echo "  $label: $n nuclide files extracted"
}

echo "========================================"
echo "  Nuclear Data Setup (Linux)"
echo "  data dir: $DATA_DIR"
echo "========================================"

if [ $only_other -eq 0 ]; then
  echo "[1/4] ENDF/B-VIII.1 (default)"
  fetch_lib "$URL_ENDF81" "$DATA_DIR/endfb-viii.1-hdf5" "endfb-viii.1-hdf5"
else
  echo "[1/4] ENDF/B-VIII.1 — skipped (only a specific other library was requested)"
fi

if [ $want_all -eq 1 ] || [ $want_endf8 -eq 1 ]; then
  echo "[2/4] ENDF/B-VIII.0"
  fetch_lib "$URL_ENDF80" "$DATA_DIR/endfb-viii.0-hdf5" "endfb-viii.0-hdf5"
else
  echo "[2/4] ENDF/B-VIII.0 — skipped (use --endf8 or --all)"
fi

if [ $want_all -eq 1 ] || [ $want_vii1 -eq 1 ]; then
  echo "[3/4] ENDF/B-VII.1 (legacy)"
  fetch_lib "$URL_VII1" "$DATA_DIR/endfb-vii.1-hdf5" "endfb-vii.1-hdf5"
else
  echo "[3/4] ENDF/B-VII.1 — skipped (use --vii1 or --all)"
fi

if [ $want_all -eq 1 ] || [ $want_jeff -eq 1 ]; then
  echo "[4/4] JEFF-3.3"
  fetch_lib "$URL_JEFF" "$DATA_DIR/jeff-3.3-hdf5" "jeff-3.3-hdf5"
else
  echo "[4/4] JEFF-3.3 — skipped (use --jeff or --all)"
fi

# ── Verify ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
for d in "$DATA_DIR"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  if [ -d "$d/neutron" ]; then
    n=$(find "$d/neutron" -maxdepth 1 -name '*.h5' | wc -l)
    sz=$(du -sh "$d/neutron" 2>/dev/null | cut -f1)
    printf "  %-22s %4s nuclides  %6s\n" "$name" "$n" "$sz"
    [ -f "$d/cross_sections.xml" ] && echo "    cross_sections.xml: present" \
                                   || echo "    cross_sections.xml: MISSING"
  else
    printf "  %-22s (no neutron/ folder)\n" "$name"
  fi
done

command -v df >/dev/null && { echo ""; df -h "$DATA_DIR" | tail -n +1; }
echo "[$(date '+%F %T')] done"
