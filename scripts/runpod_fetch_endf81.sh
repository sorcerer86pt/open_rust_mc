#!/usr/bin/env bash
# Stream-extract ENDF/B-VIII.1 HDF5 from OpenMC's box.com mirror directly
# into /workspace/open_rust_mc/data/endfb-viii.1-hdf5/, avoiding the
# intermediate ~10 GB .tar.xz file on disk. Designed for the RunPod
# pod where the 80 GB MooseFS workspace is the only volume with enough
# room for the ~30 GB uncompressed library.
#
# Uses nohup + setsid so SSH disconnects do not SIGHUP the curl|tar
# pipeline. Run via:
#     bash scripts/runpod_fetch_endf81.sh > /workspace/openmc_fetch.log 2>&1 &
#     disown
set -euo pipefail

URL="https://anl.box.com/shared/static/6qr7jezzihkj9p9esl5jn19qgpujyjyz.xz"
DEST="/workspace/open_rust_mc/data/endfb-viii.1-hdf5"

mkdir -p "$DEST"
echo "[$(date)] stream-extract from $URL to $DEST"

# Note: using --progress-bar instead of -s lets us see ETA in the log.
# --fail aborts on HTTP error so we don't silently extract a 404 page.
# tar `--no-same-owner` avoids the MooseFS uid 1000 / 4000 ownership
# warnings (uid mapping isn't supported on the remote workspace).
curl --fail -L --progress-bar "$URL" \
  | tar -xJ -C "$DEST" --no-same-owner --strip-components=1

echo "[$(date)] done"
echo "neutron:           $(ls "$DEST"/neutron/*.h5 2>/dev/null | wc -l)"
echo "thermal:           $(ls "$DEST"/thermal/*.h5 2>/dev/null | wc -l)"
echo "photon:            $(ls "$DEST"/photon/*.h5 2>/dev/null | wc -l)"
[ -f "$DEST/cross_sections.xml" ] && echo "cross_sections.xml: present" || echo "cross_sections.xml: MISSING"
df -h /workspace
