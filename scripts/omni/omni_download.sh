#!/bin/bash -l
# Download + extract OmniObject3D raw_scans for ONE category, or --all from the list.
# Usage:
#   scripts/omni/omni_download.sh knife          # one category (TEST FIRST)
#   scripts/omni/omni_download.sh --all          # every category in categories_kitchen.txt
# Bulk data goes to /scratch (18TB), NOT home.
set -euo pipefail

OXL="$HOME/.venv-openxlab/bin/openxlab"
REPO="omniobject3d/OmniObject3D-New"
DEST="/scratch/$USER/omni/raw_scans"          # tars + extracted objects live here
LIST="$(dirname "$0")/categories_kitchen.txt"

mkdir -p "$DEST"

download_one() {
  local cat="$1"
  local tdir="$DEST/$cat"
  if [ -d "$tdir/extracted" ]; then
    echo "[dl] $cat already extracted, skipping"; return 0
  fi
  echo "[dl] downloading $cat ..."
  "$OXL" dataset download --dataset-repo "$REPO" \
    --source-path "/raw/raw_scans/${cat}.tar.gz" --target-path "$tdir"
  # openxlab nests output under <repo>___.../raw/raw_scans/<cat>.tar.gz
  local tar
  tar="$(find "$tdir" -name "${cat}.tar.gz" | head -1)"
  if [ -z "$tar" ]; then echo "[dl] ERROR: ${cat}.tar.gz not found after download"; return 1; fi
  echo "[dl] extracting $cat ..."
  mkdir -p "$tdir/extracted"
  tar xzf "$tar" -C "$tdir/extracted"
  rm -f "$tar"   # save space: keep only extracted meshes
  local n
  n="$(find "$tdir/extracted" -name Scan.obj | wc -l)"
  echo "[dl] $cat: $n objects extracted"
}

if [ "${1:-}" = "--all" ]; then
  while read -r cat; do
    [ -z "$cat" ] && continue
    case "$cat" in \#*) continue;; esac
    download_one "$cat"
  done < <(grep -v '^#' "$LIST")
else
  [ -z "${1:-}" ] && { echo "usage: $0 <category>|--all"; exit 1; }
  download_one "$1"
fi
echo "[dl] DONE"
