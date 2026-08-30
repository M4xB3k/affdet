#!/bin/bash
set -euo pipefail

# One-time setup on IFI cluster.
BLENDER_TARBALL_URL="https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz"
BLENDER_ROOT="$HOME/blender-4.1.0-linux-x64"

echo "[setup] downloading Blender..."
wget -O "$HOME/blender.tar.xz" "$BLENDER_TARBALL_URL"
tar -xf "$HOME/blender.tar.xz" -C "$HOME"

echo "[setup] cloning affdet repo..."
git clone <repo_url> "$HOME/affdet"
cd "$HOME/affdet"

echo "[setup] installing affdet into Blender Python..."
"$BLENDER_ROOT/4.1/python/bin/python3.11" -m pip install -e ".[dev]"

echo "[setup] downloading meshes + HDRIs..."
python scripts/download_meshes.py --out "/scratch/$USER/data/meshes"
python scripts/download_hdri.py --out "/scratch/$USER/data/hdri"
