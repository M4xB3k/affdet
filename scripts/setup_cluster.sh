#!/bin/bash
# Run once on the IFI cluster to set up everything needed for rendering.
# Usage: bash scripts/setup_cluster.sh
set -euo pipefail

REPO="$HOME/affdet"
SCRATCH="/scratch/$USER/affdet"
BLENDER_VER="5.1.2"
BLENDER_DIR="$HOME/blender-5.1"
BLENDER_URL="https://download.blender.org/release/Blender5.1/blender-${BLENDER_VER}-linux-x64.tar.xz"

echo "=== [1/5] Pull latest repo ==="
cd "$REPO"
git pull || echo "WARN: git pull failed (auth?) — continuing with current checkout $(git rev-parse --short HEAD)"

echo "=== [2/5] Install Blender $BLENDER_VER (if not present) ==="
if [ -f "$BLENDER_DIR/blender" ]; then
    echo "Blender already at $BLENDER_DIR — skipping"
else
    echo "Downloading Blender $BLENDER_VER..."
    wget -q --show-progress "$BLENDER_URL" -O /tmp/blender.tar.xz
    tar -xf /tmp/blender.tar.xz -C "$HOME"
    mv "$HOME/blender-${BLENDER_VER}-linux-x64" "$BLENDER_DIR"
    rm /tmp/blender.tar.xz
    echo "Blender installed at $BLENDER_DIR"
fi
"$BLENDER_DIR/blender" --version | head -1

echo "=== [3/5] Install Python deps into Blender's bundled Python ==="
# blender.py imports pydantic (schemas) and OpenEXR/Imath (mask export) IN-PROCESS,
# so they must live in Blender's own interpreter — NOT system python3.
# numpy is already bundled with Blender; do not reinstall it.
BPY=$(ls "$BLENDER_DIR"/*/python/bin/python3* 2>/dev/null | head -1)
if [ -z "$BPY" ]; then
    echo "ERROR: could not locate Blender's bundled python under $BLENDER_DIR" >&2
    exit 1
fi
echo "Blender python: $BPY ($("$BPY" --version 2>&1))"
"$BPY" -m ensurepip --upgrade 2>/dev/null || true
"$BPY" -m pip install --no-input pydantic OpenEXR Imath
echo "Verifying imports:"
"$BPY" -c "import pydantic, numpy, OpenEXR, Imath; print('  deps OK — pydantic', pydantic.__version__, '/ numpy', numpy.__version__)"

echo "=== [4/5] Download YCB meshes to scratch ==="
mkdir -p "$SCRATCH"
python3 "$REPO/scripts/download_meshes.py" --out "$SCRATCH/meshes"
echo "Meshes: $(ls $SCRATCH/meshes | wc -l) objects"

echo "=== [5/5] Download table textures to scratch ==="
mkdir -p "$SCRATCH/table_textures"
TEX_BASE="https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k"
# plywood PBR set
wget -q -nc "$TEX_BASE/wood_table_001/wood_table_001_diff_2k.jpg" \
    -O "$SCRATCH/table_textures/plywood_diff_2k.jpg" || true
wget -q -nc "https://dl.polyhaven.org/file/ph-assets/Textures/exr/2k/wood_table_001/wood_table_001_nor_gl_2k.exr" \
    -O "$SCRATCH/table_textures/plywood_nor_gl_2k.exr" || true
wget -q -nc "https://dl.polyhaven.org/file/ph-assets/Textures/exr/2k/wood_table_001/wood_table_001_rough_2k.exr" \
    -O "$SCRATCH/table_textures/plywood_rough_2k.exr" || true
echo "Textures: $(ls $SCRATCH/table_textures)"

echo ""
echo "=== Setup complete ==="
echo "To render 1000 scenes:"
echo "  mkdir -p $REPO/logs"
echo "  sbatch --array=0-9 $REPO/scripts/slurm_render.sh"
