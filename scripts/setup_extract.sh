#!/usr/bin/env bash
# One-time setup for DINOv2 feature extraction on the IFI cluster.
# Run on the HEADNODE (needs internet). Idempotent — safe to re-run.
#
#   bash scripts/setup_extract.sh
#
# Does two things:
#   1. Installs the CUDA build of torch into the repo venv (cu124 wheel —
#      driver 560.35 supports CUDA <=12.6). Skips if torch already imports.
#   2. Pre-populates the torch.hub cache (~/.cache/torch/hub) with the DINOv2
#      ViT-B/14 backbone + weights. HOME is NFS-shared, so compute nodes then
#      load from cache and never need internet.

set -euo pipefail

REPO="$HOME/affdet"
PY="$REPO/.venv/bin/python"

cd "$REPO"

echo "[setup_extract] 1/2 — torch (CUDA cu124) ..."
if "$PY" -c "import torch" 2>/dev/null; then
    echo "[setup_extract]   torch already installed: $("$PY" -c 'import torch; print(torch.__version__)')"
else
    "$PY" -m pip install --no-input torch --index-url https://download.pytorch.org/whl/cu124
fi

echo "[setup_extract] 2/2 — pre-caching DINOv2 vitb14 weights into torch.hub ..."
"$PY" - <<'PY'
import torch
# Downloads facebookresearch/dinov2 repo + vitb14 weights into ~/.cache/torch/hub.
# force_reload=False (default) reuses the cache on later calls / compute nodes.
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
n = sum(p.numel() for p in m.parameters())
print(f"[setup_extract]   dinov2_vitb14 loaded OK, {n/1e6:.1f}M params, cache warm")
PY

echo "[setup_extract] done. Now submit:  sbatch --array=0-9 scripts/slurm_extract.sh"
