#!/bin/bash -l
#
# SLURM array job — extracts DINOv2 features for 1000 scenes across 10 workers.
# Mirrors slurm_render.sh: task N reads scenes from worker_N/ and writes
# features to worker_N/ (one .npz + .meta.json per scene).
#
# Submit:   sbatch --array=0-9 scripts/slurm_extract.sh   (all 10 workers)
# Subset:   sbatch --array=2-5 scripts/slurm_extract.sh   (workers 2,3,4,5)
# Single:   sbatch --array=0   scripts/slurm_extract.sh   (worker_0, test)
#
# DINOv2 inference is light (~1-2 min for 100 scenes on a 2070), so the array
# is for convenience/consistency with render, not because it's compute-bound.
# The extractor skips scenes whose .npz already exists, so re-runs are safe.
#
# Weights: dinov2.py loads the backbone via torch.hub from ~/.cache/torch/hub,
# which is on NFS-shared HOME — pre-populate it once on the headnode (see
# scripts/setup_extract.sh) so compute nodes never need internet.
#
#SBATCH --partition=IFIall
#SBATCH --gres=gpu:1
#SBATCH --job-name=affdet_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G            # DINOv2 ViT-B/14 weights + activations; generous
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/extract.%A_%a.out
#SBATCH --error=logs/extract.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=max.beckmann@student.uibk.ac.at

set -euo pipefail
mkdir -p logs

REPO="$HOME/affdet"
SCENES="/scratch/$USER/affdet/scenes"
FEATURES="/scratch/$USER/affdet/features"

TASK=${SLURM_ARRAY_TASK_ID:-0}
SCENES_DIR="$SCENES/worker_${TASK}"
FEATURES_DIR="$FEATURES/worker_${TASK}"

if [[ ! -d "$SCENES_DIR" ]]; then
    echo "[extract] ERROR: scenes dir not found: $SCENES_DIR" >&2
    exit 1
fi
mkdir -p "$FEATURES_DIR"

echo "[extract] task=$TASK node=$(hostname) scenes=$SCENES_DIR features=$FEATURES_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true

cd "$REPO"
.venv/bin/python -m affdet.scripts.extract \
    --in "$SCENES_DIR" \
    --out "$FEATURES_DIR"

echo "[extract] task=$TASK done — $(ls "$FEATURES_DIR"/*.npz 2>/dev/null | wc -l) feature files written"
