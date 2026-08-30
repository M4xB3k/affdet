#!/bin/bash -l
#
# SLURM array job — renders 1000 scenes across 10 parallel workers.
# Each task renders 100 scenes with a unique seed and non-overlapping
# scene indices (task 0 → scene_0001–0100, task 1 → scene_0101–0200, …)
#
# Submit:   sbatch --array=0-9 scripts/slurm_render.sh   (10 workers × 100 = 1000)
# Single:   sbatch --array=0   scripts/slurm_render.sh   (100 scenes, test)
#
# Partition IFIall spans all GPU nodes: SLURM uses whatever has a free GPU +
# enough CPUs (gc18 4090, gc19 L40S, gc1-16 2070 as their CPUs free up).
# The cluster is CPU-starved, so --cpus-per-task=2 (render is GPU-bound) lets
# more workers schedule. blender.py picks OPTIX>CUDA per node, so mixed
# hardware is fine — correctness is identical, only timing varies.
#
#SBATCH --partition=IFIall
#SBATCH --gres=gpu:1
#SBATCH --job-name=affdet_render
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G            # measured peak RSS ~1.6G; small ask fits RAM-tight nodes (e.g. gc18)
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/render.%A_%a.out
#SBATCH --error=logs/render.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=max.beckmann@student.uibk.ac.at

set -euo pipefail
mkdir -p logs

# --- paths (adjust if your scratch layout differs) --------------------------
REPO="$HOME/affdet"
BLENDER="$HOME/blender-5.1/blender"   # adjust to actual install path
OUT="/scratch/$USER/affdet/scenes"
MESH_DIR="/scratch/$USER/affdet/meshes"
# ----------------------------------------------------------------------------

SCENES_PER_JOB=100
TASK=${SLURM_ARRAY_TASK_ID:-0}

# Non-overlapping start index: task 0 writes scene_0001-0100, task 1 → 0101-0200 …
# blender.py auto-detects the highest existing scene and increments from there,
# so we just need unique output subdirs per task to avoid races.
TASK_OUT="$OUT/worker_${TASK}"
mkdir -p "$TASK_OUT"

SEED=$((42 + TASK))

echo "[render] task=$TASK seed=$SEED n=$SCENES_PER_JOB out=$TASK_OUT"

cd "$REPO"
"$BLENDER" --background --python affdet/render/blender.py -- \
    --out "$TASK_OUT" \
    --n-scenes "$SCENES_PER_JOB" \
    --seed "$SEED" \
    --mesh-dir "$MESH_DIR"

echo "[render] task=$TASK done — $(ls "$TASK_OUT" | grep -c scene_) scenes written"
