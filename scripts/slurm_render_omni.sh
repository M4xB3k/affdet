#!/bin/bash -l
#
# SLURM array job — renders the coverage-balanced OmniObject3D scene set across
# 10 parallel workers. Each task renders 100 scenes with a unique seed into its
# own output subdir (worker_N) so there are no write races. Each worker runs the
# coverage-aware sampler independently, so every worker's 100 scenes are already
# balanced across all affordances (incl. rare canPeel).
#
# Submit:   sbatch --array=0-9 scripts/slurm_render_omni.sh   (10 workers x 100 = 1000)
# Test:     sbatch --array=0   scripts/slurm_render_omni.sh   (single worker, 100 scenes)
# Big+kind: sbatch --array=0-79%16 --exclude=gc16 \
#             --export=ALL,OUT_ROOT=/scratch/$USER/omni_scenes_v3 \
#             scripts/slurm_render_omni.sh   (8000 scenes, <=16 workers at once)
# The %N in --array=...%N caps simultaneously-running workers so the job never
# hogs the shared cluster during work hours; the rest queue and drain as slots free.
#
# Partition IFIall spans all GPU nodes; omni_blender picks OPTIX>CUDA per node.
# Render is GPU-bound so --cpus-per-task=2 lets more workers schedule on the
# CPU-starved cluster. Meshes are large (food/container scans 150-300MB) but only
# a scene's worth are loaded at once; measured peak well under the mem ask.
#
#SBATCH --partition=IFIall
#SBATCH --gres=gpu:1
#SBATCH --job-name=affdet_omni_render
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/omni_render.%A_%a.out
#SBATCH --error=logs/omni_render.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maksym.bek@student.uibk.ac.at

set -euo pipefail
mkdir -p logs

REPO="$HOME/affdet"
BLENDER="$HOME/blender-5.1/blender"
OMNI_ROOT="/scratch/$USER/omni/raw_scans"
TEX_DIR="/scratch/$USER/affdet/table_textures"
# OUT_ROOT can be overridden at submit time, e.g.
#   sbatch --export=ALL,OUT_ROOT=/scratch/$USER/omni_scenes_v3 ...
# so a fresh batch never clobbers an existing scene set.
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/omni_scenes_v2}"

# Overridable at submit time: a small SCENES_PER_JOB spread over many array
# tasks finishes far sooner than one worker doing the whole batch, since the
# tasks run concurrently. Coverage balancing is per-worker, so going too small
# leaves the sampler little room to place rare affordances (canPeel) -- 10 is
# about the floor.
SCENES_PER_JOB="${SCENES_PER_JOB:-100}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((100 + TASK))          # unique per worker so coverage picks differ
TASK_OUT="$OUT_ROOT/worker_${TASK}"
mkdir -p "$TASK_OUT"

echo "[omni_render] task=$TASK seed=$SEED n=$SCENES_PER_JOB out=$TASK_OUT node=$(hostname)"

cd "$REPO"
"$BLENDER" --background --python affdet/render/omni_blender.py -- \
    --curation scripts/omni/curation.json \
    --omni-root "$OMNI_ROOT" \
    --out "$TASK_OUT" \
    --tex-dir "$TEX_DIR" \
    --overrides scripts/omni/scale_overrides.json \
    --labels data/omni/labels/omni_vlm_raw.jsonl \
    --n-scenes "$SCENES_PER_JOB" \
    --seed "$SEED"

echo "[omni_render] task=$TASK done — $(ls -d "$TASK_OUT"/scene_* 2>/dev/null | wc -l) scenes"
