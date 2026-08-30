#!/bin/bash -l
#
# SLURM job — renders paired-scene VLM-relabeling renders (actor+recipient
# composed together in one real scene instead of two independently
# auto-zoomed tiles). See
# tools/docs/2026-07-11-session-findings-cansupport-detector.md, section 5.
#
# Two modes:
#   Single job (small batches, e.g. the pilot): sbatch scripts/omni/slurm_pair_render.sh
#   Array job (large batches): sbatch --array=0-16 scripts/omni/slurm_pair_render.sh
#     Each array task N reads PAIRS_DIR/pairs_<N>.json (a pre-sharded slice)
#     instead of the single PAIRS file, so N workers render in parallel with
#     no write races (each writes into its own pairs -- output dirs are
#     already keyed by <actor>__<recipient>, safe to share OUT across tasks).
#
# Override grid/pairs/out at submit time, e.g.:
#   sbatch --array=0-16 --export=ALL,GRID=3x2,PAIRS_DIR=data/labels/pair_pilot/shards \
#       scripts/omni/slurm_pair_render.sh
#
#SBATCH --partition=IFIall
#SBATCH --gres=gpu:1
#SBATCH --job-name=affdet_pair_render
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-11:00:00
#SBATCH --output=logs/pair_render.%A_%a.out
#SBATCH --error=logs/pair_render.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maksym.bek@student.uibk.ac.at

set -euo pipefail
mkdir -p logs

REPO="$HOME/affdet"
BLENDER="$HOME/blender-5.1/blender"
OMNI_ROOT="/scratch/$USER/omni/raw_scans"
TEX_DIR="/scratch/$USER/affdet/table_textures"

OUT="${OUT:-/scratch/$USER/affdet/pair_pilot_renders}"
GRID="${GRID:-2x2}"
RESOLVED="${RESOLVED:-/scratch/$USER/affdet/resolved_instances_v2.json}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    PAIRS_DIR="${PAIRS_DIR:-data/labels/pair_pilot/shards}"
    PAIRS=$(printf "%s/pairs_%02d.json" "$PAIRS_DIR" "$SLURM_ARRAY_TASK_ID")
else
    PAIRS="${PAIRS:-data/labels/pair_pilot/pairs.json}"
fi

mkdir -p "$OUT"

echo "[pair_render] task=${SLURM_ARRAY_TASK_ID:-single} pairs=$PAIRS grid=$GRID out=$OUT resolved=$RESOLVED node=$(hostname)"

cd "$REPO"
"$BLENDER" --background --python affdet/render/omni_pair_render.py -- \
    --pairs "$PAIRS" \
    --resolved "$RESOLVED" \
    --omni-root "$OMNI_ROOT" \
    --out "$OUT" \
    --tex-dir "$TEX_DIR" \
    --overrides scripts/omni/scale_overrides.json \
    --grid "$GRID"

echo "[pair_render] task=${SLURM_ARRAY_TASK_ID:-single} done — $(ls -d "$OUT"/*/ 2>/dev/null | wc -l) pairs total in $OUT"
