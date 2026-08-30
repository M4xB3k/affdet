#!/usr/bin/env bash
# Label one or more workers' scenes with the VLM labeler.
#
# Usage:
#   bash scripts/vlm_label.sh <worker_id> [start_scene_id]   # single worker (resumable)
#   bash scripts/vlm_label.sh <A-B>                          # inclusive range, sequential
#   bash scripts/vlm_label.sh <id> <id> <id> ...             # explicit list, sequential
#
# Examples:
#   bash scripts/vlm_label.sh 0                 # just worker_0
#   bash scripts/vlm_label.sh 0 scene_0009      # resume worker_0 from scene_0009
#   bash scripts/vlm_label.sh 2-5               # worker_2, _3, _4, _5 in order
#   bash scripts/vlm_label.sh 2 4 7             # worker_2, _4, _7 in order
#
# Output goes to /scratch/$USER/affdet/labels/worker_<id>/
# so workers never overwrite each other (each starts scene count from 0001).
# Already-labeled scenes are skipped (safe to re-run / resume).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <worker_id> [start_scene_id] | <A-B> | <id> <id> ..." >&2
    exit 2
fi

SCRATCH="/scratch/$USER/affdet"
REPO="$HOME/affdet"

# --- resolve the list of workers + optional start_from -----------------------
# Three call shapes:
#   "2-5"                 -> range 2 3 4 5,           START_FROM=""
#   "0" "scene_0009"      -> single worker 0,         START_FROM=scene_0009
#   "2" "4" "7"           -> list 2 4 7,              START_FROM=""
WORKERS=()
START_FROM=""

if [[ "$1" == *-* && $# -eq 1 ]]; then
    # inclusive range A-B
    LO=${1%-*}
    HI=${1#*-}
    if ! [[ "$LO" =~ ^[0-9]+$ && "$HI" =~ ^[0-9]+$ ]]; then
        echo "ERROR: bad range '$1' (expected digits like 2-5)" >&2
        exit 2
    fi
    if (( LO > HI )); then
        echo "ERROR: range start $LO > end $HI" >&2
        exit 2
    fi
    for ((w = LO; w <= HI; w++)); do WORKERS+=("$w"); done
elif [[ $# -eq 2 && "$2" == scene_* ]]; then
    # single worker + resume point
    WORKERS=("$1")
    START_FROM="$2"
else
    # explicit list of worker ids
    for arg in "$@"; do
        if ! [[ "$arg" =~ ^[0-9]+$ ]]; then
            echo "ERROR: '$arg' is not a worker id (digits) — for a single resumable worker use: $0 <id> scene_XXXX" >&2
            exit 2
        fi
        WORKERS+=("$arg")
    done
fi

echo "[vlm_label] workers: ${WORKERS[*]}${START_FROM:+  (start_from=$START_FROM)}"

cd "$REPO"

# --- label each worker, fail-soft so one bad worker doesn't abort the rest ----
FAILED=()
for WORKER in "${WORKERS[@]}"; do
    SCENES_DIR="$SCRATCH/scenes/worker_${WORKER}"
    LABELS_DIR="$SCRATCH/labels/worker_${WORKER}"

    echo
    echo "==================== worker_${WORKER} ===================="
    if [[ ! -d "$SCENES_DIR" ]]; then
        echo "[vlm_label] SKIP worker_${WORKER}: scenes dir not found: $SCENES_DIR" >&2
        FAILED+=("$WORKER")
        continue
    fi
    mkdir -p "$LABELS_DIR"
    echo "[vlm_label] scenes=$SCENES_DIR  labels=$LABELS_DIR  start_from=${START_FROM:-<beginning>}"

    EXTRA_ARGS=()
    [[ -n "$START_FROM" ]] && EXTRA_ARGS=(--start-from "$START_FROM")

    if AFFDET_VLM_DEBUG=1 .venv/bin/python -m affdet.scripts.label \
        --in "$SCENES_DIR" \
        --out "$LABELS_DIR" \
        "${EXTRA_ARGS[@]}"; then
        echo "[vlm_label] worker_${WORKER} done — labels in $LABELS_DIR"
    else
        rc=$?
        echo "[vlm_label] worker_${WORKER} FAILED (exit $rc)" >&2
        FAILED+=("$WORKER")
    fi
done

echo
if (( ${#FAILED[@]} )); then
    echo "[vlm_label] finished with failures: workers ${FAILED[*]}" >&2
    exit 1
fi
echo "[vlm_label] all workers done: ${WORKERS[*]}"
