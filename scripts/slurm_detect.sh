#!/bin/bash -l
#
# SLURM job — fine-tune the object detector on the converted render dataset.
# Single job (no array): Ultralytics handles the whole train/val loop and
# prints per-class mAP at the end (also saved under $RUNS/<name>/).
#
# Prereqs (headnode, once):
#   1. pip install ultralytics into the repo venv
#   2. convert the scenes:
#        .venv/bin/python -m affdet.scripts.detect convert \
#            --in /scratch/$USER/affdet/scenes/worker_0 \
#            --out /scratch/$USER/affdet/detect_dataset \
#            --curation scripts/omni/curation.json
#      (repeat/merge workers as needed — converter takes any scenes dir)
#   3. pre-download weights so compute nodes never need internet:
#        .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
#      (weights land in the CWD / Ultralytics cache on NFS-shared HOME)
#      For YOLO-World: .venv/bin/python -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8s-worldv2.pt')"
#
# Submit:   sbatch scripts/slurm_detect.sh
# Variant:  MODEL=rtdetr-l.pt sbatch scripts/slurm_detect.sh
# Variant:  MODEL=yolov8s-worldv2.pt sbatch scripts/slurm_detect.sh
#
#SBATCH --partition=IFIall
#SBATCH --gres=gpu:1
#SBATCH --job-name=affdet_detect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/detect.%j.out
#SBATCH --error=logs/detect.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=max.beckmann@student.uibk.ac.at

set -euo pipefail
mkdir -p logs

REPO="$HOME/affdet"
DATASET="/scratch/$USER/affdet/detect_dataset"
RUNS="/scratch/$USER/affdet/detect_runs"

MODEL="${MODEL:-yolo11s.pt}"
EPOCHS="${EPOCHS:-100}"
NAME="${NAME:-${MODEL%.pt}}"

if [[ ! -f "$DATASET/data.yaml" ]]; then
    echo "[detect] ERROR: dataset not converted: $DATASET/data.yaml missing" >&2
    exit 1
fi

echo "[detect] node=$(hostname) model=$MODEL epochs=$EPOCHS dataset=$DATASET"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true

cd "$REPO"
# cuDNN 9.2.0 fails to initialize on this driver with cu124 torch
# (CUDNN_STATUS_NOT_INITIALIZED on the first conv) — disable it, native CUDA
# conv path works fine. See cluster_gpu_torch_fix notes; every GPU job needs this.
.venv/bin/python -c "
import torch; torch.backends.cudnn.enabled = False
import sys
from affdet.scripts.detect import app
sys.argv = ['detect', 'train',
    '--data', '$DATASET/data.yaml',
    '--model', '$MODEL',
    '--epochs', '$EPOCHS',
    '--project', '$RUNS',
    '--name', '$NAME',
]
app()
"

echo "[detect] done — results under $RUNS/$NAME/"
