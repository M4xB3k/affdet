#!/bin/bash
# Grasp maps on the IFI cluster (CPU only).
#   sbatch scripts/slurm_sim_grasp.sh ROLE "INDICES" STEP OUT_NAME
# Example: sbatch scripts/slurm_sim_grasp.sh frying_pan "0 1" 0.02 grasp_pans
#SBATCH --partition=IFIall
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/maksym.bek/sim/logs/%x-%j.out
set -e
ROLE="$1"; IDX="$2"; STEP="$3"; OUT="$4"
SIM=/scratch/maksym.bek/sim
mkdir -p $SIM/logs $SIM/out/$OUT
[ -d $SIM/out/$OUT/coacd ] || cp -r $SIM/coacd $SIM/out/$OUT/coacd
cd ~/affdet-sim
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/venv-sim/bin/python scripts/sim_grasp_maps.py $SIM/picks_all.json $SIM/glbs $SIM/out/$OUT --role $ROLE --index $IDX --step $STEP --procs ${SLURM_CPUS_PER_TASK:-8}
