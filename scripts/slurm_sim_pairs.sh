#!/bin/bash
# Pair support maps on the IFI cluster (CPU only). One job = one node's worth of cores.
#   sbatch --cpus-per-task=32 scripts/slurm_sim_pairs.sh "<tops>" "<bases>" OUT_NAME [extra batch args]
# Example: sbatch scripts/slurm_sim_pairs.sh "shaker:d00ef026 mug:0" "bowl:0 tray:0" run1 --adaptive 2
#SBATCH --partition=IFIall
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/maksym.bek/sim/logs/%x-%j.out
set -e
TOPS="$1"; BASES="$2"; OUT="$3"; shift 3
SIM=/scratch/maksym.bek/sim
mkdir -p $SIM/logs $SIM/out/$OUT
[ -d $SIM/out/$OUT/coacd ] || cp -r $SIM/coacd $SIM/out/$OUT/coacd
cd ~/affdet-sim
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/venv-sim/bin/python scripts/sim_pair_batch.py $SIM/picks_all.json $SIM/glbs docs/objaverse/salt_shaker_set.csv $SIM/out/$OUT \
    --tops $TOPS --bases $BASES --procs ${SLURM_CPUS_PER_TASK:-8} "$@"
