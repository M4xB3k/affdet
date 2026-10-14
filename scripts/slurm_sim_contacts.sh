#!/bin/bash
# Contact-region labels for every pair map in a directory, on the IFI cluster.
#   sbatch scripts/slurm_sim_contacts.sh PAIR_DIR_UNDER_/scratch/maksym.bek/sim OUT_NAME
#SBATCH --partition=IFIall
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/maksym.bek/sim/logs/%x-%j.out
set -e
PAIRS="$1"; OUT="$2"
SIM=/scratch/maksym.bek/sim
mkdir -p $SIM/logs $SIM/out_contacts/$OUT
cd ~/affdet-sim
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/venv-sim/bin/python -m pip install -q rtree 2>&1 | tail -1
~/venv-sim/bin/python scripts/sim_contact_regions.py $SIM/picks_all.json $SIM/glbs docs/objaverse/salt_shaker_set.csv $SIM/out_contacts/$OUT --pairs $SIM/$PAIRS --coacd $SIM/$PAIRS/coacd
