#!/bin/bash
# canSlide maps on the IFI cluster (CPU only).  sbatch scripts/slurm_sim_slide.sh "<specs>" OUT_NAME
#SBATCH --partition=IFIall
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/maksym.bek/sim/logs/%x-%j.out
set -e
SPECS="$1"; OUT="$2"
SIM=/scratch/maksym.bek/sim
mkdir -p $SIM/logs $SIM/out/$OUT
[ -d $SIM/out/$OUT/coacd ] || cp -r $SIM/coacd $SIM/out/$OUT/coacd
cd ~/affdet-sim
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/venv-sim/bin/python scripts/sim_slide_maps.py $SIM/picks_all.json $SIM/glbs docs/objaverse/salt_shaker_set.csv $SIM/out/$OUT --specs $SPECS --procs ${SLURM_CPUS_PER_TASK:-8}
