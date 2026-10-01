#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-19
#SBATCH --job-name=diana_step4_1_init
#SBATCH --output=logs/phase4_v9/linear_init_%a_%j.out
#SBATCH --error=logs/phase4_v9/linear_init_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Phase 4 step 4.1: logistic-regression initialisation per task x dev fold (19_logreg_init_v9.py).
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled
TASKS=(community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID; TASK=${TASKS[$((i / 5))]}; FOLD=$((i % 5))
echo "=== step 4.1 init $TASK fold $FOLD  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/19_logreg_init_v9.py --task "$TASK" --fold "$FOLD" --n-jobs "$SLURM_CPUS_PER_TASK"
rc=$?; echo "=== exit $rc — $(date) ==="; exit $rc
