#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-3
#SBATCH --job-name=diana_hash_matrix
#SBATCH --output=logs/hash_matrix_v9/build_%a_%j.out
#SBATCH --error=logs/hash_matrix_v9/build_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# H1: k-mer hash presence matrices from the sourmash sketches (33_hash_matrix_v9.py).
# Four arms: scaled 1,000 at floors 272 (MUSET's -F 0.1 on 2,716 runs, the v9 rule), 28
# (1 %) and 14 (the v11 rule, -N 14); scaled 100 at floor 272 (the width control). The
# 90 % ceiling (-f 0.1) is kept everywhere. Afterwards, afterok: run_baselines_hash_v9.sbatch
# and run_epoch_budget_hash_v9.sbatch.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SCALES=(1000 1000 1000 100)
FLOORS=(272 28 14 272)
S=${SCALES[$SLURM_ARRAY_TASK_ID]}; F=${FLOORS[$SLURM_ARRAY_TASK_ID]}
echo "=== H1 hash matrix, scaled $S floor $F  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/data_prep/33_hash_matrix_v9.py --scaled "$S" --min-samples "$F"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
