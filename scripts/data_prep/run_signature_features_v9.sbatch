#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --array=0-4
#SBATCH --job-name=diana_s88_sig
#SBATCH --output=logs/sketches_v9/sig_fold%a_%j.out
#SBATCH --error=logs/sketches_v9/sig_fold%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.8 / B1: per-fold class-signature features (scripts/data_prep/31_signature_features_v9.py).
# One array task per dev fold; each reads the scaled=100 training sketches (~25 min),
# builds the signatures from training-fold runs only and writes sig_fold<k>.npz.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== S8.8 signatures fold $SLURM_ARRAY_TASK_ID  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/data_prep/31_signature_features_v9.py --fold "$SLURM_ARRAY_TASK_ID"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
