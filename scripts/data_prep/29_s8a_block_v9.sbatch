#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_s8a_block
#SBATCH --output=logs/sketches_v9/s8a_block_%j.out
#SBATCH --error=logs/sketches_v9/s8a_block_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.2: append the K = 1,984,422 kept off-list hashes as 0/1 columns to the
# train and held-out fraction matrices (scripts/data_prep/28_append_offlist_hash_columns.py).
# Writes data/matrices/matrix_v9_{train,heldout}/unitigs.frac.s8a.npz; refuses
# to overwrite. The block is not standardised.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== S8.2  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/data_prep/28_append_offlist_hash_columns.py
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
