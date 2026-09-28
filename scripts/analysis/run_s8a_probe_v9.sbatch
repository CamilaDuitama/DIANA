#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --job-name=diana_s8a_probe
#SBATCH --output=logs/sketches_v9/s8a_probe_%j.out
#SBATCH --error=logs/sketches_v9/s8a_probe_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.2b: linear probe of the off-list block on the grouped dev folds
# (scripts/analysis/43_offlist_block_probe.py). Submit --dependency=afterok:<S8.2>.
# Memory: the npz expands to 2,716 x 2.09 M float32 (22.8 GB) plus a sparse copy
# of the block; 160 GB leaves room for the solver.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
[ -s data/matrices/matrix_v9_train/unitigs.frac.s8a.npz ] || { echo "S8.2 output missing" >&2; exit 1; }
echo "=== S8.2b  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/analysis/43_offlist_block_probe.py
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
