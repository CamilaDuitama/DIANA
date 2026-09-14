#!/bin/bash
#SBATCH --job-name=svd_matrices
#SBATCH --partition=edid
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/svd_build/svd_%j.out
#SBATCH --error=logs/svd_build/svd_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Build per-fold rank-192 SVD copies of the fraction and presence/absence matrices
# so DIANA can be trained on them (G5, G6). G1 tried SVD under logistic regression
# only; DIANA has never seen a reduced space.
#
# Rank 192 is chosen because it is DIANA's own trunk width, so this asks directly
# whether an unsupervised 192-dim projection is as good as the supervised one the
# network learns. One SVD per fold, fitted on that fold's four TRAINING folds only.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== per-fold SVD matrices === $(date)"
./env/bin/python scripts/data_prep/13_svd_matrices_per_fold.py \
    --input data/matrices/matrix_v9_train/unitigs.frac.mat --tag frac --ranks 192 --center
./env/bin/python scripts/data_prep/13_svd_matrices_per_fold.py \
    --input data/matrices/matrix_v9_train/unitigs.pa.mat --tag pa --ranks 192 --center
echo "=== done $(date)"
