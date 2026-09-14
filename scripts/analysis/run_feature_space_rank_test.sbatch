#!/bin/bash
#SBATCH --job-name=g1_rank_test
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/feature_space/rank_%j.out
#SBATCH --error=logs/feature_space/rank_%j.err

################################################################################
# G1a + G1b: is the v9 feature space low-rank, and does reduction help?
#
# Runs scripts/analysis/19_feature_space_rank_test.py, which compares logistic
# regression on the full 110,202 unitig fractions against truncated SVD at
# 64 / 192 / 1000 components and supervised top-5,000 / top-20,000 by ANOVA F.
#
# Dev folds only. Every transform is fit on the four training folds and applied
# to the held-back fold, so nothing crosses the fold boundary and the held-out
# set is never touched. C is fixed at 1.0 across arms: this compares feature
# spaces, not tuning.
#
# Memory: X is 2,716 x 110,202 float32 (1.2 GB) and StandardScaler plus the
# rank-1000 SVD hold several copies, hence 160 GB.
################################################################################

set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export KMP_INIT_AT_FORK=FALSE
export KMP_AFFINITY=disabled
export MPLBACKEND=Agg

echo "=== G1 rank test === node=$(hostname) start=$(date)"
./env/bin/python scripts/analysis/19_feature_space_rank_test.py
echo "=== end=$(date)"
