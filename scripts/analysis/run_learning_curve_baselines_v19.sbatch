#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-44
#SBATCH --job-name=lc_base_v19
#SBATCH --output=logs/learning_curve_v19/base_%a_%j.out
#SBATCH --error=logs/learning_curve_v19/base_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Learning curve (PROTOCOLS.md v19 amendment 2026-10-26): logistic regression and random forest at their selected settings
# on the same BioProject subsets as the network fits.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
./env/bin/python scripts/analysis/86_learning_curve_v19.py baselines --job "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
