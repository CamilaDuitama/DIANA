#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=depth_tuned_v19
#SBATCH --output=logs/depth_tuned_v19_%j.out
#SBATCH --error=logs/depth_tuned_v19_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Depth-only logistic regression and random forest, tuned like the fraction baselines (81_depth_tuned_v19.py).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
./env/bin/python scripts/analysis/81_depth_tuned_v19.py
