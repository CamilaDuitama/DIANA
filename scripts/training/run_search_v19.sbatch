#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_search_v19
#SBATCH --output=logs/search_v19/search_%a_%j.out
#SBATCH --error=logs/search_v19/search_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 Phase 2 (PROTOCOLS.md v19, C3): 100-trial search per arm, every trial fitted
# on the five dev folds to a 400-epoch cap with fold-absent classes masked, scored on
# the pooled out-of-fold curve. Resumes from results/search_v19/<arm>/study.db.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(community_type feature sample_host material multitask)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "=== v19 search $ARM  $(hostname)  $(date) ==="
./env/bin/python scripts/training/30_search_v19.py --arm "$ARM" --split-dir data/splits_v19 \
    --matrix data/matrices/matrix_v17_train/unitigs.frac.mat --out "results/search_v19/$ARM"
echo "=== COMPLETE $(date) ==="
