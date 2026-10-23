#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_rescore_v19
#SBATCH --output=logs/search_v19/rescore_%a_%j.out
#SBATCH --error=logs/search_v19/rescore_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 Phase 2 (C4): each of the five best trials per arm refitted with five seeds on
# the five dev folds; seed-averaged probabilities per epoch, same selection rule.
# Task i: arm = i / 5, rank = i % 5. Optional CAP=<n> for the doubled-cap rerun.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(community_type feature sample_host material multitask)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}
RANK=$((SLURM_ARRAY_TASK_ID % 5))
EXTRA=()
[ -n "${CAP:-}" ] && EXTRA=(--cap "$CAP")
echo "=== v19 rescore $ARM rank $RANK ${CAP:+cap $CAP}  $(hostname)  $(date) ==="
./env/bin/python scripts/training/31_rescore_top5_v19.py --arm "$ARM" --rank "$RANK" \
    --search-dir "results/search_v19/$ARM" --split-dir data/splits_v19 \
    --matrix data/matrices/matrix_v17_train/unitigs.frac.mat "${EXTRA[@]}"
echo "=== COMPLETE $(date) ==="
