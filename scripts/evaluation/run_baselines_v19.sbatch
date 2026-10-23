#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-19
#SBATCH --job-name=diana_base_v19
#SBATCH --output=logs/baselines_v19/base_%a_%j.out
#SBATCH --error=logs/baselines_v19/base_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 Phase 1 (PROTOCOLS.md v19, C2e): every baseline tuned on the dev folds with its
# pre-registered grid, selected on the shared score; out-of-fold output of the selection.
# Task i: task = i / 5, model = i % 5.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
MODELS=(LogisticRegression_Bal LinearSVM_Bal RandomForest_Bal kNN MajorityClass)
TASK=${TASKS[$((SLURM_ARRAY_TASK_ID / 5))]}
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 5))]}
echo "=== v19 baseline $TASK / $MODEL  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/40_baselines_v19.py --task "$TASK" --model "$MODEL" \
    --split-dir data/splits_v19 --matrix data/matrices/matrix_v17_train/unitigs.frac.mat \
    --out results/baselines_v19 --n-jobs "$SLURM_CPUS_PER_TASK"
echo "=== COMPLETE $(date) ==="
