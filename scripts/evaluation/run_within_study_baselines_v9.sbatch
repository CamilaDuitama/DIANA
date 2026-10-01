#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-15
#SBATCH --job-name=diana_within_baselines
#SBATCH --output=logs/within_study_v9/base_%a_%j.out
#SBATCH --error=logs/within_study_v9/base_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# W2: the tuned baselines at their read-9 configurations on the five within-study
# (sample-grouped) folds of the training set (16_within_study_baselines_v9.py). No grid.
# 4 tasks x 4 models = 16 tasks. Requires results/within_study_v9/folds.tsv (W1).
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
MODELS=(LogisticRegression_Bal LinearSVM_Bal RandomForest_Bal kNN)
i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$((i / 4))]}; MODEL=${MODELS[$((i % 4))]}
[ -s results/within_study_v9/folds.tsv ] || { echo "folds.tsv missing (W1)" >&2; exit 1; }
echo "=== W2  $TASK / $MODEL  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/16_within_study_baselines_v9.py --task "$TASK" --model "$MODEL" --n-jobs "$SLURM_CPUS_PER_TASK"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
