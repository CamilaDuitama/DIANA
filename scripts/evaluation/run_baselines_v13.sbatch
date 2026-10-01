#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-31
#SBATCH --job-name=diana_v13_baselines
#SBATCH --output=logs/baselines_v13/task_%a_%j.out
#SBATCH --error=logs/baselines_v13/task_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K2: the tuned baselines on the v13 dev folds (class set: >= 2 training BioProjects,
# data/splits_v13/, PROJECT.md section 8, K protocol). Fractions, and fractions + the
# 38 S8.8 signature columns; full v9 grid on both; same selection rule.
# 2 reps x 4 tasks x 4 models = 32 tasks. Afterwards, afterok: run_baselines_pool_v13.sbatch.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
REPS=(fraction sig)
TASKS=(community_type feature sample_host material)
MODELS=(LogisticRegression_Bal LinearSVM_Bal RandomForest_Bal kNN)
i=$SLURM_ARRAY_TASK_ID
REP=${REPS[$((i / 16))]}; TASK=${TASKS[$(((i % 16) / 4))]}; MODEL=${MODELS[$((i % 4))]}
[ -s data/splits_v13/class_eligibility.tsv ] || { echo "data/splits_v13 missing: run K1 first" >&2; exit 1; }
for k in 0 1 2 3 4; do [ -s "results/s8_signatures_v9/sig_fold$k.npz" ] || { echo "sig_fold$k.npz missing" >&2; exit 1; }; done
echo "=== K2  $REP / $TASK / $MODEL  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/14_tune_baselines_s8a_v9.py --splits data/splits_v13 \
    --rep "$REP" --task "$TASK" --model "$MODEL" \
    --n-jobs "$SLURM_CPUS_PER_TASK" --out results/baselines_v13 --rf-grid full
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
