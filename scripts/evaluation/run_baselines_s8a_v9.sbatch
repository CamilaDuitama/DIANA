#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-31
#SBATCH --job-name=diana_s8a_baselines
#SBATCH --output=logs/baselines_s8a_v9/task_%a_%j.out
#SBATCH --error=logs/baselines_s8a_v9/task_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.3a: 2 representations x 4 tasks x 4 models = 32 array tasks, each running the
# model's grid over the 5 grouped dev folds (scripts/evaluation/14_tune_baselines_s8a_v9.py).
# Afterwards: ./env/bin/python scripts/evaluation/14_tune_baselines_s8a_v9.py --pool
# Same grid and selection rule as the v9 baseline tuning; RandomForest limited to
# max_features="sqrt" on BOTH representations. No --time (edid has no limit).
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
REPS=(fraction s8a)
TASKS=(community_type feature sample_host material)
MODELS=(LogisticRegression_Bal LinearSVM_Bal RandomForest_Bal kNN)
i=$SLURM_ARRAY_TASK_ID
REP=${REPS[$((i / 16))]}
TASK=${TASKS[$(((i % 16) / 4))]}
MODEL=${MODELS[$((i % 4))]}
echo "=== S8.3a  $REP / $TASK / $MODEL  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/14_tune_baselines_s8a_v9.py --rep "$REP" --task "$TASK" --model "$MODEL" --n-jobs "$SLURM_CPUS_PER_TASK"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
