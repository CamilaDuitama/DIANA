#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-11
#SBATCH --job-name=diana_devfold_probs
#SBATCH --output=logs/devfold_probs_v9/task_%a_%j.out
#SBATCH --error=logs/devfold_probs_v9/task_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X9.0c: out-of-fold probabilities of the tuned baselines on the dev folds, selected
# configurations, no grid (17_devfold_baseline_probs_v9.py). 3 models x 4 tasks.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
MODELS=(LogisticRegression_Bal RandomForest_Bal kNN)
TASKS=(community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$((i / 4))]}; TASK=${TASKS[$((i % 4))]}
echo "=== X9.0c  $MODEL / $TASK  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/17_devfold_baseline_probs_v9.py --task "$TASK" --model "$MODEL" --n-jobs "$SLURM_CPUS_PER_TASK"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
