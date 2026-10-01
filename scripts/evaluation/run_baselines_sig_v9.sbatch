#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-47
#SBATCH --job-name=diana_sig_baselines
#SBATCH --output=logs/baselines_sig_v9/task_%a_%j.out
#SBATCH --error=logs/baselines_sig_v9/task_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.8 / B2: the tuned baselines on fractions + per-fold class-signature columns
# (`sig`) against the same models on fractions alone, plus the signature columns alone
# (`sigonly`, a diagnostic), all with the FULL v9 grid
# (RandomForest incl. max_features=0.05: the width is 110,240, not 2 M). 3 reps x 4
# tasks x 4 models = 48 tasks. Submit --dependency=afterok:<S8.8 B1 array>.
# Afterwards: 14_tune_baselines_s8a_v9.py --pool --out results/baselines_sig_v9 --rep-a sig --rep-b fraction
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
REPS=(fraction sig sigonly)
TASKS=(community_type feature sample_host material)
MODELS=(LogisticRegression_Bal LinearSVM_Bal RandomForest_Bal kNN)
i=$SLURM_ARRAY_TASK_ID
REP=${REPS[$((i / 16))]}; TASK=${TASKS[$(((i % 16) / 4))]}; MODEL=${MODELS[$((i % 4))]}
for k in 0 1 2 3 4; do [ -s "results/s8_signatures_v9/sig_fold$k.npz" ] || { echo "sig_fold$k.npz missing" >&2; exit 1; }; done
echo "=== S8.8 B2  $REP / $TASK / $MODEL  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/14_tune_baselines_s8a_v9.py --rep "$REP" --task "$TASK" --model "$MODEL" \
    --n-jobs "$SLURM_CPUS_PER_TASK" --out results/baselines_sig_v9 --rf-grid full
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
