#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-7
#SBATCH --job-name=attnpool_final
#SBATCH --output=logs/attnpool/%x_%a_%j.out
#SBATCH --error=logs/attnpool/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# The TRAIN result for S5, which the dev-fold screen cannot give.
#
# The screen's numbers come from 5 fold models per task, each trained on 4/5 of train.
# A train result means ONE final model per task fitted on all 2,716 training runs and
# scored on them, the same thing `results/final_eval_v9/train_multitask` is for the
# published model.
#
# Tasks 0-3: fit the attention-pool model on all of train, then score it on train.
# Tasks 4-7: score the EXISTING frozen v9 single-task baselines on train. Only the
#            multi-task model had a train evaluation, so the single-task comparison
#            numbers do not exist yet. Nothing is refitted and no published number
#            changes; this only adds `results/final_eval_v9/train_single_<task>/`.
#
# Held-out is NOT touched here.
#
# `gpu`, not `edid_rtx6000`: sm_120 against a cu118 torch. No --time.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

TASKS=(community_type feature sample_host material)
TRAIN_IDS=data/splits_v9/train_accessions.txt
mkdir -p logs/attnpool

if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
  T=${TASKS[$SLURM_ARRAY_TASK_ID]}
  CFG="results/attnpool_final_v9/configs/attnpool_final_${T}.json"
  echo "=== S5 final fit on all of train: $T  $(date) ==="
  ./env/bin/python scripts/training/02_train_final_model.py "$CFG"
  ./env/bin/diana-test \
    --model "results/attnpool_final_v9/fit_${T}/best_model.pth" \
    --config "$CFG" \
    --matrix data/matrices/matrix_v9_train/unitigs.frac.mat \
    --metadata data/splits_v9/train_metadata.tsv \
    --eligibility data/splits_v9/class_eligibility.tsv \
    --test-ids "$TRAIN_IDS" \
    --output "results/attnpool_final_v9/train_${T}" --device cuda
else
  T=${TASKS[$(( SLURM_ARRAY_TASK_ID - 4 ))]}
  CFG="results/search_v9_final_single_${T}/final_training_config.json"
  echo "=== baseline train evaluation (existing frozen model): $T  $(date) ==="
  ./env/bin/diana-test \
    --model "results/search_v9_final_single_${T}/final_model/best_model.pth" \
    --config "$CFG" \
    --matrix data/matrices/matrix_v9_train/unitigs.frac.mat \
    --metadata data/splits_v9/train_metadata.tsv \
    --eligibility data/splits_v9/class_eligibility.tsv \
    --test-ids "$TRAIN_IDS" \
    --output "results/final_eval_v9/train_single_${T}" --device cuda
fi
echo "=== COMPLETE — $(date) ==="
