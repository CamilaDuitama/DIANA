#!/bin/bash
#SBATCH --partition=edid_rtx6000,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-24
#SBATCH --job-name=diana_search_v9
#SBATCH --output=logs/search_v9/arm_%a_%j.out
#SBATCH --error=logs/search_v9/arm_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# DIANA v9 — grouped hyperparameter search, multi-task vs single-task
#
# Optuna over the 5 BioProject-disjoint dev folds. Identical budget and identical
# search space for the multi-task arm and every single-task control, so the
# comparison is best-vs-best rather than an artefact of who got tuned.
#
# tau (logit-adjustment strength) is INSIDE the search space. tau=1.0, the
# parameter-free default, over-corrects at this imbalance: dev fold 2 single-task
# `feature` scored 0.009 at tau=1.0 against 0.484 at tau=0.25.
#
#   arm 0   : multi-task, 4 heads
#   arms 1-4: single-task controls, one head each
#
# 25 tasks = 5 configs x 5 outer folds. The 922-run held-out set is NOT touched.
################################################################################
set -e
CONFIGS=(
    "configs/train_config_v9_search_multitask.json"
    "configs/train_config_v9_search_single_community_type.json"
    "configs/train_config_v9_search_single_feature.json"
    "configs/train_config_v9_search_single_sample_host.json"
    "configs/train_config_v9_search_single_material.json"
)
RUN_CONFIG="${CONFIGS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}"
FOLD_ID=$(( SLURM_ARRAY_TASK_ID % 5 ))

echo "=== $RUN_CONFIG  fold $FOLD_ID  $(hostname)  $(date) ==="
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
mkdir -p logs/search_v9
./env/bin/python scripts/training/01_train_multitask_single_fold.py \
    --run-config "$RUN_CONFIG" --fold_id "$FOLD_ID" --use_gpu
echo "=== TASK $SLURM_ARRAY_TASK_ID COMPLETE — $(date) ==="
