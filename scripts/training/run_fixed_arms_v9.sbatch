#!/bin/bash
#SBATCH --partition=edid_rtx6000,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --job-name=diana_fixed_v9
#SBATCH --output=logs/fixed_v9/arm_%a_%j.out
#SBATCH --error=logs/fixed_v9/arm_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify

################################################################################
# DIANA v9 — fixed-hyperparameter arms
#
# No Optuna search. Both arms use the identical `fixed_hyperparameters` block, so
# any difference between them is task sharing rather than a luckier draw from the
# search space.
#
#   arm 0  : multi-task, 4 heads
#   arms 1-4: single-task controls, one head each
#
# 25 tasks = 5 configs x 5 outer folds. Fold index is the outer BioProject-grouped
# CV fold within train (2,716 runs); the 922-run held-out set is NOT touched here.
#
# v7's sbatch had --chdir=/pasteur/appa/... , which is not this project root.
#
# Both GPU partitions are listed: edid_rtx6000 alone caps this array at ~2
# concurrent tasks (QOSGrpGRES), so 23 of 25 sat pending. SLURM takes whichever
# frees first, keeping edid_rtx6000 as the preferred one.
#
# USAGE:
#   sbatch scripts/training/run_fixed_arms_v9.sbatch
################################################################################

set -e

CONFIGS=(
    "configs/train_config_v9_fixed_multitask.json"
    "configs/train_config_v9_fixed_single_community_type.json"
    "configs/train_config_v9_fixed_single_feature.json"
    "configs/train_config_v9_fixed_single_sample_host.json"
    "configs/train_config_v9_fixed_single_material.json"
)

CFG_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD_ID=$(( SLURM_ARRAY_TASK_ID % 5 ))
RUN_CONFIG="${CONFIGS[$CFG_IDX]}"

echo "=== DIANA v9 fixed arms ==="
echo "Array task:  $SLURM_ARRAY_TASK_ID  (config $CFG_IDX, fold $FOLD_ID)"
echo "Run config:  $RUN_CONFIG"
echo "Node:        $(hostname)   Start: $(date)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE
export KMP_AFFINITY=disabled
export MPLBACKEND=Agg

PY=./env/bin/python
OUTPUT_DIR=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$RUN_CONFIG")
mkdir -p "$OUTPUT_DIR/cv_results" logs/fixed_v9
echo "Output dir:  $OUTPUT_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY scripts/training/01_train_multitask_single_fold.py \
    --run-config "$RUN_CONFIG" \
    --fold_id    "$FOLD_ID" \
    --use_gpu

echo "=== TASK $SLURM_ARRAY_TASK_ID COMPLETE — $(date) ==="
