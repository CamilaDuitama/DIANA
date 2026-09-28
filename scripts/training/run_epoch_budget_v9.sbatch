#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --job-name=diana_epoch_budget
#SBATCH --output=logs/epoch_budget_v9/fit_%a_%j.out
#SBATCH --error=logs/epoch_budget_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X1: the epoch budget for every v9 arm, chosen on the grouped dev folds.
#
# 5 arms x 5 folds. Each task fits one arm on four folds for a fixed 600 epochs
# with no inner validation split and records the eval fold's predictions after
# every epoch. scripts/analysis/37_epoch_budget_select.py then pools the folds,
# picks the budget per arm, and runs the dev-fold A6 comparison. Rules are
# pre-registered in PROJECT.md (section 4, X1) and in the two scripts.
#
# Same cap, criterion and hyperparameters (configs/final_fixed_v9/) for all arms.
# --partition=gpu, one partition only, no --time: see CLAUDE.md.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

ARMS=(multitask community_type feature sample_host material)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
CFG=configs/final_fixed_v9/${ARM}.json
OUT=results/epoch_budget_v9/${ARM}/fold${FOLD}
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }
[ -f "$OUT/preds_by_epoch.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }

echo "=== epoch budget  arm $ARM  fold $FOLD  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" \
    --fold "$FOLD" --out "$OUT" --max-epochs 600
echo "=== COMPLETE — $(date) ==="
