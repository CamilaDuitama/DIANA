#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --job-name=diana_sig_budget
#SBATCH --output=logs/epoch_budget_sig_v9/fit_%a_%j.out
#SBATCH --error=logs/epoch_budget_sig_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.8 / B3: DIANA's five arms on fractions + per-fold class-signature columns, the
# X1 epoch-budget recipe (04_epoch_budget_dev_folds.py, cap 600, same guard in 37_).
# Hyperparameters are those of configs/final_fixed_v9/ unchanged: the input grows by
# 38 columns in [0, 1] (0.03 % of the width), unlike S6's 3,072 standardised columns,
# so the searched settings apply; this is the one stated inequality. One config per
# (arm, fold) because the block is fold-specific: configs/sig_v9/<arm>_fold<k>.json.
# Submit --dependency=afterok:<S8.8 B1 array>. --partition=gpu, one partition, no --time.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(multitask community_type feature sample_host material)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
CFG=configs/sig_v9/${ARM}_fold${FOLD}.json
OUT=results/epoch_budget_sig_v9/${ARM}/fold${FOLD}
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }
[ -s "results/s8_signatures_v9/sig_fold${FOLD}.npz" ] || { echo "ERROR: sig_fold${FOLD}.npz missing" >&2; exit 1; }
[ -f "$OUT/preds_by_epoch.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== S8.8 B3  arm $ARM  fold $FOLD  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" --max-epochs 600
echo "=== COMPLETE — $(date) ==="
