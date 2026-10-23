#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-29
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_budget_v17
#SBATCH --output=logs/epoch_budget_v17/fit_%a_%j.out
#SBATCH --error=logs/epoch_budget_v17/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# v17 budget curves (PROTOCOLS.md v17): 600-epoch dev-fold curves at the searched
# hyperparameters. 6 arms x 5 folds: the four tasks, the multitask control
# (v9 architecture carried, for the single-vs-multi re-test), and the
# equal-study-weight feature arm. Budgets are then selected by the frozen rule.
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(community_type feature sample_host material multitask feature_sw)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
EXTRA=""
if [ "$ARM" = "feature_sw" ]; then
    CFG=configs/step4_2_v17/feature.json
    EXTRA="--study-weights"
    OUT=results/epoch_budget_v17_sw/feature/fold${FOLD}
else
    CFG=configs/final_fixed_v17/${ARM}.json
    OUT=results/epoch_budget_v17/${ARM}/fold${FOLD}
fi
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 1; }
[ -f "$OUT/preds_by_epoch.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== v17 budget curve  arm $ARM  fold $FOLD  $(hostname)  $(date) ==="
# shellcheck disable=SC2086
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" \
    --fold "$FOLD" --out "$OUT" --max-epochs 600 \
    --dev-folds data/splits_v17/dev_folds.tsv $EXTRA
echo "=== COMPLETE — $(date) ==="
