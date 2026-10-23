#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_screens_v19
#SBATCH --output=logs/screens_v19/fit_%a_%j.out
#SBATCH --error=logs/screens_v19/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 dev-fold screens (PROTOCOLS.md, v19 amendment of 2026-10-21): the final DIANA network
# with one change per arm (configs/screens_v19/arms.tsv from screen_configs_v19.py), seeds
# 42, 1, 2, 3, 4, every dev fold. Submit with --array=0-$((25 * n_arms - 1)).
# Task i: arm = line i / 25 of arms.tsv, seed = SEEDS[(i % 25) / 5], fold = i % 5.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4)
i=$SLURM_ARRAY_TASK_ID
ARMS=${ARMS:-configs/screens_v19/arms.tsv}   # arms_adapted.tsv for the five arms adapted to the shared network
LINE=$(sed -n "$((i / 25 + 1))p" "$ARMS")
[ -n "$LINE" ] || { echo "no arm for task $i" >&2; exit 1; }
IFS=$'\t' read -r ARM TMPL EPOCHS EXTRA <<< "$LINE"
SEED=${SEEDS[$(((i % 25) / 5))]}
FOLD=$((i % 5))
CFG=${TMPL//\{fold\}/$FOLD}
[ -f "$CFG" ] || { echo "missing config $CFG" >&2; exit 1; }
OUT=results/screens_v19/$ARM/seed${SEED}/fold${FOLD}
# SMOKE=1: two epochs into results/_smoke_screens_v19 (wiring check before the real array)
if [ "${SMOKE:-0}" = 1 ]; then EPOCHS=2; OUT=results/_smoke_screens_v19/$ARM/seed${SEED}/fold${FOLD}; fi
[ -e "$OUT/history.json" ] && { echo "$OUT exists; refusing to overwrite" >&2; exit 1; }
echo "=== screen $ARM seed $SEED fold $FOLD, cap $EPOCHS, extra '$EXTRA', config $CFG -> $OUT  $(hostname)  $(date) ==="
# shellcheck disable=SC2086
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --dev-folds data/splits_v19/dev_folds.tsv --max-epochs "$EPOCHS" --seed "$SEED" --save-probs $EXTRA
echo "=== COMPLETE $(date) ==="
