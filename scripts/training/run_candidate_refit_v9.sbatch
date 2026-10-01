#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-99
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_candidate
#SBATCH --output=logs/candidates_v9/%x_%a_%j.out
#SBATCH --error=logs/candidates_v9/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X9 / X10: one candidate recipe, run exactly like the frozen reference (per-arm lr,
# schedule and epoch budget from results/reference_v9/epoch_budget.tsv; 5 seeds; probabilities
# saved) with ONE thing changed, given through environment variables:
#   CFG_DIR   configs for the four arms (<arm>.json), e.g. configs/x9b_v9
#   OUT_DIR   results directory, e.g. results/x9b_v9
#   EXTRA     optional extra arguments for 04_epoch_budget_dev_folds.py (e.g. --offsets-dir ...)
# Submit:  sbatch --export=ALL,CFG_DIR=...,OUT_DIR=...,EXTRA="..." --job-name=diana_x9b scripts/training/run_candidate_refit_v9.sbatch
# Afterwards: 54_seed_ensemble.py --base $OUT_DIR --budgets results/reference_v9/epoch_budget.tsv --out ${OUT_DIR}_oof
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
: "${CFG_DIR:?set CFG_DIR}"; : "${OUT_DIR:?set OUT_DIR}"
SEEDS=(42 1 2 3 4)
TASKS=(community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((i / 20))]}; rem=$((i % 20)); TASK=${TASKS[$((rem / 5))]}; FOLD=$((rem % 5))
read -r BUDGET LR SCHED HORIZON < <(awk -F'\t' -v a="$TASK" 'NR>1 && $1==a {print $2, $3, $4, ($5==""?"-":$5)}' results/reference_v9/epoch_budget.tsv)
[ -n "$BUDGET" ] || { echo "no reference settings for $TASK" >&2; exit 1; }
ARGS=(--lr "$LR" --lr-schedule "$SCHED"); [ "$HORIZON" != "-" ] && ARGS+=(--lr-horizon "$HORIZON")
OUT=${OUT_DIR}/${TASK}/seed${SEED}/fold${FOLD}
[ -f "$CFG_DIR/${TASK}.json" ] || { echo "no config $CFG_DIR/${TASK}.json" >&2; exit 1; }
[ -f "$OUT/probs_last.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== candidate $(basename $OUT_DIR)  $TASK seed $SEED fold $FOLD: $BUDGET epochs, lr $LR $SCHED ${HORIZON}  extra: ${EXTRA:-}  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
# shellcheck disable=SC2086
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG_DIR/${TASK}.json" --fold "$FOLD" --out "$OUT" \
    --max-epochs "$BUDGET" --seed "$SEED" --save-probs "${ARGS[@]}" ${EXTRA:-}
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
