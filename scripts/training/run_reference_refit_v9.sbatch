#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-99
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_reference_refit
#SBATCH --output=logs/reference_v9/fit_%a_%j.out
#SBATCH --error=logs/reference_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X8: the frozen reference recipe (PROJECT.md section 2, "Reference DIANA model") refit
# on the five grouped dev folds as its declared 5-seed ensemble, so that X9, X10 and every
# later screen pair against one set of out-of-fold predictions.
#   per arm: lr / schedule / epochs from results/reference_v9/epoch_budget.tsv
#   5 seeds (42, 1, 2, 3, 4) x 4 arms x 5 folds = 100 fits, probabilities saved
# Afterwards: 54_seed_ensemble.py --base results/reference_v9 --budgets
#   results/reference_v9/epoch_budget.tsv --out results/reference_v9_oof
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4)
TASKS=(community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((i / 20))]}; rem=$((i % 20)); TASK=${TASKS[$((rem / 5))]}; FOLD=$((rem % 5))
read -r BUDGET LR SCHED HORIZON < <(awk -F'\t' -v a="$TASK" 'NR>1 && $1==a {print $2, $3, $4, ($5==""?"-":$5)}' results/reference_v9/epoch_budget.tsv)
[ -n "$BUDGET" ] || { echo "no reference settings for $TASK" >&2; exit 1; }
EXTRA=(--lr "$LR" --lr-schedule "$SCHED"); [ "$HORIZON" != "-" ] && EXTRA+=(--lr-horizon "$HORIZON")
OUT=results/reference_v9/${TASK}/seed${SEED}/fold${FOLD}
[ -f "$OUT/probs_last.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== X8 reference  $TASK seed $SEED fold $FOLD: $BUDGET epochs, lr $LR $SCHED ${HORIZON}  -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py configs/final_fixed_v9/${TASK}.json --fold "$FOLD" --out "$OUT" \
    --max-epochs "$BUDGET" --seed "$SEED" --save-probs "${EXTRA[@]}"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
