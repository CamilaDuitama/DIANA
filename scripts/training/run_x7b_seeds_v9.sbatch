#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-99
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_x7b_seeds
#SBATCH --output=logs/x7_v9/seed_%a_%j.out
#SBATCH --error=logs/x7_v9/seed_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X7b: five seeds per single-task arm, each fitted for exactly the arm's read-9 epoch
# budget (results/epoch_budget_v9/epoch_budget.tsv) on each dev fold, softmax probabilities
# saved at the last epoch. 54_seed_ensemble.py averages the five seeds (declared ensemble)
# and reports the seed-to-seed spread. 5 seeds x 4 tasks x 5 folds = 100 fits.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4)
TASKS=(community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((i / 20))]}; rem=$((i % 20)); TASK=${TASKS[$((rem / 5))]}; FOLD=$((rem % 5))
CFG=configs/final_fixed_v9/${TASK}.json
BUDGET=$(awk -F'\t' -v a="$TASK" 'NR>1 && $1==a {print $3}' results/epoch_budget_v9/epoch_budget.tsv)
[ -n "$BUDGET" ] || { echo "no budget for $TASK" >&2; exit 1; }
OUT=results/x7b_seeds_v9/${TASK}/seed${SEED}/fold${FOLD}
[ -f "$OUT/probs_last.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== X7b  seed $SEED  $TASK fold $FOLD, $BUDGET epochs -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --max-epochs "$BUDGET" --seed "$SEED" --save-probs
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
