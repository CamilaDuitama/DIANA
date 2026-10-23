#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-124
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_seeds_v17
#SBATCH --output=logs/seeds_v17/seed_%a_%j.out
#SBATCH --error=logs/seeds_v17/seed_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v17 seed ensembles on the dev folds, both architectures (Camila 2026-10-09):
# five seeds per arm at the final settings, probabilities saved at the budget.
# Settings per arm after the rate re-check (results/x6_selected_v17, adoption by
# the written whole-task rule): material adopts lr 3e-3 cosine (300-epoch
# horizon, run 5 epochs); the other arms keep their searched settings at their
# budget-stage budgets. 5 seeds x 5 arms x 5 folds = 125 fits.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4)
ARMS=(community_type feature sample_host material multitask)
BUDGETS=(71 2 304 5 60)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((i / 25))]}; rem=$((i % 25)); k=$((rem / 5)); ARM=${ARMS[$k]}; FOLD=$((rem % 5))
BUDGET=${BUDGETS[$k]}
CFG=configs/final_fixed_v17/${ARM}.json
EXTRA=""
if [ "$ARM" = "material" ]; then
    EXTRA="--lr 3e-3 --lr-schedule cosine --lr-horizon 300"
fi
OUT=results/seeds_v17/${ARM}/seed${SEED}/fold${FOLD}
[ -f "$OUT/probs_last.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== v17 seeds  seed $SEED  $ARM fold $FOLD, $BUDGET epochs $EXTRA -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
# shellcheck disable=SC2086
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --max-epochs "$BUDGET" --seed "$SEED" --save-probs \
    --dev-folds data/splits_v17/dev_folds.tsv $EXTRA
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
