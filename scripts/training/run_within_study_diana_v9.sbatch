#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_within_fits
#SBATCH --output=logs/within_study_v9/fit_%a_%j.out
#SBATCH --error=logs/within_study_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# W3: DIANA's five read-9 arms on the five within-study (sample-grouped) folds of the
# training set, each fitted for exactly its read-9 epoch budget
# (results/epoch_budget_v9/epoch_budget.tsv; the last epoch is the prediction used).
# Hyperparameters unchanged (configs/final_fixed_v9/). Studies sit on both sides of
# every fold by design: --allow-shared-projects, logged as a warning by 04_.
# The rtx6000 nodes are excluded (sm_120, unsupported by this env's torch).
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(multitask community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$((i / 5))]}; FOLD=$((i % 5))
CFG=configs/final_fixed_v9/${ARM}.json
OUT=results/within_study_v9/diana/${ARM}/fold${FOLD}
BUDGET=$(awk -F'\t' -v a="$ARM" 'NR>1 && $1==a {print $3}' results/epoch_budget_v9/epoch_budget.tsv)
[ -n "$BUDGET" ] || { echo "no budget for $ARM in results/epoch_budget_v9/epoch_budget.tsv" >&2; exit 1; }
[ -s results/within_study_v9/folds.tsv ] || { echo "folds.tsv missing (W1)" >&2; exit 1; }
[ -f "$OUT/preds_by_epoch.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== W3  $ARM fold $FOLD, budget $BUDGET epochs -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --max-epochs "$BUDGET" --dev-folds results/within_study_v9/folds.tsv --allow-shared-projects
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
