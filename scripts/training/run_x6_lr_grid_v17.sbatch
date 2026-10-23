#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-199
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_x6_lrgrid_v17
#SBATCH --output=logs/x6_v17/fit_%a_%j.out
#SBATCH --error=logs/x6_v17/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v17 learning-rate re-check under the fixed budget, on the v17 dev folds, for
# the four single-task arms AND the multitask arm (both architectures carried to
# the end; Camila 2026-10-09). Grid: lr in {1e-4, 3e-4, 1e-3, 3e-3} x schedule in
# {constant, cosine to zero over the 300-epoch cap}; every other hyperparameter
# unchanged (configs/final_fixed_v17). Budget per grid point by the X1 rule
# (37_), best point per arm by 55_x6_select.py, tie keeps the searched rate.
# 8 configs x 5 arms x 5 folds = 200 fits.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
LRS=(1e-4 3e-4 1e-3 3e-3)
SCHEDS=(constant cosine)
ARMS=(community_type feature sample_host material multitask)
i=$SLURM_ARRAY_TASK_ID
c=$((i / 25)); rem=$((i % 25)); ARM=${ARMS[$((rem / 5))]}; FOLD=$((rem % 5))
LR=${LRS[$((c / 2))]}; SCHED=${SCHEDS[$((c % 2))]}
CFG=configs/final_fixed_v17/${ARM}.json
OUT=results/x6_grid_v17/lr${LR}_${SCHED}/${ARM}/fold${FOLD}
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 1; }
[ -f "$OUT/preds_by_epoch.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== v17 X6  lr $LR $SCHED  $ARM fold $FOLD -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" \
    --fold "$FOLD" --out "$OUT" --max-epochs 300 \
    --dev-folds data/splits_v17/dev_folds.tsv --lr "$LR" --lr-schedule "$SCHED"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
