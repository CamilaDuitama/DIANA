#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-159
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_x6_lrgrid
#SBATCH --output=logs/x6_v9/fit_%a_%j.out
#SBATCH --error=logs/x6_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X6: re-tune the learning rate and add a schedule UNDER the fixed-budget recipe, on the
# grouped dev folds, for the four single-task arms. Grid: lr in {1e-4, 3e-4, 1e-3, 3e-3}
# x schedule in {constant, cosine to zero over the 300-epoch cap}; every other
# hyperparameter unchanged (configs/final_fixed_v9). The budget of each grid point is then
# chosen by 37_ (same rule as X1) and the best point per arm by 55_x6_select.py.
# 8 configs x 4 tasks x 5 folds = 160 fits.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
LRS=(1e-4 3e-4 1e-3 3e-3)
SCHEDS=(constant cosine)
TASKS=(community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
c=$((i / 20)); rem=$((i % 20)); TASK=${TASKS[$((rem / 5))]}; FOLD=$((rem % 5))
LR=${LRS[$((c / 2))]}; SCHED=${SCHEDS[$((c % 2))]}
CFG=configs/final_fixed_v9/${TASK}.json
OUT=results/x6_grid_v9/lr${LR}_${SCHED}/${TASK}/fold${FOLD}
[ -f "$OUT/history.json" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== X6  lr $LR $SCHED  $TASK fold $FOLD -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --max-epochs 300 --lr "$LR" --lr-schedule "$SCHED"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
