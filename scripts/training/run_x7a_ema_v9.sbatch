#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_x7a_ema
#SBATCH --output=logs/x7_v9/ema_%a_%j.out
#SBATCH --error=logs/x7_v9/ema_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X7a: weight averaging. The five arms under the X1 recipe (cap 600, same hyperparameters),
# with predictions taken from a per-epoch exponential moving average of the weights
# (decay 0.9, started after the first epoch) instead of the raw weights. Budgets are then
# chosen by 37_ on the averaged predictions and compared with results/epoch_budget_v9.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(multitask community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$((i / 5))]}; FOLD=$((i % 5))
CFG=configs/final_fixed_v9/${ARM}.json
OUT=results/x7a_ema_v9/${ARM}/fold${FOLD}
[ -f "$OUT/history.json" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== X7a  EMA 0.9  $ARM fold $FOLD -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --max-epochs 600 --ema 0.9
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
