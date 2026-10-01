#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-49
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_x5_stopping
#SBATCH --output=logs/x5_v9/fit_%a_%j.out
#SBATCH --error=logs/x5_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X5: the two early-stopping rules on the grouped dev folds, to be compared fairly with
# the fixed budget (results/epoch_budget_v9). Published settings: 10 % inner validation
# split, patience 20, cap 200 epochs, monitor = inner macro-F1.
#   tasks  0-24: inner split RANDOM over runs (the published, leaky-within-study rule)
#   tasks 25-49: inner split GROUPED by BioProject with the class floor (read-8 rule)
# Each task = one arm x one dev fold; predictions saved at every epoch, selected epoch in
# history.json. Afterwards: 53_pool_selected_epoch.py and 48_ (run_x567_verdict_v9.sbatch).
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
RULES=(random grouped)
ARMS=(multitask community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
RULE=${RULES[$((i / 25))]}; j=$((i % 25)); ARM=${ARMS[$((j / 5))]}; FOLD=$((j % 5))
CFG=configs/final_fixed_v9/${ARM}.json
OUT=results/x5_${RULE}_v9/${ARM}/fold${FOLD}
[ -f "$OUT/history.json" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== X5  $RULE  $ARM fold $FOLD -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --max-epochs 200 --early-stop "$RULE" --val-split 0.1 --patience 20
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
