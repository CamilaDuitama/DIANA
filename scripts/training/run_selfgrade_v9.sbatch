#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --array=0-19
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_selfgrade
#SBATCH --output=logs/selfgrade_v9/fit_%a_%j.out
#SBATCH --error=logs/selfgrade_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Self-grading demonstration (PROTOCOLS.md 2026-10-05): reference recipe trained on ALL runs
# with planting-s1 corrupted labels, resubstitution prediction. Never a result.
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4); TASKS=(community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID; SEED=${SEEDS[$((i / 4))]}; TASK=${TASKS[$((i % 4))]}
read -r BUDGET LR SCHED HORIZON < <(awk -F'\t' -v a="$TASK" 'NR>1 && $1==a {print $2, $3, $4, ($5==""?"-":$5)}' results/reference_v9/epoch_budget.tsv)
[ -n "$BUDGET" ] || { echo "no settings for $TASK" >&2; exit 1; }
ARGS=(--lr "$LR" --lr-schedule "$SCHED"); [ "$HORIZON" != "-" ] && ARGS+=(--lr-horizon "$HORIZON")
OUT=results/selfgrade_v9/$TASK/seed$SEED
[ -f "$OUT/probs_last.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== selfgrade $TASK seed $SEED budget $BUDGET  job $SLURM_JOB_ID  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "configs/selfgrade_v9/${TASK}.json" --fold 0 --out "$OUT" \
    --max-epochs "$BUDGET" --seed "$SEED" --save-probs "${ARGS[@]}" --resubstitution
rc=$?; echo "=== exit $rc — $(date) ==="; exit $rc
