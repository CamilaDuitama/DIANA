#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_final_test
#SBATCH --output=logs/final_test_v9/%x_%a_%j.out
#SBATCH --error=logs/final_test_v9/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# HELD-OUT READ 11 (PROJECT.md ledger). Submit ONLY with Camila's explicit approval.
# The final DIANA: the frozen reference recipe (per-arm lr, schedule, epoch budget from
# results/reference_v9/epoch_budget.tsv; 5 seeds) fitted on ALL 2,716 training runs and
# applied once to the 922 held-out runs. Tasks 0-19: the four arms x 5 seeds with the
# reference recipe; tasks 20-24: the feature arm x 5 seeds with equal weight per study
# (Phase 4 step 4.2, the one adopted change). The scoring (20_final_test_score_v9.py)
# composes the final DIANA per task: community_type, sample_host, material from the
# reference fits, feature from the study-weight fits.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4); TASKS=(community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 20 ]; then
    TASK=${TASKS[$((i / 5))]}; SEED=${SEEDS[$((i % 5))]}; CFG=configs/final_fixed_v9/${TASK}.json; OUT=results/final_test_v9/reference/${TASK}/seed${SEED}; EXTRA=""
else
    TASK=feature; SEED=${SEEDS[$((i - 20))]}; CFG=configs/step4_2_v9/feature.json; OUT=results/final_test_v9/studyweight/${TASK}/seed${SEED}; EXTRA="--study-weights"
fi
read -r BUDGET LR SCHED HORIZON < <(awk -F'\t' -v a="$TASK" 'NR>1 && $1==a {print $2, $3, $4, ($5==""?"-":$5)}' results/reference_v9/epoch_budget.tsv)
[ -n "$BUDGET" ] || { echo "no reference settings for $TASK" >&2; exit 1; }
ARGS=(--lr "$LR" --lr-schedule "$SCHED"); [ "$HORIZON" != "-" ] && ARGS+=(--lr-horizon "$HORIZON")
[ -f "$OUT/probs_last.npz" ] && { echo "refusing to re-read held-out into $OUT" >&2; exit 1; }
echo "=== FINAL TEST (read 11)  $TASK seed $SEED  budget $BUDGET lr $LR $SCHED $HORIZON  ${EXTRA}  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
# shellcheck disable=SC2086
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold 0 --out "$OUT" --max-epochs "$BUDGET" --seed "$SEED" --save-probs "${ARGS[@]}" $EXTRA \
    --heldout data/matrices/matrix_v9_heldout/unitigs.frac.mat --heldout-ids data/splits_v9/test_accessions.txt --heldout-metadata data/splits_v9/test_metadata.tsv
rc=$?; echo "=== exit $rc — $(date) ==="; exit $rc
