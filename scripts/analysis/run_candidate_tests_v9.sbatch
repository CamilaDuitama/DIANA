#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=diana_candidate_tests
#SBATCH --output=logs/phase4_v9/%x_%j.out
#SBATCH --error=logs/phase4_v9/%x_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Phase 4: score one candidate (100 fits in OUT_DIR, reference layout) against the starting model.
#   NAME      short name (step4_1, step4_2, ...)
#   OUT_DIR   results directory of the fits (results/<NAME>_..._v9)
#   START     starting model's OOF directory (default results/reference_v9_oof)
#   PROBS     optional: name=dir of logistic-regression probabilities to score beside (4.2)
#   PAIR      optional: extra paired rows a:b for 59_ (space-separated)
#   REPORT    1 -> 63_watchlist_shallow_report.py (4.2, 4.3)
# 54_ pools the 5 seeds -> 48_ macro-F1 paired (vs start, vs logistic regression) -> 59_ flags.
#   sbatch --export=ALL,NAME=step4_2,OUT_DIR=results/step4_2_studyweight_v9,PROBS=logreg_w=results/step4_2_studyweight_v9/logreg_probs,REPORT=1 ...
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
: "${NAME:?set NAME}"; : "${OUT_DIR:?set OUT_DIR}"; START=${START:-results/reference_v9_oof}
OOF=${OUT_DIR}_oof
echo "=== Phase 4 tests for $NAME ($OUT_DIR) against $START  job $SLURM_JOB_ID  $(date) ==="
if [ -f "$OOF/epoch_budget.tsv" ]; then echo "pooled OOF exists in $OOF, kept"; else
    ./env/bin/python scripts/analysis/54_seed_ensemble.py --base "$OUT_DIR" --seeds 42,1,2,3,4 --budgets results/reference_v9/epoch_budget.tsv --out "$OOF" || exit 1
fi
./env/bin/python scripts/analysis/48_paired_sig_vs_fraction.py --sig "$OOF" --frac "$START" --name-a "$NAME" --single-only || exit 1
./env/bin/python scripts/analysis/48_paired_sig_vs_fraction.py --sig "$OOF" --frac results/logreg_oof_v9 --name-a "${NAME}_vs_logreg" --single-only || exit 1
FLAGS=(--extra "$NAME=$OUT_DIR" --out "results/devfold_flagging_${NAME}_v9")
[ -n "${PROBS:-}" ] && FLAGS+=(--extra-probs "$PROBS")
for p in ${PAIR:-}; do FLAGS+=(--pair "$p"); done
./env/bin/python scripts/analysis/59_devfold_flagging_score.py "${FLAGS[@]}" || exit 1
if [ -n "${REPORT:-}" ]; then
    ./env/bin/python scripts/analysis/63_watchlist_shallow_report.py --a "$START" --b "$OOF" --name-b "$NAME" --out "$OUT_DIR" || exit 1
fi
echo "=== exit 0 — $(date) ==="
