#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=diana_v13_baseline_preds
#SBATCH --output=logs/baseline_predictions_v13/preds_%j.out
#SBATCH --error=logs/baseline_predictions_v13/preds_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K5, baseline half of held-out read 10: the tuned baselines fitted on all training runs
# with the v13 labels (data/splits_v13/) at the configs selected on the v13 dev folds
# (results/baselines_v13/best_per_model_fraction.tsv, written by run_baselines_pool_v13),
# predicting the 922 held-out runs once, per-run predictions and probabilities kept.
# Held-out runs of the dropped classes are out-of-vocabulary: counted, not scored.
# afterok run_baselines_pool_v13.sbatch. Same n-boot as read 9 (run_baseline_preds_v9).
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TUNED=results/baselines_v13/best_per_model_fraction.tsv
[ -s "$TUNED" ] || { echo "$TUNED missing: the K2 pool did not run" >&2; exit 1; }
[ -e results/baseline_predictions_v13/heldout_predictions.tsv ] && \
    { echo "refusing to re-read held-out into results/baseline_predictions_v13" >&2; exit 1; }
echo "=== K5 baselines, held-out read 10  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/09_baselines_v9.py --splits data/splits_v13 --tuned "$TUNED" \
    --output results/baseline_predictions_v13 --n-boot 1000
echo "=== COMPLETE — $(date) ==="
