#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=diana_baseline_preds
#SBATCH --output=logs/baseline_preds_v9/preds_%j.out
#SBATCH --error=logs/baseline_preds_v9/preds_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Re-run the tuned baselines to keep their PER-RUN held-out predictions.
#
# The first run saved only aggregate metrics, and two aggregate numbers cannot be
# paired. A superiority test has to bootstrap the difference on the same runs, so
# the predictions themselves are needed.
#
# Writes to a NEW directory: results/baseline_comparison_v9 is cited by Table 8 and
# must not be overwritten. Everything is seeded, so the metrics should reproduce
# exactly; that is checked against the old summary.csv afterwards.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== baselines with predictions  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/09_baselines_v9.py \
    --output results/baseline_predictions_v9 --n-boot 1000
echo "=== COMPLETE — $(date) ==="
