#!/bin/bash
#SBATCH --partition=seqbio
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=diana_baselines
#SBATCH --output=logs/baseline_comparison_%j.out
#SBATCH --error=logs/baseline_comparison_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# Baseline Classifier Comparison (CPU, no GPU needed)
################################################################################
#
# PURPOSE:
#   Runs 5-fold cross-validation on the training set for linear baseline
#   classifiers (LogisticRegression, LogisticRegression_Bal, LinearSVM,
#   LinearSVM_Bal, RidgeClassifier, RidgeClassifier_Bal) across the four
#   DIANA tasks. Results + paired t-tests vs DIANA CV saved to OUTPUT_DIR.
#
# INPUT:
#   - data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat
#   - paper/metadata/train_metadata.tsv
#   - results/training/cv_results/aggregated_results.json (DIANA reference)
#
# OUTPUT:
#   - results/baseline_comparison/  (cv_results.json, summary.csv/tex,
#     ttest_results.json, aggregated_metrics.json, confusion_matrices/,
#     classification_reports/, comparison_figure.html, timing.json, ...)
#
# USAGE:
#   sbatch scripts/evaluation/run_baseline_comparison.sbatch
#
################################################################################

set -euo pipefail

echo "=========================================="
echo "DIANA Baseline Comparison"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Time:    $(date)"
echo "=========================================="

cd /pasteur/appa/scratch/cduitama/EDID/decOM-classify

mkdir -p logs results/baseline_comparison

mamba run -p ./env python scripts/evaluation/07_baseline_comparison.py

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
