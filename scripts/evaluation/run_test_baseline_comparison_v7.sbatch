#!/bin/bash
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=diana_baseline_v7
#SBATCH --output=logs/baseline/diana_baseline_v7_%j.out
#SBATCH --error=logs/baseline/diana_baseline_v7_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr
#SBATCH --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify

################################################################################
# DIANA Test-Set Baseline Comparison — v7
#
# Trains linear baselines on v7 training set (2,838 samples, 78,430 features)
# and evaluates on v7 test set (352 samples).
#
# Tasks (from config):
#   community_type, sample_host, material, sample_age, latitude, longitude
#   (sample_type excluded — all v7 samples are "ancient_metagenome")
#
# CONFIGURATION:
#   Edit configs/baseline_config_v7.yaml to change:
#   - Data paths (matrix, metadata files)
#   - Task list
#   - Model hyperparameters
#   - Output directory
#
# USAGE:
#   sbatch scripts/evaluation/run_test_baseline_comparison_v7.sbatch
#
# AFTER COMPLETION:
#   Results in: results/baseline_comparison_v7/
#   - metrics.json         : Raw metrics with bootstrap CIs
#   - summary.csv          : Human-readable table
#   - summary.tex          : LaTeX table for paper
#   - baseline_comparison.log : Full log
################################################################################

set -e

echo "=== DIANA BASELINE COMPARISON (v7) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PYTHON=/pasteur/appa/scratch/cduitama/EDID/decOM-classify/env/bin/python

echo "Python: $PYTHON"
echo "Config: configs/baseline_config_v7.yaml"
echo ""

# Create directories
mkdir -p logs/baseline results/baseline_comparison_v7

# Run baseline comparison with config file
$PYTHON scripts/evaluation/08_test_set_baseline_comparison.py \
    --config configs/baseline_config_v7.yaml

echo ""
echo "=== BASELINE COMPARISON COMPLETE — $(date) ==="
echo "Results in: results/baseline_comparison_v7/"
