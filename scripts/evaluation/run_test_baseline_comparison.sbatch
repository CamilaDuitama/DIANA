#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_baseline_bp
#SBATCH --output=logs/final_training/diana_baseline_bp_%j.out
#SBATCH --error=logs/final_training/diana_baseline_bp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# DIANA Test-Set Baseline Comparison — BioProject-Disjoint Split
################################################################################
#
# Trains linear + Random Forest baselines on the full training set and evaluates
# on the held-out BioProject-disjoint test set. Addresses reviewer comments
# requesting non-linear baselines evaluated on the test set (not training CV).
#
# USAGE:
#   sbatch scripts/evaluation/run_test_baseline_comparison.sbatch
################################################################################

set -e

echo "=== DIANA BASELINE COMPARISON (BioProject-disjoint test set) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

python scripts/evaluation/08_test_set_baseline_comparison.py

echo ""
echo "=== BASELINE COMPARISON COMPLETE — $(date) ==="
echo "Results in: results/baseline_comparison_bioproject/"
