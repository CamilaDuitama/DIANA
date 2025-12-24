#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --output=logs/feature_analysis_%j.out
#SBATCH --error=logs/feature_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Feature Analysis Pipeline for DIANA Model
# ==========================================
# Extracts and analyzes most discriminant features from trained model

set -e  # Exit on error

echo "Starting feature analysis pipeline..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./env

# Configuration file
CONFIG_FILE="configs/feature_analysis.yaml"

echo ""
echo "Using configuration: $CONFIG_FILE"
echo ""

# Create output directories
mkdir -p logs

echo ""
echo "=== Step 1: Extract Feature Importance ==="
echo ""

python scripts/feature_analysis/01_extract_feature_importance.py \
    --config $CONFIG_FILE

echo ""
echo "=== Step 2: Analyze Feature Sequences ==="
echo ""

python scripts/feature_analysis/02_analyze_feature_sequences.py \
    --config $CONFIG_FILE \
    --method weight_based

echo ""
echo "=== Feature Analysis Complete! ==="
echo "Results saved to:"
echo "  - Figures: paper/figures/feature_analyses/"
echo "  - Tables: paper/tables/feature_analyses/"
echo ""
echo "Finished at: $(date)"
