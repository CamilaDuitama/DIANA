#!/bin/bash
#SBATCH --job-name=blast_annotate
#SBATCH --partition=common
#SBATCH --output=logs/blast_annotation_%j.out
#SBATCH --error=logs/blast_annotation_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# SLURM script to run BLAST annotation of discriminant features
# Uses BLAST+ 2.17.0 to avoid gi overflow errors

set -e

echo "=== BLAST Feature Annotation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Load required modules
module purge
module load blast+/2.17.0

# Verify BLAST is loaded
echo "BLAST version:"
blastn -version
echo ""

# Activate conda environment
source /pasteur/appa/homes/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /pasteur/appa/scratch/cduitama/EDID/decOM-classify/env

# Create output directory
mkdir -p paper/blast_results
mkdir -p logs

# Run annotation script
echo "Running BLAST annotation..."
python scripts/feature_analysis/03_annotate_features.py \
    --config configs/feature_analysis.yaml

echo ""
echo "Completed: $(date)"
