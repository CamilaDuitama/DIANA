#!/bin/bash
#SBATCH --job-name=feat_imp_v5
#SBATCH --output=logs/feat_imp_v5_%j.out
#SBATCH --error=logs/feat_imp_v5_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=edid
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

# Feature Importance Extraction — v5 model
# ==========================================
# Runs gradient-based and weight-based feature importance analysis
# on the v5 BioProject-disjoint model.
#
# Prerequisite: BLAST annotations already exist in results/feature_analysis/
# (blast_annotations.tsv was computed on the old model but annotates unitig
# sequences which are model-independent — no need to re-run BLAST)
#
# Steps:
#   1. Extract feature importance (gradient + weight) → paper/figures/feature_analysis/
#   2. Aggregate for paper figures → results/feature_analysis/feature_importance_by_genus.tsv
#
# Runtime: ~1-2 h (gradient method on 107K features × 523 test samples)

set -eo pipefail

DIANA_ROOT="/pasteur/appa/scratch/cduitama/EDID/decOM-classify"
cd "${DIANA_ROOT}"

echo "========================================"
echo "Feature Importance Extraction — v5"
echo "Job ${SLURM_JOB_ID}"
echo "========================================"
echo "Start: $(date)"
echo ""

# Activate environment
source /pasteur/appa/homes/cduitama/miniconda3/bin/activate
conda activate "${DIANA_ROOT}/env"

echo "Step 1: Extract feature importance (gradient + weight)"
echo "  Model: results/training_bioproject_v5/best_model.pth"
echo "  Matrix: data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat"
echo "  Test split: data/splits_v5/test_ids.txt (523 samples)"
echo ""

python scripts/feature_analysis/01_extract_feature_importance.py \
    --config configs/feature_analysis.yaml

echo ""
echo "Step 2: Aggregate for paper figures"
python scripts/feature_analysis/05_aggregate_for_validation.py

echo ""
echo "Done: $(date)"
echo "Outputs:"
echo "  paper/figures/feature_analysis/"
echo "  results/feature_analysis/feature_importance_by_genus.tsv"
