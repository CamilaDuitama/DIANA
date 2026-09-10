#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=diana_final_bp_v7
#SBATCH --output=logs/final_training/diana_final_bp_v7_%j.out
#SBATCH --error=logs/final_training/diana_final_bp_v7_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# DIANA v7 Final Model Training
#
# Config-driven: all settings read from configs/train_config_bioproject_v7.json
# Tasks: community_type, sample_host, material (classification)
#        sample_age, latitude, longitude (regression)
# Matrix: data/matrices/matrix_v7_3190/unitigs.frac.mat  (78,430 unitigs)
# Splits: data/splits_v7/  (2,838 train / 352 test / 702 val)
#
# PREREQUISITES — run in order:
#
#   1. HPO (5 folds, array job):
#        sbatch scripts/training/run_hyperopt_bioproject_v7.sbatch
#
#   2. Aggregate CV results:
#        conda run -p ./env python scripts/training/aggregate_cv_results.py \
#            --cv_dir results/training_bioproject_v7/cv_results \
#            --n_folds 5
#      → writes results/training_bioproject_v7/final_training_config.json
#
#   3. Submit this script:
#        sbatch scripts/training/run_final_train_edid_v7.sbatch
#
# NEXT STEPS after training:
#   1. Test evaluation:
#        sbatch scripts/evaluation/run_test_eval_v7.sbatch
#   2. Validation inference (once FASTQs ready):
#        sbatch scripts/validation/run_inference_bioproject_v7.sbatch
#   3. Baseline comparison (already configured):
#        sbatch scripts/evaluation/run_test_baseline_comparison_v7.sbatch
################################################################################

set -e

echo "=== DIANA v7 FINAL TRAINING ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

TRAIN_CONFIG="results/training_bioproject_v7/final_training_config.json"

if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "ERROR: Training config not found: $TRAIN_CONFIG"
    echo "Run HPO + aggregate_cv_results.py first (see script header)."
    exit 1
fi

echo "Config: $TRAIN_CONFIG"
OUTPUT_DIR=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['output_dir'])")
echo "Output:  $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR" logs/final_training

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "Starting final training..."
python scripts/training/02_train_final_model.py "$TRAIN_CONFIG"

echo ""
echo "=== TRAINING COMPLETE — $(date) ==="
