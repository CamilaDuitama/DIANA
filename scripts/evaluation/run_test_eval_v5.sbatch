#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=diana_test_eval
#SBATCH --output=logs/final_training/diana_test_eval_%j.out
#SBATCH --error=logs/final_training/diana_test_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr
#SBATCH --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify

################################################################################
# DIANA Test Evaluation — generic, config-driven
#
# Evaluates a trained DIANA model on the held-out test set.
# All paths are passed as environment variables — do NOT hardcode versions here.
#
# Required environment variables (set before sbatch, or export before sourcing):
#   MODEL      path to best_model.pth
#   CONFIG     path to final_training_config.json
#   MATRIX     path to unitigs.frac.mat
#   METADATA   path to test_metadata.tsv
#   TEST_IDS   path to test_ids.txt
#   OUTPUT     output directory
#
# Example (v7):
#   MODEL=results/training_bioproject_v7/best_model.pth \
#   CONFIG=results/training_bioproject_v7/final_training_config.json \
#   MATRIX=data/matrices/matrix_v7_3190/unitigs.frac.mat \
#   METADATA=data/splits_v7/test_metadata.tsv \
#   TEST_IDS=data/splits_v7/test_ids.txt \
#   OUTPUT=results/test_evaluation_bioproject_v7 \
#   sbatch scripts/evaluation/run_test_eval_v5.sbatch
#
# OUTPUT:
#   $OUTPUT/test_predictions.tsv
#   $OUTPUT/test_metrics.json
################################################################################

set -e

# Validate required variables
for var in MODEL CONFIG MATRIX METADATA TEST_IDS OUTPUT; do
    if [ -z "${!var}" ]; then
        echo "ERROR: environment variable '$var' is not set."
        echo "See script header for usage."
        exit 1
    fi
done

echo "=== DIANA TEST EVALUATION ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PYTHON=/pasteur/appa/scratch/cduitama/EDID/decOM-classify/env/bin/python
DIANA_TEST=/pasteur/appa/scratch/cduitama/EDID/decOM-classify/env/bin/diana-test
echo "Python: $PYTHON"

mkdir -p "$OUTPUT" logs/final_training

echo "Model:    $MODEL"
echo "Config:   $CONFIG"
echo "Matrix:   $MATRIX"
echo "Metadata: $METADATA"
echo "Test IDs: $TEST_IDS"
echo "Output:   $OUTPUT"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

$DIANA_TEST \
    --model "$MODEL" \
    --config "$CONFIG" \
    --matrix "$MATRIX" \
    --metadata "$METADATA" \
    --test-ids "$TEST_IDS" \
    --output "$OUTPUT" \
    --device cpu

echo ""
echo "=== TEST EVALUATION COMPLETE — $(date) ==="
echo "Results in: $OUTPUT"
