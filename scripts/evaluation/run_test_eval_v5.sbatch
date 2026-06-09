#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=diana_test_bp_v5
#SBATCH --output=logs/final_training/diana_test_bp_v5_%j.out
#SBATCH --error=logs/final_training/diana_test_bp_v5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr
#SBATCH --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify

################################################################################
# DIANA v5 Test Evaluation — BioProject-Disjoint Split
#
# Evaluates the v5 model on the held-out test set (523 samples).
# Run AFTER run_final_train_edid_v5.sbatch completes.
#
# OUTPUT:
#   results/test_evaluation_bioproject_v5/test_predictions.tsv
#   results/test_evaluation_bioproject_v5/test_metrics.json
#
# NEXT STEPS:
#   sbatch --array=1-611 scripts/validation/run_inference_bioproject_v5.sbatch
################################################################################

set -e

echo "=== DIANA v5 TEST EVALUATION (BioProject-disjoint split) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

MODEL="results/training_bioproject_v5/best_model.pth"
CONFIG="results/training_bioproject_v5/final_training_config.json"
MATRIX="data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat"
METADATA="data/splits_bioproject/test_metadata.tsv"
TEST_IDS="data/splits_bioproject/test_ids.txt"
OUTPUT="results/test_evaluation_bioproject_v5"

mkdir -p "$OUTPUT" logs/final_training

echo "Model:    $MODEL"
echo "Config:   $CONFIG"
echo "Output:   $OUTPUT"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

diana-test \
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
echo ""
echo "Next: sbatch --array=1-611 scripts/validation/run_inference_bioproject_v5.sbatch"
