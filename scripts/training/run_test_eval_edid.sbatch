#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=diana_test_bp
#SBATCH --output=logs/final_training/diana_test_bp_%j.out
#SBATCH --error=logs/final_training/diana_test_bp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# DIANA Test Evaluation — BioProject-Disjoint Split (edid_rtx6000)
################################################################################
#
# Evaluates the final trained model on the held-out BioProject-disjoint test set
# (543 samples across 26 BioProjects).
#
# Run AFTER run_final_train_edid.sbatch completes.
#
# USAGE:
#   sbatch scripts/training/run_test_eval_edid.sbatch
################################################################################

set -e

echo "=== DIANA TEST EVALUATION (BioProject-disjoint split) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate environment
eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

MODEL=${MODEL:-"results/training_bioproject/best_model.pth"}
CONFIG=${CONFIG:-"results/training_bioproject/final_training_config.json"}
MATRIX=${MATRIX:-"data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat"}
METADATA=${METADATA:-"data/splits_bioproject/test_metadata.tsv"}
TEST_IDS=${TEST_IDS:-"data/splits_bioproject/test_ids.txt"}
OUTPUT=${OUTPUT:-"results/test_evaluation_bioproject"}

mkdir -p "$OUTPUT"

echo "Model:    $MODEL"
echo "Config:   $CONFIG"
echo "Matrix:   $MATRIX"
echo "Metadata: $METADATA"
echo "Test IDs: $TEST_IDS"
echo "Output:   $OUTPUT"
echo ""

echo "GPU:"
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
