#!/bin/bash
#SBATCH --job-name=ood_calibration_v5
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/calibration/ood_calibration_v5_%j.out
#SBATCH --error=logs/calibration/ood_calibration_v5_%j.err
#SBATCH --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify

################################################################################
# OOD Calibration Comparison for v5 DIANA model
#
# Usage:
#   sbatch scripts/calibration/run_08_ood_calibration.sbatch [dataset]
#
# Arguments:
#   dataset: validation (default) or test
#
# Compares four confidence metrics for calibration quality and
# wrong-prediction detection:
#   1. Softmax max-probability
#   2. Energy score
#   3. Mahalanobis distance
#   4. MC Dropout (can be skipped with --skip-mc-dropout flag)
#
# REQUIRES:
#   - results/training_bioproject_v5/train_embeddings.npz
#   - results/training_bioproject_v5/mahalanobis_params.npz
#   - results/calibration_{dataset}_v5/embeddings.npz
#
# OUTPUT (depends on dataset):
#   results/calibration_{dataset}_v5/
#   paper/assets/calibration_{dataset}/
################################################################################

set -e

DATASET=${1:-validation}  # Default to validation if not specified

echo "======================================"
echo "Script 08: OOD Calibration Comparison"
echo "Dataset: ${DATASET}"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

mkdir -p logs/calibration

# Activate environment
export LD_LIBRARY_PATH="$(pwd)/env/lib:$LD_LIBRARY_PATH"
eval "$(conda shell.bash hook)"
conda activate ./env
export PATH="$(pwd)/env/bin:$PATH"

RESULTS_DIR="results/training_bioproject_v5"

# Read dataset-specific paths
case "${DATASET}" in
    validation)
        METADATA="data/splits_v5/validation_metadata.tsv"
        PRED_DIR="results/validation_predictions_bioproject_v5_full"
        EVAL_DIR="results/calibration_validation_v5"
        FIG_DIR="paper/assets/calibration_validation"
        TSV_ARG=""  # No TSV for validation (uses JSON files)
        MATRIX_ARG=""  # No matrix for validation (uses prediction directories)
        ;;
    test)
        METADATA="data/splits_v5/test_metadata.tsv"
        PRED_DIR="results/test_predictions_bioproject_v5"
        EVAL_DIR="results/calibration_test_v5"
        FIG_DIR="paper/assets/calibration_test"
        TSV_ARG="--predictions-tsv results/test_evaluation_bioproject_v5/test_predictions.tsv"
        MATRIX_ARG="--matrix data/matrices/large_matrix_3070_with_frac"
        ;;
    *)
        echo "ERROR: Unknown dataset '${DATASET}'. Use 'validation' or 'test'."
        exit 1
        ;;
esac

# Check prerequisites
echo "Checking prerequisites..."
for file in \
    "${RESULTS_DIR}/train_embeddings.npz" \
    "${RESULTS_DIR}/mahalanobis_params.npz" \
    "${EVAL_DIR}/embeddings.npz" \
    "${RESULTS_DIR}/best_model.pth" \
    "${RESULTS_DIR}/label_encoders.json" \
    "${METADATA}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Missing prerequisite file: $file"
        echo "Run script 07 first: sbatch scripts/calibration/run_07_extract_embeddings.sbatch ${DATASET}"
        exit 1
    fi
done
echo "✓ All prerequisites found"
echo ""

echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  Metadata: ${METADATA}"
echo "  Predictions: ${PRED_DIR}"
echo "  Output: ${EVAL_DIR}"
echo "  Figures: ${FIG_DIR}"
echo ""

mkdir -p "${EVAL_DIR}"
mkdir -p "${FIG_DIR}"

echo "=== Running calibration comparison ==="
echo ""
echo "NOTE: MC Dropout enabled (30 stochastic passes per sample). This may take ~30 minutes."
echo ""

# Run with MC Dropout enabled for complete calibration analysis
python scripts/calibration/08_ood_calibration_comparison.py \
    --train-embeddings ${RESULTS_DIR}/train_embeddings.npz \
    --val-embeddings ${EVAL_DIR}/embeddings.npz \
    --mahal-params ${RESULTS_DIR}/mahalanobis_params.npz \
    --val-pred-dir ${PRED_DIR} \
    --val-metadata ${METADATA} \
    --model ${RESULTS_DIR}/best_model.pth \
    --label-encoders ${RESULTS_DIR}/label_encoders.json \
    --out-dir ${EVAL_DIR} \
    --figure-dir ${FIG_DIR} \
    ${TSV_ARG} \
    ${MATRIX_ARG}

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "======================================"
echo ""
echo "Generated outputs:"
echo ""
echo "Analysis results:"
ls -lh ${EVAL_DIR}/
echo ""
echo "Figures:"
ls -lh ${FIG_DIR}/
echo ""
echo "To re-run WITH MC Dropout (slower but more complete):"
echo "  Edit the sbatch script and remove the --skip-mc-dropout flag, or:"
echo "  sbatch scripts/calibration/run_08_ood_calibration.sbatch ${DATASET}  # and manually edit"
