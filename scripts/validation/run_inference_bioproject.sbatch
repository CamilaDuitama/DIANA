#!/bin/bash
#SBATCH --job-name=diana_val_bp
#SBATCH --partition=edid_rtx6000
#SBATCH --array=1-611
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/diana_val_bp_%A_%a.out
#SBATCH --error=logs/validation/diana_val_bp_%A_%a.err

set -e

ACCESSIONS="data/splits_bioproject/validation_accessions.txt"
INPUT_DIR="results/validation_predictions"
OUTPUT_DIR="results/validation_predictions_bioproject"
MODEL="results/training_bioproject/best_model.pth"
LABEL_ENCODERS="results/training_bioproject/label_encoders.json"

# Get accession for this task
ACC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACCESSIONS")
if [ -z "$ACC" ]; then
    echo "ERROR: No accession for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: $ACC"

# Output paths
SAMPLE_OUT="${OUTPUT_DIR}/${ACC}"
PRED_FILE="${SAMPLE_OUT}/${ACC}_predictions.json"

# Skip if already done
if [ -f "$PRED_FILE" ]; then
    echo "Already exists, skipping: $PRED_FILE"
    exit 0
fi

mkdir -p "$SAMPLE_OUT"

# Input fraction file (already computed)
FRACTION_FILE="${INPUT_DIR}/${ACC}/${ACC}_unitig_fraction.txt"
if [ ! -f "$FRACTION_FILE" ]; then
    echo "ERROR: Fraction file not found: $FRACTION_FILE"
    exit 1
fi

# Setup environment
export LD_LIBRARY_PATH="$(pwd)/env/lib:$LD_LIBRARY_PATH"
eval "$(conda shell.bash hook)"
conda activate ./env
export PATH="$(pwd)/env/bin:$PATH"

echo "Running inference..."
./env/bin/python scripts/inference/03_run_inference.py \
    --model "$MODEL" \
    --input "$FRACTION_FILE" \
    --output "$PRED_FILE" \
    --sample-id "$ACC" \
    --label-encoders "$LABEL_ENCODERS" \
    --device cpu

echo "Done: $PRED_FILE"
