#!/bin/bash
#SBATCH --job-name=diana_val_bp_v5
#SBATCH --partition=edid_rtx6000
#SBATCH --array=1-611
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/validation/diana_val_bp_v5_%A_%a.out
#SBATCH --error=logs/validation/diana_val_bp_v5_%A_%a.err
#SBATCH --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify

################################################################################
# DIANA v5 Validation Inference — BioProject-Disjoint Split
#
# Runs per-sample inference on 611 validation accessions using the v5 model.
# Fraction files must already exist in results/validation_predictions/
# (created by the original unitig counting step).
#
# Run AFTER run_test_eval_v5.sbatch completes.
#
# OUTPUT:
#   results/validation_predictions_bioproject_v5/<ACC>/<ACC>_predictions.json
#
# NEXT STEPS (all from project root):
#   mamba run -p ./env python scripts/evaluation/06_compute_training_metrics.py
#   mamba run -p ./env python scripts/evaluation/patch_diana_in_metrics.py
#   bash scripts/paper/generate_all_paper_materials.sh
################################################################################

set -e

ACCESSIONS="data/splits_v5/all_validation_accessions.txt"
INPUT_DIR="results/validation_predictions"
OUTPUT_DIR="results/validation_predictions_bioproject_v5"
MODEL="results/training_bioproject_v5/best_model.pth"
LABEL_ENCODERS="results/training_bioproject_v5/label_encoders.json"

ACC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACCESSIONS")
if [ -z "$ACC" ]; then
    echo "ERROR: No accession for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: $ACC"

SAMPLE_OUT="${OUTPUT_DIR}/${ACC}"
PRED_FILE="${SAMPLE_OUT}/${ACC}_predictions.json"

if [ -f "$PRED_FILE" ]; then
    echo "Already exists, skipping: $PRED_FILE"
    exit 0
fi

mkdir -p "$SAMPLE_OUT"

FRACTION_FILE="${INPUT_DIR}/${ACC}/${ACC}_unitig_fraction.txt"
if [ ! -f "$FRACTION_FILE" ]; then
    echo "ERROR: Fraction file not found: $FRACTION_FILE"
    exit 1
fi

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
