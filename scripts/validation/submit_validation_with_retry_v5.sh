#!/bin/bash
#
# Submit v5 validation predictions (full pipeline, with .jobinfo) with OOM retry logic
#
# Runs diana-predict from FASTQs using the v5 model on all 611 bioproject-disjoint
# validation samples. Generates .jobinfo files for the runtime/memory figure.
#
# Usage:
#   bash scripts/validation/submit_validation_with_retry_v5.sh
#
# Idempotent — re-run after OOM failures to retry with doubled memory.

set -e

echo "========================================"
echo "DIANA v5 Validation Predictions (full)"
echo "========================================"
echo "Started: $(date)"
echo ""

ACCESSIONS="data/splits_v5/all_validation_accessions.txt"
OUTPUT_DIR="results/validation_predictions_bioproject_v5_full"
MAX_MEMORY=512000
MEMORY_TIERS=(32000 64000 128000 256000 512000)

declare -a TIER_32GB=()
declare -a TIER_64GB=()
declare -a TIER_128GB=()
declare -a TIER_256GB=()
declare -a TIER_512GB=()

TOTAL_SAMPLES=$(wc -l < "$ACCESSIONS")
echo "Analyzing ${TOTAL_SAMPLES} samples and determining memory requirements..."

for TASK_ID in $(seq 1 $TOTAL_SAMPLES); do
    RUN_ACCESSION=$(sed -n "${TASK_ID}p" "$ACCESSIONS")

    SAMPLE_DIR="${OUTPUT_DIR}/${RUN_ACCESSION}"
    PREDICTION_FILE="${SAMPLE_DIR}/${RUN_ACCESSION}_predictions.json"
    JOBINFO_FILE="${SAMPLE_DIR}/.jobinfo"
    MEMORY_HISTORY="${SAMPLE_DIR}/.memory_history"

    # Skip completed
    if [ -f "$PREDICTION_FILE" ] && [ -f "$JOBINFO_FILE" ]; then
        if grep -q '"status": "SUCCESS"' "$JOBINFO_FILE" 2>/dev/null; then
            continue
        fi
    fi

    # Determine memory tier
    if [ -f "$MEMORY_HISTORY" ]; then
        LAST_MEM=$(tail -n 1 "$MEMORY_HISTORY")
        NEXT_MEM=$((LAST_MEM * 2))
        if [ $NEXT_MEM -gt $MAX_MEMORY ]; then
            echo "WARNING: $RUN_ACCESSION exceeded max memory. Skipping."
            continue
        fi
    else
        NEXT_MEM=32000
    fi

    if [ $NEXT_MEM -le 32000 ]; then
        TIER_32GB+=($TASK_ID)
    elif [ $NEXT_MEM -le 64000 ]; then
        TIER_64GB+=($TASK_ID)
    elif [ $NEXT_MEM -le 128000 ]; then
        TIER_128GB+=($TASK_ID)
    elif [ $NEXT_MEM -le 256000 ]; then
        TIER_256GB+=($TASK_ID)
    else
        TIER_512GB+=($TASK_ID)
    fi
done

echo ""
echo "Sample distribution by memory tier:"
echo "  32GB:  ${#TIER_32GB[@]} samples"
echo "  64GB:  ${#TIER_64GB[@]} samples"
echo "  128GB: ${#TIER_128GB[@]} samples"
echo "  256GB: ${#TIER_256GB[@]} samples"
echo "  512GB: ${#TIER_512GB[@]} samples"
echo ""

declare -a SUBMITTED_JOBS=()

submit_tier() {
    local MEMORY=$1
    local TIER_NAME=$2
    shift 2
    local SAMPLES=("$@")

    if [ ${#SAMPLES[@]} -eq 0 ]; then
        echo "No samples for ${TIER_NAME} tier - skipping"
        return
    fi

    local NUM_SAMPLES=${#SAMPLES[@]}
    local TASK_LIST=$(IFS=,; echo "${SAMPLES[*]}")

    echo "Submitting ${TIER_NAME} tier: ${NUM_SAMPLES} samples @ ${MEMORY}MB"

    JOB_ID=$(sbatch \
        --array=${TASK_LIST}%10 \
        --mem=${MEMORY} \
        --job-name=diana-v5-${TIER_NAME} \
        --output=logs/validation/diana_v5_%A_%a.out \
        --error=logs/validation/diana_v5_%A_%a.err \
        --cpus-per-task=6 \
        --partition=edid_rtx6000 \
        --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify \
        scripts/validation/05_run_predictions_single_v5.sbatch | awk '{print $4}')

    echo "  → Job ID: $JOB_ID"
    SUBMITTED_JOBS+=($JOB_ID)
}

submit_tier 32000  "32GB"  "${TIER_32GB[@]}"
submit_tier 64000  "64GB"  "${TIER_64GB[@]}"
submit_tier 128000 "128GB" "${TIER_128GB[@]}"
submit_tier 256000 "256GB" "${TIER_256GB[@]}"
submit_tier 512000 "512GB" "${TIER_512GB[@]}"

echo ""
echo "========================================"
if [ ${#SUBMITTED_JOBS[@]} -eq 0 ]; then
    echo "No jobs submitted - all samples already completed!"
else
    echo "Submitted ${#SUBMITTED_JOBS[@]} job(s): ${SUBMITTED_JOBS[*]}"
fi
echo ""
echo "Monitor with: squeue -u $USER"
echo "After jobs complete, re-run this script to retry OOM failures."
echo ""
echo "When all done, update config.py predictions_dir to:"
echo "  results/validation_predictions_bioproject_v5_full"
echo "then re-run: bash scripts/paper/generate_all_paper_materials.sh"
echo "========================================"
