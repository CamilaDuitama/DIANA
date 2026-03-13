#!/bin/bash
#
# Submit validation predictions with automatic OOM retry logic
#
# This script:
# 1. Scans all 957 validation samples
# 2. Groups by required memory tier (32GB → 64GB → 128GB → 256GB → 512GB)
# 3. Submits separate SLURM array jobs for each tier
# 4. Can be re-run to automatically retry OOM failures
#
# Usage:
#   bash scripts/validation/submit_validation_with_retry.sh
#
# The script is idempotent - completed predictions are skipped.
# After OOM failures, re-run this script to retry with doubled memory.

set -e

echo "========================================"
echo "DIANA Validation Predictions with Retry"
echo "========================================"
echo "Started: $(date)"
echo ""

METADATA="paper/metadata/validation_metadata.tsv"
OUTPUT_DIR="results/validation_predictions"
MAX_MEMORY=512000  # 512GB in MB
MEMORY_TIERS=(32000 64000 128000 256000 512000)  # Memory in MB

# Create arrays to hold sample indices for each tier
declare -a TIER_32GB=()
declare -a TIER_64GB=()
declare -a TIER_128GB=()
declare -a TIER_256GB=()
declare -a TIER_512GB=()

echo "Analyzing samples and determining memory requirements..."

# Read metadata and classify each sample
TOTAL_SAMPLES=$(wc -l < "$METADATA")
TOTAL_SAMPLES=$((TOTAL_SAMPLES - 1))  # Exclude header

for TASK_ID in $(seq 1 $TOTAL_SAMPLES); do
    # Get run accession (column 2)
    SAMPLE_LINE=$(sed -n "$((TASK_ID + 1))p" "$METADATA")
    RUN_ACCESSION=$(echo "$SAMPLE_LINE" | cut -f2)
    
    SAMPLE_DIR="${OUTPUT_DIR}/${RUN_ACCESSION}"
    PREDICTION_FILE="${SAMPLE_DIR}/${RUN_ACCESSION}_predictions.json"
    JOBINFO_FILE="${SAMPLE_DIR}/.jobinfo"
    MEMORY_HISTORY="${SAMPLE_DIR}/.memory_history"
    
    # Skip if prediction exists AND was successful
    if [ -f "$PREDICTION_FILE" ] && [ -f "$JOBINFO_FILE" ]; then
        if grep -q '"status": "SUCCESS"' "$JOBINFO_FILE" 2>/dev/null; then
            continue
        fi
    fi
    
    # Determine required memory
    if [ -f "$MEMORY_HISTORY" ]; then
        # Get last attempted memory
        LAST_MEM=$(tail -n 1 "$MEMORY_HISTORY")
        
        # Double the memory for next attempt
        NEXT_MEM=$((LAST_MEM * 2))
        
        # Cap at max memory
        if [ $NEXT_MEM -gt $MAX_MEMORY ]; then
            echo "WARNING: Sample $RUN_ACCESSION exceeded max memory ($MAX_MEMORY MB). Skipping."
            continue
        fi
    else
        # First attempt - start with 32GB
        NEXT_MEM=32000
    fi
    
    # Assign to appropriate tier
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

# Track submitted job IDs
declare -a SUBMITTED_JOBS=()

# Function to submit array job for a tier
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
    
    # Convert array to comma-separated list for SLURM --array
    local TASK_LIST=$(IFS=,; echo "${SAMPLES[*]}")
    
    # For large lists, truncate for display
    if [ $NUM_SAMPLES -le 20 ]; then
        echo "Submitting ${TIER_NAME} tier: ${NUM_SAMPLES} samples with ${MEMORY}MB memory (tasks: $TASK_LIST)"
    else
        # Show first 10 and last 10 task IDs
        local FIRST_TEN="${SAMPLES[@]:0:10}"
        local LAST_TEN_START=$((NUM_SAMPLES - 10))
        local LAST_TEN="${SAMPLES[@]:$LAST_TEN_START:10}"
        echo "Submitting ${TIER_NAME} tier: ${NUM_SAMPLES} samples with ${MEMORY}MB memory"
        echo "  First 10 tasks: $(IFS=,; echo "${FIRST_TEN// /,}")"
        echo "  Last 10 tasks: $(IFS=,; echo "${LAST_TEN// /,}")"
    fi
    
    # Submit with specific task IDs (no mapping file needed!)
    JOB_ID=$(sbatch \
        --array=${TASK_LIST}%10 \
        --mem=${MEMORY} \
        --job-name=diana-val-${TIER_NAME} \
        --output=logs/validation/diana_predict_%A_%a.out \
        --error=logs/validation/diana_predict_%A_%a.err \
        --cpus-per-task=6 \
        --partition=common \
        --exclude=maestro-2010 \
        scripts/validation/05_run_predictions_single.sbatch | awk '{print $4}')
    
    echo "  → Job ID: $JOB_ID"
    
    SUBMITTED_JOBS+=($JOB_ID)
}

# Submit jobs for each tier
submit_tier 32000 "32GB" "${TIER_32GB[@]}"
submit_tier 64000 "64GB" "${TIER_64GB[@]}"
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
if [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    echo "Check efficiency: reportseff ${SUBMITTED_JOBS[0]}"
fi
echo ""
echo "After jobs complete, re-run this script to retry OOM failures."
echo "========================================"
