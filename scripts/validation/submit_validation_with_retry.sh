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
    # Get run accession
    SAMPLE_LINE=$(sed -n "$((TASK_ID + 1))p" "$METADATA")
    RUN_ACCESSION=$(echo "$SAMPLE_LINE" | cut -f17)
    
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
    
    # Create temporary task mapping file
    MAPPING_FILE=$(mktemp)
    for i in "${!SAMPLES[@]}"; do
        echo "$((i + 1)) ${SAMPLES[$i]}" >> "$MAPPING_FILE"
    done
    
    local ARRAY_SIZE=${#SAMPLES[@]}
    
    echo "Submitting ${TIER_NAME} tier: ${ARRAY_SIZE} samples with ${MEMORY}MB memory"
    
    # Create modified sbatch script that reads from mapping file
    TEMP_SBATCH=$(mktemp --suffix=.sbatch)
    cat > "$TEMP_SBATCH" <<'SBATCH_SCRIPT'
#!/bin/bash
#SBATCH --job-name=diana-val-TIER
#SBATCH --output=logs/validation/diana_predict_%A_%a.out
#SBATCH --error=logs/validation/diana_predict_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --partition=seqbio
#SBATCH --exclude=maestro-2010

# Read actual task ID from mapping file
ACTUAL_TASK_ID=$(awk -v idx="$SLURM_ARRAY_TASK_ID" '$1 == idx {print $2}' "$TASK_MAPPING")

# Export for the main script to use
export OVERRIDE_TASK_ID=$ACTUAL_TASK_ID

# Run main prediction script
bash scripts/validation/05_run_predictions_single.sbatch
SBATCH_SCRIPT
    
    # Replace TIER placeholder
    sed -i "s/diana-val-TIER/diana-val-${TIER_NAME}/" "$TEMP_SBATCH"
    
    # Submit with memory override and mapping file
    JOB_ID=$(sbatch \
        --array=1-${ARRAY_SIZE}%10 \
        --mem=${MEMORY} \
        --export=TASK_MAPPING="$MAPPING_FILE" \
        "$TEMP_SBATCH" | awk '{print $4}')
    
    echo "  → Job ID: $JOB_ID"
    echo "  → Mapping file: $MAPPING_FILE (saved for job duration)"
    
    # Save mapping file with job ID for cleanup later
    cp "$MAPPING_FILE" "data/validation/task_mapping_${JOB_ID}.txt"
    
    rm "$TEMP_SBATCH"
}

# Submit jobs for each tier
submit_tier 32000 "32GB" "${TIER_32GB[@]}"
submit_tier 64000 "64GB" "${TIER_64GB[@]}"
submit_tier 128000 "128GB" "${TIER_128GB[@]}"
submit_tier 256000 "256GB" "${TIER_256GB[@]}"
submit_tier 512000 "512GB" "${TIER_512GB[@]}"

echo ""
echo "========================================"
echo "All jobs submitted!"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check efficiency: reportseff <job_id>"
echo ""
echo "After jobs complete, re-run this script to retry OOM failures."
echo "========================================"
