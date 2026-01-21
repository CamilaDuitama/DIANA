#!/bin/bash
# Pre-filter validation samples to only submit jobs for missing predictions

set -e

METADATA="paper/metadata/validation_metadata.tsv"
OUTPUT_DIR="results/validation_predictions"
MISSING_LIST="/tmp/missing_predictions_$$.txt"

echo "Checking which predictions are missing..."

# Extract all run_accessions from metadata (skip header, get column 17)
tail -n +2 "$METADATA" | cut -f17 > /tmp/all_samples_$$.txt

# Check which predictions already exist (faster with find)
> "$MISSING_LIST"
TOTAL=$(wc -l < /tmp/all_samples_$$.txt)

# Get list of existing predictions
find "$OUTPUT_DIR" -name "*_predictions.json" | sed 's|.*/\([^/]*\)_predictions.json|\1|' | sort > /tmp/existing_$$.txt

# Find missing samples (in metadata but not in predictions)
comm -23 <(sort /tmp/all_samples_$$.txt) /tmp/existing_$$.txt > "$MISSING_LIST"

EXISTING=$(wc -l < /tmp/existing_$$.txt)
MISSING=$(wc -l < "$MISSING_LIST")

echo ""
echo "Summary:"
echo "  Total samples: $TOTAL"
echo "  Existing predictions: $EXISTING"
echo "  Missing predictions: $MISSING"
echo ""

if [ $MISSING -eq 0 ]; then
    echo "All predictions complete! Nothing to submit."
    rm -f /tmp/all_samples_$$.txt "$MISSING_LIST"
    exit 0
fi

echo "Missing samples list saved to: $MISSING_LIST"
echo ""
echo "Creating filtered array mapping..."

# Create a mapping file: array_index -> run_accession
MAPPING_FILE="data/validation/missing_samples_mapping.txt"
> "$MAPPING_FILE"

INDEX=1
while IFS= read -r RUN_ACCESSION; do
    echo "$INDEX $RUN_ACCESSION" >> "$MAPPING_FILE"
    ((INDEX++))
done < "$MISSING_LIST"

echo "Mapping file created: $MAPPING_FILE"
echo "Array size: $MISSING"
echo ""

# Submit job with filtered array
echo "Submitting SLURM array job for $MISSING missing samples..."
sbatch --array=1-${MISSING}%10 scripts/validation/05_run_predictions_filtered.sbatch

# Cleanup temp files
rm -f /tmp/all_samples_$$.txt /tmp/existing_$$.txt "$MISSING_LIST"

echo "Job submitted!"
