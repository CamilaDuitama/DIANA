#!/bin/bash
# Step 1: Count reference k-mers in new sample using back_to_sequences

set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <reference_kmers_fasta> <sample_fastq> <output_counts> <threads> [min_abundance]"
    echo ""
    echo "Count reference k-mers in a new sample"
    echo ""
    echo "Arguments:"
    echo "  min_abundance - Minimum k-mer count to consider present (default: 2)"
    echo "                  K-mers with count < min_abundance are set to 0"
    echo "                  This filters sequencing errors"
    exit 1
fi

REFERENCE_KMERS=$1
SAMPLE_FASTQ=$2
OUTPUT_COUNTS=$3
THREADS=$4
MIN_ABUNDANCE=${5:-2}  # Default minimum abundance = 2 (filter sequencing errors)

SAMPLE_NAME=$(basename "$SAMPLE_FASTQ" | sed 's/\.[^.]*$//')

echo "=== Step 1: Count K-mers in Sample ==="
echo "Sample: $SAMPLE_NAME"
echo "Reference k-mers: $REFERENCE_KMERS"
echo "Minimum abundance: $MIN_ABUNDANCE"
echo "Output: $OUTPUT_COUNTS"

# Run back_to_sequences
TMP_COUNTS="${OUTPUT_COUNTS}.tmp"
~/.local/bin/back_to_sequences \
    --in-kmers "$REFERENCE_KMERS" \
    --in-sequences "$SAMPLE_FASTQ" \
    --out-kmers "$TMP_COUNTS" \
    --threads "$THREADS"

# Filter k-mers by minimum abundance (avoid sequencing errors)
# Set count to 0 if below threshold
awk -v min_ab="$MIN_ABUNDANCE" '{
    if ($2 >= min_ab) {
        print $0
    } else {
        print $1, 0
    }
}' "$TMP_COUNTS" > "$OUTPUT_COUNTS"

rm "$TMP_COUNTS"

# Report filtering stats
TOTAL_KMERS=$(wc -l < "$OUTPUT_COUNTS")
KEPT_KMERS=$(awk '$2 > 0' "$OUTPUT_COUNTS" | wc -l)
FILTERED_KMERS=$((TOTAL_KMERS - KEPT_KMERS))
echo "K-mers with count >= $MIN_ABUNDANCE: $KEPT_KMERS"
echo "K-mers filtered (count < $MIN_ABUNDANCE): $FILTERED_KMERS"

echo "✓ Done!"
