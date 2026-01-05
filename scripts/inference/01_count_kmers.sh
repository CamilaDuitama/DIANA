#!/bin/bash
# Step 1: Count reference k-mers in new sample using back_to_sequences

set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <reference_kmers_fasta> <sample_fastq_or_filelist> <output_counts> <threads> [min_abundance]"
    echo ""
    echo "Count reference k-mers in a new sample"
    echo ""
    echo "Arguments:"
    echo "  sample_fastq_or_filelist - Single FASTQ file OR path to file list (one FASTQ per line)"
    echo "  min_abundance - Minimum k-mer count to consider present (default: 2)"
    echo "                  K-mers with count < min_abundance are set to 0"
    echo "                  This filters sequencing errors"
    exit 1
fi

REFERENCE_KMERS=$1
SAMPLE_INPUT=$2  # Can be single FASTQ or file list
OUTPUT_COUNTS=$3
THREADS=$4
MIN_ABUNDANCE=${5:-2}  # Default minimum abundance = 2 (filter sequencing errors)

SAMPLE_NAME=$(basename "$SAMPLE_INPUT" | sed 's/\.[^.]*$//' | sed 's/_filelist$//')

# Use back_to_sequences from external directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B2S_PATH="$SCRIPT_DIR/../../external/back_to_sequences/target/release/back_to_sequences"

if [ ! -x "$B2S_PATH" ]; then
    echo "[ERROR] back_to_sequences not found at $B2S_PATH"
    exit 2
fi

TMP_COUNTS="${OUTPUT_COUNTS}.tmp"

# Check if input is a file list (text file with multiple FASTQ paths)
if [ -f "$SAMPLE_INPUT" ] && file "$SAMPLE_INPUT" | grep -q "ASCII text"; then
    # Check if it contains FASTQ paths (likely a file list)
    if head -1 "$SAMPLE_INPUT" | grep -qE '\.(fastq|fq)(\.gz)?$'; then
        # Use seqkit concat for robust FASTQ concatenation
        # seqkit handles compression automatically and preserves FASTQ format
        seqkit concat $(cat "$SAMPLE_INPUT") | "$B2S_PATH" \
            --in-kmers "$REFERENCE_KMERS" \
            --out-kmers "$TMP_COUNTS" \
            --threads "$THREADS"
    else
        # Single FASTQ file
        "$B2S_PATH" \
            --in-kmers "$REFERENCE_KMERS" \
            --in-sequences "$SAMPLE_INPUT" \
            --out-kmers "$TMP_COUNTS" \
            --threads "$THREADS"
    fi
else
    # Single FASTQ file
    "$B2S_PATH" \
        --in-kmers "$REFERENCE_KMERS" \
        --in-sequences "$SAMPLE_INPUT" \
        --out-kmers "$TMP_COUNTS" \
        --threads "$THREADS"
fi

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
