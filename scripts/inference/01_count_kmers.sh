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

# Check if back_to_sequences is available in PATH
if ! command -v back_to_sequences >/dev/null 2>&1; then
    echo "[ERROR] back_to_sequences not found in PATH"
    echo "Please ensure back_to_sequences is installed and available"
    exit 2
fi

# Check if input is a file list based on extension
if [[ "$SAMPLE_INPUT" == *.txt ]] || [[ "$SAMPLE_INPUT" == *.list ]]; then
    # File list: use seqkit concat and pipe to back_to_sequences via stdin
    seqkit concat $(cat "$SAMPLE_INPUT") | back_to_sequences \
        --in-kmers "$REFERENCE_KMERS" \
        --out-kmers "$OUTPUT_COUNTS" \
        --counted-kmer-threshold "$MIN_ABUNDANCE" \
        --threads "$THREADS"
else
    # Single FASTQ file
    back_to_sequences \
        --in-kmers "$REFERENCE_KMERS" \
        --in-sequences "$SAMPLE_INPUT" \
        --out-kmers "$OUTPUT_COUNTS" \
        --counted-kmer-threshold "$MIN_ABUNDANCE" \
        --threads "$THREADS"
fi

echo "✓ Done!"
