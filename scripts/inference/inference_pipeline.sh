#!/bin/bash
# Complete inference pipeline for a single new sample

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_file>"
    echo ""
    echo "Config file should define:"
    echo "  MUSET_OUTPUT_DIR - Path to MUSET training output"
    echo "  SAMPLE_FASTQ - New sample FASTQ file"
    echo "  OUTPUT_DIR - Output directory"
    echo "  K - K-mer size"
    echo "  THREADS - Number of threads"
    echo "  MIN_ABUNDANCE - Minimum k-mer count (default: 2, filters sequencing errors)"
    exit 1
fi

CONFIG_FILE=$1
source "$CONFIG_FILE"

# Validate config
: ${MUSET_OUTPUT_DIR:?Error: MUSET_OUTPUT_DIR not set}
: ${SAMPLE_FASTQ:?Error: SAMPLE_FASTQ not set}
: ${OUTPUT_DIR:?Error: OUTPUT_DIR not set}
: ${K:?Error: K not set}
: ${THREADS:=4}
: ${MIN_ABUNDANCE:=2}  # Default: filter k-mers with count < 2

SAMPLE_NAME=$(basename "$SAMPLE_FASTQ" | sed 's/\.[^.]*$//')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "   DIANA Inference Pipeline"
echo "============================================"
echo "Sample: $SAMPLE_NAME"
echo "MUSET output: $MUSET_OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "K-mer size: $K"
echo "Minimum abundance: $MIN_ABUNDANCE"
echo "Threads: $THREADS"
echo "============================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 0: Extract reference k-mers (if not already done)
REFERENCE_KMERS="$OUTPUT_DIR/reference_kmers.fasta"
if [ ! -f "$REFERENCE_KMERS" ]; then
    echo ">>> Step 0: Extracting reference k-mers..."
    bash "$SCRIPT_DIR/00_extract_reference_kmers.sh" \
        "$MUSET_OUTPUT_DIR" \
        "$REFERENCE_KMERS"
    echo ""
else
    echo ">>> Step 0: Reference k-mers already extracted"
    echo ""
fi

# Step 1: Count k-mers in new sample
echo ">>> Step 1: Counting k-mers in sample..."
KMER_COUNTS="$OUTPUT_DIR/${SAMPLE_NAME}_kmer_counts.txt"
bash "$SCRIPT_DIR/01_count_kmers.sh" \
    "$REFERENCE_KMERS" \
    "$SAMPLE_FASTQ" \
    "$KMER_COUNTS" \
    "$THREADS" \
    "$MIN_ABUNDANCE"
echo ""

# Step 2: Aggregate to unitigs
echo ">>> Step 2: Aggregating k-mers to unitigs..."
UNITIGS_FA="$MUSET_OUTPUT_DIR/unitigs.fa"
OUT_ABUNDANCE="$OUTPUT_DIR/${SAMPLE_NAME}_unitig_abundance.txt"
OUT_FRACTION="$OUTPUT_DIR/${SAMPLE_NAME}_unitig_fraction.txt"

bash "$SCRIPT_DIR/02_aggregate_to_unitigs.sh" \
    "$KMER_COUNTS" \
    "$UNITIGS_FA" \
    "$K" \
    "$OUT_ABUNDANCE" \
    "$OUT_FRACTION"
echo ""

echo "============================================"
echo "   Pipeline Complete!"
echo "============================================"
echo "Outputs:"
echo "  - Abundance: $OUT_ABUNDANCE"
echo "  - Fraction: $OUT_FRACTION"
echo "============================================"
