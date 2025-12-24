#!/bin/bash
# Complete inference pipeline for a single new sample

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_file>"
    echo ""
    echo "Config file should define:"
    echo "  MUSET_OUTPUT_DIR - Path to MUSET training output (e.g., data/matrices/large_matrix_3070_with_frac)"
    echo "  MODEL_PATH - Path to trained model checkpoint (e.g., results/training/best_model.pth)"
    echo "  SAMPLE_FASTQ - New sample FASTQ file"
    echo "  OUTPUT_DIR - Output directory"
    echo "  K - K-mer size (default: 31)"
    echo "  THREADS - Number of threads (default: 4)"
    echo "  MIN_ABUNDANCE - Minimum k-mer count (default: 2, filters sequencing errors)"
    exit 1
fi

CONFIG_FILE=$1
source "$CONFIG_FILE"

# Validate config
: ${MUSET_OUTPUT_DIR:?Error: MUSET_OUTPUT_DIR not set}
: ${MODEL_PATH:?Error: MODEL_PATH not set}
: ${SAMPLE_FASTQ:?Error: SAMPLE_FASTQ not set}
: ${OUTPUT_DIR:?Error: OUTPUT_DIR not set}
: ${K:=31}
: ${THREADS:=4}
: ${MIN_ABUNDANCE:=2}  # Default: filter k-mers with count < 2

SAMPLE_NAME=$(basename "$SAMPLE_FASTQ" | sed 's/\.[^.]*$//')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "   DIANA Inference Pipeline"
echo "============================================"
echo "Sample: $SAMPLE_NAME"
echo "MUSET output: $MUSET_OUTPUT_DIR"
echo "Model: $MODEL_PATH"
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

# Step 3: Run model inference
echo ">>> Step 3: Running model inference..."
PREDICTIONS_JSON="$OUTPUT_DIR/${SAMPLE_NAME}_predictions.json"

python "$SCRIPT_DIR/03_run_inference.py" \
    --model "$MODEL_PATH" \
    --input "$OUT_FRACTION" \
    --output "$PREDICTIONS_JSON" \
    --sample-id "$SAMPLE_NAME"
echo ""

echo "============================================"
echo "   Pipeline Complete!"
echo "============================================"
echo "Outputs:"
echo "  - K-mer counts: $KMER_COUNTS"
echo "  - Unitig abundance: $OUT_ABUNDANCE"
echo "  - Unitig fractions: $OUT_FRACTION"
echo "  - Predictions: $PREDICTIONS_JSON"
echo "============================================"
echo ""
echo "View predictions:"
echo "  cat $PREDICTIONS_JSON"
echo "============================================"
