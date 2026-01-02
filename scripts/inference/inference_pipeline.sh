#!/bin/bash
# DIANA inference pipeline for processing new samples
# Runs k-mer counting → unitig aggregation → model prediction

set -e

timestamp() {
    date '+[%Y-%m-%d %H:%M:%S]'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

time_step() {
    local step_name="$1"
    shift
    local start_time=$(date +%s)
    echo "$(timestamp) Starting: $step_name"
    "$@"
    local status=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "$(timestamp) Finished: $step_name (Duration: ${duration}s, Status: $status)"
    return $status
}

# ============================================================================
# CONFIGURATION & VALIDATION
# ============================================================================

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <sample_fastq1> [sample_fastq2 ...] <sample_id>"
    echo ""
    echo "Run DIANA inference on a new sample"
    echo ""
    echo "Arguments:"
    echo "  sample_fastq1   - First FASTQ file (required)"
    echo "  sample_fastq2   - Second FASTQ file (optional, for paired-end)"
    echo "  sample_id       - Sample identifier (last argument)"
    echo ""
    echo "Environment variables (optional):"
    echo "  MUSET_MATRIX_DIR - MUSET matrix directory (default: data/matrices/large_matrix_3070_with_frac)"
    echo "  MODEL_PATH      - Model checkpoint (default: results/training/best_model.pth)"
    echo "  OUTPUT_DIR      - Output directory (default: results/inference/<sample_id>)"
    echo "  KMER_SIZE       - K-mer size (default: 31)"
    echo "  MIN_ABUNDANCE   - Minimum k-mer count (default: 2)"
    echo "  THREADS         - Number of threads (default: 10)"
    exit 1
fi

# Parse arguments - all but last are FASTQ files, last is sample_id
SAMPLE_ID="${@: -1}"  # Last argument
FASTQ_FILES=("${@:1:$#-1}")  # All but last argument

# Set defaults from environment or use hardcoded defaults
MUSET_OUTPUT_DIR="${MUSET_MATRIX_DIR:-data/matrices/large_matrix_3070_with_frac}"
MODEL_PATH="${MODEL_PATH:-results/training/best_model.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/inference/$SAMPLE_ID}"
K="${KMER_SIZE:-31}"
THREADS="${THREADS:-10}"
MIN_ABUNDANCE="${MIN_ABUNDANCE:-2}"

# Validate input files
for FASTQ in "${FASTQ_FILES[@]}"; do
    if [ ! -f "$FASTQ" ]; then
        echo "[ERROR] Input FASTQ file not found: $FASTQ" >&2
        exit 2
    fi
done
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "   DIANA Inference Pipeline"
echo "============================================"
echo "Sample: $SAMPLE_ID"
echo "Input files: ${#FASTQ_FILES[@]}"
for i in "${!FASTQ_FILES[@]}"; do
    echo "  [$((i+1))] $(basename ${FASTQ_FILES[$i]})"
done
echo "MUSET output: $MUSET_OUTPUT_DIR"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "K-mer size: $K"
echo "Minimum abundance: $MIN_ABUNDANCE"
echo "Threads: $THREADS"
echo "============================================"
echo ""

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

# Step 0: Extract reference k-mers from training matrix (one-time setup)
REFERENCE_KMERS="$OUTPUT_DIR/reference_kmers.fasta"
if [ ! -f "$REFERENCE_KMERS" ]; then
    time_step "Step 0: Extracting reference k-mers" \
        bash "$SCRIPT_DIR/00_extract_reference_kmers.sh" \
            "$MUSET_OUTPUT_DIR" \
            "$REFERENCE_KMERS"
    echo ""
else
    echo "$(timestamp) Step 0: Reference k-mers already extracted"
    echo ""
fi

# Step 1: Count reference k-mers in new sample using back_to_sequences
KMER_COUNTS="$OUTPUT_DIR/${SAMPLE_ID}_kmer_counts.txt"

# If multiple FASTQ files, create a file list for back_to_sequences
if [ "${#FASTQ_FILES[@]}" -gt 1 ]; then
    FASTQ_FILELIST="$OUTPUT_DIR/${SAMPLE_ID}_fastq_filelist.txt"
    printf "%s\n" "${FASTQ_FILES[@]}" > "$FASTQ_FILELIST"
    KMER_INPUT="$FASTQ_FILELIST"
else
    KMER_INPUT="${FASTQ_FILES[0]}"
fi

time_step "Step 1: Counting k-mers in sample" \
    bash "$SCRIPT_DIR/01_count_kmers.sh" \
        "$REFERENCE_KMERS" \
        "$KMER_INPUT" \
        "$KMER_COUNTS" \
        "$THREADS" \
        "$MIN_ABUNDANCE"
echo ""

# Step 2: Aggregate k-mer counts to unitig-level abundances/fractions
UNITIGS_FA="$MUSET_OUTPUT_DIR/unitigs.fa"
OUT_ABUNDANCE="$OUTPUT_DIR/${SAMPLE_ID}_unitig_abundance.txt"
OUT_FRACTION="$OUTPUT_DIR/${SAMPLE_ID}_unitig_fraction.txt"
time_step "Step 2: Aggregating k-mers to unitigs" \
    bash "$SCRIPT_DIR/02_aggregate_to_unitigs.sh" \
        "$KMER_COUNTS" \
        "$UNITIGS_FA" \
        "$K" \
        "$OUT_ABUNDANCE" \
        "$OUT_FRACTION"
echo ""

# Step 3: Run trained model on unitig fractions to predict OM composition
PREDICTIONS_JSON="$OUTPUT_DIR/${SAMPLE_ID}_predictions.json"
time_step "Step 3: Running model inference" \
    python "$SCRIPT_DIR/03_run_inference.py" \
        --model "$MODEL_PATH" \
        --input "$OUT_FRACTION" \
        --output "$PREDICTIONS_JSON" \
        --sample-id "$SAMPLE_ID"
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


# Step 4: Plot results
PLOTS_DIR="$OUTPUT_DIR/plots"
time_step "Step 4: Plotting results" \
    python "$SCRIPT_DIR/04_plot_results.py" \
        --predictions "$PREDICTIONS_JSON" \
        --output_dir "$PLOTS_DIR" \
        --sample_id "$SAMPLE_ID"
echo "Barplots saved in: $PLOTS_DIR"
echo "============================================"
