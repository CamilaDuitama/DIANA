#!/bin/bash
# DIANA inference pipeline for processing new samples
# Runs k-mer counting → unitig aggregation → model prediction

set -e

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Redirect all output to log file in output directory
if [ "$#" -ge 1 ]; then
    CONFIG_FILE=$1
    source "$CONFIG_FILE"
    SAMPLE_NAME=$(basename "$SAMPLE_FASTQ" | sed -E 's/(\.f(ast)?q(\.gz)?|\.fq(\.gz)?)$//')
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    : ${OUTPUT_DIR:?Error: OUTPUT_DIR not set}
    LOG_FILE="$OUTPUT_DIR/inference_pipeline.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

timestamp() {
    date '+[%Y-%m-%d %H:%M:%S]'
}

# ============================================================================
# BINARY COMPATIBILITY CHECK & REBUILD
# ============================================================================
# On heterogeneous clusters, pre-compiled binaries may not work on all nodes
# due to different CPU architectures. Check compatibility and rebuild if needed.

# Check back_to_sequences (Rust binary for k-mer extraction)
B2S_PATH="$SCRIPT_DIR/../../external/back_to_sequences/target/release/back_to_sequences"
NEED_B2S_BUILD=false

if [ ! -x "$B2S_PATH" ]; then
    echo "$(timestamp) [BUILD] back_to_sequences binary not found."
    NEED_B2S_BUILD=true
elif ! "$B2S_PATH" --help >/dev/null 2>&1; then
    echo "$(timestamp) [BUILD] back_to_sequences binary incompatible with this node."
    NEED_B2S_BUILD=true
else
    echo "$(timestamp) [BUILD] back_to_sequences is compatible, skipping rebuild."
fi

if [ "$NEED_B2S_BUILD" = true ]; then
    echo "$(timestamp) [BUILD] Building back_to_sequences for this node..."
    (
        cd "$SCRIPT_DIR/../../external/back_to_sequences" || exit 2
        if command -v module >/dev/null 2>&1; then
            module load llvm/ || true
        fi
        export CC=$(which clang)
        rm -rf target
        cargo clean
        cargo build --release
    )
    if [ ! -x "$B2S_PATH" ]; then
        echo "[ERROR] Failed to build back_to_sequences. Exiting." >&2
        exit 2
    fi
    echo "$(timestamp) [BUILD] back_to_sequences built successfully."
fi

# Check MUSET tools (C++ binaries for k-mer matrix operations)
MUSET_DIR="$SCRIPT_DIR/../../external/muset"
KMAT_TOOLS="$MUSET_DIR/bin/kmat_tools"
NEED_MUSET_BUILD=false

if [ ! -x "$KMAT_TOOLS" ]; then
    echo "$(timestamp) [BUILD] kmat_tools binary not found."
    NEED_MUSET_BUILD=true
else
    # Test binary compatibility: check if --version runs without "Illegal instruction"
    # Note: kmat_tools returns exit code 1 even on success, so check output instead
    COMPAT_TEST=$("$KMAT_TOOLS" --version 2>&1 || true)
    if echo "$COMPAT_TEST" | grep -qi "illegal"; then
        echo "$(timestamp) [BUILD] kmat_tools incompatible (illegal instruction)."
        NEED_MUSET_BUILD=true
    elif echo "$COMPAT_TEST" | grep -q "v0\."; then
        echo "$(timestamp) [BUILD] kmat_tools compatible, skipping rebuild."
        NEED_MUSET_BUILD=false
    else
        echo "$(timestamp) [BUILD] kmat_tools test failed, rebuilding."
        NEED_MUSET_BUILD=true
    fi
fi

if [ "$NEED_MUSET_BUILD" = true ]; then
    echo "$(timestamp) [BUILD] Building MUSET tools for this node..."
    (
        cd "$MUSET_DIR" || exit 2
        if command -v module >/dev/null 2>&1; then
            module load cmake/3.26.3 || module load cmake || true
            module load gcc/13.2.0 || true
        fi
        rm -rf build
        mkdir -p build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j4
    )
    if [ ! -x "$KMAT_TOOLS" ]; then
        echo "[ERROR] Failed to build MUSET tools. Exiting." >&2
        exit 2
    fi
    echo "$(timestamp) [BUILD] MUSET tools built successfully."
fi

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

# Validate required variables
: ${MUSET_OUTPUT_DIR:?Error: MUSET_OUTPUT_DIR not set}
: ${MODEL_PATH:?Error: MODEL_PATH not set}
: ${SAMPLE_FASTQ:?Error: SAMPLE_FASTQ not set}
: ${OUTPUT_DIR:?Error: OUTPUT_DIR not set}
: ${K:=31}
: ${THREADS:=4}
: ${MIN_ABUNDANCE:=2}  # Filter sequencing errors (k-mers with abundance < 2)

# Validate input files
if [ ! -f "$SAMPLE_FASTQ" ]; then
    echo "[ERROR] Input FASTQ file not found: $SAMPLE_FASTQ" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

SAMPLE_NAME=$(basename "$SAMPLE_FASTQ" | sed -E 's/(\.f(ast)?q(\.gz)?|\.fq(\.gz)?)$//')
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
KMER_COUNTS="$OUTPUT_DIR/${SAMPLE_NAME}_kmer_counts.txt"
time_step "Step 1: Counting k-mers in sample" \
    bash "$SCRIPT_DIR/01_count_kmers.sh" \
        "$REFERENCE_KMERS" \
        "$SAMPLE_FASTQ" \
        "$KMER_COUNTS" \
        "$THREADS" \
        "$MIN_ABUNDANCE"
echo ""

# Step 2: Aggregate k-mer counts to unitig-level abundances/fractions
UNITIGS_FA="$MUSET_OUTPUT_DIR/unitigs.fa"
OUT_ABUNDANCE="$OUTPUT_DIR/${SAMPLE_NAME}_unitig_abundance.txt"
OUT_FRACTION="$OUTPUT_DIR/${SAMPLE_NAME}_unitig_fraction.txt"
time_step "Step 2: Aggregating k-mers to unitigs" \
    bash "$SCRIPT_DIR/02_aggregate_to_unitigs.sh" \
        "$KMER_COUNTS" \
        "$UNITIGS_FA" \
        "$K" \
        "$OUT_ABUNDANCE" \
        "$OUT_FRACTION"
echo ""

# Step 3: Run trained model on unitig fractions to predict OM composition
PREDICTIONS_JSON="$OUTPUT_DIR/${SAMPLE_NAME}_predictions.json"
time_step "Step 3: Running model inference" \
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

# Step 4: Generate visualization plots
PLOTS_DIR="$OUTPUT_DIR/plots"
time_step "Step 4: Plotting results" \
    python "$SCRIPT_DIR/04_plot_results.py" \
        --predictions "$PREDICTIONS_JSON" \
        --output_dir "$PLOTS_DIR" \
        --sample_id "$SAMPLE_NAME"
echo "Barplots saved in: $PLOTS_DIR"
echo "============================================"
