#!/bin/bash
# Build the v9 feature vocabulary from TRAINING SAMPLES ONLY (PROJECT.md step 10, R3.4).
#
# Copied from 03_generate_matrix_v7.sbatch; the only substantive change is the
# input file-of-files. v7 was built over data/train_test_samples_v7.fof -- train
# AND test together -- and MUSET's -f/--min-frac-absent and -F/--min-frac-present
# both default to 0.1, which is the paper's "present in 10-90 % of samples"
# window. So the test samples helped decide which unitigs exist at all, which is
# exactly R3.4.
#
# Here the input is data/train_samples_v9.fof: 2,716 v9 TRAINING runs, verified to
# contain zero test and zero val runs. Validation and test are projected onto the
# resulting vocabulary afterwards with back_to_sequences + kmat_tools unitig, the
# same per-sample path already used for the 512 validation vectors.
#
# The feature dimension will differ from v7's 78,430; that is expected.
#SBATCH --job-name=muset_v9_matrix
#SBATCH --output=logs/muset_v9_%j.out
#SBATCH --error=logs/muset_v9_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=650G
#SBATCH --partition=edid
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

# Generate MUSET matrix for v7 train+test samples (3,191 samples)
# This creates the unitig fraction matrix needed for v7 model training
# Uses modified MUSET that saves SSHash dictionary for inference

echo "========================================"
echo "MUSET v7 Matrix Generation"
echo "Train + Test samples only (3,190)"
echo "Job ${SLURM_JOB_ID}"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Paths
DIANA_ROOT="/pasteur/appa/scratch/cduitama/EDID/decOM-classify"
MUSET_BIN="$DIANA_ROOT/external/muset/bin/muset"
INPUT_FOF="${DIANA_ROOT}/data/train_samples_v9.fof"
OUTPUT_DIR="${DIANA_ROOT}/data/matrices/matrix_v9_train"

# Create output directory; clean only kmer_matrix so MUSET can start fresh
# (keeps the parent OUTPUT_DIR and any logs from previous runs)
mkdir -p "${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}/kmer_matrix"
mkdir -p "${DIANA_ROOT}/logs"

# Verify input file exists
if [[ ! -f "$INPUT_FOF" ]]; then
    echo "❌ ERROR: Input FOF file not found: $INPUT_FOF"
    echo "   Expected: 3,190 samples from v7 train+test split"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "$INPUT_FOF")
echo "Processing $SAMPLE_COUNT unitig files"

if [[ $SAMPLE_COUNT -ne 3191 ]]; then
    echo "⚠️  WARNING: Expected 3,191 samples but found $SAMPLE_COUNT"
fi

# Activate local DIANA environment (has ggcat and dependencies)
echo "Activating DIANA environment..."
source /pasteur/appa/homes/cduitama/miniconda3/bin/activate
conda activate "${DIANA_ROOT}/env"

# Verify GGCAT is available (required dependency)
if ! command -v ggcat >/dev/null 2>&1; then
    echo "❌ ERROR: GGCAT not found in environment"
    conda list | grep -i ggcat
    exit 1
fi

echo "✅ GGCAT available: $(which ggcat)"

# Check if modified MUSET is built
if [ ! -f "$MUSET_BIN" ]; then
    echo "❌ ERROR: Modified MUSET not built."
    echo "Please run: cd $DIANA_ROOT && bash scripts/create_umat/01_build_muset.sh"
    exit 1
fi

echo "✅ Using modified MUSET: $MUSET_BIN"
echo "   (Saves SSHash dictionary for inference)"
echo ""

# Raise open-file limit as high as the hard limit allows.
# The merge step in kmtricks opens ~threads × samples files simultaneously.
# With 32 threads × 3190 samples = 102,080 — must stay under ulimit -n.
CURRENT_ULIMIT=$(ulimit -Hn)
ulimit -n "${CURRENT_ULIMIT}"
echo "Open file limit (ulimit -n): $(ulimit -n)"

# Load GCC 13.2 for runtime compatibility (MUSET was built with it)
if command -v module >/dev/null 2>&1; then
    module load gcc/13.2.0 2>/dev/null || true
    export LD_LIBRARY_PATH=/opt/gensoft/exe/gcc/13.2.0/lib64:$LD_LIBRARY_PATH
fi

# Set working directory
cd "${DIANA_ROOT}"

echo "Configuration:"
echo "  Input FOF: $INPUT_FOF"
echo "  Output directory: $OUTPUT_DIR"
echo "  Threads: 32 (reduced from 64 to stay under ulimit -n 131072; merge step needs threads × samples < ulimit)"
echo "  K-mer size: 31"
echo "  Min abundance: 2"
echo "  Min unitig length: 61"
echo "  Fraction absent: 0.1 (keep if present in >10% samples)"
echo "  Fraction present: 0.1 (keep if present in >10% samples)"
echo "  Output fraction matrix: YES ✓"
echo "  Output SSHash dict: YES ✓"
echo ""

# Run modified MUSET with --out-frac flag
echo "🚀 Starting MUSET pipeline..."
echo "⏱️  Estimated time: ~40-50 hours (32 threads, resuming from merge step)"
echo "   Previous run: 76.7h with 32 threads, 3,070 samples"
echo ""

$MUSET_BIN \
    --file "$INPUT_FOF" \
    -o "$OUTPUT_DIR" \
    -k 31 \
    -m 15 \
    -a 2 \
    -l 61 \
    -r 0 \
    -f 0.1 \
    -F 0.1 \
    --out-frac \
    --logan \
    -t 32 \
    --keep-temp \
    2>&1 | tee logs/muset_v9_${SLURM_JOB_ID}_live.log ; MUSET_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "📊 MUSET completed with exit code: $MUSET_EXIT_CODE"
echo "End time: $(date)"
echo ""

if [[ $MUSET_EXIT_CODE -eq 0 ]]; then
    echo "✅ SUCCESS: MUSET matrix building completed!"
    echo ""
    
    # Check output files
    echo "📁 Output directory contents:"
    ls -lah "$OUTPUT_DIR"
    echo ""
    
    # Verify critical files for Diana
    echo "🔍 Verifying Diana-required files:"
    
    if [[ -f "${OUTPUT_DIR}/unitigs.abundance.mat" ]]; then
        echo "  ✓ unitigs.abundance.mat ($(du -h ${OUTPUT_DIR}/unitigs.abundance.mat | cut -f1))"
    else
        echo "  ✗ unitigs.abundance.mat MISSING!"
    fi
    
    if [[ -f "${OUTPUT_DIR}/unitigs.frac.mat" ]]; then
        echo "  ✓ unitigs.frac.mat ($(du -h ${OUTPUT_DIR}/unitigs.frac.mat | cut -f1))"
    else
        echo "  ✗ unitigs.frac.mat MISSING!"
    fi
    
    if [[ -f "${OUTPUT_DIR}/unitigs.fa" ]]; then
        echo "  ✓ unitigs.fa ($(du -h ${OUTPUT_DIR}/unitigs.fa | cut -f1))"
    else
        echo "  ✗ unitigs.fa MISSING!"
    fi
    
    if [[ -f "${OUTPUT_DIR}/unitigs.sshash.dict" ]]; then
        echo "  ✓ unitigs.sshash.dict ($(du -h ${OUTPUT_DIR}/unitigs.sshash.dict | cut -f1))"
    else
        echo "  ✗ unitigs.sshash.dict MISSING!"
    fi
    
    echo ""
    echo "📊 Matrix statistics:"
    if [[ -f "${OUTPUT_DIR}/unitigs.frac.mat" ]]; then
        head -1 "${OUTPUT_DIR}/unitigs.frac.mat"
    fi
    
else
    echo "❌ FAILED: MUSET matrix building failed with exit code $MUSET_EXIT_CODE"
    echo ""
    echo "Check logs for details:"
    echo "  Output: logs/muset_v9_${SLURM_JOB_ID}.out"
    echo "  Error:  logs/muset_v9_${SLURM_JOB_ID}.err"
fi

echo ""
echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
