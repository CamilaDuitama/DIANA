#!/bin/bash
#SBATCH --job-name=muset_frac_matrix
#SBATCH --output=logs/muset_fraction_%j.out
#SBATCH --error=logs/muset_fraction_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --partition=seqbio
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

# Re-run MUSET with --out-frac flag to generate fraction matrix
# This is needed for Diana inference pipeline consistency
# Uses modified MUSET from external/muset/ that saves SSHash dictionary

echo "========================================"
echo "MUSET Re-run with Fraction Matrix"
echo "Diana Project - Job ${SLURM_JOB_ID}"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Paths
PROJECT_ROOT="/pasteur/appa/scratch/cduitama/decOM"
DIANA_ROOT="/pasteur/appa/scratch/cduitama/EDID/decOM-classify"
MUSET_BIN="$DIANA_ROOT/external/muset/bin/muset"
INPUT_FOF="${DIANA_ROOT}/data/diana_samples.fof"
OUTPUT_DIR="${DIANA_ROOT}/data/matrices/large_matrix_3070_with_frac"

# Create directories (clean if exists to avoid conflicts)
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DIANA_ROOT}/logs"
mkdir -p "${DIANA_ROOT}/data"

# Verify input file exists
if [[ ! -f "$INPUT_FOF" ]]; then
    echo "❌ ERROR: Input FOF file not found: $INPUT_FOF"
    echo "   Expected: 3,070 samples from data/metadata/DIANA_metadata.tsv"
    exit 1
fi

echo "Processing $(wc -l < $INPUT_FOF) unitig files"

# Activate muset environment (needed for GGCAT and dependencies)
echo "Activating muset environment..."
source /pasteur/appa/homes/cduitama/miniconda3/bin/activate
conda activate /pasteur/appa/homes/cduitama/.local/share/mamba/envs/muset_env

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
    echo "Please run: cd $DIANA_ROOT && bash scripts/build_muset.sh"
    exit 1
fi

echo "✅ Using modified MUSET: $MUSET_BIN"
echo "   (Saves SSHash dictionary for inference)"
echo ""

# Load GCC 13.2 for runtime compatibility (MUSET was built with it)
if command -v module >/dev/null 2>&1; then
    module load gcc/13.2.0 2>/dev/null || true
    export LD_LIBRARY_PATH=/opt/gensoft/exe/gcc/13.2.0/lib64:$LD_LIBRARY_PATH
fi

# Set working directory
cd "${PROJECT_ROOT}"

echo "Configuration:"
echo "  Input FOF: $INPUT_FOF"
echo "  Output directory: $OUTPUT_DIR"
echo "  Threads: ${SLURM_CPUS_PER_TASK}"
echo "  K-mer size: 31"
echo "  Min abundance: 2"
echo "  Output fraction matrix: YES ✓"
echo "  Output SSHash dict: YES ✓ (NEW!)"
echo ""

# Run modified MUSET with --out-frac flag
echo "🚀 Starting MUSET pipeline..."
echo "⏱️  Estimated time: ~60 hours"
echo ""

$MUSET_BIN \
    --file "$INPUT_FOF" \
    -o "$OUTPUT_DIR" \
    -k 31 \
    -m 15 \
    -a 2 \
    -l 61 \
    -r 0 \
    --out-frac \
    --logan \
    -t ${SLURM_CPUS_PER_TASK} \
    --keep-temp

MUSET_EXIT_CODE=$?

echo ""
echo "📊 MUSET completed with exit code: $MUSET_EXIT_CODE"
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
        echo "  ✓ unitigs.frac.mat ($(du -h ${OUTPUT_DIR}/unitigs.frac.mat | cut -f1)) - NEW!"
    else
        echo "  ✗ unitigs.frac.mat MISSING!"
    fi
    
    if [[ -f "${OUTPUT_DIR}/unitigs.fa" ]]; then
        echo "  ✓ unitigs.fa ($(du -h ${OUTPUT_DIR}/unitigs.fa | cut -f1))"
    else
        echo "  ✗ unitigs.fa MISSING!"
    fi
    
    if [[ -f "${OUTPUT_DIR}/unitigs.sshash.dict" ]]; then
        echo "  ✓ unitigs.sshash.dict ($(du -h ${OUTPUT_DIR}/unitigs.sshash.dict | cut -f1)) - NEW!"
    else
        echo "  ✗ unitigs.sshash.dict MISSING!"
    fi
    
else
    echo "❌ FAILED: MUSET matrix building failed with exit code $MUSET_EXIT_CODE"
fi

# Final storage usage
echo ""
echo "💾 Storage analysis:"
echo "  Output directory: $(du -sh $OUTPUT_DIR | cut -f1)"

echo ""
echo "🎉 Matrix building job completed!"
echo "End time: $(date)"
echo "📁 Results available in: ${OUTPUT_DIR}"
echo ""
echo "========================================"
echo "📋 Next steps for Diana project:"
echo "========================================"
echo "1. Train models using unitigs.frac.mat"
echo "   cd $DIANA_ROOT"
echo "   python scripts/01_create_splits.py"
echo "   python scripts/03_train_multitask.py --matrix ${OUTPUT_DIR}/unitigs.frac.mat"
echo ""
echo "2. Inference on new samples will use:"
echo "   - unitigs.fa (unitig sequences)"
echo "   - unitigs.sshash.dict (k-mer lookup)"
echo "========================================"
