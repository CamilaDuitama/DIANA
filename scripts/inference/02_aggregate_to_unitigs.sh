#!/bin/bash
# Step 2: Aggregate k-mer counts to unitig-level features using kmat_tools

set -e

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <kmer_counts> <unitigs_fa> <kmer_size> <out_abundance> [out_fraction]"
    echo ""
    echo "Aggregate k-mer counts to unitig level using existing kmat_tools unitig"
    exit 1
fi

KMER_COUNTS=$1
UNITIGS_FA=$2
KMER_SIZE=$3
OUT_ABUNDANCE=$4
OUT_FRACTION=$5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_NAME=$(basename "$OUT_ABUNDANCE" | sed 's/_unitig_abundance.txt//')

echo "=== Step 2: Aggregate K-mers to Unitigs ==="
echo "Sample: $SAMPLE_NAME"
echo "K-mer counts: $KMER_COUNTS"
echo "Unitigs: $UNITIGS_FA"
echo "K-mer size: $KMER_SIZE"
echo "Output: $OUT_ABUNDANCE"

# Use kmat_tools from external directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KMAT_TOOLS="$SCRIPT_DIR/../../external/muset/bin/kmat_tools"

if [ ! -x "$KMAT_TOOLS" ]; then
    echo "[ERROR] kmat_tools not found at $KMAT_TOOLS"
    exit 2
fi

OUTPUT_PREFIX="${OUT_ABUNDANCE%_abundance.txt}"

CMD="$KMAT_TOOLS unitig \
    -k $KMER_SIZE \
    -p $OUTPUT_PREFIX"

if [ -n "$OUT_FRACTION" ]; then
    CMD="$CMD --out-frac"
fi

# Pass back_to_sequences output directly to kmat_tools
CMD="$CMD $UNITIGS_FA $KMER_COUNTS"

echo "Running: $CMD"
eval $CMD

# Rename outputs to match expected names
if [ -f "${OUTPUT_PREFIX}.abundance.mat" ]; then
    # Extract just the single column (skip unitig ID column)
    awk '{print $2}' "${OUTPUT_PREFIX}.abundance.mat" > "$OUT_ABUNDANCE"
    rm "${OUTPUT_PREFIX}.abundance.mat"
fi

if [ -n "$OUT_FRACTION" ] && [ -f "${OUTPUT_PREFIX}.frac.mat" ]; then
    awk '{print $2}' "${OUTPUT_PREFIX}.frac.mat" > "$OUT_FRACTION"
    rm "${OUTPUT_PREFIX}.frac.mat"
fi

# Cleanup
rm -f "$MATRIX_FILE"

echo "✓ Done!"
