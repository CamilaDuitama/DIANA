#!/bin/bash
# Step 0: Extract reference k-mers from MUSET output
# If matrix.filtered.fasta exists, use it directly
# Otherwise, regenerate from matrix.filtered.mat

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <muset_output_dir> <output_fasta>"
    echo ""
    echo "Extract reference k-mer set from MUSET training output"
    exit 1
fi

MUSET_DIR=$1
OUTPUT_FASTA=$2

echo "=== Step 0: Extract Reference K-mers ==="
echo "MUSET directory: $MUSET_DIR"
echo "Output: $OUTPUT_FASTA"

# Check if matrix.filtered.fasta exists
if [ -f "$MUSET_DIR/matrix.filtered.fasta" ]; then
    echo "✓ Found matrix.filtered.fasta"
    cp "$MUSET_DIR/matrix.filtered.fasta" "$OUTPUT_FASTA"
else
    echo "✓ matrix.filtered.fasta not found, will regenerate from matrix.filtered.mat"
    
    if [ ! -f "$MUSET_DIR/matrix.filtered.mat" ]; then
        echo "ERROR: Neither matrix.filtered.fasta nor matrix.filtered.mat found in $MUSET_DIR"
        exit 1
    fi
    
    # Use kmat_tools to convert matrix to FASTA
    kmat_tools fasta -o "$OUTPUT_FASTA" "$MUSET_DIR/matrix.filtered.mat"
fi

NUM_KMERS=$(grep -c "^>" "$OUTPUT_FASTA")
echo "✓ Reference set contains $NUM_KMERS k-mers"
echo "Done!"
