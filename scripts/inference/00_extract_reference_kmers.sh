#!/bin/bash
# Step 0: Verify that reference_kmers.fasta is present.
#
# This file must be downloaded from Zenodo via install.sh before running
# diana-predict.  On-the-fly regeneration is intentionally NOT supported:
# the file is large (179 MB compressed) and tied to a specific matrix
# version; using any other file would produce incorrect feature vectors.

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <muset_output_dir> <output_fasta>"
    exit 1
fi

MUSET_DIR=$1
OUTPUT_FASTA=$2

echo "=== Step 0: Verify Reference K-mers ==="

if [ ! -f "$OUTPUT_FASTA" ]; then
    echo ""
    echo "ERROR: reference_kmers.fasta not found at:"
    echo "  $OUTPUT_FASTA"
    echo ""
    echo "This file must be downloaded from Zenodo before running diana-predict."
    echo "Run the installer to download it automatically:"
    echo ""
    echo "    bash install.sh"
    echo ""
    echo "Or download it manually (179 MB compressed) and place it at the path above:"
    echo "    https://zenodo.org/records/18157419/files/reference_kmers.fasta.gz"
    echo ""
    echo "Note: do NOT use a different k-mer file — it must match the training matrix exactly."
    exit 1
fi

NUM_KMERS=$(grep -c "^>" "$OUTPUT_FASTA")
echo "✓ Reference k-mers found: $NUM_KMERS sequences"
echo "Done!"
