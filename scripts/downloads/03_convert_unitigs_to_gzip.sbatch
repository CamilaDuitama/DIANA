#!/bin/bash
#SBATCH --job-name=zst2gz
#SBATCH --output=logs/downloads/convert_%A_%a.out
#SBATCH --error=logs/downloads/convert_%A_%a.err
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# Convert zstd compressed unitigs to gzip format
# Usage: 
#   1. Generate list of .zst files: find data/unitigs -name "*.zst" > scripts/downloads/unitigs_to_convert.txt
#   2. sbatch --array=1-N%50 scripts/downloads/03_convert_unitigs_to_gzip.sbatch

set -euo pipefail

# Create log directory
mkdir -p logs/downloads

# Load file for this array task
FILE_LIST="scripts/downloads/unitigs_to_convert.txt"
ZST_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$FILE_LIST")

if [ -z "$ZST_FILE" ]; then
    echo "ERROR: No file found at line $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=================================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "File: $ZST_FILE"
echo "Node: $SLURM_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "=================================================="

# Check if zst file exists
if [ ! -f "$ZST_FILE" ]; then
    echo "✗ File not found: $ZST_FILE"
    exit 1
fi

# Output filenames
FA_FILE="${ZST_FILE%.zst}"
GZ_FILE="${FA_FILE}.gz"

# Skip if gzip file already exists
if [ -f "$GZ_FILE" ]; then
    echo "✓ $GZ_FILE already exists, skipping conversion"
    rm -f "$ZST_FILE"  # Clean up zst file
    exit 0
fi

# Decompress zst and compress to gzip in one pipeline
echo "Converting $ZST_FILE → $GZ_FILE"
if zstd -d -c "$ZST_FILE" | pigz -p "$SLURM_CPUS_PER_TASK" > "$GZ_FILE"; then
    echo "✓ Successfully converted to gzip"
    
    # Verify gzip file
    if gzip -t "$GZ_FILE"; then
        echo "✓ Gzip file verified"
        
        # Clean up zst file
        rm -f "$ZST_FILE"
        echo "✓ Removed zst file"
        
        # Show file size
        SIZE=$(stat -f%z "$GZ_FILE" 2>/dev/null || stat -c%s "$GZ_FILE")
        echo "  Final size: $(numfmt --to=iec-i --suffix=B $SIZE)"
    else
        echo "✗ Gzip verification failed!" >&2
        rm -f "$GZ_FILE"
        exit 1
    fi
else
    echo "✗ Conversion failed!" >&2
    rm -f "$GZ_FILE"
    exit 1
fi

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
