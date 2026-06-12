#!/bin/bash
#SBATCH --job-name=dl_unitigs
#SBATCH --output=logs/downloads/unitigs_%A_%a.out
#SBATCH --error=logs/downloads/unitigs_%A_%a.err
#SBATCH --partition=edid
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00

# Download unitig files from Logan S3 bucket (SLURM array job)
# Usage: sbatch --array=1-N%50 scripts/downloads/02_download_unitigs.sbatch
# where N = number of missing accessions

set -euo pipefail

# Create log directory
mkdir -p logs/downloads

# Load accession for this array task
ACCESSION_LIST="scripts/downloads/missing_unitigs.txt"
ACCESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACCESSION_LIST")

if [ -z "$ACCESSION" ]; then
    echo "ERROR: No accession found at line $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=================================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Accession: $ACCESSION"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================================================="

# Output directory
UNITIG_DIR="data/unitigs"
mkdir -p "$UNITIG_DIR"
cd "$UNITIG_DIR"

OUTPUT="${ACCESSION}.unitigs.fa.zst"

# Skip if already downloaded
if [ -f "$OUTPUT" ] || [ -f "${ACCESSION}.unitigs.fa.gz" ]; then
    echo "✓ $ACCESSION already exists, skipping download"
    exit 0
fi

# Download from Logan S3 bucket
URL="s3://logan-pub/u/${ACCESSION}/${ACCESSION}.unitigs.fa.zst"

echo "Downloading: $URL"
if aws s3 cp "$URL" . --no-sign-request; then
    echo "✓ Successfully downloaded $ACCESSION"
    
    # Verify file size
    SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT")
    echo "  File size: $(numfmt --to=iec-i --suffix=B $SIZE)"
else
    echo "✗ Failed to download $ACCESSION" >&2
    exit 1
fi

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
