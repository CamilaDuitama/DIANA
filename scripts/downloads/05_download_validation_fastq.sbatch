#!/bin/bash
#SBATCH --job-name=dl_fastq
#SBATCH --output=logs/downloads/fastq_%A_%a.out
#SBATCH --error=logs/downloads/fastq_%A_%a.err
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Download and convert SRA files to FASTQ for validation set
# Usage: sbatch --array=1-N%20 scripts/downloads/05_download_validation_fastq.sbatch
# where N = number of missing accessions

set -euo pipefail

# Create directories
mkdir -p logs/downloads
mkdir -p data/validation/sra
mkdir -p data/validation/raw

# Load accession for this array task
ACCESSION_LIST="scripts/downloads/missing_validation_accessions.txt"
ACCESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACCESSION_LIST")

if [ -z "$ACCESSION" ]; then
    echo "ERROR: No accession found at line $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=================================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Accession: $ACCESSION"
echo "Node: $SLURM_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "=================================================="

# Output directory
OUT_DIR="data/validation/raw/${ACCESSION}"
SRA_DIR="data/validation/sra/${ACCESSION}"

# Skip if FASTQ already exists
if [ -d "$OUT_DIR" ] && [ -n "$(find "$OUT_DIR" -name '*.fastq.gz' -print -quit)" ]; then
    echo "✓ $ACCESSION FASTQ already exists, skipping"
    exit 0
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Step 1: Download SRA file using prefetch
echo "Step 1: Downloading SRA file..."
if prefetch "$ACCESSION" -O data/validation/sra/; then
    echo "✓ SRA download complete"
else
    echo "✗ prefetch failed for $ACCESSION" >&2
    exit 1
fi

# Find SRA file
SRA_FILE=$(find "$SRA_DIR" -name "*.sra" -print -quit)
if [ -z "$SRA_FILE" ]; then
    echo "✗ SRA file not found for $ACCESSION" >&2
    exit 1
fi
echo "  SRA file: $SRA_FILE"

# Step 2: Convert to FASTQ using fasterq-dump
echo "Step 2: Converting SRA to FASTQ..."
if fasterq-dump "$ACCESSION" \
    --split-files \
    --threads "$SLURM_CPUS_PER_TASK" \
    --outdir "$OUT_DIR" \
    --temp /tmp \
    --progress; then
    echo "✓ FASTQ conversion complete"
else
    echo "✗ fasterq-dump failed for $ACCESSION" >&2
    exit 1
fi

# Step 3: Compress FASTQ files with pigz
echo "Step 3: Compressing FASTQ files..."
FASTQ_FILES=$(find "$OUT_DIR" -name "*.fastq" -type f)
if [ -z "$FASTQ_FILES" ]; then
    echo "✗ No FASTQ files found to compress" >&2
    exit 1
fi

for fastq in $FASTQ_FILES; do
    echo "  Compressing: $(basename $fastq)"
    if pigz -p "$SLURM_CPUS_PER_TASK" "$fastq"; then
        echo "  ✓ Compressed: ${fastq}.gz"
    else
        echo "  ✗ Compression failed: $fastq" >&2
    fi
done

# Step 4: Clean up SRA file to save space
echo "Step 4: Cleaning up SRA file..."
rm -rf "$SRA_DIR"
echo "✓ Removed SRA directory"

# Verify output
FINAL_FILES=$(find "$OUT_DIR" -name "*.fastq.gz" -type f | wc -l)
echo "✓ Final output: $FINAL_FILES compressed FASTQ file(s)"

# Show total size
TOTAL_SIZE=$(du -sh "$OUT_DIR" | cut -f1)
echo "  Total size: $TOTAL_SIZE"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
