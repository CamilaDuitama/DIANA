#!/bin/bash
#SBATCH --job-name=fastq_conv
#SBATCH --output=logs/downloads/fastq_%A_%a.out
#SBATCH --error=logs/downloads/fastq_%A_%a.err
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Convert SRA files to FASTQ for validation set (v7 split)
# IMPORTANT: SRA files MUST be pre-downloaded using 05a_prefetch_validation_sra.sh
#            (compute nodes don't have internet access)
# Usage: sbatch --array=1-539%15 scripts/downloads/05_download_validation_fastq.sbatch
# Total: 539 missing accessions
# Resources: 8 CPUs for fasterq-dump/pigz threading, 32G for large files

set -euo pipefail

# Load required modules
module load sra-tools/3.2.0

# Disable internet access for fasterq-dump (it should use local SRA files only)
vdb-config --prefetch-to-cwd
vdb-config --set /repository/user/main/public/root="$PWD/data/validation/sra"

# Create directories
mkdir -p logs/downloads
mkdir -p data/validation/sra
mkdir -p data/validation/raw

# Load accession for this array task
ACCESSION_LIST="data/metadata/missing_validation_fastq_v7.txt"
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
TEMP_DIR="data/validation/temp/${ACCESSION}_${SLURM_JOB_ID}"

# Skip if FASTQ already exists
if [ -d "$OUT_DIR" ] && [ -n "$(find "$OUT_DIR" -name '*.fastq.gz' -print -quit)" ]; then
    echo "✓ $ACCESSION FASTQ already exists, skipping"
    exit 0
fi

# Create output and temp directories
mkdir -p "$OUT_DIR"
mkdir -p "$TEMP_DIR"

# Check if SRA file exists (should have been downloaded by prefetch)
SRA_FILE=$(find "$SRA_DIR" -name "${ACCESSION}.sra" -o -name "*.sra" 2>/dev/null | head -1)
if [ -z "$SRA_FILE" ]; then
    echo "✗ SRA file not found for $ACCESSION in $SRA_DIR" >&2
    echo "   Run 05a_prefetch_validation_sra.sh first to download SRA files" >&2
    exit 1
fi
echo "Found SRA file: $SRA_FILE"

# Step 1: Convert to FASTQ using fasterq-dump-orig (uses all $SLURM_CPUS_PER_TASK cores)
echo "Step 1: Converting SRA to FASTQ..."
if fasterq-dump-orig "$ACCESSION" \
    --split-files \
    --threads "$SLURM_CPUS_PER_TASK" \
    --outdir "$OUT_DIR" \
    --temp "$TEMP_DIR" \
    --bufsize 1024MB \
    --curcache 512MB \
    --mem 28GB \
    --progress; then
    echo "✓ FASTQ conversion complete"
else
    echo "✗ fasterq-dump-orig failed for $ACCESSION" >&2
    exit 1
fi

# Step 2: Compress FASTQ files with pigz
echo "Step 2: Compressing FASTQ files..."
FASTQ_FILES=$(find "$OUT_DIR" -name "*.fastq" -type f)
if [ -z "$FASTQ_FILES" ]; then
    echo "✗ No FASTQ files found to compress" >&2
    exit 1
fi

for fastq in $FASTQ_FILES; do
    echo "  Compressing: $(basename $fastq)"
    # pigz uses all $SLURM_CPUS_PER_TASK cores for parallel compression
    if pigz -p "$SLURM_CPUS_PER_TASK" "$fastq"; then
        echo "  ✓ Compressed: ${fastq}.gz"
    else
        echo "  ✗ Compression failed: $fastq" >&2
    fi
done

# Step 3: Clean up temp files to save space
echo "Step 3: Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo "✓ Removed temp directory"

# Verify output
FINAL_FILES=$(find "$OUT_DIR" -name "*.fastq.gz" -type f | wc -l)
echo "✓ Final output: $FINAL_FILES compressed FASTQ file(s)"

# Show total size
TOTAL_SIZE=$(du -sh "$OUT_DIR" | cut -f1)
echo "  Total size: $TOTAL_SIZE"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
