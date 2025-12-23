#!/bin/bash
# Wrapper script for downloading validation files
# Usage: ./scripts/data_prep/submit_validation_download.sh [prepare|download|all] [num_tasks]

set -e

STEP=${1:-all}
NUM_TASKS=${2:-10}

echo "=================================================="
echo "DIANA Validation Data Download"
echo "=================================================="
echo "Step: $STEP"
echo "=================================================="
echo ""

# Create directories
mkdir -p logs/validation_download
mkdir -p data/validation/raw

if [ "$STEP" = "prepare" ] || [ "$STEP" = "all" ]; then
    echo "Step 1: Expanding metadata and fetching run accessions from ENA..."
    echo "This queries ENA API for each archive accession (SRS/ERS) to get run accessions (SRR/ERR)"
    echo ""
    
    python scripts/data_prep/09_expand_validation_metadata.py \
        --input data/validation/validation_metadata_v25.09.0.tsv \
        --output data/validation/validation_metadata_expanded.tsv \
        --cache data/validation/ena_cache.json
    
    echo ""
    echo "✓ Metadata expanded successfully"
    echo "  Output: data/validation/validation_metadata_expanded.tsv"
    echo ""
    
    if [ "$STEP" = "prepare" ]; then
        exit 0
    fi
fi

if [ "$STEP" = "download" ] || [ "$STEP" = "all" ]; then
    # Check if expanded metadata exists
    if [ ! -f "data/validation/validation_metadata_expanded.tsv" ]; then
        echo "Error: Expanded metadata not found!"
        echo "Please run: $0 prepare"
        exit 1
    fi
    
    echo "Step 2: Submitting SLURM array job to download FASTQ files..."
    echo "Array tasks: $NUM_TASKS"
    echo ""
    
    JOB_ID=$(sbatch --array=0-$((NUM_TASKS-1)) \
        scripts/data_prep/download_validation.sbatch | awk '{print $NF}')
    
    echo ""
    echo "✓ Job submitted: $JOB_ID"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -j $JOB_ID"
    echo "  watch -n 10 'squeue -j $JOB_ID'"
    echo ""
    echo "Check logs:"
    echo "  tail -f logs/validation_download/download_${JOB_ID}_*.out"
    echo "  ls -lh logs/validation_download/"
    echo ""
    echo "Check downloaded files:"
    echo "  ls -lh data/validation/raw/"
    echo "  find data/validation/raw/ -type f | wc -l"
    echo ""
else
    echo "Error: Step must be 'prepare', 'download', or 'all'"
    echo "Usage: $0 [prepare|download|all] [num_tasks]"
    exit 1
fi
