#!/bin/bash
#SBATCH --job-name=diana-val
#SBATCH --output=logs/validation/diana_predict_%A_%a.out
#SBATCH --error=logs/validation/diana_predict_%A_%a.err
#SBATCH --array=1-629%10
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --partition=seqbio

# SLURM array job to run diana-predict on validation samples (seqbio partition, no time limit)

set -e

echo "======================================"
echo "DIANA Validation Predictions"
echo "======================================"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Create log directory
mkdir -p logs/validation

# Get sample from metadata file (skip header, get Nth line)
METADATA="data/validation/validation_metadata_expanded.tsv"
SAMPLE_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$METADATA")

# Extract run_accession (column 17)
RUN_ACCESSION=$(echo "$SAMPLE_LINE" | cut -f17)

echo "Sample ${SLURM_ARRAY_TASK_ID}/629: $RUN_ACCESSION"

# Find FASTQ file(s) for this run accession
SAMPLE_DIR="data/validation/raw/${RUN_ACCESSION}"

if [ ! -d "$SAMPLE_DIR" ]; then
    echo "ERROR: Sample directory not found: $SAMPLE_DIR"
    exit 1
fi

# Find fastq.gz files
FASTQ_FILES=$(find "$SAMPLE_DIR" -name "*.fastq.gz" -o -name "*.fq.gz")

if [ -z "$FASTQ_FILES" ]; then
    echo "ERROR: No FASTQ files found in $SAMPLE_DIR"
    exit 1
fi

echo "Found FASTQ files:"
echo "$FASTQ_FILES"
echo ""

# Run diana-predict
echo "Running diana-predict..."
mamba run -p ./env diana-predict \
  --sample $FASTQ_FILES \
  --model results/full_training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/validation_predictions \
  --threads $SLURM_CPUS_PER_TASK \
  --verbose

echo ""
echo "Completed: $(date)"
echo "======================================"
