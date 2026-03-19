#!/bin/bash
#SBATCH --job-name=diana-val
#SBATCH --output=logs/validation/diana_predict_%A_%a.out
#SBATCH --error=logs/validation/diana_predict_%A_%a.err
#SBATCH --array=1-950%10
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --partition=common

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
METADATA="paper/metadata/validation_metadata.tsv"
SAMPLE_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$METADATA")

# Extract run_accession (column 17)
RUN_ACCESSION=$(echo "$SAMPLE_LINE" | cut -f17)

echo "Sample ${SLURM_ARRAY_TASK_ID}/950: $RUN_ACCESSION"

# Find FASTQ file(s) for this run accession
SAMPLE_DIR="data/validation/raw/${RUN_ACCESSION}"

if [ ! -d "$SAMPLE_DIR" ]; then
    echo "ERROR: Sample directory not found: $SAMPLE_DIR"
    exit 1
fi

# Find fastq.gz files and convert to array
mapfile -t FASTQ_ARRAY < <(find "$SAMPLE_DIR" -name "*.fastq.gz" -o -name "*.fq.gz" | sort)

if [ ${#FASTQ_ARRAY[@]} -eq 0 ]; then
    echo "ERROR: No FASTQ files found in $SAMPLE_DIR"
    exit 1
fi

echo "Found FASTQ files:"
printf '%s\n' "${FASTQ_ARRAY[@]}"
echo ""

# Check if prediction already exists
PREDICTION_FILE="results/validation_predictions/${RUN_ACCESSION}/${RUN_ACCESSION}_predictions.json"
if [ -f "$PREDICTION_FILE" ]; then
    echo "Prediction already exists: $PREDICTION_FILE"
    echo "Skipping..."
    echo "Completed: $(date)"
    echo "======================================"
    exit 0
fi

# Add inference scripts to PATH
export PATH="$(pwd)/scripts/inference:$PATH"

# Fix libssh2 library issue - must be set before mamba is invoked
export LD_LIBRARY_PATH="$(pwd)/env/lib:$LD_LIBRARY_PATH"

# Run diana-predict
echo "Running diana-predict..."
mamba run -p ./env diana-predict \
  --sample "${FASTQ_ARRAY[@]}" \
  --model results/training/best_model.pth \
  --training-matrix data/matrices/training_matrix \
  --output results/validation_predictions \
  --threads $SLURM_CPUS_PER_TASK \
  --verbose

echo ""
echo "Completed: $(date)"
echo "======================================"
