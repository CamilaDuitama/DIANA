#!/bin/bash
#SBATCH --job-name=diana-val
#SBATCH --output=logs/validation/diana_predict_%A_%a.out
#SBATCH --error=logs/validation/diana_predict_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --partition=seqbio
#SBATCH --exclude=maestro-2010

# SLURM array job for MISSING validation predictions only (pre-filtered)

set -e

# Change to project directory
cd /pasteur/appa/scratch/cduitama/EDID/decOM-classify

# Fix libssh2 library issue on some nodes - MUST be set early!
export LD_LIBRARY_PATH="$(pwd)/env/lib:$LD_LIBRARY_PATH"

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

# Get run_accession from mapping file (array_index -> run_accession)
MAPPING_FILE="data/validation/missing_samples_mapping.txt"
RUN_ACCESSION=$(awk -v idx="$SLURM_ARRAY_TASK_ID" '$1 == idx {print $2}' "$MAPPING_FILE")

if [ -z "$RUN_ACCESSION" ]; then
    echo "ERROR: Could not find run_accession for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Sample ${SLURM_ARRAY_TASK_ID}: $RUN_ACCESSION"

# Double-check prediction doesn't exist (safety check)
PREDICTION_FILE="results/validation_predictions/${RUN_ACCESSION}/${RUN_ACCESSION}_predictions.json"
if [ -f "$PREDICTION_FILE" ]; then
    echo "WARNING: Prediction already exists: $PREDICTION_FILE"
    echo "Skipping..."
    echo "Completed: $(date)"
    echo "======================================"
    exit 0
fi

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

# Add inference scripts to PATH
export PATH="$(pwd)/scripts/inference:$PATH"

# Run diana-predict with mamba run
echo "Running diana-predict..."
mamba run -p ./env diana-predict \
  --sample "${FASTQ_ARRAY[@]}" \
  --model results/full_training/best_model.pth \
  --training-matrix data/matrices/training_matrix \
  --output results/validation_predictions \
  --threads $SLURM_CPUS_PER_TASK \
  --verbose

echo ""
echo "Completed: $(date)"
echo "======================================"
