#!/bin/bash
#SBATCH --job-name=diana-val
#SBATCH --output=logs/validation/diana_predict_%A_%a.out
#SBATCH --error=logs/validation/diana_predict_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=seqbio

# SLURM array job to run diana-predict ONLY on samples that need prediction
# NOTE: --array parameter should be set via sbatch command line (see 04b_find_samples_to_predict.py)

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

# Use filtered metadata (only samples needing prediction)
METADATA="data/validation/samples_to_predict.tsv"

if [ ! -f "$METADATA" ]; then
    echo "ERROR: Filtered metadata not found: $METADATA"
    echo "Run: ./env/bin/python scripts/validation/04b_find_samples_to_predict.py"
    exit 1
fi

# Get sample from filtered metadata (skip header, get Nth line)
SAMPLE_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$METADATA")

# Extract run_accession (column 17)
RUN_ACCESSION=$(echo "$SAMPLE_LINE" | cut -f17)

echo "Sample ${SLURM_ARRAY_TASK_ID}: $RUN_ACCESSION"

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

# Calculate total file size
TOTAL_SIZE_BYTES=0
for f in "${FASTQ_ARRAY[@]}"; do
    SIZE=$(stat -c%s "$f")
    TOTAL_SIZE_BYTES=$((TOTAL_SIZE_BYTES + SIZE))
done
TOTAL_SIZE_GB=$(echo "scale=2; $TOTAL_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
echo "Total FASTQ size: ${TOTAL_SIZE_GB} GB"
echo ""

# Double-check if predictions already exist (safety check)
PREDICTION_FILE="results/validation_predictions/${RUN_ACCESSION}/${RUN_ACCESSION}_predictions.json"
if [ -f "$PREDICTION_FILE" ]; then
    echo "⚠️  Predictions already exist (should not happen with filtered list): $PREDICTION_FILE"
    echo "Completed: $(date)"
    exit 0
fi

# Warn if file is extremely large (>50GB)
if (( $(echo "$TOTAL_SIZE_GB > 50" | bc -l) )); then
    echo "⚠️  WARNING: Large file size (${TOTAL_SIZE_GB} GB) - may take several hours"
fi

# Add inference scripts to PATH
export PATH="$(pwd)/scripts/inference:$PATH"

# Run diana-predict
echo "Running diana-predict..."
mamba run -p ./env diana-predict \
  --sample "${FASTQ_ARRAY[@]}" \
  --model results/full_training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/validation_predictions \
  --threads $SLURM_CPUS_PER_TASK \
  --verbose

echo ""
echo "Completed: $(date)"
echo "======================================"
