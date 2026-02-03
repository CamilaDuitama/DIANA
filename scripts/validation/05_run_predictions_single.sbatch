#!/bin/bash
# Single-sample prediction script called by submit_validation_with_retry.sh
# Uses OVERRIDE_TASK_ID environment variable to map array task to actual sample

set -e

# Use override task ID if provided (from mapping file), otherwise use SLURM array task ID
TASK_ID=${OVERRIDE_TASK_ID:-$SLURM_ARRAY_TASK_ID}

echo "======================================"
echo "DIANA Validation Predictions"
echo "======================================"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Actual Sample ID: ${TASK_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Create log directory
mkdir -p logs/validation

# Get sample from metadata file (skip header, get Nth line)
METADATA="paper/metadata/validation_metadata.tsv"
SAMPLE_LINE=$(sed -n "$((TASK_ID + 1))p" "$METADATA")

# Extract run_accession (column 2)
RUN_ACCESSION=$(echo "$SAMPLE_LINE" | cut -f2)

echo "Sample ${TASK_ID}/957: $RUN_ACCESSION"
echo "Memory allocation: ${SLURM_MEM_PER_NODE}MB"

# Create output directory
OUTPUT_DIR="results/validation_predictions/${RUN_ACCESSION}"
mkdir -p "$OUTPUT_DIR"

# Check if prediction already exists AND was successful
PREDICTION_FILE="${OUTPUT_DIR}/${RUN_ACCESSION}_predictions.json"
JOBINFO_FILE="${OUTPUT_DIR}/.jobinfo"
MEMORY_HISTORY="${OUTPUT_DIR}/.memory_history"

# Only skip if both prediction file exists AND jobinfo shows SUCCESS
if [ -f "$PREDICTION_FILE" ] && [ -f "$JOBINFO_FILE" ]; then
    if grep -q '"status": "SUCCESS"' "$JOBINFO_FILE"; then
        PREV_JOB=$(grep '"job_id"' "$JOBINFO_FILE" | cut -d'"' -f4)
        echo "Prediction already completed successfully in job $PREV_JOB"
        echo "Prediction file: $PREDICTION_FILE"
        echo "Skipping..."
        echo "Completed: $(date)"
        echo "======================================"
        exit 0
    else
        echo "Previous attempt failed or incomplete - retrying..."
    fi
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

# Fix libssh2 library issue - must be set before mamba is invoked
export LD_LIBRARY_PATH="$(pwd)/env/lib:$LD_LIBRARY_PATH"

# Initialize conda/mamba
eval "$(conda shell.bash hook)"
conda activate ./env

# Ensure conda environment binaries take precedence over ~/.local/bin
export PATH="$(pwd)/env/bin:$PATH"

# Write job metadata before running
cat > "$JOBINFO_FILE" <<EOF
{
  "job_id": "${SLURM_ARRAY_JOB_ID}",
  "task_id": "${SLURM_ARRAY_TASK_ID}",
  "actual_sample_id": ${TASK_ID},
  "run_accession": "${RUN_ACCESSION}",
  "memory_mb": ${SLURM_MEM_PER_NODE},
  "cpus": ${SLURM_CPUS_PER_TASK},
  "node": "$(hostname)",
  "start_time": "$(date -Iseconds)",
  "status": "RUNNING"
}
EOF

# Record memory attempt
echo "${SLURM_MEM_PER_NODE}" >> "$MEMORY_HISTORY"

# Run diana-predict
echo "Running diana-predict..."
START_TIME=$(date +%s)

if diana-predict \
  --sample "${FASTQ_ARRAY[@]}" \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/validation_predictions \
  --threads $SLURM_CPUS_PER_TASK \
  --verbose; then
  
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  
  # Update jobinfo with completion time
  cat > "$JOBINFO_FILE" <<EOF
{
  "job_id": "${SLURM_ARRAY_JOB_ID}",
  "task_id": "${SLURM_ARRAY_TASK_ID}",
  "actual_sample_id": ${TASK_ID},
  "run_accession": "${RUN_ACCESSION}",
  "memory_mb": ${SLURM_MEM_PER_NODE},
  "cpus": ${SLURM_CPUS_PER_TASK},
  "node": "$(hostname)",
  "start_time": "$(date -Iseconds)",
  "elapsed_seconds": ${ELAPSED},
  "status": "SUCCESS"
}
EOF
  
  echo ""
  echo "Completed successfully in ${ELAPSED}s"
  echo "Completed: $(date)"
  echo "======================================"
  exit 0
else
  EXIT_CODE=$?
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  
  # Record failure
  cat > "$JOBINFO_FILE" <<EOF
{
  "job_id": "${SLURM_ARRAY_JOB_ID}",
  "task_id": "${SLURM_ARRAY_TASK_ID}",
  "actual_sample_id": ${TASK_ID},
  "run_accession": "${RUN_ACCESSION}",
  "memory_mb": ${SLURM_MEM_PER_NODE},
  "cpus": ${SLURM_CPUS_PER_TASK},
  "node": "$(hostname)",
  "start_time": "$(date -Iseconds)",
  "elapsed_seconds": ${ELAPSED},
  "status": "FAILED",
  "exit_code": ${EXIT_CODE}
}
EOF
  
  echo ""
  echo "FAILED with exit code ${EXIT_CODE}"
  echo "Completed: $(date)"
  echo "======================================"
  exit $EXIT_CODE
fi
