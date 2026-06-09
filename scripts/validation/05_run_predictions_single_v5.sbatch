#!/bin/bash
# Single-sample prediction script (v5 model) called by submit_validation_with_retry_v5.sh
# Uses OVERRIDE_TASK_ID environment variable to map array task to actual sample

set -e

# Use override task ID if provided (from mapping file), otherwise use SLURM array task ID
TASK_ID=${OVERRIDE_TASK_ID:-$SLURM_ARRAY_TASK_ID}

echo "======================================"
echo "DIANA v5 Validation Predictions"
echo "======================================"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Actual Sample ID: ${TASK_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

mkdir -p logs/validation

# Get sample from accessions file (v5 uses bioproject-disjoint split: 611 samples)
ACCESSIONS="data/splits_bioproject/all_validation_accessions.txt"
RUN_ACCESSION=$(sed -n "${TASK_ID}p" "$ACCESSIONS")

if [ -z "$RUN_ACCESSION" ]; then
    echo "ERROR: No accession for task ${TASK_ID}"
    exit 1
fi

echo "Sample ${TASK_ID}/611: $RUN_ACCESSION"
echo "Memory allocation: ${SLURM_MEM_PER_NODE}MB"

OUTPUT_DIR="results/validation_predictions_bioproject_v5_full/${RUN_ACCESSION}"
mkdir -p "$OUTPUT_DIR"

PREDICTION_FILE="${OUTPUT_DIR}/${RUN_ACCESSION}_predictions.json"
JOBINFO_FILE="${OUTPUT_DIR}/.jobinfo"
MEMORY_HISTORY="${OUTPUT_DIR}/.memory_history"

if [ -f "$PREDICTION_FILE" ] && [ -f "$JOBINFO_FILE" ]; then
    if grep -q '"status": "SUCCESS"' "$JOBINFO_FILE"; then
        PREV_JOB=$(grep '"job_id"' "$JOBINFO_FILE" | cut -d'"' -f4)
        echo "Prediction already completed successfully in job $PREV_JOB"
        echo "Skipping..."
        echo "Completed: $(date)"
        echo "======================================"
        exit 0
    else
        echo "Previous attempt failed or incomplete - retrying..."
    fi
fi

SAMPLE_DIR="data/validation/raw/${RUN_ACCESSION}"
if [ ! -d "$SAMPLE_DIR" ]; then
    echo "ERROR: Sample directory not found: $SAMPLE_DIR"
    exit 1
fi

mapfile -t FASTQ_ARRAY < <(find "$SAMPLE_DIR" -name "*.fastq.gz" -o -name "*.fq.gz" | sort)
if [ ${#FASTQ_ARRAY[@]} -eq 0 ]; then
    echo "ERROR: No FASTQ files found in $SAMPLE_DIR"
    exit 1
fi

echo "Found FASTQ files:"
printf '%s\n' "${FASTQ_ARRAY[@]}"
echo ""

export LD_LIBRARY_PATH="$(pwd)/env/lib:$LD_LIBRARY_PATH"
eval "$(conda shell.bash hook)"
conda activate ./env
export PATH="$(pwd)/env/bin:$PATH"

# Write initial jobinfo
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

echo "${SLURM_MEM_PER_NODE}" >> "$MEMORY_HISTORY"

echo "Running diana-predict (v5 model)..."
START_TIME=$(date +%s)

if diana-predict \
  --sample "${FASTQ_ARRAY[@]}" \
  --model results/training_bioproject_v5/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output "results/validation_predictions_bioproject_v5_full" \
  --threads $SLURM_CPUS_PER_TASK \
  --verbose; then

  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))

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
