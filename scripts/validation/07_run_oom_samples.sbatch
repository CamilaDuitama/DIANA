#!/bin/bash
#SBATCH --job-name=diana-oom
#SBATCH --output=logs/validation/diana_oom_%A_%a.out
#SBATCH --error=logs/validation/diana_oom_%A_%a.err
#SBATCH --array=1-45%5
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --partition=seqbio

# SLURM array job to retry OOM failures with 192GB memory (3x 64GB)

set -e

echo "======================================"
echo "DIANA OOM Retry (192GB)"
echo "======================================"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Get the run_accession from the OOM samples list
OOM_LIST="data/validation/oom_samples.txt"
RUN_ACCESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$OOM_LIST")

echo "Retrying OOM sample: $RUN_ACCESSION"

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

echo "Sample size: $(du -sh $SAMPLE_DIR | cut -f1)"
echo "Found FASTQ files:"
printf '%s\n' "${FASTQ_ARRAY[@]}"
echo ""

# Add inference scripts to PATH
export PATH="$(pwd)/scripts/inference:$PATH"

# Run diana-predict with 192GB
echo "Running diana-predict (192GB memory)..."
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
