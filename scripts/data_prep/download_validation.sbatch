#!/bin/bash
#SBATCH --job-name=download_validation
#SBATCH --output=logs/validation_download/download_%A_%a.out
#SBATCH --error=logs/validation_download/download_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=seqbio

# Download validation FASTA files in parallel using SLURM array jobs
# Each array task downloads a subset of samples

# Load environment
source /pasteur/appa/scratch/cduitama/EDID/decOM-classify/env/bin/activate

# Configuration
METADATA="data/validation/validation_metadata_expanded.tsv"
OUTPUT_DIR="data/validation/raw"
LOG_DIR="logs/validation_download"

# Count total rows in expanded metadata (excluding header)
TOTAL_RUNS=$(wc -l < $METADATA)
TOTAL_RUNS=$((TOTAL_RUNS - 1))

echo "Total runs to download: $TOTAL_RUNS"

# Calculate samples per task (for 10 array tasks, ~76 samples each)
SAMPLES_PER_TASK=$((TOTAL_RUNS / ${SLURM_ARRAY_TASK_COUNT:-1}))
START_IDX=$((SLURM_ARRAY_TASK_ID * SAMPLES_PER_TASK))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * SAMPLES_PER_TASK))

# Last task gets any remaining samples
if [ $SLURM_ARRAY_TASK_ID -eq $((${SLURM_ARRAY_TASK_COUNT:-1} - 1)) ]; then
    END_IDX=$TOTAL_RUNS
fi

echo "==============================================="
echo "Download Validation Files - Array Task $SLURM_ARRAY_TASK_ID"
echo "==============================================="
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Runs: $START_IDX to $END_IDX"
echo "==============================================="

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Run download script
python scripts/data_prep/10_download_validation_files.py \
    --metadata $METADATA \
    --output $OUTPUT_DIR \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --log-dir $LOG_DIR

echo "==============================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "==============================================="
