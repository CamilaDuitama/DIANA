#!/bin/bash
#SBATCH --job-name=blast_all_features
#SBATCH --output=logs/blast_all_features_%j.out
#SBATCH --error=logs/blast_all_features_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=common

# BLAST All Unitig Features - Comprehensive Annotation
# =====================================================
# 
# Annotates all 107,480 unitig features against NCBI nt database
# to provide comprehensive statistics for reviewers
#
# Expected runtime: 4-12 hours depending on database version and load
# Memory: 64GB (nt database requires significant RAM)
# CPUs: 32 threads for parallel BLAST

set -e
set -u
set -o pipefail

echo "================================================================"
echo "BLAST All Features Job Started"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "================================================================"

# Paths
UNITIGS_FA="data/matrices/training_matrix/unitigs.fa"
# Database files are in 2.10.0 directory but can be used with newer BLAST
BLAST_DB="/local/databases/rel/nt/nt_2026-1-19/blast+/2.10.0/nt"
OUTPUT_DIR="results/feature_analysis/all_features_blast"

# Verify unitigs.fa exists
if [ ! -f "$UNITIGS_FA" ]; then
    echo "ERROR: unitigs.fa not found at $UNITIGS_FA"
    exit 1
fi

# Load BLAST module (use 2.16.0+ for better compatibility with newer databases)
echo "Loading BLAST module..."
module load blast+/2.16.0
module list

# Verify BLAST database exists
if [ ! -f "${BLAST_DB}.nal" ]; then
    echo "ERROR: BLAST database not found at $BLAST_DB"
    echo "Looked for: ${BLAST_DB}.nal"
    exit 1
fi

echo "BLAST database: $BLAST_DB"
echo "Query file: $UNITIGS_FA"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Run BLAST annotation with checkpointing
# Processes 10,000 features per chunk - can resume if interrupted
echo ""
echo "Starting BLAST annotation with checkpointing..."
echo "Chunk size: 10,000 features per chunk"
echo "================================================================"

mamba run -p ./env python scripts/feature_analysis/04_blast_all_features.py \
    --unitigs-fa "$UNITIGS_FA" \
    --blast-db "$BLAST_DB" \
    --output "$OUTPUT_DIR" \
    --num-threads "$SLURM_CPUS_PER_TASK" \
    --chunk-size 10000 \
    --evalue 1e-5

exit_code=$?

echo "================================================================"
echo "Job completed"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "================================================================"

if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: Results saved to $OUTPUT_DIR"
    echo ""
    echo "View summary:"
    echo "  cat $OUTPUT_DIR/hit_statistics.txt"
    echo ""
    echo "View detailed results:"
    echo "  head $OUTPUT_DIR/blast_results.txt"
    echo "  cat $OUTPUT_DIR/blast_summary.json"
else
    echo "ERROR: BLAST annotation failed with exit code $exit_code"
    echo "Check error log: logs/blast_all_features_${SLURM_JOB_ID}.err"
fi

exit $exit_code
