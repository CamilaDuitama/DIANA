#!/bin/bash
#SBATCH --job-name=diana-test
#SBATCH --output=logs/validation/test_prediction.out
#SBATCH --error=logs/validation/test_prediction.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --partition=seqbio
#SBATCH --array=1

# Test single prediction on compute node

set -e

cd /pasteur/appa/scratch/cduitama/EDID/decOM-classify

echo "======================================"
echo "DIANA Test Prediction"
echo "======================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Get first missing sample
RUN_ACCESSION=$(head -1 data/validation/missing_samples_mapping.txt | awk '{print $2}')
echo "Sample: $RUN_ACCESSION"

# Find FASTQ files
SAMPLE_DIR="data/validation/raw/${RUN_ACCESSION}"
mapfile -t FASTQ_ARRAY < <(find "$SAMPLE_DIR" -name "*.fastq.gz" -o -name "*.fq.gz" | sort)

echo "Found ${#FASTQ_ARRAY[@]} FASTQ file(s)"
echo ""

# Test library paths
echo "=== Testing library configuration ==="
echo "LD_LIBRARY_PATH (before): $LD_LIBRARY_PATH"

export LD_LIBRARY_PATH="/pasteur/appa/scratch/cduitama/EDID/decOM-classify/env/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH (after): $LD_LIBRARY_PATH"
echo ""

# Check if library exists
echo "=== Checking libssh2 library ==="
ls -lh /pasteur/appa/scratch/cduitama/EDID/decOM-classify/env/lib/libssh2.so.1* 2>&1
echo ""

# Test mamba
echo "=== Testing mamba ==="
which mamba
ldd $(which mamba) | grep libssh2 || echo "No libssh2 dependency found"
echo ""

# Try to run mamba
echo "=== Running mamba test ==="
mamba run -p ./env python -c "print('Mamba works!')"

echo ""
echo "Completed: $(date)"
echo "======================================"
