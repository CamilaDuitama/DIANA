#!/bin/bash
# Test single prediction to verify library fix

set -e

# Change to project directory
cd /pasteur/appa/scratch/cduitama/EDID/decOM-classify

# Get first missing sample
RUN_ACCESSION=$(head -1 data/validation/missing_samples_mapping.txt | awk '{print $2}')

echo "Testing prediction for: $RUN_ACCESSION"

# Find FASTQ files
SAMPLE_DIR="data/validation/raw/${RUN_ACCESSION}"

if [ ! -d "$SAMPLE_DIR" ]; then
    echo "ERROR: Sample directory not found: $SAMPLE_DIR"
    exit 1
fi

mapfile -t FASTQ_ARRAY < <(find "$SAMPLE_DIR" -name "*.fastq.gz" -o -name "*.fq.gz" | sort)

if [ ${#FASTQ_ARRAY[@]} -eq 0 ]; then
    echo "ERROR: No FASTQ files found"
    exit 1
fi

echo "Found FASTQ files:"
printf '%s\n' "${FASTQ_ARRAY[@]}"

# Test with mamba (old way - should fail on maestro-2010)
echo ""
echo "=== Test 1: Using mamba run ==="
export LD_LIBRARY_PATH="/pasteur/appa/scratch/cduitama/EDID/decOM-classify/env/lib:$LD_LIBRARY_PATH"
mamba run -p ./env python -c "print('mamba works!')" 2>&1 | head -5

# Test with direct activation (new way - should work)
echo ""
echo "=== Test 2: Using source activate ==="
source /pasteur/appa/scratch/cduitama/EDID/decOM-classify/env/bin/activate
python -c "print('Direct activation works!')"
which diana-predict

echo ""
echo "Tests complete!"
