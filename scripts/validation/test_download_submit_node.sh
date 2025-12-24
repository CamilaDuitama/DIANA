#!/bin/bash
# Test download on submit node

set -e

# Configuration
ACCESSION_LIST="data/validation/accessions.txt"
OUTPUT_DIR="data/validation/raw_test"
TEMP_DIR="/tmp/download_test_$$"

# Load modules
module purge
module load sra-tools/3.2.0

# Set paths to actual executables
PREFETCH="/opt/gensoft/exe/sra-tools/3.2.0/bin/prefetch-orig"
FASTERQ_DUMP="/opt/gensoft/exe/sra-tools/3.2.0/bin/fasterq-dump-orig"

# Get first 2 accessions
ACCESSION1=$(sed -n "1p" ${ACCESSION_LIST})
ACCESSION2=$(sed -n "2p" ${ACCESSION_LIST})

echo "Testing download on submit node"
echo "================================"
echo "Accession 1: ${ACCESSION1}"
echo "Accession 2: ${ACCESSION2}"
echo ""

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${TEMP_DIR}

# Function to download one accession
download_accession() {
    local ACC=$1
    echo "================================================================"
    echo "Processing: ${ACC}"
    echo "================================================================"
    
    # Step 1: Download with prefetch
    echo "Step 1: Downloading with prefetch..."
    if ! $PREFETCH ${ACC} \
        --output-directory ${TEMP_DIR} \
        --max-size u \
        --progress; then
        echo "ERROR: prefetch failed for ${ACC}"
        return 1
    fi
    
    # Step 2: Convert to FASTQ
    echo "Step 2: Converting to FASTQ..."
    if ! $FASTERQ_DUMP \
        ${TEMP_DIR}/${ACC}/${ACC}.sra \
        --outdir ${TEMP_DIR} \
        --split-3 \
        --skip-technical \
        --threads 2 \
        --progress; then
        echo "ERROR: fasterq-dump failed for ${ACC}"
        return 1
    fi
    
    # Remove .sra file to save space
    rm -f ${TEMP_DIR}/${ACC}/${ACC}.sra
    
    # Step 3: Compress
    echo "Step 3: Compressing..."
    gzip ${TEMP_DIR}/*.fastq
    
    # Step 4: Move to final location
    echo "Step 4: Moving to final location..."
    mkdir -p ${OUTPUT_DIR}/${ACC}
    mv ${TEMP_DIR}/*.fastq.gz ${OUTPUT_DIR}/${ACC}/
    
    echo "✓ Completed: ${ACC}"
    ls -lh ${OUTPUT_DIR}/${ACC}/
    echo ""
    
    return 0
}

# Download first accession
if download_accession ${ACCESSION1}; then
    echo "SUCCESS: ${ACCESSION1}"
else
    echo "FAILED: ${ACCESSION1}"
fi

# Download second accession
if download_accession ${ACCESSION2}; then
    echo "SUCCESS: ${ACCESSION2}"
else
    echo "FAILED: ${ACCESSION2}"
fi

# Cleanup
rm -rf ${TEMP_DIR}

echo ""
echo "Test complete!"
echo "Check results in: ${OUTPUT_DIR}"
