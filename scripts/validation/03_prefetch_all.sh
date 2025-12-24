#!/bin/bash
# Step 1: Prefetch all .sra files on submit node
# ===============================================
# This downloads .sra files from NCBI to local storage
# Must run on submit node (has internet access)

ACCESSION_LIST="data/validation/accessions.txt"
SRA_DIR="data/validation/sra"
FAILED_LOG="${SRA_DIR}/failed_prefetch.txt"

# Load module
module purge
module load sra-tools/3.2.0

PREFETCH="/opt/gensoft/exe/sra-tools/3.2.0/bin/prefetch-orig"

# Create output directory
mkdir -p ${SRA_DIR}

# Clear failed log if exists
> ${FAILED_LOG}

echo "=========================================="
echo "Prefetch Validation Data (Submit Node)"
echo "=========================================="
echo "Accession list: ${ACCESSION_LIST}"
echo "Output directory: ${SRA_DIR}"
echo ""

TOTAL=$(wc -l < ${ACCESSION_LIST})
SUCCESS=0
FAILED=0

while IFS= read -r ACCESSION; do
    echo "[$((SUCCESS+FAILED+1))/${TOTAL}] Processing: ${ACCESSION}"
    
    # Check if already downloaded
    if [ -f "${SRA_DIR}/${ACCESSION}/${ACCESSION}.sra" ]; then
        echo "  ✓ Already exists, skipping"
        ((SUCCESS++))
        continue
    fi
    
    # Download with prefetch (redirect stdin to prevent it from consuming the file)
    if $PREFETCH ${ACCESSION} \
        --output-directory ${SRA_DIR} \
        --max-size u \
        --progress < /dev/null; then
        echo "  ✓ Downloaded successfully"
        ((SUCCESS++))
    else
        echo "  ✗ Failed"
        echo "${ACCESSION}" >> ${FAILED_LOG}
        ((FAILED++))
    fi
    echo ""
done < ${ACCESSION_LIST}

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total accessions: ${TOTAL}"
echo "Successfully downloaded: ${SUCCESS}"
echo "Failed: ${FAILED}"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "Failed accessions written to: ${FAILED_LOG}"
    echo "You can retry those later"
fi

echo ""
echo "✓ Prefetch complete!"
echo "Next step: Submit SLURM job for fasterq-dump conversion"
echo "  sbatch --array=1-${TOTAL}%20 scripts/validation/04_convert_sra_to_fastq.sbatch"
