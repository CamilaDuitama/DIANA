#!/bin/bash
#SBATCH --job-name=sra2fastq
#SBATCH --partition=seqbio
#SBATCH --output=logs/downloads/convert_%A_%a.out
#SBATCH --error=logs/downloads/convert_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-880%20

# Step 2: Convert .sra to FASTQ on seqbio partition
# ==================================================
# Reads local .sra files (no internet needed)
# Converts to FASTQ and compresses
#
# PREREQUISITE: Run 03_prefetch_all.sh first!

# Configuration
ACCESSION_LIST="data/validation/accessions.txt"
SRA_DIR="data/validation/sra"
OUTPUT_DIR="data/validation/raw"
TEMP_DIR="/tmp/convert_${SLURM_ARRAY_TASK_ID}_${RANDOM}"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${TEMP_DIR}
mkdir -p logs/downloads

# Load modules
module purge
module load sra-tools/3.2.0

FASTERQ_DUMP="/opt/gensoft/exe/sra-tools/3.2.0/bin/fasterq-dump-orig"

# Get accession for this array task
ACCESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${ACCESSION_LIST})

if [ -z "${ACCESSION}" ]; then
    echo "ERROR: No accession found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Accession: ${ACCESSION}"
echo "SRA file: ${SRA_DIR}/${ACCESSION}/${ACCESSION}.sra"
echo "Output: ${OUTPUT_DIR}/${ACCESSION}/"
echo "Node: $(hostname)"
echo "================================================================"

# Check if already converted (resume capability)
if [ -f "${OUTPUT_DIR}/${ACCESSION}/${ACCESSION}_1.fastq.gz" ] || [ -f "${OUTPUT_DIR}/${ACCESSION}/${ACCESSION}.fastq.gz" ]; then
    echo "✓ Already converted: ${ACCESSION}"
    exit 0
fi

# Check if .sra file exists
if [ ! -f "${SRA_DIR}/${ACCESSION}/${ACCESSION}.sra" ]; then
    echo "ERROR: SRA file not found!"
    echo "Run 03_prefetch_all.sh first on submit node"
    exit 1
fi

# Convert to FASTQ with fasterq-dump (NO INTERNET NEEDED)
echo "Converting to FASTQ..."
time $FASTERQ_DUMP \
    ${SRA_DIR}/${ACCESSION}/${ACCESSION}.sra \
    --outdir ${TEMP_DIR} \
    --split-3 \
    --skip-technical \
    --threads ${SLURM_CPUS_PER_TASK} \
    --progress

if [ $? -ne 0 ]; then
    echo "ERROR: fasterq-dump failed"
    rm -rf ${TEMP_DIR}
    exit 1
fi

# Compress with gzip
echo "Compressing..."
time gzip ${TEMP_DIR}/*.fastq

# Move to final location
echo "Moving to final location..."
mkdir -p ${OUTPUT_DIR}/${ACCESSION}
mv ${TEMP_DIR}/*.fastq.gz ${OUTPUT_DIR}/${ACCESSION}/

# Cleanup
rm -rf ${TEMP_DIR}

echo "✓ Completed: ${ACCESSION}"
ls -lh ${OUTPUT_DIR}/${ACCESSION}/
