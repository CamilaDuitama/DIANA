#!/bin/bash
#SBATCH --job-name=sra2fastq_v7
#SBATCH --partition=edid
#SBATCH --account=edid
#SBATCH --output=logs/downloads/convert_v7_%A_%a.out
#SBATCH --error=logs/downloads/convert_v7_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-402%20

# Convert .sra → FASTQ for v7 validation samples not yet converted
# Accessions list: data/validation/accessions_to_convert.txt (402 samples)
#
# USAGE:
#   sbatch scripts/validation/04b_convert_sra_to_fastq_v7.sbatch

set -e

ACCESSION_LIST="data/validation/accessions_to_convert.txt"
SRA_DIR="data/validation/sra"
OUTPUT_DIR="data/validation/raw"
TEMP_DIR="/tmp/convert_v7_${SLURM_ARRAY_TASK_ID}_${RANDOM}"
FASTERQ_DUMP="/opt/gensoft/exe/sra-tools/3.2.0/bin/fasterq-dump-orig"

mkdir -p "${OUTPUT_DIR}" "${TEMP_DIR}" logs/downloads

ACCESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ACCESSION_LIST}")

if [ -z "${ACCESSION}" ]; then
    echo "ERROR: No accession for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "================================================================"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Accession: ${ACCESSION}"
echo "Node: $(hostname)"
echo "================================================================"

# Skip if already converted
if ls "${OUTPUT_DIR}/${ACCESSION}/"*.fastq.gz &>/dev/null; then
    echo "Already converted: ${ACCESSION} — skipping"
    exit 0
fi

SRA_FILE="${SRA_DIR}/${ACCESSION}/${ACCESSION}.sra"
if [ ! -f "${SRA_FILE}" ]; then
    echo "ERROR: SRA file not found: ${SRA_FILE}"
    exit 1
fi

echo "Converting to FASTQ..."
time $FASTERQ_DUMP \
    "${SRA_FILE}" \
    --outdir "${TEMP_DIR}" \
    --split-3 \
    --skip-technical \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --progress

echo "Compressing..."
gzip "${TEMP_DIR}"/*.fastq

echo "Moving to output..."
mkdir -p "${OUTPUT_DIR}/${ACCESSION}"
mv "${TEMP_DIR}"/*.fastq.gz "${OUTPUT_DIR}/${ACCESSION}/"

rm -rf "${TEMP_DIR}"

echo "Done: ${ACCESSION}"
ls -lh "${OUTPUT_DIR}/${ACCESSION}/"
