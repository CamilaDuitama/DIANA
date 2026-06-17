#!/bin/bash
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=diana_inference
#SBATCH --output=logs/validation/inference_%A_%a.out
#SBATCH --error=logs/validation/inference_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# DIANA Sample Inference (Version-Agnostic)
#
# Runs per-sample inference on validation/test samples using any trained model.
# Generates per-sample predictions and unitig abundance files.
#
# CONFIGURATION:
#   All paths and settings are in a YAML config file.
#   Default: configs/validation_inference_config.yaml
#
#   To use a different config:
#   sbatch --array=1-N scripts/validation/run_sample_inference.sbatch <path/to/config.yaml>
#
# USAGE:
#   # Default config (v7)
#   sbatch --array=1-702 scripts/validation/run_sample_inference.sbatch
#
#   # Custom config
#   sbatch --array=1-500 scripts/validation/run_sample_inference.sbatch configs/my_config.yaml
#
# OUTPUT:
#   <output_dir>/<ACC>/<ACC>_predictions.json
#   <output_dir>/<ACC>/<ACC>_unitig_abundance.txt
################################################################################

set -e

# Configuration file (default or from argument)
CONFIG="${1:-configs/validation_inference_config.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "=== DIANA Sample Inference ==="
echo "Config:  $CONFIG"
echo "Job ID:  $SLURM_JOB_ID"
echo "Task:    $SLURM_ARRAY_TASK_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

# Parse config with Python
ACCESSIONS_FILE=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['accessions_file'])")
OUTPUT_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['output_dir'])")
MODEL=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['weights'])")
LABEL_ENCODERS=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['label_encoders'])")
FASTQ_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['fastq_dir'])")

# Get accession for this array task
ACC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACCESSIONS_FILE")

if [ -z "$ACC" ]; then
    echo "ERROR: No accession for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Accession: $ACC"
echo "Output:    $OUTPUT_DIR/$ACC/"
echo ""

# Activate environment
eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env

# Create output directory
mkdir -p "$OUTPUT_DIR/$ACC"

# Check if already completed
if [ -f "$OUTPUT_DIR/$ACC/${ACC}_predictions.json" ]; then
    echo "SKIP: Predictions already exist for $ACC"
    exit 0
fi

# Find FASTQ file (handle both .fastq.gz and .fq.gz)
FASTQ_FILE=$(find "$FASTQ_DIR/$ACC" -name "*.fastq.gz" -o -name "*.fq.gz" 2>/dev/null | head -1)

if [ -z "$FASTQ_FILE" ] || [ ! -f "$FASTQ_FILE" ]; then
    echo "ERROR: No FASTQ file found for $ACC in $FASTQ_DIR/$ACC"
    echo "Tip: Run validation download scripts first if samples are missing"
    exit 1
fi

echo "FASTQ: $FASTQ_FILE"
echo ""
echo "Running inference..."

# Run inference using diana predict CLI
diana predict \
    --model "$MODEL" \
    --label-encoders "$LABEL_ENCODERS" \
    --input "$FASTQ_FILE" \
    --output "$OUTPUT_DIR/$ACC" \
    --accession "$ACC"

echo ""
echo "=== COMPLETE — $(date) ==="
echo "Output: $OUTPUT_DIR/$ACC/"
ls -la "$OUTPUT_DIR/$ACC/"
