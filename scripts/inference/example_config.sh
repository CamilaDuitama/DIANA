# Example configuration for DIANA inference pipeline
# Copy and modify this file for your samples

# Path to MUSET training output directory
# This should contain: unitigs.fa, unitigs.sshash.dict, etc.
MUSET_OUTPUT_DIR="data/matrices/large_matrix_3070_with_frac"

# Path to trained model checkpoint
MODEL_PATH="results/training/best_model.pth"

# Input FASTQ file (new sample to classify)
# Can be gzipped (.fastq.gz) or plain text (.fastq)
SAMPLE_FASTQ="data/validation/raw/SAMPLE001.fastq.gz"

# Output directory for results
OUTPUT_DIR="results/inference/SAMPLE001"

# K-mer size (must match training data)
K=31

# Number of threads for k-mer counting
THREADS=8

# Minimum k-mer abundance (filters sequencing errors)
# K-mers with count < MIN_ABUNDANCE are ignored
MIN_ABUNDANCE=2
