#!/bin/bash
# Step 1: Count reference k-mers in new sample using back_to_sequences

set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <reference_kmers_fasta> <sample_fastq_or_filelist> <output_counts> <threads> [min_abundance]"
    echo ""
    echo "  min_abundance defaults to 2, which is correct for RAW READS."
    echo "  Pass 1 for ASSEMBLED UNITIGS -- each k-mer occurs once, so 2 discards"
    echo "  all of them and produces an all-zero feature vector without erroring."
    echo ""
    echo "Count reference k-mers in a new sample"
    echo ""
    echo "Arguments:"
    echo "  sample_fastq_or_filelist - Single FASTQ file OR path to file list (one FASTQ per line)"
    echo "  min_abundance - Minimum k-mer count to consider present (default: 2)"
    echo "                  K-mers with count < min_abundance are set to 0"
    echo "                  This filters sequencing errors"
    exit 1
fi

REFERENCE_KMERS=$1
SAMPLE_INPUT=$2  # Can be single FASTQ or file list
OUTPUT_COUNTS=$3
THREADS=$4
# Minimum times a k-mer must be seen before it counts. THIS DEPENDS ON THE INPUT
# TYPE and getting it wrong fails silently:
#
#   raw reads (FASTQ)   -> 2.  A genuine k-mer appears in several overlapping
#                              reads; a sequencing error usually appears once. 2
#                              filters errors and is the right default.
#
#   assembled unitigs   -> 1.  Assembly has already collapsed and error-corrected
#                              the reads, so each k-mer occurs exactly ONCE. A
#                              threshold of 2 discards every k-mer, the counts
#                              file comes out empty, and the downstream unitig
#                              matrix is written as all zeros -- while
#                              back_to_sequences still reports millions of
#                              matches. That cost a debugging round on 2026-09-09.
#
# CAVEAT for user-supplied unitigs: we know Logan unitigs are built this way, but
# a FASTA the user assembled themselves may carry redundant or abundance-weighted
# sequences, in which case neither default is obviously right. If you pass
# assembled sequences of unknown provenance, check that the resulting vector has a
# plausible number of non-zero unitigs before trusting it -- diana-predict warns
# when the vector is all-zero or near-empty, but it cannot tell an under-counted
# sample from a genuinely sparse one.
MIN_ABUNDANCE=${5:-2}

SAMPLE_NAME=$(basename "$SAMPLE_INPUT" | sed 's/\.[^.]*$//' | sed 's/_filelist$//')

# Check if back_to_sequences is available in PATH
if ! command -v back_to_sequences >/dev/null 2>&1; then
    echo "[ERROR] back_to_sequences not found in PATH"
    echo "Please ensure back_to_sequences is installed and available"
    exit 2
fi

# Check if input is a file list based on extension
if [[ "$SAMPLE_INPUT" == *.txt ]] || [[ "$SAMPLE_INPUT" == *.list ]]; then
    # File list: use seqkit --infile-list to read paths line-by-line.
    # This is safe for paths containing spaces and handles arbitrarily long lists
    # without hitting shell argument-length limits.
    seqkit seq --infile-list "$SAMPLE_INPUT" | back_to_sequences \
        --in-kmers "$REFERENCE_KMERS" \
        --out-kmers "$OUTPUT_COUNTS" \
        --counted-kmer-threshold "$MIN_ABUNDANCE" \
        --threads "$THREADS"
else
    # Single FASTQ file
    back_to_sequences \
        --in-kmers "$REFERENCE_KMERS" \
        --in-sequences "$SAMPLE_INPUT" \
        --out-kmers "$OUTPUT_COUNTS" \
        --counted-kmer-threshold "$MIN_ABUNDANCE" \
        --threads "$THREADS"
fi

echo "✓ Done!"
