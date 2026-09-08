#!/bin/bash
#SBATCH --job-name=diana_valvec
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/val_vectors/valvec_%A_%a.out
#SBATCH --error=logs/val_vectors/valvec_%A_%a.err

################################################################################
# Build v7-dimension unitig feature vectors for the usable validation runs.
#
# Why: the v7 matrix contains train+test only (3,190 = 2,838 + 352). The 611
# vectors under results/validation_predictions_bioproject_v5_full/ are
# 107,480-dimensional (the v5 unitig set); v7 needs 78,430. They cannot be
# reused. Without these vectors tau -- and every other hyperparameter -- can
# only be selected on the test set, which is the leakage the referees flagged.
#
# Inputs are already on disk; this job downloads nothing.
#
# Usage:
#   mkdir -p logs/val_vectors
#   sbatch --array=1-439%20 scripts/validation/10_build_val_vectors_v7.sbatch
#   ACC_LIST=data/splits_v7/val_remaining_accessions.txt \
#     sbatch --array=1-73%20 --export=ALL,ACC_LIST scripts/validation/10_build_val_vectors_v7.sbatch
#
# Output per run:
#   results/validation_vectors_v7/<ACC>/<ACC>_unitig_fraction.txt   (78,430 lines)
################################################################################

set -euo pipefail

ROOT=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
cd "$ROOT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export KMP_INIT_AT_FORK=FALSE
export KMP_AFFINITY=disabled
export MPLBACKEND=Agg

# Override with ACC_LIST=<file> sbatch ... to extract a different set.
ACC_LIST=${ACC_LIST:-data/splits_v7/val_usable_accessions.txt}
REF_KMERS=data/matrices/matrix_v7_3190/reference_kmers.fasta
UNITIGS=data/matrices/matrix_v7_3190/unitigs.fa
KMER_SIZE=31
OUTDIR=results/validation_vectors_v7

ACC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACC_LIST")
if [ -z "$ACC" ]; then
    echo "ERROR: no accession at line $SLURM_ARRAY_TASK_ID of $ACC_LIST" >&2
    exit 1
fi

SAMPLE_DIR="$OUTDIR/$ACC"
FRACTION="$SAMPLE_DIR/${ACC}_unitig_fraction.txt"
mkdir -p "$SAMPLE_DIR"

# Idempotent: a completed run is skipped, so the array can be resubmitted safely.
if [ -s "$FRACTION" ] && [ "$(wc -l < "$FRACTION")" -eq 78430 ]; then
    echo "$ACC already has a complete 78,430-line vector; skipping."
    exit 0
fi

# Every FASTQ for this run (single-end, paired, or with an unpaired mate).
FILELIST="$SAMPLE_DIR/${ACC}_files.txt"
find data/validation/raw -type f -name "${ACC}*.fastq.gz" | sort > "$FILELIST"
if [ ! -s "$FILELIST" ]; then
    echo "ERROR: no FASTQ found for $ACC under data/validation/raw" >&2
    exit 1
fi

echo "=== $ACC ==="
echo "Node:  $(hostname)"
echo "Files: $(wc -l < "$FILELIST")"
echo "Start: $(date)"

COUNTS="$SAMPLE_DIR/${ACC}_kmer_counts.txt"
ABUND="$SAMPLE_DIR/${ACC}_unitig_abundance.txt"

bash scripts/inference/01_count_kmers.sh \
    "$REF_KMERS" "$FILELIST" "$COUNTS" "${SLURM_CPUS_PER_TASK:-8}"

# kmat_tools' sshash step writes a ~23 MB temp file into the CURRENT working
# directory ("./sshash.tmp.run_<id>.minimizers.0.bin"). Every array task used to
# run from the project root, so 20 concurrent tasks hammered one directory on a
# shared network filesystem and 35 of 439 died with "cannot open file" -- after
# the expensive k-mer counting had already completed. Re-running a failed task
# alone succeeded unchanged, which is what identified it as contention.
#
# Give each task its own working directory on node-local disk and pass absolute
# paths, so the temp files never share a directory.
ABS_UNITIGS=$(readlink -f "$UNITIGS")
ABS_COUNTS=$(readlink -f "$COUNTS")
ABS_ABUND="$ROOT/$ABUND"
ABS_FRACTION="$ROOT/$FRACTION"
AGG_SCRIPT="$ROOT/scripts/inference/02_aggregate_to_unitigs.sh"

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/diana_valvec_${ACC}_XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

(
    cd "$WORKDIR"
    bash "$AGG_SCRIPT" "$ABS_COUNTS" "$ABS_UNITIGS" "$KMER_SIZE" "$ABS_ABUND" "$ABS_FRACTION"
)

N=$(wc -l < "$FRACTION")
if [ "$N" -ne 78430 ]; then
    echo "ERROR: $ACC produced $N lines, expected 78430" >&2
    exit 1
fi

# The k-mer count file is large and reproducible; keep only the feature vectors.
rm -f "$COUNTS"

echo "OK: $ACC -> $FRACTION ($N features)"
echo "End: $(date)"
