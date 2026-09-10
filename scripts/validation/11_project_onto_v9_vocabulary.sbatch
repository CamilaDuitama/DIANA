#!/bin/bash
#SBATCH --job-name=diana_proj_v9
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/project_v9/proj_%A_%a.out
#SBATCH --error=logs/project_v9/proj_%A_%a.err

################################################################################
# Build unitig feature vectors for val/test (and 59 leftover train) runs against
# the v9 TRAIN-ONLY vocabulary.
#
# This is the same feature-extraction the `diana-predict` CLI runs for any unseen
# sample -- steps 1 and 2 of that pipeline, 01_count_kmers.sh followed by
# 02_aggregate_to_unitigs.sh -- just batched over the split instead of one sample
# at a time, and pointed at the v9 vocabulary.
#
# It is needed because matrix_v9_train was built from the 2,716 training runs only,
# so its 110,202-unitig vocabulary carries no information from val or test (R3.4).
# Those runs were deliberately excluded from the matrix, so their feature vectors
# have to be produced the same way a brand-new sample's would be.
#
# Input source, in order of preference:
#   1. Logan unitigs  (data/unitigs/<ACC>.unitigs.fa.gz) — same input type the
#      matrix was built from, so the k-mer content is directly comparable;
#   2. raw FASTQ      (data/validation/raw/<ACC>*.fastq.gz) for the 152 runs with
#      no Logan assembly.
#
# Usage:
#   ACC_LIST=data/project_v9_from_unitigs.txt \
#     sbatch --array=1-830%25 --export=ALL,ACC_LIST scripts/validation/11_project_onto_v9_vocabulary.sbatch
#   ACC_LIST=data/project_v9_from_fastq.txt \
#     sbatch --array=1-152%25 --export=ALL,ACC_LIST scripts/validation/11_project_onto_v9_vocabulary.sbatch
################################################################################

set -euo pipefail
ROOT=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
cd "$ROOT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export KMP_INIT_AT_FORK=FALSE
export KMP_AFFINITY=disabled
export MPLBACKEND=Agg

ACC_LIST=${ACC_LIST:-data/project_v9_from_unitigs.txt}
REF_KMERS=data/matrices/matrix_v9_train/matrix.filtered.fasta
UNITIGS=data/matrices/matrix_v9_train/unitigs.fa
KMER_SIZE=31
N_FEATURES=110202
OUTDIR=${OUTDIR:-results/v9_vectors}
LOGAN_DIR=/pasteur/appa/scratch/cduitama/EDID/decOM-classify/data/unitigs

ACC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACC_LIST")
[ -n "$ACC" ] || { echo "ERROR: no accession at line $SLURM_ARRAY_TASK_ID" >&2; exit 1; }

SAMPLE_DIR="$OUTDIR/$ACC"
FRACTION="$SAMPLE_DIR/${ACC}_unitig_fraction.txt"
mkdir -p "$SAMPLE_DIR"

# Idempotent: a complete vector is skipped, so the array can be resubmitted.
if [ -s "$FRACTION" ] && [ "$(wc -l < "$FRACTION")" -eq "$N_FEATURES" ]; then
    echo "$ACC already complete; skipping."
    exit 0
fi

# The abundance threshold must match the input type.
#
# back_to_sequences --counted-kmer-threshold defaults to 2, which filters
# sequencing errors in raw reads: a genuine k-mer appears in several overlapping
# reads, an error usually once. Logan unitigs are ASSEMBLED, so each k-mer occurs
# exactly once and a threshold of 2 discards all of them. The first run of this
# job wrote 110,202 all-zero features while back_to_sequences reported 4.28 M
# matches, because the counts file was empty and kmat_tools then emitted the
# unitig-id column alone.
#
# Assembly has already done the error filtering, and --out-frac asks only which
# k-mers are PRESENT, so threshold 1 is correct for unitig input.
FILELIST="$SAMPLE_DIR/${ACC}_files.txt"
# FORCE_SRC=fastq projects a run from raw reads even when Logan unitigs exist. Used
# to measure whether the input route biases the features: 238 held-out runs have
# both, so the same run can be pushed through each route and the vectors compared.
if [ "${FORCE_SRC:-}" = "fastq" ]; then
    find data/validation/raw -type f -name "${ACC}*.fastq.gz" | sort > "$FILELIST"
    SRC="fastq-forced"
    MIN_ABUNDANCE=2
elif [ -e "$LOGAN_DIR/${ACC}.unitigs.fa.gz" ]; then
    echo "$LOGAN_DIR/${ACC}.unitigs.fa.gz" > "$FILELIST"
    SRC="logan-unitigs"
    MIN_ABUNDANCE=1
else
    find data/validation/raw -type f -name "${ACC}*.fastq.gz" | sort > "$FILELIST"
    SRC="fastq"
    MIN_ABUNDANCE=2
fi
[ -s "$FILELIST" ] || { echo "ERROR: no input found for $ACC" >&2; exit 1; }

echo "=== $ACC ($SRC) === node=$(hostname) start=$(date)"

COUNTS="$SAMPLE_DIR/${ACC}_kmer_counts.txt"
ABUND="$SAMPLE_DIR/${ACC}_unitig_abundance.txt"

bash scripts/inference/01_count_kmers.sh \
    "$REF_KMERS" "$FILELIST" "$COUNTS" "${SLURM_CPUS_PER_TASK:-8}" "$MIN_ABUNDANCE"

# An empty counts file yields a silently all-zero vector; fail loudly instead.
if [ ! -s "$COUNTS" ]; then
    echo "ERROR: $ACC empty k-mer counts (src=$SRC threshold=$MIN_ABUNDANCE)" >&2
    exit 1
fi

# Each task gets its own working directory: kmat_tools' sshash step writes a large
# temp file into the CWD, and concurrent tasks sharing one directory killed 35 of
# 439 in the previous array with "cannot open file".
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/diana_proj_${ACC}_XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT
(
    cd "$WORKDIR"
    bash "$ROOT/scripts/inference/02_aggregate_to_unitigs.sh" \
        "$ROOT/$COUNTS" "$(readlink -f "$ROOT/$UNITIGS")" "$KMER_SIZE" \
        "$ROOT/$ABUND" "$ROOT/$FRACTION"
)

N=$(wc -l < "$FRACTION")
[ "$N" -eq "$N_FEATURES" ] || { echo "ERROR: $ACC produced $N lines, expected $N_FEATURES" >&2; exit 1; }
NZ=$(awk '$1>0' "$FRACTION" | wc -l)
[ "$NZ" -gt 0 ] || { echo "ERROR: $ACC produced an all-zero vector" >&2; exit 1; }
rm -f "$COUNTS"
echo "OK: $ACC ($SRC) -> $N features, $NZ nonzero. end=$(date)"
