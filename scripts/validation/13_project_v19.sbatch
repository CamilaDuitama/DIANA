#!/bin/bash
#SBATCH --job-name=diana_proj_v19
#SBATCH --partition=common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/project_v19/proj_%A_%a.out
#SBATCH --error=logs/project_v19/proj_%A_%a.err
################################################################################
# v19 C5: project Logan-unitig runs onto the v17 training vocabulary with the rule
# that built the training matrix (MUSET v0.6.1, --logan -a 2): unitigs whose ka:f
# coverage rounds below 2 contribute no k-mer. Route = diana-predict's Logan route:
# k-mer positions (01_), ka:f coverage reconstruction with min abundance 2 (01b_),
# aggregation with the kmat_tools that built the matrix (02_, KMAT_TOOLS_BIN).
# The maestro-20xx nodes are excluded: back_to_sequences dies there with
# "Illegal instruction" (2026-10-09).
#
#   ACC_LIST=<file> OUTDIR=<dir> sbatch --array=1-<n> --export=ALL,ACC_LIST,OUTDIR \
#       scripts/validation/13_project_v19.sbatch
################################################################################
set -euo pipefail
ROOT=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
cd "$ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
export KMAT_TOOLS_BIN="$ROOT/external/muset/bin/kmat_tools"

ACC_LIST=${ACC_LIST:?ACC_LIST is required}
OUTDIR=${OUTDIR:?OUTDIR is required}
REF_KMERS=data/matrices/matrix_v17_train/matrix.filtered.fasta
UNITIGS=data/matrices/matrix_v17_train/unitigs.fa
N_FEATURES=85986
LOGAN_DIR=/pasteur/appa/scratch/cduitama/EDID/decOM-classify/data/unitigs

ACC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ACC_LIST")
[ -n "$ACC" ] || { echo "ERROR: no accession at line $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
SAMPLE_DIR="$OUTDIR/$ACC"
FRACTION="$SAMPLE_DIR/${ACC}_unitig_fraction.txt"
mkdir -p "$SAMPLE_DIR"
if [ -s "$FRACTION" ] && [ "$(wc -l < "$FRACTION")" -eq "$N_FEATURES" ]; then
    echo "$ACC already complete; skipping."
    exit 0
fi
IN="$LOGAN_DIR/${ACC}.unitigs.fa.gz"
[ -e "$IN" ] || { echo "ERROR: no Logan unitigs for $ACC" >&2; exit 1; }
POS="$SAMPLE_DIR/${ACC}_kmer_positions.txt"
COUNTS="$SAMPLE_DIR/${ACC}_kmer_counts.txt"
ABUND="$SAMPLE_DIR/${ACC}_unitig_abundance.txt"
echo "=== $ACC === node=$(hostname) start=$(date)"

bash scripts/inference/01_count_kmers.sh "$REF_KMERS" "$IN" "$POS" "${SLURM_CPUS_PER_TASK:-8}" 1 --output-kmer-positions
./env/bin/python scripts/inference/01b_logan_abundance_counts.py "$POS" "$IN" "$COUNTS" --min-abundance 2 --rule round
[ -s "$COUNTS" ] || { echo "ERROR: $ACC empty k-mer counts" >&2; exit 1; }

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/diana_proj_v19_${ACC}_XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT
(
    cd "$WORKDIR"
    bash "$ROOT/scripts/inference/02_aggregate_to_unitigs.sh" \
        "$ROOT/$COUNTS" "$(readlink -f "$ROOT/$UNITIGS")" 31 "$ROOT/$ABUND" "$ROOT/$FRACTION"
)
N=$(wc -l < "$FRACTION")
[ "$N" -eq "$N_FEATURES" ] || { echo "ERROR: $ACC produced $N lines, expected $N_FEATURES" >&2; exit 1; }
NZ=$(awk '$1>0' "$FRACTION" | wc -l)
[ "$NZ" -gt 0 ] || { echo "ERROR: $ACC produced an all-zero vector" >&2; exit 1; }
rm -f "$POS" "$COUNTS"
echo "OK: $ACC -> $N features, $NZ nonzero. end=$(date)"
