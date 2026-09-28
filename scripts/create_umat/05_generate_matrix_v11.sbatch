#!/bin/bash
# Build the v11 feature vocabulary: the v9 build with a lower prevalence floor.
#
# v9 (04_generate_matrix_v9.sbatch) kept a k-mer only if present in >= 10 % of the
# 2,716 training samples (272). Counting on the training side, only 5 of the 38
# eligible classes have 272 runs, so a k-mer private to any other class was deleted
# before assembly and no column could carry it (PROJECT.md section 4, M1). Here the
# floor is an absolute count, -N 14: present in >= 14 samples. Everything else is
# identical: the same 2,716 TRAINING runs (data/train_samples_v9.fof, zero held-out
# runs, R3.4), the same split, k=31, -a 2, -l 61, ceiling -f 0.1.
#
# This changes the benchmark, not DIANA: every model is refit on the new matrix.
#
# --keep-temp so other floors (-N 28) can be run from the kmtricks directory with
# `muset -i` (about 4.5 h) instead of re-counting (about 12.5 h). Check after the
# run that kmer_matrix/ actually holds the matrices; v9's did not survive.
#
# v9 job 10888376: 16 h 35 m on 32 threads, peak RSS 38.8 GB. Memory is raised
# because the filtered k-mer set and ggcat's input will be larger. No --time: edid
# has no limit. Threads stay at 32: kmtricks' merge opens threads x samples files
# (32 x 2,716 = 86,912 < ulimit -n 131,072).
#SBATCH --job-name=muset_v11_matrix
#SBATCH --output=logs/muset_v11_%j.out
#SBATCH --error=logs/muset_v11_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=edid
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# No `set -u`: the env's conda activate.d scripts reference unset variables.
set -o pipefail

DIANA_ROOT="/pasteur/helix/scratch/cduitama/EDID/decOM-classify"
MUSET_BIN="$DIANA_ROOT/external/muset/bin/muset"
INPUT_FOF="${DIANA_ROOT}/data/train_samples_v9.fof"
OUTPUT_DIR="${DIANA_ROOT}/data/matrices/matrix_v11_train"
MIN_NB_PRESENT=14

echo "=== MUSET v11 matrix  job ${SLURM_JOB_ID}  $(hostname)  $(date) ==="
[[ -f "$INPUT_FOF" ]] || { echo "ERROR: no input fof at $INPUT_FOF" >&2; exit 1; }
[[ -e "$OUTPUT_DIR/unitigs.frac.mat" ]] && { echo "refusing to overwrite $OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR" "$DIANA_ROOT/logs"

SAMPLE_COUNT=$(wc -l < "$INPUT_FOF")
[[ $SAMPLE_COUNT -eq 2716 ]] || { echo "ERROR: expected 2,716 training runs, fof has $SAMPLE_COUNT" >&2; exit 1; }
MISSING=0
while read -r line; do
    f=$(echo "$line" | awk '{print $NF}')
    [[ -e "$f" ]] || MISSING=$((MISSING+1))
done < "$INPUT_FOF"
[[ $MISSING -eq 0 ]] || { echo "ERROR: $MISSING of $SAMPLE_COUNT input files missing" >&2; exit 1; }
# The vocabulary must come from training runs only.
if [[ $(comm -12 <(awk '{print $1}' "$INPUT_FOF" | sed 's/[:\/].*//' | sort) <(sort data/splits_v9/test_accessions.txt) | wc -l) -ne 0 ]]; then
    echo "ERROR: held-out runs in the input fof" >&2; exit 1
fi
echo "All $SAMPLE_COUNT training unitig files present, 0 held-out runs."

ulimit -n "$(ulimit -Hn)"
if command -v module >/dev/null 2>&1; then
    module load gcc/13.2.0 2>/dev/null || true
    export LD_LIBRARY_PATH=/opt/gensoft/exe/gcc/13.2.0/lib64:${LD_LIBRARY_PATH:-}
fi
source /pasteur/appa/homes/cduitama/miniconda3/bin/activate
conda activate "${DIANA_ROOT}/env"
command -v ggcat >/dev/null 2>&1 || { echo "ERROR: ggcat not on PATH" >&2; exit 1; }
[[ -f "$MUSET_BIN" ]] || { echo "ERROR: muset not built at $MUSET_BIN" >&2; exit 1; }
echo "ggcat: $(command -v ggcat); muset: $MUSET_BIN; open files: $(ulimit -n)"

echo "Configuration: k 31, -m 15, -a 2, -l 61, -r 0, ceiling -f 0.1, floor -N ${MIN_NB_PRESENT}, --logan, --out-frac, --keep-temp, 32 threads"
"$MUSET_BIN" \
    --file "$INPUT_FOF" \
    -o "$OUTPUT_DIR" \
    -k 31 \
    -m 15 \
    -a 2 \
    -l 61 \
    -r 0 \
    -f 0.1 \
    -N "$MIN_NB_PRESENT" \
    --out-frac \
    --logan \
    -t 32 \
    --keep-temp \
    2>&1 | tee "logs/muset_v11_${SLURM_JOB_ID}_live.log" ; MUSET_EXIT_CODE=${PIPESTATUS[0]}

echo "MUSET exit code: $MUSET_EXIT_CODE  $(date)"
[[ $MUSET_EXIT_CODE -eq 0 ]] || exit "$MUSET_EXIT_CODE"
for f in unitigs.abundance.mat unitigs.frac.mat unitigs.fa unitigs.sshash.dict matrix.filtered.fasta; do
    [[ -f "$OUTPUT_DIR/$f" ]] && echo "  ok  $f ($(du -h "$OUTPUT_DIR/$f" | cut -f1))" || echo "  MISSING $f"
done
echo "unitigs: $(grep -c '^>' "$OUTPUT_DIR/unitigs.fa")"
echo "kmtricks matrix kept: $(du -sh "$OUTPUT_DIR/kmer_matrix" | cut -f1)"
grep -h "kmers retained\|sequences retained" "$OUTPUT_DIR"/muset_*.log | tail -2
echo "=== COMPLETE — $(date) ==="
