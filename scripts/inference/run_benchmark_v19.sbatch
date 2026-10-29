#!/bin/bash
#SBATCH --job-name=diana_bench_v19
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/benchmark_v19/bench_%A_%a.out
#SBATCH --error=logs/benchmark_v19/bench_%A_%a.err
################################################################################
# End-to-end runtime and memory of diana-predict with the released DIANA model
# (results/final_v19_weights: five networks),
# one held-out run per array task, under /usr/bin/time -v.
#   ROUTE=logan (default)  results/benchmark_v19/runs.txt        Logan unitigs
#   ROUTE=reads            results/benchmark_v19/runs_reads.txt  raw FASTQ (data/validation/raw)
# Run lists and their selection rule: collect_benchmark_v19.py --select.
# Features as the held-out projection (scripts/validation/13_project_v19.sbatch):
# matrix_v17_train reference k-mers (matrix.filtered.fasta) and unitigs.fa, symlinked in
# results/benchmark_v19/matrix/, and the MUSET v0.6.1 kmat_tools (KMAT_TOOLS_BIN).
# maestro-20xx are excluded: back_to_sequences dies there with "Illegal instruction".
# Each task runs in its own node-local directory: kmat_tools writes a large temp file
# into the CWD, and concurrent tasks sharing one directory fail.
# A task whose time_v.txt exists is skipped, so the array can be resubmitted.
#
#   sbatch --array=0-29 scripts/inference/run_benchmark_v19.sbatch
#   ROUTE=reads sbatch --array=0-9 --export=ALL,ROUTE scripts/inference/run_benchmark_v19.sbatch
# BENCH=results/benchmark_v19net (with --export=ALL,ROUTE,BENCH): the rerun with DIANA as the shared network alone
# (PROTOCOLS.md v19 amendment 2026-10-26); run lists and matrix stay in results/benchmark_v19.
#   ./env/bin/python scripts/inference/collect_benchmark_v19.py
################################################################################
set -euo pipefail
ROOT=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
cd "$ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
export KMAT_TOOLS_BIN="$ROOT/external/muset/bin/kmat_tools"
export PATH="$ROOT/env/bin:$PATH"

ROUTE=${ROUTE:-logan}
case "$ROUTE" in
    logan) LIST=results/benchmark_v19/runs.txt ;;
    reads) LIST=results/benchmark_v19/runs_reads.txt ;;
    *) echo "ERROR: ROUTE must be logan or reads, got $ROUTE" >&2; exit 1 ;;
esac
LINE=$(grep -v '^#' "$LIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$LINE" ] || { echo "ERROR: no run at index $SLURM_ARRAY_TASK_ID of $LIST" >&2; exit 1; }
RUN=$(cut -f1 <<< "$LINE")
FIRST=$(cut -f3 <<< "$LINE" | cut -d, -f1)   # diana-predict finds the _2 mate itself
MATRIX="$ROOT/results/benchmark_v19/matrix"
OUT="$ROOT/${BENCH:-results/benchmark_v19}/$ROUTE/$RUN"
if [ -e "$OUT/time_v.txt" ]; then
    echo "$ROUTE $RUN already timed ($OUT/time_v.txt); skipping."
    exit 0
fi
for f in "$MATRIX/reference_kmers.fasta" "$MATRIX/unitigs.fa" "$FIRST"; do
    [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
command -v back_to_sequences >/dev/null || { echo "ERROR: back_to_sequences not on PATH" >&2; exit 1; }
mkdir -p "$OUT"

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/diana_bench_v19_${RUN}_XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT
echo "=== $ROUTE $RUN === input=$(cut -f2 <<< "$LINE") bytes node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p') start=$(date)"
cd "$WORKDIR"
set +e
/usr/bin/time -v -o "$OUT/time_v.txt.part" diana-predict \
    --sample "$FIRST" \
    --model "$ROOT/results/final_v19_weights" \
    --muset-matrix "$MATRIX" \
    --output "$OUT" \
    --threads "${SLURM_CPUS_PER_TASK:-8}"
STATUS=$?
set -e
mv "$OUT/time_v.txt.part" "$OUT/time_v.txt"
echo "exit=$STATUS end=$(date)"
exit $STATUS
