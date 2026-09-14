#!/bin/bash
#SBATCH --job-name=logan_abund_probe
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/logan_abund_probe/probe_%A_%a.out
#SBATCH --error=logs/logan_abund_probe/probe_%A_%a.err

################################################################################
# Price the fix for the diana-predict Logan abundance bug, on one held-out run.
#
# The bug: 01_count_kmers.sh counts reference k-mer OCCURRENCES. Logan input is
# assembled, so every occurrence count is 1 and kmat_tools' abundance equals the
# completeness fraction. Logan does record coverage in the ka:f: header field.
#
# The fix under test: add --output-kmer-positions to the existing
# back_to_sequences call, then weight each reference k-mer by the ka:f: of the
# sample unitig it was found in (01b_logan_abundance_counts.py) before the
# unchanged 02_aggregate_to_unitigs.sh step.
#
# Reports the wall time of each step and checks the resulting FRACTION vector is
# byte-identical to the one already in results/v9_vectors: presence is unchanged
# by the reweighting, so any difference is a bug in the fix.
#
# Usage: ACC_LIST=<file> sbatch --array=1-N scripts/inference/run_logan_abundance_probe.sbatch
################################################################################

set -euo pipefail
ROOT=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
cd "$ROOT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export KMP_INIT_AT_FORK=FALSE
export KMP_AFFINITY=disabled
export MPLBACKEND=Agg

ACC_LIST=${ACC_LIST:?set ACC_LIST}
REF_KMERS=data/matrices/matrix_v9_train/matrix.filtered.fasta
UNITIGS=data/matrices/matrix_v9_train/unitigs.fa
KMER_SIZE=31
N_FEATURES=110202
B2S=./external/back_to_sequences/target/release/back_to_sequences
OUTDIR=results/logan_abundance_probe

ACC=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$ACC_LIST")
[ -n "$ACC" ] || { echo "ERROR: no accession at line ${SLURM_ARRAY_TASK_ID:-1}" >&2; exit 1; }
SRC=data/unitigs/${ACC}.unitigs.fa.gz
[ -e "$SRC" ] || { echo "ERROR: no Logan unitigs for $ACC" >&2; exit 1; }

D="$OUTDIR/$ACC"
mkdir -p "$D"
echo "=== $ACC === node=$(hostname) gz_bytes=$(stat -Lc %s "$SRC") start=$(date)"

t0=$SECONDS
"$B2S" --in-kmers "$REF_KMERS" --in-sequences "$SRC" \
       --out-kmers "$D/positions.txt" --counted-kmer-threshold 1 \
       --output-kmer-positions --threads "${SLURM_CPUS_PER_TASK:-8}"
t1=$SECONDS
echo "STEP back_to_sequences_positions $((t1-t0)) s  positions_bytes=$(stat -c %s "$D/positions.txt")"

./env/bin/python scripts/inference/01b_logan_abundance_counts.py \
    "$D/positions.txt" "$SRC" "$D/weighted_counts.txt"
t2=$SECONDS
echo "STEP reweight_by_ka $((t2-t1)) s"

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/logan_abund_${ACC}_XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT
(
  cd "$WORKDIR"
  bash "$ROOT/scripts/inference/02_aggregate_to_unitigs.sh" \
      "$ROOT/$D/weighted_counts.txt" "$(readlink -f "$ROOT/$UNITIGS")" "$KMER_SIZE" \
      "$ROOT/$D/${ACC}_unitig_abundance.txt" "$ROOT/$D/${ACC}_unitig_fraction.txt"
)
t3=$SECONDS
echo "STEP kmat_tools_unitig $((t3-t2)) s"
echo "STEP total $((t3-t0)) s"

N=$(wc -l < "$D/${ACC}_unitig_fraction.txt")
[ "$N" -eq "$N_FEATURES" ] || { echo "ERROR: $ACC produced $N lines, expected $N_FEATURES" >&2; exit 1; }

OLD=results/v9_vectors/$ACC/${ACC}_unitig_fraction.txt
if [ -s "$OLD" ]; then
  if cmp -s "$OLD" "$D/${ACC}_unitig_fraction.txt"; then
    echo "CHECK fraction identical to results/v9_vectors"
  else
    echo "CHECK fraction DIFFERS from results/v9_vectors"
    paste "$OLD" "$D/${ACC}_unitig_fraction.txt" | awk '$1!=$2{n++} END{print "  differing rows:", n+0}'
  fi
else
  echo "CHECK no existing vector to compare"
fi

./env/bin/python - "$D/${ACC}_unitig_abundance.txt" "$D/${ACC}_unitig_fraction.txt" <<'PY'
import sys
import numpy as np
a = np.loadtxt(sys.argv[1]); f = np.loadtxt(sys.argv[2])
nz = a > 0
print(f"ABUND nonzero={nz.sum()} max={a.max():.1f} mean_nonzero={a[nz].mean():.3f}")
print(f"FRAC  nonzero={(f>0).sum()} max={f.max():.4f}")
print(f"IDENTICAL_TO_FRACTION={bool(np.allclose(a, f))}")
PY

rm -f "$D/positions.txt" "$D/weighted_counts.txt"
echo "OK: $ACC end=$(date)"
