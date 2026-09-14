#!/bin/bash
#SBATCH --job-name=logan_abund_val
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/logan_abund_probe/validate_%j.out
#SBATCH --error=logs/logan_abund_probe/validate_%j.err

################################################################################
# Pin down the exact abundance convention the v9 TRAINING matrix used, by
# reconstructing a TRAINING run's abundance column from its own Logan unitigs
# and comparing against the column MUSET already wrote.
#
# Why this is needed. The training matrix was built by MUSET with -a 2 and
# --logan, and the kmtricks options record c_ab_min=2 with logan=1, so a k-mer
# whose containing sample unitig had ka:f: < 2 was thresholded. Three candidate
# rules are indistinguishable from the logs:
#
#   drop   coverage below 2 is absent               (kmtricks hard-min)
#   round  round to integer first, then threshold   (keeps ka:f: in [1.5, 2))
#   floor  no threshold, present k-mers count >= 1  (what the prototype did)
#
# The training column is ground truth, so whichever rule reproduces it is the
# convention the held-out vectors must use for G2 to be a like-for-like test.
#
# Usage: ACC=<train accession> COL=<1-based column in train_samples_v9.fof> \
#          sbatch --export=ALL,ACC,COL scripts/inference/run_logan_abundance_validate.sbatch
################################################################################

set -euo pipefail
ROOT=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
cd "$ROOT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export KMP_INIT_AT_FORK=FALSE
export KMP_AFFINITY=disabled
export MPLBACKEND=Agg

ACC=${ACC:?set ACC}
COL=${COL:?set COL}
MAT=data/matrices/matrix_v9_train
REF=$MAT/matrix.filtered.fasta
UNITIGS=$MAT/unitigs.fa
B2S=./external/back_to_sequences/target/release/back_to_sequences
SRC=data/unitigs/${ACC}.unitigs.fa.gz
D=results/logan_abundance_validate/$ACC
mkdir -p "$D"

echo "=== $ACC (fof column $COL) === node=$(hostname) start=$(date)"

# ground truth: column COL of the abundance and fraction matrices (col 1 is the unitig id)
awk -v c=$((COL+1)) '{print $c}' "$MAT/unitigs.abundance.mat" > "$D/truth_abundance.txt"
awk -v c=$((COL+1)) '{print $c}' "$MAT/unitigs.frac.mat"      > "$D/truth_fraction.txt"
echo "truth rows: $(wc -l < "$D/truth_abundance.txt")"

t0=$SECONDS
"$B2S" --in-kmers "$REF" --in-sequences "$SRC" \
       --out-kmers "$D/positions.txt" --counted-kmer-threshold 1 \
       --output-kmer-positions --threads "${SLURM_CPUS_PER_TASK:-8}"
echo "STEP back_to_sequences $((SECONDS-t0)) s"

for RULE in drop round round_each floor; do
    ./env/bin/python scripts/inference/01b_logan_abundance_counts.py \
        "$D/positions.txt" "$SRC" "$D/counts_$RULE.txt" --rule "$RULE" --min-abundance 2
    W=$(mktemp -d "${TMPDIR:-/tmp}/val_${ACC}_${RULE}_XXXXXX")
    ( cd "$W"
      bash "$ROOT/scripts/inference/02_aggregate_to_unitigs.sh" \
          "$ROOT/$D/counts_$RULE.txt" "$(readlink -f "$ROOT/$UNITIGS")" 31 \
          "$ROOT/$D/${RULE}_abundance.txt" "$ROOT/$D/${RULE}_fraction.txt" )
    rm -rf "$W"
done

./env/bin/python - "$D" <<'PY'
import sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
truth_a = np.loadtxt(d / "truth_abundance.txt")
truth_f = np.loadtxt(d / "truth_fraction.txt")
print(f"\ntruth: abundance nonzero={int((truth_a>0).sum())} max={truth_a.max():.2f}  "
      f"fraction nonzero={int((truth_f>0).sum())} max={truth_f.max():.4f}")
for rule in ("drop", "round", "round_each", "floor"):
    a = np.loadtxt(d / f"{rule}_abundance.txt")
    f = np.loadtxt(d / f"{rule}_fraction.txt")
    both = (truth_a > 0) & (a > 0)
    r = np.corrcoef(truth_a[both], a[both])[0, 1] if both.sum() > 2 else float("nan")
    print(f"\n  rule={rule}")
    print(f"    abundance nonzero={int((a>0).sum())} (truth {int((truth_a>0).sum())})  "
          f"max={a.max():.2f}  exact={int(np.isclose(a, truth_a, atol=0.005).sum())}/{len(a)}")
    print(f"    support agreement: both={int(both.sum())}  "
          f"only-truth={int(((truth_a>0)&(a==0)).sum())}  only-mine={int(((truth_a==0)&(a>0)).sum())}")
    print(f"    corr on shared support={r:.5f}  mean abs diff={np.abs(a[both]-truth_a[both]).mean():.4f}")
    print(f"    fraction exact={int(np.isclose(f, truth_f, atol=0.0005).sum())}/{len(f)}")
PY
rm -f "$D/positions.txt" "$D"/counts_*.txt
echo "=== end=$(date)"
