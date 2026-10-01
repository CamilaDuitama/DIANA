#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-63
#SBATCH --job-name=diana_hash_baselines
#SBATCH --output=logs/baselines_hash_v9/task_%a_%j.out
#SBATCH --error=logs/baselines_hash_v9/task_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# H2: the tuned baselines on the four k-mer hash matrices (H1), v9 grid, same grouped dev
# folds and selection rule as the fraction side. 4 arms x 4 tasks x 4 models = 64 tasks:
#   0-15 s1000_f272 | 16-31 s1000_f28 | 32-47 s1000_f14 | 48-63 s100_f272
# RandomForest: the full grid (max_features 0.05 included) at floor 272; `sqrt` only at
# floors 28 and 14, whose width is ~1 to 2 M columns (as S8.3a did on the s8a input).
# The fraction side (results/baselines_sig_v9/{pred,score}_fraction_*, v9 splits, full
# grid) is linked into each output directory so the pool compares like for like.
# afterok run_hash_matrix_v9.sbatch.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TAGS=(s1000_f272 s1000_f28 s1000_f14 s100_f272)
TASKS=(community_type feature sample_host material)
MODELS=(LogisticRegression_Bal LinearSVM_Bal RandomForest_Bal kNN)
i=$SLURM_ARRAY_TASK_ID
TAG=${TAGS[$((i / 16))]}; TASK=${TASKS[$(((i % 16) / 4))]}; MODEL=${MODELS[$((i % 4))]}
case "$TAG" in *_f272) RF=full ;; *) RF=sqrt ;; esac
NPZ=results/hash_matrix_v9/hash_${TAG}_train.npz
OUT=results/baselines_hash_${TAG}_v9
[ -s "$NPZ" ] || { echo "$NPZ missing (H1)" >&2; exit 1; }
mkdir -p "$OUT"
for f in results/baselines_sig_v9/pred_fraction_*.tsv results/baselines_sig_v9/score_fraction_*.tsv; do
  ln -sf "$PWD/$f" "$OUT/$(basename "$f")"
done
echo "=== H2  $TAG / $TASK / $MODEL  rf-grid $RF  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/14_tune_baselines_s8a_v9.py --splits data/splits_v9 --rep hash --hash-npz "$NPZ" \
    --task "$TASK" --model "$MODEL" --n-jobs "$SLURM_CPUS_PER_TASK" --out "$OUT" --rf-grid "$RF"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
