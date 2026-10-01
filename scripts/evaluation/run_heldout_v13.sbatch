#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_v13_heldout
#SBATCH --output=logs/final_eval_v13/heldout_%j.out
#SBATCH --error=logs/final_eval_v13/heldout_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K5: HELD-OUT READ 10 (PROJECT.md, "Read first", ledger). The five v13 fraction arms
# over the 922 held-out runs, models frozen, budgets and A6 from the v13 dev folds.
# Chained afterok run_traineval_v13.sbatch (the train-set gate) on Camila's decision of
# 2026-09-30. Held-out runs of the dropped classes are out-of-vocabulary: counted in
# test_metrics.json, not scored. Taken once: refuses to re-read into an existing output.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
for a in multitask community_type feature sample_host material; do
  out="results/final_eval_v13/heldout_$a"
  [ -e "$out/test_metrics.json" ] && { echo "refusing to re-read held-out into $out" >&2; exit 1; }
  echo "===================== heldout / $a ====================="
  ./env/bin/diana-test \
    --model   "results/final_budget_v13/$a/final_model/best_model.pth" \
    --config  "results/final_budget_v13/$a/final_model/final_training_config.json" \
    --matrix  data/matrices/matrix_v9_heldout/unitigs.frac.mat \
    --metadata data/splits_v13/test_metadata.tsv \
    --eligibility data/splits_v13/class_eligibility.tsv \
    --test-ids data/splits_v13/test_accessions.txt \
    --output  "$out" --device cpu
done
echo "=== COMPLETE — read 10 taken $(date) ==="
