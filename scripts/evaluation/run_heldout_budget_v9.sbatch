#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_budget_heldout
#SBATCH --output=logs/final_eval_budget_v9/heldout_%j.out
#SBATCH --error=logs/final_eval_budget_v9/heldout_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# HELD-OUT READ 9 (PROJECT.md, "Read first", ledger). The five epoch-budget arms
# over the 922 held-out runs, models frozen, taken once after the dev-fold A6
# decision. Submit only after the train-set gate (run_traineval_budget_v9.sbatch)
# has shown every arm trained, and only once.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
for a in multitask community_type feature sample_host material; do
  out="results/final_eval_budget_v9/heldout_$a"
  [ -e "$out/test_metrics.json" ] && { echo "refusing to re-read held-out into $out" >&2; exit 1; }
  echo "===================== heldout / $a ====================="
  ./env/bin/diana-test \
    --model   "results/final_budget_v9/$a/final_model/best_model.pth" \
    --config  "results/final_budget_v9/$a/final_model/final_training_config.json" \
    --matrix  data/matrices/matrix_v9_heldout/unitigs.frac.mat \
    --metadata data/splits_v9/test_metadata.tsv \
    --eligibility data/splits_v9/class_eligibility.tsv \
    --test-ids data/splits_v9/test_accessions.txt \
    --output  "$out" --device cpu
done
echo "=== COMPLETE — $(date) ==="
