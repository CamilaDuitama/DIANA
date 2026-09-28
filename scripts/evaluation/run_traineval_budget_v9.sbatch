#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_budget_traineval
#SBATCH --output=logs/final_eval_budget_v9/traineval_%j.out
#SBATCH --error=logs/final_eval_budget_v9/traineval_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Train-set scores of the five epoch-budget arms (results/final_budget_v9/).
# Reads only the 2,716 training runs: this is the gate that shows each arm
# actually trained, and it is NOT a held-out read. The held-out read is a
# separate submission (run_heldout_budget_v9.sbatch) taken once, afterwards.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
for a in multitask community_type feature sample_host material; do
  echo "===================== train / $a ====================="
  ./env/bin/diana-test \
    --model   "results/final_budget_v9/$a/final_model/best_model.pth" \
    --config  "results/final_budget_v9/$a/final_model/final_training_config.json" \
    --matrix  data/matrices/matrix_v9_train/unitigs.frac.mat \
    --metadata data/splits_v9/train_metadata.tsv \
    --eligibility data/splits_v9/class_eligibility.tsv \
    --test-ids data/splits_v9/train_accessions.txt \
    --output  "results/final_eval_budget_v9/train_$a" --device cpu
done
echo "=== COMPLETE — $(date) ==="
