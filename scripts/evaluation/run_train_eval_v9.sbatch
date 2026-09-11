#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=diana_train_eval
#SBATCH --output=logs/heldout_eval_v9/traineval_%j.out
#SBATCH --error=logs/heldout_eval_v9/traineval_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Train-set evaluation of the four single-task models, to complete the table.
# Train is not the protected set; the multi-task arm was already run.
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
for t in community_type feature sample_host material; do
  echo "===================== single_$t ====================="
  ./env/bin/diana-test \
    --model  "results/search_v9_final_single_$t/final_model/best_model.pth" \
    --config "results/search_v9_final_single_$t/final_training_config.json" \
    --matrix data/matrices/matrix_v9_train/unitigs.frac.mat \
    --metadata data/splits_v9/train_metadata.tsv \
    --eligibility data/splits_v9/class_eligibility.tsv \
    --test-ids data/splits_v9/train_accessions.txt \
    --output "results/final_eval_v9/train_single_$t" --device cpu
done
echo "=== COMPLETE — $(date) ==="
