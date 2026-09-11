#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --job-name=diana_biology
#SBATCH --output=logs/biology/bio_%a_%j.out
#SBATCH --error=logs/biology/bio_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# B6 and B7: what signal does the model actually use? (R1.1, R3.11)
#
# The only open referee point that no accuracy tuning can address. The measurements
# so far say the features separate these classes WITHIN the training distribution
# (0.95-1.00 macro-F1 on train) and that the separation does not transfer across
# studies: tau-normalisation showed the deficit is in the representation rather than
# the classifier, and the label-shift work showed p(x|y) itself moves between
# studies. So the question is what that signal is made of.
#
# B6 ranks unitigs by how much they push each class logit and emits their sequences
# for BLAST. B7 characterises each trunk dimension: eta-squared against the task's
# classes, and Spearman against data-derived depth and library length, which is the
# confounder check.
#
# Run per task, since DIANA is four single-task networks.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

TASKS=(community_type feature sample_host material)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
D="results/search_v9_final_single_${T}"

echo "=== B6 attribution: $T  $(date) ==="
./env/bin/python scripts/analysis/12_attribute_to_unitigs.py \
    --model  "$D/final_model/best_model.pth" \
    --config "$D/final_training_config.json" \
    --method both \
    --output "results/attribution_v9/$T"

echo "=== B7 trunk dimensions: $T  $(date) ==="
./env/bin/python scripts/analysis/13_characterise_trunk_dims.py \
    --model  "$D/final_model/best_model.pth" \
    --config "$D/final_training_config.json" \
    --output "results/trunk_dims_v9/$T"
echo "=== COMPLETE — $(date) ==="
