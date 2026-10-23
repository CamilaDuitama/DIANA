#!/bin/bash
# v19 composition test on the dev folds: network alone vs network + logistic regression.
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
D=results/devfold_probs_v19
./env/bin/python scripts/analysis/45_composition_probs_v19.py
MODELS=(--model "Separate=$D/probs_Separate_{task}.tsv" --model "SeparateLR=$D/probs_SeparateLR_{task}.tsv"
        --model "Shared=$D/probs_Shared_{task}.tsv" --model "SharedLR=$D/probs_SharedLR_{task}.tsv")
PAIRS=(--pair SeparateLR:Separate --pair SharedLR:Shared --pair SeparateLR:SharedLR)
./env/bin/python scripts/analysis/43_classification_v19.py "${MODELS[@]}" "${PAIRS[@]}" --out results/classification_v19/dev_composition
./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
    --pattern "planted_train_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" --out results/flagging_v19/dev_single_composition
