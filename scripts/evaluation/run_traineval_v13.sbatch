#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_v13_traingate
#SBATCH --output=logs/final_eval_v13/train_%j.out
#SBATCH --error=logs/final_eval_v13/train_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K4, second step (afterok run_final_fits_v13.sbatch): train-set scores of the five
# v13 arms. Reads only the 2,716 training runs; NOT a held-out read. Gate before the
# chained held-out read: every arm must predict at least two classes on every one of
# its tasks (the X1 divergence collapsed an arm onto the majority class; that must stop
# the chain, not reach held-out).
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
for a in multitask community_type feature sample_host material; do
  echo "===================== train / $a ====================="
  ./env/bin/diana-test \
    --model   "results/final_budget_v13/$a/final_model/best_model.pth" \
    --config  "results/final_budget_v13/$a/final_model/final_training_config.json" \
    --matrix  data/matrices/matrix_v9_train/unitigs.frac.mat \
    --metadata data/splits_v13/train_metadata.tsv \
    --eligibility data/splits_v13/class_eligibility.tsv \
    --test-ids data/splits_v13/train_accessions.txt \
    --output  "results/final_eval_v13/train_$a" --device cpu
done
echo "===================== gate ====================="
./env/bin/python - <<'EOF'
import json, sys
import pandas as pd
arms = {"multitask": ["community_type", "feature", "sample_host", "material"],
        **{t: [t] for t in ["community_type", "feature", "sample_host", "material"]}}
bad = []
for arm, tasks in arms.items():
    p = pd.read_csv(f"results/final_eval_v13/train_{arm}/test_predictions.tsv", sep="\t")
    m = json.load(open(f"results/final_eval_v13/train_{arm}/test_metrics.json"))
    for t in tasks:
        n = p[f"{t}_pred"].nunique()
        f1 = m.get(t, {}).get("f1_macro_eligible")
        print(f"{arm:15s} {t:15s} predicted classes on train: {n:2d}  f1_macro_eligible(train) = {f1}")
        if n < 2:
            bad.append((arm, t))
if bad:
    print("GATE FAILED: an arm predicts a single class on train:", bad, file=sys.stderr)
    sys.exit(1)
print("gate passed")
EOF
echo "=== COMPLETE — $(date) ==="
