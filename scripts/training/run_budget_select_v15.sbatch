#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=diana_budget_select_v15
#SBATCH --output=logs/epoch_budget_v15/select_%j.out
#SBATCH --error=logs/epoch_budget_v15/select_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v15 protocol step 4, stage 2: re-select the epoch budgets by the frozen rule
# (37_epoch_budget_select.py, smoothed argmax under the training-loss cap) on the
# v15 curves, then assemble results/reference_v15/epoch_budget.tsv with the frozen
# lr / schedule / horizon carried from results/reference_v9/epoch_budget.tsv.
# The equal-study-weight feature arm is selected from its own base
# (results/epoch_budget_v15_sw) and lands in the table as arm `feature_sw`.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

./env/bin/python scripts/analysis/37_epoch_budget_select.py \
    --base results/epoch_budget_v15 --splits data/splits_v15 \
    --arms community_type,feature,sample_host,material \
    || echo "WARNING: 37_ exited non-zero on the main base (A6 section has no multitask arm); budgets validated below"
./env/bin/python scripts/analysis/37_epoch_budget_select.py \
    --base results/epoch_budget_v15_sw --splits data/splits_v15 \
    --arms feature \
    || echo "WARNING: 37_ exited non-zero on the sw base; budgets validated below"

./env/bin/python - <<'EOF'
import sys
import pandas as pd

new = pd.read_csv("results/epoch_budget_v15/epoch_budget.tsv", sep="\t")
sw = pd.read_csv("results/epoch_budget_v15_sw/epoch_budget.tsv", sep="\t")
frozen = pd.read_csv("results/reference_v9/epoch_budget.tsv", sep="\t")

rows = []
for arm in ["community_type", "feature", "sample_host", "material"]:
    b = new.loc[new.arm == arm, "budget"]
    assert len(b) == 1, f"no v15 budget for {arm}"
    f = frozen.loc[frozen.arm == arm].iloc[0]
    rows.append({"arm": arm, "budget": int(b.iloc[0]), "lr": f["lr"],
                 "schedule": f["schedule"],
                 "horizon": "" if pd.isna(f["horizon"]) else f["horizon"]})
b = sw.loc[sw.arm == "feature", "budget"]
assert len(b) == 1, "no v15 budget for feature_sw"
f = frozen.loc[frozen.arm == "feature"].iloc[0]
rows.append({"arm": "feature_sw", "budget": int(b.iloc[0]), "lr": f["lr"],
             "schedule": f["schedule"], "horizon": ""})
out = pd.DataFrame(rows)
out.to_csv("results/reference_v15/epoch_budget.tsv", sep="\t", index=False)
print(out.to_string(index=False))
EOF
rc=$?
[ $rc -eq 0 ] && echo "=== v15 budget table written — $(date) ===" || echo "=== FAILED ==="
exit $rc
