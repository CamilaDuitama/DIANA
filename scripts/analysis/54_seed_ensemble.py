#!/usr/bin/env python3
"""X7b: a declared ensemble of seeds on the dev folds, and the seed-to-seed spread.

Input: <base>/<arm>/seed<s>/fold<k>/probs_last.npz from 04_epoch_budget_dev_folds.py run with
--seed s --max-epochs <that arm's fixed budget> --save-probs, for every seed in --seeds. The
ensemble averages the softmax probabilities over seeds and takes the argmax. Output, in
--out: oof_<arm>_<task>.tsv and epoch_budget.tsv (the arms' fixed budgets, copied from
--budgets) in the 37_ format, so 48_paired_sig_vs_fraction.py --single-only compares the
ensemble with the single-seed arms; and seed_spread.tsv: per arm and task, the pooled
f1_macro_eligible of each single seed, their mean and standard deviation (the noise floor
for any comparison between variants), and the ensemble's value.

    ./env/bin/python scripts/analysis/54_seed_ensemble.py --base results/x7b_seeds_v9 \\
        --seeds 42,1,2,3,4 --out results/x7b_ensemble_v9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
N_FOLDS = 5


def f1_elig(y, p, eligible):
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--arms", default=",".join(TASKS), help="single-task arms (default: the four tasks)")
    ap.add_argument("--budgets", type=Path, default=ROOT / "results/epoch_budget_v9/epoch_budget.tsv")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits_v9")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    meta = pd.read_csv(a.splits / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    elig_tbl = pd.read_csv(a.splits / "class_eligibility.tsv", sep="\t")
    bud = pd.read_csv(a.budgets, sep="\t").set_index("arm")
    spread, budgets = [], []
    for arm in a.arms.split(","):
        task = arm
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        rows, per_seed = [], {s: [] for s in seeds}
        for k in range(N_FOLDS):
            dirs = {s: a.base / arm / f"seed{s}" / f"fold{k}" for s in seeds}
            runs = [l.strip() for l in open(dirs[seeds[0]] / "eval_runs.txt") if l.strip()]
            classes = json.load(open(dirs[seeds[0]] / "label_classes.json"))[task]
            P = []
            for s in seeds:
                r2 = [l.strip() for l in open(dirs[s] / "eval_runs.txt") if l.strip()]
                if r2 != runs:
                    raise SystemExit(f"{arm} fold {k}: seed {s} scored different runs")
                pr = np.load(dirs[s] / "probs_last.npz")[task]
                P.append(pr)
                per_seed[s].append(pd.DataFrame({"Run_accession": runs, "pred": [classes[i] for i in pr.argmax(1)]}))
            mean_p = np.mean(P, axis=0)
            truth = meta.loc[runs, task]
            rows.append(pd.DataFrame({"Run_accession": runs, "fold": k,
                                      f"{task}_pred": [classes[i] for i in mean_p.argmax(1)],
                                      f"{task}_true": [None if pd.isna(v) else str(v) for v in truth]}))
        oof = pd.concat(rows)
        oof.to_csv(a.out / f"oof_{arm}_{task}.tsv", sep="\t", index=False)
        budgets.append({"arm": arm, "budget": int(bud.loc[arm, "budget"]), "peaked": True,
                        "rule": f"ensemble of seeds {a.seeds} at the fixed budget"})
        lab = oof[f"{task}_true"].notna()
        y = oof.loc[lab, f"{task}_true"].astype(str).to_numpy()
        ens = f1_elig(y, oof.loc[lab, f"{task}_pred"].astype(str).to_numpy(), eligible)
        singles = []
        for s in seeds:
            d = pd.concat(per_seed[s]).set_index("Run_accession").loc[oof.loc[lab, "Run_accession"]]
            singles.append(f1_elig(y, d.pred.astype(str).to_numpy(), eligible))
        spread.append({"arm": arm, "task": task, "n_seeds": len(seeds), "single_seed_mean": float(np.mean(singles)),
                       "single_seed_std": float(np.std(singles, ddof=1)), "single_seed_min": float(np.min(singles)),
                       "single_seed_max": float(np.max(singles)), "ensemble": float(ens),
                       **{f"seed_{s}": float(v) for s, v in zip(seeds, singles)}})
    pd.DataFrame(budgets).to_csv(a.out / "epoch_budget.tsv", sep="\t", index=False)
    sp = pd.DataFrame(spread)
    sp.to_csv(a.out / "seed_spread.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(sp.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
