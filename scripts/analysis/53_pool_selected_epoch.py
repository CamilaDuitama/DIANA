#!/usr/bin/env python3
"""X5: pool the out-of-fold predictions of early-stopped fold fits at each fold's selected epoch.

Input: <base>/<arm>/fold<k>/ written by 04_epoch_budget_dev_folds.py with --early-stop, whose
history.json carries `selected_epoch` (the inner-validation argmax). Output, in <base>:
oof_<arm>_<task>.tsv in the 37_ format, epoch_budget.tsv (budget = median selected epoch,
peaked = True) so that 48_paired_sig_vs_fraction.py can compare the set against the
fixed-budget arms, and selected_epochs.tsv with the per-fold epochs.

    ./env/bin/python scripts/analysis/53_pool_selected_epoch.py --base results/x5_random_v9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
ARMS = {"multitask": TASKS, **{t: [t] for t in TASKS}}
N_FOLDS = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits_v9")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    meta = pd.read_csv(a.splits / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    budgets, epochs_rows = [], []
    for arm in a.arms.split(","):
        tasks = ARMS[arm]
        per_task = {t: [] for t in tasks}
        sel = []
        for k in range(N_FOLDS):
            d = a.base / arm / f"fold{k}"
            h = json.load(open(d / "history.json"))
            e = h.get("selected_epoch")
            if e is None:
                raise SystemExit(f"{d}: no selected_epoch (was 04_ run with --early-stop?)")
            sel.append(int(e))
            epochs_rows.append({"arm": arm, "fold": k, "selected_epoch": int(e), "stopped_epoch": h.get("stopped_epoch"),
                                "best_val_macro_f1": h.get("best_val_macro_f1"), "n_fit": h["recipe"]["n_fit"], "n_val": h["recipe"]["n_val"]})
            classes = json.load(open(d / "label_classes.json"))
            runs = [l.strip() for l in open(d / "eval_runs.txt") if l.strip()]
            z = np.load(d / "preds_by_epoch.npz")
            for t in tasks:
                pred_idx = z[t][e - 1]
                truth = meta.loc[runs, t]
                per_task[t].append(pd.DataFrame({
                    "Run_accession": runs, "fold": k,
                    f"{t}_pred": [classes[t][i] if i >= 0 else None for i in pred_idx],
                    f"{t}_true": [None if pd.isna(v) else str(v) for v in truth]}))
        for t in tasks:
            pd.concat(per_task[t]).to_csv(a.base / f"oof_{arm}_{t}.tsv", sep="\t", index=False)
        budgets.append({"arm": arm, "budget": int(np.median(sel)), "selected_epochs": ",".join(map(str, sel)),
                        "peaked": True, "rule": "inner early stopping, per-fold selected epoch"})
    pd.DataFrame(budgets).to_csv(a.base / "epoch_budget.tsv", sep="\t", index=False)
    pd.DataFrame(epochs_rows).to_csv(a.base / "selected_epochs.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(epochs_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
