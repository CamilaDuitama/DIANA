#!/usr/bin/env python3
"""Matthews correlation coefficient for every model, held-out and train.

MCC is recommended alongside macro-F1 for imbalanced multi-class problems because it
is computed from the whole confusion matrix at once: it is high only when the model
does well on all of true positives, true negatives, false positives and false
negatives, so it cannot be inflated by doing well on one large class alone the way
accuracy can.

It is NOT a replacement for `f1_macro_eligible` here, and this script does not treat
it as one. Multiclass MCC is derived from the global confusion matrix, so it is still
influenced by how frequent each class is; macro-F1 averages per-class F1 with equal
weight per class, which is the quantity this project cares about. MCC is reported as
a companion summary. Switching the headline to it after seeing the numbers would be
the metric-switching CLAUDE.md forbids.

Intervals resample whole BioProjects, as everywhere else in v9.

    ./env/bin/python scripts/evaluation/13_mcc_heldout_v9.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
GROUP = "archive_project"


def mcc_ci(y_true, y_pred, groups, n_boot: int, seed: int) -> tuple[float, float, float]:
    obs = float(matthews_corrcoef(y_true, y_pred))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in drawn])
        # a resample can leave one class only, where MCC is undefined
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(matthews_corrcoef(y_true[idx], y_pred[idx]))
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return obs, float("nan"), float("nan")
    return obs, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/final_eval_v9/mcc.tsv")
    args = ap.parse_args()

    base = pd.read_csv(PROJECT_ROOT / "results/baseline_predictions_v9/heldout_predictions.tsv",
                       sep="\t")
    rows = []
    for task in TASKS:
        bt = base[base.task == task]
        ref = bt[bt.model == bt.model.iloc[0]][["Run_accession", "y_true", GROUP]]
        order = ref.Run_accession.to_numpy()
        y_true = ref.y_true.astype(str).to_numpy()
        groups = ref[GROUP].astype(str).to_numpy()

        preds = {m: g.set_index("Run_accession").y_pred.astype(str).reindex(order).to_numpy()
                 for m, g in bt.groupby("model")}
        for label, sub in [("DIANA", f"heldout_single_{task}"),
                           ("DIANA multi-task", "heldout_multitask")]:
            f = PROJECT_ROOT / f"results/final_eval_v9/{sub}/test_predictions.tsv"
            d = pd.read_csv(f, sep="\t")
            preds[label] = d.set_index("Run_accession")[f"{task}_pred"].astype(str) \
                            .reindex(order).to_numpy()

        for model, pv in preds.items():
            if pd.isna(pv).any():
                raise SystemExit(f"{task}/{model}: predictions missing for scored rows")
            obs, lo, hi = mcc_ci(y_true, pv, groups, args.n_boot, args.seed)
            rows.append({"task": task, "model": model, "split": "test",
                         "mcc": obs, "mcc_ci_low": lo, "mcc_ci_high": hi,
                         "n_scored": len(y_true), "n_groups": int(len(np.unique(groups)))})
            print(f"  {task:<16}{model:<24}MCC={obs:.3f} [{lo:.3f}, {hi:.3f}]")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
