#!/usr/bin/env python3
"""Can a simpler model flag mislabels as well as DIANA? (R1.3, R3.9 for the detector)

Table 2 shows DIANA ties the tuned baselines as a *classifier* on three of four
tasks. The remaining justification for a neural network is that it makes a better
*detector*. That claim was never tested against a baseline, so this tests it.

The detector is identical for every model: flag a run when the model gives the label
on file less than some probability, i.e. score = 1 - P(stated label), thresholded on
a false-flag budget. Ground truth is the planted-mislabel flags. Nothing about the
scoring differs between DIANA and a baseline, so any difference is the model's.

LinearSVM has no `predict_proba` and therefore cannot be used this way at all. That
is reported rather than repaired with `CalibratedClassifierCV`, which would be a
different model from the one in the classification tables. It matters because
LinearSVM is the strongest classification baseline on `sample_host` and `material`.

    ./env/bin/python scripts/analysis/19_baseline_anomaly_detection.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
TARGET_FPR = [0.05, 0.10]


def score_one(y_planted, score) -> dict:
    out = {"roc_auc": float(roc_auc_score(y_planted, score)),
           "average_precision": float(average_precision_score(y_planted, score)),
           "n_scored": int(len(y_planted)), "n_planted": int(y_planted.sum())}
    fpr, tpr, thr = roc_curve(y_planted, score)
    for t in TARGET_FPR:
        i = max(int(np.searchsorted(fpr, t, side="right") - 1), 0)
        out[f"tpr_at_fpr{int(t*100)}"] = float(tpr[i])
        out[f"threshold_at_fpr{int(t*100)}"] = float(thr[i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", type=Path,
                    default=PROJECT_ROOT / "results/baseline_predictions_v9")
    ap.add_argument("--planted", type=Path,
                    default=PROJECT_ROOT / "results/planted_mislabels_v9/planted_test_mixed_r0.1.tsv")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/anomaly_detection_baselines")
    args = ap.parse_args()

    plant = pd.read_csv(args.planted, sep="\t")
    rows, skipped = [], []
    for task in TASKS:
        f = args.baseline_dir / f"heldout_probabilities_{task}.tsv"
        if not f.exists():
            raise SystemExit(f"missing {f}; re-run 09_baselines_v9.py")
        prob = pd.read_csv(f, sep="\t")
        flag_col = f"{task}_planted"
        pl = plant[["Run_accession", task, flag_col]].dropna(subset=[task, flag_col])

        for model, g in prob.groupby("model"):
            m = g.merge(pl, on="Run_accession", how="inner", suffixes=("", "_stated"))
            if m.empty:
                continue
            # the label a curator would see is the possibly-corrupted one
            stated = m[task].astype(str).to_numpy()
            cols = {c[2:]: c for c in m.columns if c.startswith("p_")}
            p_stated = np.array([m[cols[s]].iloc[i] if s in cols else np.nan
                                 for i, s in enumerate(stated)], dtype=float)
            ok = ~np.isnan(p_stated)
            y = m.loc[ok, flag_col].astype(int).to_numpy()
            if y.sum() == 0 or y.sum() == len(y):
                continue
            r = score_one(y, 1.0 - p_stated[ok])
            rows.append({"task": task, "model": model, **r})

    # DIANA, scored the same way, for the comparison
    for task in TASKS:
        f = PROJECT_ROOT / f"results/anomaly_detection/roc_DIANA_{task}.tsv"
        if f.exists():
            d = pd.read_csv(f, sep="\t").iloc[0]
            rows.append({"task": task, "model": "DIANA",
                         "roc_auc": d.roc_auc, "average_precision": d.average_precision,
                         "n_scored": int(d.n_scored), "n_planted": int(d.n_planted),
                         "tpr_at_fpr5": d.tpr_at_fpr5, "tpr_at_fpr10": d.tpr_at_fpr10,
                         "threshold_at_fpr5": d.threshold_at_fpr5,
                         "threshold_at_fpr10": d.threshold_at_fpr10})

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "detector_comparison.tsv", sep="\t", index=False)

    L = ["Anomaly detection: DIANA against the tuned baselines", "",
         "Identical detector for every model: flag when 1 - P(stated label) exceeds a",
         "threshold set by false-flag budget. Ground truth is the planted flags.", "",
         f"{'task':<16}{'model':<24}{'AUC':>7}{'AP':>7}{'TPR@5%':>9}{'TPR@10%':>9}"]
    for task in TASKS:
        g = df[df.task == task].sort_values("roc_auc", ascending=False)
        for _, r in g.iterrows():
            star = "  <-- DIANA" if r.model == "DIANA" else ""
            L.append(f"{task:<16}{r.model:<24}{r.roc_auc:>7.3f}{r.average_precision:>7.3f}"
                     f"{r.tpr_at_fpr5:>9.3f}{r.tpr_at_fpr10:>9.3f}{star}")
        L.append("")
    L += ["LinearSVM_Bal is absent: LinearSVC has no predict_proba, so it cannot be",
          "used as a detector at all. It is the strongest classification baseline on",
          "sample_host and material, so that is a point in DIANA's favour and should be",
          "reported rather than worked around."]
    report = "\n".join(L)
    print(report)
    (args.output / "summary.txt").write_text(report + "\n")
    json.dump(rows, open(args.output / "detector_comparison.json", "w"), indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
