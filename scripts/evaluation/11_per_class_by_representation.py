#!/usr/bin/env python3
"""Per-class metrics stratified by how well the class is represented in training.

Referee 3 (**R3.13**) argues the corpus is oral-heavy and that headline numbers
therefore say little about the rare classes -- the ones an anomaly detector will
actually be asked about. A macro average already gives every class equal weight,
but it does not show *where* the model fails. This does.

For each task it reports, per class: how many training runs it has, how many
held-out runs, and precision / recall / F1. Classes are then binned by training
representation so the relationship is legible rather than buried in a long table.

Reads the output of `diana-test` (or of the baseline sweep, same format), so it runs
in minutes once a model has been evaluated. It never re-runs a model.

    ./env/bin/python scripts/evaluation/11_per_class_by_representation.py \\
        --predictions results/final_eval_v9/predictions.csv \\
        --label DIANA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
# Bin edges on training-run count. The first bin is the one R3.13 is about.
BINS = [0, 10, 50, 200, np.inf]
BIN_LABELS = ["1-9", "10-49", "50-199", "200+"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="CSV with Run_accession plus <task>_true / <task>_pred columns")
    ap.add_argument("--label", default="model", help="name for this model in the output")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/per_class_by_representation")
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions)
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")

    rows = []
    for task in TASKS:
        tcol, pcol = f"{task}_true", f"{task}_pred"
        if tcol not in pred.columns or pcol not in pred.columns:
            print(f"  {task}: no {tcol}/{pcol} columns, skipping")
            continue
        sub = pred[[tcol, pcol]].dropna(subset=[tcol])
        if sub.empty:
            continue
        y_true = sub[tcol].astype(str).to_numpy()
        y_pred = sub[pcol].astype(str).to_numpy()

        n_train = train[task].value_counts()
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"])
        classes = sorted(set(y_true))
        pr, rc, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0)
        for c, p_, r_, f_, s_ in zip(classes, pr, rc, f1, sup):
            rows.append({"model": args.label, "task": task, "class": c,
                         "n_train": int(n_train.get(c, 0)), "n_heldout": int(s_),
                         "eligible": c in eligible,
                         "precision": float(p_), "recall": float(r_), "f1": float(f_)})

    if not rows:
        raise SystemExit("no task had both true and predicted columns; check --predictions")

    df = pd.DataFrame(rows)
    df["representation"] = pd.cut(df.n_train, bins=BINS, labels=BIN_LABELS, right=False)

    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / f"per_class_{args.label}.tsv", sep="\t", index=False)

    lines = [f"Per-class metrics by training representation — {args.label}  (R3.13)", ""]
    lines.append("Recall averaged over classes within each training-count bin.")
    lines.append("A model that only works on well-represented classes shows a steep")
    lines.append("gradient here even when its macro-F1 looks acceptable.")
    lines.append("")
    piv = (df[df.eligible]
           .groupby(["task", "representation"], observed=True)
           .agg(n_classes=("class", "size"), mean_recall=("recall", "mean"),
                mean_f1=("f1", "mean"), heldout_runs=("n_heldout", "sum"))
           .reset_index())
    lines.append(piv.to_string(index=False))
    lines.append("")
    lines.append("Worst-served eligible classes (lowest F1, >=1 held-out run):")
    worst = df[(df.eligible) & (df.n_heldout > 0)].nsmallest(12, "f1")
    lines.append(worst[["task", "class", "n_train", "n_heldout",
                        "precision", "recall", "f1"]].to_string(index=False))

    report = "\n".join(lines)
    print(report)
    (args.output / f"summary_{args.label}.txt").write_text(report + "\n")
    json.dump(rows, open(args.output / f"per_class_{args.label}.json", "w"), indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
