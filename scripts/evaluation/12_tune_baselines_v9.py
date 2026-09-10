#!/usr/bin/env python3
"""Tune the baselines on the same dev folds the networks were tuned on.

Why this is necessary
---------------------
`09_baselines_v9.py` uses fixed scikit-learn settings -- `LogisticRegression(
max_iter=2000)`, `RandomForest(n_estimators=300, max_features='sqrt')`, `kNN(k=5)`
-- while every MLP arm receives 100 Optuna trials over 5 folds. Reporting a network
win against out-of-the-box baselines would let a referee say the win was bought with
tuning budget, and **R1.3 is exactly that referee asking whether the added
complexity is justified**. The comparison has to be fair in both directions.

So each baseline gets a grid searched over the **same 5 BioProject-grouped dev
folds** the networks used, scored on the **same metric** (`f1_macro_eligible`), with
the held-out set untouched. Grids are small because these models have few
consequential knobs -- that asymmetry is inherent, not something this script hides.

If a tuned baseline beats the fixed one, Table 2's comparator gets *stronger* and
DIANA's job gets harder. That is the point.

    ./env/bin/python scripts/evaluation/12_tune_baselines_v9.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from diana.evaluation.metrics import classification_metrics  # noqa: E402

SPLITS = PROJECT_ROOT / "data/splits_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]

# Deliberately compact. These models have few knobs that matter; inflating the grid
# to "match" 100 Optuna trials would be theatre, not fairness.
GRIDS = {
    "LogisticRegression_Bal": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "LinearSVM_Bal": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)],
    "RandomForest_Bal": [{"n_estimators": n, "max_features": f, "min_samples_leaf": l}
                         for n in (300, 800) for f in ("sqrt", 0.05) for l in (1, 3)],
    "kNN": [{"n_neighbors": k, "weights": w}
            for k in (1, 3, 5, 10, 20) for w in ("uniform", "distance")],
}


def build(name: str, params: dict, seed: int):
    if name == "LogisticRegression_Bal":
        return LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, **params)
    if name == "LinearSVM_Bal":
        return LinearSVC(max_iter=8000, class_weight="balanced", **params)
    if name == "RandomForest_Bal":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed, **params)
    if name == "kNN":
        return KNeighborsClassifier(n_jobs=-1, **params)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/baselines_tuned_v9")
    args = ap.parse_args()

    from load_v9_features import load_features

    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    dev = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t")
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    train = train.merge(dev[["Run_accession", "fold"]], on="Run_accession", how="inner")
    X, kept = load_features(train.Run_accession.tolist())
    train = train[train.Run_accession.isin(kept)].reset_index(drop=True)
    print(f"train {X.shape}, folds {sorted(train.fold.unique())}")

    rows = []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"])
        labelled = train[task].notna().to_numpy()
        for name, grid in GRIDS.items():
            for params in grid:
                scores = []
                for f in sorted(train.fold.unique()):
                    tr = labelled & (train.fold != f).to_numpy()
                    te = labelled & (train.fold == f).to_numpy()
                    if te.sum() < 5 or tr.sum() < 20:
                        continue
                    ytr = train.loc[tr, task].astype(str).to_numpy()
                    yte = train.loc[te, task].astype(str).to_numpy()
                    seen = set(ytr)
                    keep = np.array([y in seen for y in yte])   # OOV excluded, as elsewhere
                    if keep.sum() < 5:
                        continue
                    m = build(name, params, args.seed).fit(X[tr], ytr)
                    pred = m.predict(X[te][keep])
                    scores.append(classification_metrics(
                        yte[keep], pred, eligible)["f1_macro_eligible"])
                if scores:
                    rows.append({"task": task, "model": name,
                                 "params": json.dumps(params),
                                 "mean_f1_eligible": float(np.nanmean(scores)),
                                 "n_folds": len(scores)})
                    print(f"  {task:<16}{name:<24}{json.dumps(params):<52}"
                          f"{np.nanmean(scores):.4f}")

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "grid_scores.tsv", sep="\t", index=False)

    best = (df.sort_values("mean_f1_eligible", ascending=False)
              .groupby(["task", "model"], as_index=False).first())
    best.to_csv(args.output / "best_per_model.tsv", sep="\t", index=False)

    lines = ["Tuned baselines — selected on the 5 BioProject-grouped dev folds", "",
             "Same folds and same metric (f1_macro_eligible) as the network arms.",
             "Held-out was not used. Apply these settings when re-running",
             "09_baselines_v9.py to produce the Table 2 comparator.", ""]
    for task in TASKS:
        sub = best[best.task == task].sort_values("mean_f1_eligible", ascending=False)
        if sub.empty:
            continue
        lines.append(f"{task}")
        for _, r in sub.iterrows():
            lines.append(f"    {r.model:<24}{r.mean_f1_eligible:.4f}   {r.params}")
        lines.append("")
    report = "\n".join(lines)
    print("\n" + report)
    (args.output / "summary.txt").write_text(report + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
