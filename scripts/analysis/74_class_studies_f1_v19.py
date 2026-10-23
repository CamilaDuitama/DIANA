#!/usr/bin/env python3
"""Dev-fold per-class F1 against the number of training studies of the class (v19, descriptive).

For every class present in at least two training BioProjects (a single-study class has no dev-fold
prediction made by a model that saw it), the one-vs-rest F1 of the out-of-fold predictions (argmax)
of DIANA, logistic regression and random forest on the v19 dev folds, and the best of the three.
No held-out run is used. Writes results/class_studies_v19/per_class_f1.tsv and by_studies.tsv.

    ./env/bin/python scripts/analysis/74_class_studies_f1_v19.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/class_studies_v19"
TASKS = ["community_type", "feature", "sample_host", "material"]
MODELS = {"DIANA": "results/devfold_probs_v19/probs_DIANA_{t}.tsv",
          "LogReg": "results/baselines_v19/{t}/oof_probs_LogisticRegression_Bal.tsv",
          "RandomForest": "results/baselines_v19/{t}/oof_probs_RandomForest_Bal.tsv"}
BINS, LABELS = [1, 2, 3, 5, 10, 1000], ["2", "3", "4-5", "6-10", ">10"]


def main() -> None:
    meta = pd.read_csv(ROOT / "data/splits_v19/train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    rows = []
    for t in TASKS:
        studies = meta.dropna(subset=[t]).groupby(t).archive_project.nunique()
        runs = meta[t].value_counts()
        for m, path in MODELS.items():
            d = pd.read_csv(ROOT / path.format(t=t), sep="\t")
            pred = d[[c for c in d.columns if c.startswith("p_")]].idxmax(axis=1).str[2:]
            for c in studies[studies >= 2].index:
                rows.append({"field": t, "class": c, "training_studies": int(studies[c]), "training_runs": int(runs[c]),
                             "model": m, "f1": f1_score(d.y_true == c, pred == c, zero_division=0)})
    r = pd.DataFrame(rows)
    wide = r.pivot_table(index=["field", "class", "training_studies", "training_runs"], columns="model", values="f1").reset_index()
    wide["best"] = wide[list(MODELS)].max(axis=1)
    wide["studies_bin"] = pd.cut(wide.training_studies, BINS, labels=LABELS)
    by = wide.groupby("studies_bin", observed=True).agg(classes=("class", "size"), median_best_f1=("best", "median"),
                                                        share_best_f1_at_least_0_5=("best", lambda x: (x >= 0.5).mean()))
    OUT.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT / "per_class_f1.tsv", sep="\t", index=False)
    by.to_csv(OUT / "by_studies.tsv", sep="\t")
    print(by.round(2).to_string())
    print("Spearman, training studies vs best F1:", round(wide[["training_studies", "best"]].corr("spearman").iloc[0, 1], 2), "over", len(wide), "classes")


if __name__ == "__main__":
    main()
