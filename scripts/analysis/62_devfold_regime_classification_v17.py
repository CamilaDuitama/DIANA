#!/usr/bin/env python3
"""Dev-fold classification by training-support regime, per model (v17).

For every probs_<model>_<task>.tsv in results/devfold_probs_v17 (DIANA, the
shared network and the tuned baselines): macro-F1 over the eligible classes of
each regime, with the regime of a class set by its training-run count
(few-shot < 20, medium-shot 20-100, many-shot > 100; the 48_ definition).
Predictions are the argmax of the stored out-of-fold probabilities.
Output: results/devfold_regime_v17/classification_by_regime.tsv, printed.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v17"
PROBS = ROOT / "results/devfold_probs_v17"
OUT = ROOT / "results/devfold_regime_v17"
TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    rows = []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        support = train[task].dropna().astype(str).value_counts()
        for f in sorted(glob.glob(str(PROBS / f"probs_*_{task}.tsv"))):
            model = re.fullmatch(rf"probs_(.+)_{task}\.tsv", Path(f).name).group(1)
            d = pd.read_csv(f, sep="\t")
            d = d[d.y_true.notna()]
            p_cols = [c for c in d.columns if c.startswith("p_")]
            pred = np.array([p_cols[i][2:] for i in d[p_cols].to_numpy().argmax(1)])
            y = d.y_true.astype(str).to_numpy()
            for name, lo, hi in [("all", 0, np.inf)] + REGIMES:
                cls = sorted(c for c in eligible if lo <= support.get(c, 0) < hi and c in set(y))
                if not cls:
                    continue
                f1 = f1_score(y, pred, labels=cls, average="macro", zero_division=0)
                rows.append({"task": task, "model": model, "regime": name,
                             "n_classes": len(cls), "f1_macro_eligible": round(float(f1), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "classification_by_regime.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    for task in TASKS:
        sub = out[out.task == task].pivot_table(index="model", columns="regime",
                                                values="f1_macro_eligible")
        cols = [c for c in ["all", "few-shot", "medium-shot", "many-shot"] if c in sub.columns]
        print(f"\n{task} (dev folds, macro-F1 over eligible classes of the regime):")
        print(sub[cols].round(3).to_string())


if __name__ == "__main__":
    main()
