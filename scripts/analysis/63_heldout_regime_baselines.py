#!/usr/bin/env python3
"""Held-out classification by training-support regime for the tuned baselines,
with BioProject bootstrap intervals; run identically on any split.

Inputs: <baselines>/heldout_probabilities_<task>.tsv (model, Run_accession, p_*)
and <baselines>/heldout_predictions.tsv (LinearSVM_Bal y_pred), as written by
09_baselines_v9.py; <splits>/{train,test}_metadata.tsv, class_eligibility.tsv.
Regime of a class = its training-run count (few < 20, medium 20-100, many > 100,
as 23_). Cell = f1_macro_eligible over the eligible classes of the regime present
in the held-out labels; 2,000 resamples of held-out BioProjects.

    ./env/bin/python scripts/analysis/63_heldout_regime_baselines.py \\
        --baselines results/baseline_predictions_v17 --splits data/splits_v17 --out <tsv>
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TASKS = ["community_type", "feature", "sample_host", "material"]
MODELS = ["LogisticRegression_Bal", "LinearSVM_Bal", "RandomForest_Bal", "kNN", "MajorityClass"]
REGIMES = [("all", 0, np.inf), ("few", 0, 20), ("medium", 20, 100), ("many", 100, np.inf)]
N_BOOT = 2000


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def f1(y: np.ndarray, p: np.ndarray, cls: list) -> float:
    labs = [c for c in cls if c in set(y.tolist())]
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def predictions(base: Path, task: str) -> dict[str, pd.Series]:
    out = {}
    pr = base / f"heldout_probabilities_{task}.tsv"
    if pr.exists():
        d = pd.read_csv(pr, sep="\t")
        for m, g in d.groupby("model"):
            g = g.set_index("Run_accession")
            cols = [c for c in g.columns if c.startswith("p_") and g[c].notna().any()]
            out[m] = g[cols].idxmax(axis=1).str[2:]
    sv = base / "heldout_predictions.tsv"
    if sv.exists():
        d = pd.read_csv(sv, sep="\t")
        d = d[(d.task == task) & (d.model == "LinearSVM_Bal")]
        if len(d):
            out["LinearSVM_Bal"] = d.set_index("Run_accession").y_pred.astype(str)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr = pd.read_csv(a.splits / "train_metadata.tsv", sep="\t", low_memory=False)
    te = pd.read_csv(a.splits / "test_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    el = pd.read_csv(a.splits / "class_eligibility.tsv", sep="\t")
    rows = []
    for task in TASKS:
        eligible = set(el[(el.target == task) & el.evaluable]["class"].astype(str))
        support = tr[task].dropna().astype(str).value_counts()
        for model, pred in predictions(a.baselines, task).items():
            runs = pred.index.intersection(te.index[te[task].notna()])
            y = te.loc[runs, task].astype(str).to_numpy()
            p = pred.loc[runs].astype(str).to_numpy()
            g = te.loc[runs, "archive_project"].astype(str).to_numpy()
            projects = np.unique(g)
            by = {q: np.where(g == q)[0] for q in projects}
            for name, lo, hi in REGIMES:
                cls = sorted(c for c in eligible if lo <= support.get(c, 0) < hi)
                point = f1(y, p, cls)
                if not np.isfinite(point):
                    continue
                rng = rng_for(task, model, name)
                bs = []
                for _ in range(N_BOOT):
                    idx = np.concatenate([by[q] for q in rng.choice(projects, size=len(projects), replace=True)])
                    bs.append(f1(y[idx], p[idx], cls))
                bs = np.asarray(bs, float)
                bs = bs[np.isfinite(bs)]
                rows.append({"task": task, "model": model, "regime": name,
                             "n_classes": sum(c in set(y.tolist()) for c in cls),
                             "f1_macro_eligible": round(float(point), 3),
                             "ci_low": round(float(np.percentile(bs, 2.5)), 3) if len(bs) else np.nan,
                             "ci_high": round(float(np.percentile(bs, 97.5)), 3) if len(bs) else np.nan})
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
