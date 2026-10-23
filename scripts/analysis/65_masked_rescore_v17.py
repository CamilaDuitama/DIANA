#!/usr/bin/env python3
"""Re-score the v17 seed ensembles with classes absent from a fold's training rows
masked out at prediction time (probability 0), as a model fitted on that fold's
classes would behave and as the scikit-learn baselines do by construction.

Reason: the trainer clamps a zero prior to 1e-12, so under logit adjustment a class
with no training row in the fold is never pushed down and can win every prediction
(fold 2: the single-task sample_host net calls all 573 runs Ursus arctos).

Paired over BioProjects (2,000 resamples), both eligibility rules (current, and
two-study: >= 2 training BioProjects): 4 separate minus shared network, and
DIANA (masked shared network + logistic regression, half each) minus shared network.

    ./env/bin/python scripts/analysis/65_masked_rescore_v17.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
R = __import__("64_devfold_rescore_two_study_rule")

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v17"
SEEDS = [42, 1, 2, 3, 4]
TASKS = R.TASKS
OUT = ROOT / "results/devfold_rescore_two_study_v17"


def masked_probs(arm_dir: Path, task: str, meta: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    parts = []
    for k in range(5):
        dirs = [arm_dir / f"seed{s}" / f"fold{k}" for s in SEEDS]
        runs = [l.strip() for l in open(dirs[0] / "eval_runs.txt") if l.strip()]
        classes = json.load(open(dirs[0] / "label_classes.json"))[task]
        P = np.mean([np.load(d / "probs_last.npz")[task] for d in dirs], axis=0)
        fit = meta.loc[folds.index[folds != k]]
        present = set(fit[task].dropna().astype(str))
        keep = np.array([c in present for c in classes])
        P = P * keep
        P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
        parts.append(pd.DataFrame(P, index=runs, columns=classes))
    return pd.concat(parts)


def to_pred(P: pd.DataFrame, meta: pd.DataFrame, task: str) -> pd.DataFrame:
    lab = meta.loc[P.index, task].notna()
    P = P[lab.to_numpy()]
    return pd.DataFrame({"y": meta.loc[P.index, task].astype(str), "pred": P.idxmax(axis=1)}, index=P.index)


def main() -> None:
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold
    E = R.eligibility("v17")
    rows, score_rows = [], []
    for t in TASKS:
        single = masked_probs(ROOT / f"results/seeds_v17/{t}", t, meta, folds)
        shared = masked_probs(ROOT / "results/seeds_v17/multitask", t, meta, folds)
        lr = pd.read_csv(ROOT / f"results/devfold_probs_v17/probs_LogisticRegression_Bal_{t}.tsv", sep="\t").set_index("Run_accession")
        lr = lr[[c for c in lr.columns if c.startswith("p_")]].rename(columns=lambda c: c[2:])
        idx = shared.index.intersection(lr.index)
        cols = list(shared.columns)
        diana = 0.5 * shared.loc[idx, cols] + 0.5 * lr.loc[idx].reindex(columns=cols).fillna(0.0)
        preds = {"4 separate (masked)": to_pred(single, meta, t), "shared network (masked)": to_pred(shared, meta, t),
                 "DIANA (masked net + logreg)": to_pred(diana, meta, t)}
        for rule in ["current", "two-study"]:
            el = E[t][rule]
            for m, d in preds.items():
                score_rows.append({"task": t, "rule": rule, "model": m,
                                   "f1": round(R.f1_elig(d.y.to_numpy(), d.pred.to_numpy(), el), 3)})
            rows.append({"test": "4 separate minus shared", "task": t, "rule": rule,
                         **R.paired(preds["4 separate (masked)"], preds["shared network (masked)"], meta.archive_project, el, t, "a6:single-multi")})
            rows.append({"test": "DIANA minus shared network", "task": t, "rule": rule,
                         **R.paired(preds["DIANA (masked net + logreg)"], preds["shared network (masked)"], meta.archive_project, el, t, "diana-vs-net")})
    S, P = pd.DataFrame(score_rows), pd.DataFrame(rows)
    S.to_csv(OUT / "masked_scores.tsv", sep="\t", index=False)
    P.to_csv(OUT / "masked_paired_tests.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("pooled dev-fold macro-F1 with fold-absent classes masked:")
    print(S.pivot_table(index="model", columns=["task", "rule"], values="f1").round(3).to_string())
    print("\npaired tests (first minus second):")
    print(P.to_string(index=False))


if __name__ == "__main__":
    main()
