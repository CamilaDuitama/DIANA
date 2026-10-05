#!/usr/bin/env python3
"""Bagged-logistic-regression control (PROTOCOLS.md, 2026-10-02): B = 10 logistic regressions
on bootstrap resamples of the training folds, probabilities averaged, out-of-fold over the 5
grouped dev folds; --resample runs|studies. Writes probs_LogReg_bag_<resample>_<task>.tsv in
the devfold_probs layout.

    ./env/bin/python scripts/evaluation/21_bagged_logreg_v9.py --task material --resample studies
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
NPZ = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
SELECTED = ROOT / "results/baselines_sig_v9/selected_configs_sig_vs_fraction.tsv"
OUT = ROOT / "results/bagged_logreg_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
B, SEED = 10, 42


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--resample", choices=["runs", "studies"], required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"probs_LogReg_bag_{a.resample}_{a.task}.tsv"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    sel = pd.read_csv(SELECTED, sep="\t")
    params = json.loads(sel[(sel.task == a.task) & (sel.model == "LogisticRegression_Bal") & (sel.rep == "fraction")].cfg.iloc[0])
    with np.load(NPZ, allow_pickle=False) as z:
        ids = np.asarray(z["sample_ids"].astype(str)); X = z["frac"].astype(np.float32)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    lab = meta[a.task].notna().to_numpy(); y = meta[a.task].astype(object).to_numpy()
    groups = meta["archive_project"].astype(str).to_numpy()
    classes = sorted(set(y[lab].astype(str)))
    parts = []
    for f in sorted(set(folds)):
        tr, te = np.flatnonzero(lab & (folds != f)), np.flatnonzero(lab & (folds == f))
        acc = np.zeros((len(te), len(classes)), dtype=np.float64)
        t0 = time.time()
        for b in range(B):
            rng = np.random.default_rng(SEED * 1000 + f * 100 + b)
            if a.resample == "runs":
                bag = rng.choice(tr, size=len(tr), replace=True)
            else:
                studies = np.unique(groups[tr])
                draw = rng.choice(studies, size=len(studies), replace=True)
                bag = np.concatenate([tr[groups[tr] == s_] for s_ in draw])
            if len(set(y[bag].astype(str))) < 2:
                continue
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=a.n_jobs, **params).fit(X[bag], y[bag].astype(str))
            Pb = clf.predict_proba(X[te])
            for j, c in enumerate(clf.classes_):
                acc[:, classes.index(c)] += Pb[:, j]
        acc /= acc.sum(axis=1, keepdims=True)
        d = pd.DataFrame(acc, columns=[f"p_{c}" for c in classes])
        d.insert(0, "y_true", y[te].astype(str)); d.insert(0, "fold", f); d.insert(0, "Run_accession", ids[te])
        parts.append(d)
        logger.info("fold %d: %d bags on %d rows in %.0f s", f, B, len(tr), time.time() - t0)
    pd.concat(parts).to_csv(out, sep="\t", index=False)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
