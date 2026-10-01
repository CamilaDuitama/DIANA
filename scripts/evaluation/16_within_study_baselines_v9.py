#!/usr/bin/env python3
"""W2: the tuned baselines on the within-study folds, fixed configurations, no grid.

One (model, task): for each of the five sample-grouped folds (34_within_study_folds_v9.py),
fit the model with the hyperparameters selected on the grouped dev folds for read 9
(results/baselines_tuned_v9/best_per_model.tsv) on the other four folds of the v9 unitig
fractions, predict the held-out fold, pool. Nothing is re-tuned here.

Writes results/within_study_v9/pred_<model>_<task>.tsv (Run_accession, fold, y_true, y_pred).

    ./env/bin/python scripts/evaluation/16_within_study_baselines_v9.py --task material --model kNN
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS, OUT = ROOT / "data/splits_v9", ROOT / "results/within_study_v9"
NPZ = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
TUNED = ROOT / "results/baselines_tuned_v9/best_per_model.tsv"
TASKS = ["community_type", "feature", "sample_host", "material"]
MODELS = ["LogisticRegression_Bal", "LinearSVM_Bal", "RandomForest_Bal", "kNN"]
SEED = 42


def build(name: str, params: dict, n_jobs: int):
    if name == "LogisticRegression_Bal":
        return LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=n_jobs, **params)
    if name == "LinearSVM_Bal":
        return LinearSVC(max_iter=8000, class_weight="balanced", **params)
    if name == "RandomForest_Bal":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=n_jobs, random_state=SEED, **params)
    if name == "kNN":
        return KNeighborsClassifier(n_jobs=n_jobs, **params)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = OUT / f"pred_{a.model}_{a.task}.tsv"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    tuned = pd.read_csv(TUNED, sep="\t")
    row = tuned[(tuned.task == a.task) & (tuned.model == a.model)]
    if len(row) != 1:
        raise SystemExit(f"no unique tuned config for {a.task} / {a.model} in {TUNED}")
    params = json.loads(row.params.iloc[0])
    t0 = time.time()
    with np.load(NPZ, allow_pickle=False) as z:
        ids = list(z["sample_ids"].astype(str)); X = z["frac"].astype(np.float32)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(OUT / "folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    lab = meta[a.task].notna().to_numpy()
    y_all = meta[a.task].astype(object).to_numpy()
    logger.info("%s / %s: X %s, params %s, loaded in %.0f s", a.model, a.task, X.shape, params, time.time() - t0)
    rows = []
    for f in sorted(set(folds)):
        tr, te = lab & (folds != f), lab & (folds == f)
        t1 = time.time()
        clf = build(a.model, params, a.n_jobs).fit(X[tr], y_all[tr].astype(str))
        pred = clf.predict(X[te])
        rows.append(pd.DataFrame({"Run_accession": np.asarray(ids)[te], "fold": f, "y_true": y_all[te].astype(str), "y_pred": pred}))
        logger.info("fold %d: %d train / %d eval runs, %.0f s", f, int(tr.sum()), int(te.sum()), time.time() - t1)
    pd.concat(rows).to_csv(out, sep="\t", index=False)
    logger.info("wrote %s in %.0f s", out, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
