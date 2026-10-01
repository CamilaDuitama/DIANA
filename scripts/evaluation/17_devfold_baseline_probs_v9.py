#!/usr/bin/env python3
"""X9.0c: out-of-fold class probabilities of the tuned baselines on the grouped dev folds.

The baseline tuner kept argmax predictions only. The flagging score needs the probability
of the stated label, so for one (model, task) this refits the configuration selected on
the dev folds (results/baselines_sig_v9/selected_configs_sig_vs_fraction.tsv, rep
`fraction`, the same grid and rule as the read-9 tuning) on four folds and writes
predict_proba on the fifth, pooled over the five folds. Linear SVM has no probabilities
and is not included.

Writes results/devfold_probs_v9/probs_<model>_<task>.tsv: Run_accession, fold, y_true,
then one column p_<class> per training class of the task.

    ./env/bin/python scripts/evaluation/17_devfold_baseline_probs_v9.py --task material --model kNN
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

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS, OUT = ROOT / "data/splits_v9", ROOT / "results/devfold_probs_v9"
NPZ = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
SELECTED = ROOT / "results/baselines_sig_v9/selected_configs_sig_vs_fraction.tsv"
TASKS = ["community_type", "feature", "sample_host", "material"]
MODELS = ["LogisticRegression_Bal", "RandomForest_Bal", "kNN"]
SEED = 42


def build(name: str, params: dict, n_jobs: int):
    if name == "LogisticRegression_Bal":
        return LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=n_jobs, **params)
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
    ap.add_argument("--study-weights", action="store_true",
                    help="Phase 4 step 4.2: sample_weight 1/max(n_study, 5) over the training rows, mean 1 (logistic regression only)")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.study_weights and a.model != "LogisticRegression_Bal":
        raise SystemExit("--study-weights is defined for LogisticRegression_Bal")
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"probs_{a.model}_{a.task}.tsv"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    sel = pd.read_csv(SELECTED, sep="\t")
    row = sel[(sel.task == a.task) & (sel.model == a.model) & (sel.rep == "fraction")]
    if len(row) != 1:
        raise SystemExit(f"no unique selected config for {a.task} / {a.model}")
    params = json.loads(row.cfg.iloc[0])
    t0 = time.time()
    with np.load(NPZ, allow_pickle=False) as z:
        ids = list(z["sample_ids"].astype(str)); X = z["frac"].astype(np.float32)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    lab = meta[a.task].notna().to_numpy()
    y_all = meta[a.task].astype(object).to_numpy()
    classes = sorted(set(y_all[lab].astype(str)))
    logger.info("%s / %s: X %s, params %s, %d classes, loaded in %.0f s", a.model, a.task, X.shape, params, len(classes), time.time() - t0)
    parts = []
    for f in sorted(set(folds)):
        tr, te = lab & (folds != f), lab & (folds == f)
        t1 = time.time()
        sw = None
        if a.study_weights:
            g = meta["archive_project"].astype(str).to_numpy()[tr]
            n = pd.Series(g).value_counts().loc[g].to_numpy(dtype=float)
            sw = 1.0 / np.maximum(n, 5); sw *= len(sw) / sw.sum()
        clf = build(a.model, params, a.n_jobs).fit(X[tr], y_all[tr].astype(str), sample_weight=sw)
        P = clf.predict_proba(X[te])
        full = np.zeros((int(te.sum()), len(classes)), dtype=np.float32)   # classes absent from the fit get 0
        for j, c in enumerate(clf.classes_):
            full[:, classes.index(c)] = P[:, j]
        d = pd.DataFrame(full, columns=[f"p_{c}" for c in classes])
        d.insert(0, "y_true", y_all[te].astype(str)); d.insert(0, "fold", f); d.insert(0, "Run_accession", np.asarray(ids)[te])
        parts.append(d)
        logger.info("fold %d: %d train / %d eval, %.0f s", f, int(tr.sum()), int(te.sum()), time.time() - t1)
    pd.concat(parts).to_csv(out, sep="\t", index=False)
    logger.info("wrote %s in %.0f s", out, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
