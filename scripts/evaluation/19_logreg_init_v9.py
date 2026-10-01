#!/usr/bin/env python3
"""Phase 4 step 4.1: the logistic regression that initialises the linear part of the residual
network, one fit per task and dev fold, on the fit rows of that fold only (folds != k, labelled
rows), at the tuned C with class_weight balanced, exactly as the baseline. Writes
results/step4_1_residual_v9/linear_init/linear_<task>_fold<k>.npz with coef (C x F), intercept,
classes (sklearn order), C and n_fit; 04_epoch_budget_dev_folds.py --linear-init-dir reads it.

    ./env/bin/python scripts/evaluation/19_logreg_init_v9.py --task material --fold 0
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
SPLITS, OUT = ROOT / "data/splits_v9", ROOT / "results/step4_1_residual_v9/linear_init"
NPZ = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
SELECTED = ROOT / "results/baselines_sig_v9/selected_configs_sig_vs_fraction.tsv"
TASKS = ["community_type", "feature", "sample_host", "material"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"linear_{a.task}_fold{a.fold}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    sel = pd.read_csv(SELECTED, sep="\t")
    params = json.loads(sel[(sel.task == a.task) & (sel.model == "LogisticRegression_Bal") & (sel.rep == "fraction")].cfg.iloc[0])
    t0 = time.time()
    with np.load(NPZ, allow_pickle=False) as z:
        ids = np.asarray(z["sample_ids"].astype(str)); X = z["frac"].astype(np.float32)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    fit = meta[a.task].notna().to_numpy() & (folds != a.fold)
    y = meta[a.task].astype(object).to_numpy()[fit].astype(str)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=a.n_jobs, **params).fit(X[fit], y)
    np.savez(out, coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32), classes=np.array(clf.classes_),
             C=np.array(float(params["C"])), n_fit=np.array(int(fit.sum())), fold=np.array(a.fold))
    logger.info("%s fold %d: %d fit rows, %d classes, C %s, coef %s in %.0f s -> %s", a.task, a.fold, int(fit.sum()), len(clf.classes_),
                params["C"], clf.coef_.shape, time.time() - t0, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
