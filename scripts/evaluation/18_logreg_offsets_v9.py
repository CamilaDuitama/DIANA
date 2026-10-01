#!/usr/bin/env python3
"""X9a: logistic-regression logits for every training run, cross-fitted, to be used as a
fixed offset that the residual network adds to.

For one task and one outer dev fold k:
  * eval rows (fold k): decision_function of the tuned logistic regression fitted on the
    four training folds;
  * training rows (folds != k): an inner 4-fold cross-fit over those folds, so a training
    row's offset never comes from a model that saw that row, and no model that produced
    any offset saw a fold-k row.
Class order = sorted training classes of the task (the order DIANA's LabelEncoder uses);
classes absent from a fit get logit 0 in that fit. Writes
results/x9a_offsets_v9/offsets_<task>_fold<k>.npz with runs, classes, offsets (n x C float32).

    ./env/bin/python scripts/evaluation/18_logreg_offsets_v9.py --task feature --fold 0
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
SPLITS, OUT = ROOT / "data/splits_v9", ROOT / "results/x9a_offsets_v9"
NPZ = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
SELECTED = ROOT / "results/baselines_sig_v9/selected_configs_sig_vs_fraction.tsv"
TASKS = ["community_type", "feature", "sample_host", "material"]
N_INNER = 4


def fit_logits(X, y, Xp, classes, params, n_jobs):
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=n_jobs, **params).fit(X, y)
    d = clf.decision_function(Xp)
    if d.ndim == 1:   # two classes: sklearn gives one column
        d = np.column_stack([-d / 2, d / 2])
    out = np.zeros((Xp.shape[0], len(classes)), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        out[:, classes.index(c)] = d[:, j]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"offsets_{a.task}_fold{a.fold}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    sel = pd.read_csv(SELECTED, sep="\t")
    params = json.loads(sel[(sel.task == a.task) & (sel.model == "LogisticRegression_Bal") & (sel.rep == "fraction")].cfg.iloc[0])
    t0 = time.time()
    with np.load(NPZ, allow_pickle=False) as z:
        ids = np.asarray(z["sample_ids"].astype(str)); X = z["frac"].astype(np.float32)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    lab = meta[a.task].notna().to_numpy(); y = meta[a.task].astype(object).to_numpy()
    classes = sorted(set(y[lab].astype(str)))
    offsets = np.zeros((len(ids), len(classes)), dtype=np.float32)
    covered = np.zeros(len(ids), dtype=bool)
    train_folds = sorted(f for f in set(folds) if f != a.fold)
    # eval rows: fitted on all training folds (labelled rows)
    tr = lab & (folds != a.fold); ev = folds == a.fold
    offsets[ev] = fit_logits(X[tr], y[tr].astype(str), X[ev], classes, params, a.n_jobs); covered[ev] = True
    logger.info("%s fold %d: eval offsets from %d training rows in %.0f s", a.task, a.fold, int(tr.sum()), time.time() - t0)
    # training rows: inner cross-fit over the training folds (one inner fold = one dev fold)
    for g in train_folds:
        fit_rows = lab & (folds != a.fold) & (folds != g); pred_rows = folds == g
        offsets[pred_rows] = fit_logits(X[fit_rows], y[fit_rows].astype(str), X[pred_rows], classes, params, a.n_jobs); covered[pred_rows] = True
        logger.info("  inner fold %d: %d fit rows -> %d rows", g, int(fit_rows.sum()), int(pred_rows.sum()))
    assert covered.all()
    np.savez(out, runs=ids, classes=np.array(classes), offsets=offsets, outer_fold=np.array(a.fold))
    logger.info("wrote %s in %.0f s", out, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
