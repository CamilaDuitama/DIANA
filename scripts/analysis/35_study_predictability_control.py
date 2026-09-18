#!/usr/bin/env python3
"""Is study-predictability special to the off-list columns, or true of any representation?

The off-list columns recover `archive_project` at 0.806 against a 0.090 majority. That was
read as a laboratory fingerprint, which is an interpretation, not a measurement: a BioProject
is usually one material from one site, so its samples share real biology, and predicting the
study may be predicting the biology.

The control is the representation DIANA already uses. If the fractions predict the study just
as well, 0.806 says nothing about the off-list columns and the fingerprint reading is
unsupported. Matched on feature count as well, since 3,072 columns and 110,202 columns are
not comparable at fixed regularisation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def study_accuracy(X, y, label):
    sc = StandardScaler().fit(X)
    pred = cross_val_predict(LogisticRegression(max_iter=500, n_jobs=-1), sc.transform(X), y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1)
    acc = float((pred == y).mean())
    logger.info("%-42s study accuracy %.3f", label, acc)
    return acc


def main() -> None:
    ids, rows = [], []
    with (ROOT / "data/matrices/matrix_v9_train/unitigs.frac.mat").open() as fh:
        for line in fh:
            rows.append(np.asarray(line.split()[1:], dtype=np.float32))
    F = np.vstack(rows).T
    accs = [l.split(" : ")[0].strip()
            for l in (ROOT / "data/train_samples_v9.fof").open() if l.strip()]
    meta = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False)
    meta = meta.drop_duplicates("Run_accession").set_index("Run_accession")
    keep = [i for i, a in enumerate(accs) if a in meta.index]
    F = F[keep]; m = meta.loc[[accs[i] for i in keep]]
    y = m.archive_project.astype(str).to_numpy()
    vc = pd.Series(y).value_counts()
    ok = pd.Series(y).map(vc).to_numpy() >= 5
    F, y = F[ok], y[ok]
    logger.info("%d training samples, %d studies, majority %.3f",
                len(y), len(set(y)), vc.iloc[0] / len(y))

    # the off-list rows for exactly these samples, in the same order, so the three
    # representations are compared on one sample set rather than three different ones
    off_all = np.load(ROOT / "data/sequences_v9/offlist/pooled.npy")
    oids = [l.strip() for l in
            (ROOT / "data/sequences_v9/offlist/pooled.samples.txt").open() if l.strip()]
    opos = {a: i for i, a in enumerate(oids)}
    names = np.asarray([accs[i] for i in keep])[ok]
    have = np.asarray([a in opos for a in names])
    F, y, names = F[have], y[have], names[have]
    off = off_all[[opos[a] for a in names]]
    logger.info("comparing on ONE sample set: %d samples, %d studies, majority %.3f",
                len(y), len(set(y)), pd.Series(y).value_counts().iloc[0] / len(y))

    rng = np.random.default_rng(0)
    cols = rng.choice(F.shape[1], off.shape[1], replace=False)
    a1 = study_accuracy(F[:, cols], y, f"{off.shape[1]} RANDOM fraction columns")
    a2 = study_accuracy(off, y, f"{off.shape[1]} off-list columns")
    a3 = study_accuracy(F, y, f"all {F.shape[1]} fraction columns")
    logger.info("")
    logger.info("off-list minus matched fractions: %+.3f", a2 - a1)
    if a2 - a1 < 0.05:
        logger.info("the off-list columns are NOT more study-revealing than the "
                    "representation DIANA already uses; the fingerprint reading is "
                    "unsupported and the overfitting needs another explanation")


if __name__ == "__main__":
    main()
