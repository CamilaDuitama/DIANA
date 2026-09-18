#!/usr/bin/env python3
"""Three questions about the S7 arm, answered from the data rather than argued.

Q3  Did the held-out evaluation actually use DNA-derived columns, or did those samples fall
    through to the training mean? `25_append_offlist_columns.py` substitutes the training
    mean for any sample with no pooled row, so a sample missing from the extraction gets a
    constant, information-free block. If many held-out samples are in that state the whole
    held-out comparison is meaningless.

Q1/Q2  Do the off-list columns carry CLASS information or STUDY information? Predict
    `archive_project` from the 3,072 columns alone, grouped nowhere because study identity is
    exactly what we are testing for. A model that recovers the study from these columns is
    reading a fingerprint, and the final fit's ungrouped early-stopping split rewards that.
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
OFF = ROOT / "data/sequences_v9/offlist"


def block(path: Path, n_unitigs=110202):
    rows = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            if i >= n_unitigs:
                rows.append(np.asarray(line.split()[1:], dtype=np.float32))
    return np.vstack(rows).T


def main() -> None:
    logger.info("=== Q3: did held-out get real DNA columns? ===")
    H = block(ROOT / "data/matrices/matrix_v9_heldout/unitigs.frac.off.mat")
    logger.info("held-out appended block: %s", H.shape)
    # a sample that fell through to the training mean is IDENTICAL to every other such sample
    uniq, counts = np.unique(H.round(6), axis=0, return_counts=True)
    dup = counts.max()
    logger.info("distinct appended rows: %d of %d; largest identical group: %d",
                len(uniq), len(H), dup)
    if dup > 1:
        logger.error("Q3 PROBLEM: %d held-out samples share one identical block, i.e. they fell "
                     "through to the training mean and carry NO DNA information", dup)
    else:
        logger.info("Q3 OK: every held-out sample has its own DNA-derived block")
    pooled_ids = {l.strip() for l in (OFF / "pooled.samples.txt").open() if l.strip()}
    held = [l.strip() for l in (ROOT / "data/splits_v9/test_accessions.txt").open() if l.strip()]
    miss = [h for h in held if h not in pooled_ids]
    logger.info("held-out accessions with no pooled row: %d of %d %s",
                len(miss), len(held), miss[:5])

    logger.info("=== Q1/Q2: do the off-list columns encode the STUDY? ===")
    X = np.load(OFF / "pooled.npy")
    ids = [l.strip() for l in (OFF / "pooled.samples.txt").open() if l.strip()]
    meta = pd.concat([pd.read_csv(ROOT / f"data/splits_v9/{n}", sep="\t", low_memory=False)
                      for n in ("train_metadata.tsv", "test_metadata.tsv")])
    meta = meta.drop_duplicates("Run_accession").set_index("Run_accession")
    keep = [i for i, a in enumerate(ids) if a in meta.index]
    X = X[keep]; m = meta.loc[[ids[i] for i in keep]]
    proj = m.archive_project.astype(str).to_numpy()
    vc = pd.Series(proj).value_counts()
    ok = pd.Series(proj).map(vc).to_numpy() >= 5
    Xp, yp = X[ok], proj[ok]
    logger.info("predicting archive_project from the 3,072 off-list columns: "
                "%d samples, %d studies", len(yp), len(set(yp)))
    sc = StandardScaler().fit(Xp)
    pred = cross_val_predict(LogisticRegression(max_iter=500, n_jobs=-1),
                             sc.transform(Xp), yp,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1)
    acc = float((pred == yp).mean())
    chance = float(pd.Series(yp).value_counts().iloc[0] / len(yp))
    logger.info("STUDY accuracy %.3f against a %.3f majority baseline over %d studies",
                acc, chance, len(set(yp)))
    if acc > 3 * chance:
        logger.error("the off-list columns identify the STUDY, which is the fingerprint the "
                     "ungrouped early-stopping split rewards")


if __name__ == "__main__":
    main()
