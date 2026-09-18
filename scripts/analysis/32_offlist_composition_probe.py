#!/usr/bin/env python3
"""S7 step 1b: does the off-list pile carry class information at all?

The cheap question before the expensive arm. Each sample's off-list sequences are summarised
by their canonical 4-mer composition, 136 numbers, and a plain logistic regression is asked
to predict the label from those alone. Not appended to the fractions, alone. If crude
composition of the discarded sequence cannot separate classes, a richer description of the
same sequence is unlikely to rescue it; if it can, the full arm is justified.

Two things this probe exists to guard against:

**Assembly debris.** The off-list sequences average 48 bp, shorter than the vocabulary's 95 bp
median. That is consistent with rare biology and equally consistent with tips and
low-coverage fragments that any assembler emits and that carry nothing.

**BioProject fingerprints.** Off-list means rare, and rare often means study-specific:
adapters, host contamination, one lab's reagents. Cross-validation is therefore grouped by
`archive_project`, so a model is always scored on studies it never saw. An ungrouped split
would report a fingerprint as a discovery, which is the defect this repository exists to
correct.
"""
from __future__ import annotations

import glob
import itertools
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
K = 4


def canon_index():
    comp = {0: 3, 1: 2, 2: 1, 3: 0}
    idx, cols = {}, []
    for t in itertools.product(range(4), repeat=K):
        rc = tuple(comp[b] for b in reversed(t))
        key = min(t, rc)
        if key not in idx:
            idx[key] = len(cols); cols.append(key)
        idx[t] = idx[key]
    return idx, len(cols)


def main() -> None:
    idx, ncol = canon_index()
    rows, ids = [], []
    for f in sorted(glob.glob(str(ROOT / "data/sequences_v9/offlist/shard*.npz"))):
        z = np.load(f, allow_pickle=True)
        bases, offs, samples, counts = z["bases"], z["offsets"], z["samples"], z["counts"]
        at = 0
        for s, c in zip(samples, counts):
            prof = np.zeros(ncol, dtype=np.float64)
            for j in range(at, at + c):
                seq = bases[offs[j]:offs[j + 1]].astype(np.int64)
                if len(seq) < K:
                    continue
                w = np.lib.stride_tricks.sliding_window_view(seq, K)
                for t in map(tuple, w):
                    prof[idx[t]] += 1.0
            at += c
            tot = prof.sum()
            rows.append(prof / tot if tot else prof); ids.append(str(s))
    X = np.vstack(rows)
    logger.info("off-list composition: %s over %d samples", X.shape, len(ids))

    meta = pd.concat([pd.read_csv(ROOT / f"data/splits_v9/{n}", sep="\t", low_memory=False)
                      for n in ("train_metadata.tsv", "test_metadata.tsv")])
    meta = meta.drop_duplicates("Run_accession").set_index("Run_accession")
    keep = [i for i, a in enumerate(ids) if a in meta.index]
    X, ids = X[keep], [ids[i] for i in keep]
    m = meta.loc[ids]
    logger.info("matched metadata for %d samples, %d BioProjects",
                len(ids), m.archive_project.nunique())

    print(f"\n{'task':16s} {'n':>5s} {'classes':>8s} {'majority':>9s} "
          f"{'offlist macro-F1':>17s} {'offlist accuracy':>17s}")
    for task in TASKS:
        ok = m[task].notna().to_numpy()
        y = m[task].astype(str).to_numpy()[ok]
        g = m.archive_project.astype(str).to_numpy()[ok]
        Xt = X[ok]
        vc = pd.Series(y).value_counts()
        y2 = np.where(pd.Series(y).map(vc).to_numpy() >= 5, y, "__rare__")
        if len(set(y2)) < 2:
            print(f"{task:16s} too few classes"); continue
        pred = np.empty_like(y2)
        for tr, te in GroupKFold(n_splits=5).split(Xt, y2, groups=g):
            sc = StandardScaler().fit(Xt[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(Xt[tr]), y2[tr])
            pred[te] = clf.predict(sc.transform(Xt[te]))
        maj = vc.iloc[0] / len(y)
        print(f"{task:16s} {len(y):5d} {len(set(y2)):8d} {maj:9.3f} "
              f"{f1_score(y2, pred, average='macro', zero_division=0):17.3f} "
              f"{(pred == y2).mean():17.3f}")


if __name__ == "__main__":
    main()
