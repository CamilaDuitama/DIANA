#!/usr/bin/env python3
"""S7 gate: can the off-list description predict the label ON ITS OWN?

The decisive cheap test before S7e spends a GPU day. Four columns of evidence, all under the
identical protocol, so each number has something to be compared against — which the first
version of this probe lacked, having compared a macro-F1 against a majority accuracy share:

  real       the pooled DNABERT description of each sample's off-list sequences
  scrambled  the same, from sequences whose bases were permuted. Isolates whether the DNA
             CONTENT matters or merely having 3,072 extra columns
  null       real columns with the labels shuffled within the CV, repeated, which is what
             chance produces on 5, 12, 18 and 20 classes
  on-list    the model's actual input, the 110,202 unitig fractions, under the same protocol,
             so "is 0.37 a lot" has an answer

Grouped by `archive_project` throughout. Off-list means rare, and rare often means
study-specific: adapters, host contamination, one lab's reagents. An ungrouped split would
report a BioProject fingerprint as a discovery, which is the defect this repository exists to
correct. A result that appears only without grouping is a fingerprint and is reported as one.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "data/sequences_v9/offlist"
TASKS = ["community_type", "feature", "sample_host", "material"]
N_NULL = 3


def scored(X, y, g, rng=None):
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        yy = y[tr]
        if rng is not None:
            yy = rng.permutation(yy)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
        clf.fit(sc.transform(X[tr]), yy)
        pred[te] = clf.predict(sc.transform(X[te]))
    return f1_score(y, pred, average="macro", zero_division=0)


def load_onlist(ids):
    rows = []
    with (ROOT / "data/matrices/matrix_v9_train/unitigs.frac.mat").open() as fh:
        for line in fh:
            rows.append(np.asarray(line.split()[1:], dtype=np.float32))
    F = np.vstack(rows).T
    accs = [l.split(" : ")[0].strip()
            for l in (ROOT / "data/train_samples_v9.fof").open() if l.strip()]
    pos = {a: i for i, a in enumerate(accs)}
    take = [(k, pos[a]) for k, a in enumerate(ids) if a in pos]
    return np.asarray([k for k, _ in take]), F[[i for _, i in take]]


def main() -> None:
    X = np.load(OFF / "pooled.npy")
    Xs = np.load(OFF / "pooled.shuffled.npy")
    ids = [l.strip() for l in (OFF / "pooled.samples.txt").open() if l.strip()]
    logger.info("pooled off-list %s, scrambled %s, %d samples", X.shape, Xs.shape, len(ids))

    meta = pd.concat([pd.read_csv(ROOT / f"data/splits_v9/{n}", sep="\t", low_memory=False)
                      for n in ("train_metadata.tsv", "test_metadata.tsv")])
    meta = meta.drop_duplicates("Run_accession").set_index("Run_accession")
    keep = [i for i, a in enumerate(ids) if a in meta.index]
    X, Xs, ids = X[keep], Xs[keep], [ids[i] for i in keep]
    m = meta.loc[ids]

    on_idx, On = load_onlist(ids)
    logger.info("on-list reference available for %d of %d samples", len(on_idx), len(ids))

    print(f"\n{'task':16s} {'n':>5s} {'cls':>4s} {'real':>7s} {'scram':>7s} "
          f"{'null':>7s} {'on-list':>8s}")
    for task in TASKS:
        ok = m[task].notna().to_numpy()
        y = m[task].astype(str).to_numpy()[ok]
        g = m.archive_project.astype(str).to_numpy()[ok]
        vc = pd.Series(y).value_counts()
        y = np.where(pd.Series(y).map(vc).to_numpy() >= 5, y, "__rare__")
        if len(set(y)) < 2 or len(set(g)) < 5:
            print(f"{task:16s} skipped"); continue
        r = scored(X[ok], y, g)
        s = scored(Xs[ok], y, g)
        rng = np.random.default_rng(0)
        nulls = [scored(X[ok], y, g, rng) for _ in range(N_NULL)]
        sel = np.isin(np.arange(len(ids)), on_idx) & ok
        sub = np.isin(on_idx, np.flatnonzero(sel))
        o = (scored(On[sub], m[task].astype(str).to_numpy()[sel], 
                    m.archive_project.astype(str).to_numpy()[sel])
             if sub.sum() > 50 else float("nan"))
        print(f"{task:16s} {ok.sum():5d} {len(set(y)):4d} {r:7.3f} {s:7.3f} "
              f"{np.mean(nulls):7.3f} {o:8.3f}")


if __name__ == "__main__":
    main()
