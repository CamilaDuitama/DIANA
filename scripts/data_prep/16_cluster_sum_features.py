#!/usr/bin/env python3
"""S2 step 2: append cluster-summed fractions to the v9 fraction matrix.

The borrowing mechanism. Unitigs are grouped by approximate sequence similarity
(`15_unitig_clusters.py`, containment >= 0.30 fixed in advance), and each multi-member
cluster contributes one column: the sum of its members' fractions in that sample. A class
with 11 training runs cannot identify which of 110,202 individual features matter;
pooling look-alikes raises the effective count behind each column, which is the one
mechanism in the S plan that addresses few-shot rather than many-shot.

Only clusters with **two or more members** get a column. A singleton's sum is its own
fraction, which is already in the matrix, so adding it would duplicate a column and cost
parameters for nothing. At the pre-committed threshold that is 6,733 columns covering
19,364 unitigs.

Appended, never substituted: replacing the fractions with cluster sums is a fixed linear
map from 110,202 to d, the family that has tied 36 times. Appending adds information,
though 6,733 extra columns are 6,733 extra parameters per first-layer unit, so this
**can** lose a little -- more so than S1's 136.

One matrix per dev fold: the appended columns are standardised with the mean and sd of
that fold's four TRAINING folds only. Fitting on all of train leaks across the boundary.

    ./env/bin/python scripts/data_prep/16_cluster_sum_features.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data/matrices/matrix_v9_train"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path,
                    default=ROOT / "results/unitig_redundancy/cluster_labels_k15_t0.30.npy")
    ap.add_argument("--base", type=Path, default=MAT / "unitigs.frac.mat")
    ap.add_argument("--tag", default="frac")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    lab = np.load(a.labels)
    ids, rows = [], []
    for line in open(a.base):
        p = line.split()
        if p:
            ids.append(p[0]); rows.append(np.asarray(p[1:], dtype=np.float32))
    F = np.vstack(rows).T                                   # samples x unitigs
    if F.shape[1] != lab.size:
        logger.error("matrix has %d unitigs, labels cover %d", F.shape[1], lab.size)
        return 1

    sizes = np.bincount(lab)
    multi = np.flatnonzero(sizes > 1)
    logger.info("%d clusters, %d with >= 2 members covering %d unitigs",
                sizes.size, multi.size, int(sizes[multi].sum()))

    # one column per multi-member cluster: the sum of its members' fractions
    S = np.zeros((F.shape[0], multi.size), dtype=np.float32)
    pos = {c: j for j, c in enumerate(multi)}
    for u in range(lab.size):
        j = pos.get(lab[u])
        if j is not None:
            S[:, j] += F[:, u]
    logger.info("cluster-sum block %s, range %.3f to %.3f", S.shape, S.min(), S.max())

    accs = [l.split(" : ")[0].strip() for l in open(ROOT / "data/train_samples_v9.fof") if l.strip()]
    folds = pd.read_csv(ROOT / "data/splits_v9/dev_folds.tsv", sep="\t")
    fmap = folds.set_index("Run_accession")["fold"].to_dict()
    fold_of = np.array([fmap.get(x, -1) for x in accs])

    for f in sorted(set(fold_of[fold_of >= 0])):
        out = MAT / f"unitigs.{a.tag}.clust.fold{f}.mat"
        if out.exists():
            logger.info("  %s exists, skipping", out.name); continue
        tr = fold_of != f
        mu, sd = S[tr].mean(axis=0), S[tr].std(axis=0)
        sd[sd == 0] = 1.0
        Z = (S - mu) / sd * float(F[tr].std()) + float(F[tr].mean())
        with open(out, "w") as fh:
            for i, uid in enumerate(ids):
                fh.write(f"{uid} " + " ".join(f"{v:.6g}" for v in F[:, i]) + "\n")
            for j, c in enumerate(multi):
                fh.write(f"clust_{c} " + " ".join(f"{v:.6g}" for v in Z[:, j]) + "\n")
        logger.info("  %s: %d rows (%d unitigs + %d clusters), fit on %d samples",
                    out.name, len(ids) + multi.size, len(ids), multi.size, int(tr.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
