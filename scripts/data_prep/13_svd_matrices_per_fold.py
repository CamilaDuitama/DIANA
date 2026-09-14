#!/usr/bin/env python3
"""Per-fold truncated-SVD copies of a unitig matrix, for DIANA to train on.

G1 tried SVD only under logistic regression. This makes the same reduced spaces
available to DIANA, and also to the presence/absence matrix, which pooled testing
showed beats the fraction on `feature`.

**The SVD is fitted on the four TRAINING folds of each dev split and applied to the
held-back fold.** Fitting one SVD on all of train would let the held-back fold's runs
shape the basis, which is leakage across the fold boundary and would invalidate the
screen. That is why there is one matrix per (fold, rank) rather than one per rank.

Held-out is never touched here; projecting it would need the same per-fold basis and is
only relevant after a winner is chosen.

    ./env/bin/python scripts/data_prep/13_svd_matrices_per_fold.py \
        --input data/matrices/matrix_v9_train/unitigs.pa.mat --tag pa --ranks 192
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (samples x unitigs) plus the unitig ids; the .mat is unitigs x samples."""
    ids, rows = [], []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            ids.append(p[0])
            rows.append(np.asarray(p[1:], dtype=np.float32))
    return np.vstack(rows).T, ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="short name used in the output filenames")
    ap.add_argument("--ranks", type=int, nargs="+", default=[192])
    ap.add_argument("--center", action="store_true",
                    help="subtract the training-fold feature means first, i.e. PCA rather than "
                         "truncated SVD. Required here: measured on v9, the UNCENTRED component 1 "
                         "correlates with the per-sample non-zero unitig count at 0.956 (fraction) "
                         "and 1.000 (presence/absence), so it is a library-size axis and would "
                         "hand DIANA sequencing depth as its strongest feature (R3.7, R3.8).")
    ap.add_argument("--fof", type=Path, default=ROOT / "data/train_samples_v9.fof")
    ap.add_argument("--folds", type=Path, default=ROOT / "data/splits_v9/dev_folds.tsv")
    ap.add_argument("--outdir", type=Path, default=ROOT / "data/matrices/matrix_v9_train")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    accs = [l.split(" : ")[0].strip() for l in open(args.fof) if l.strip()]
    folds = pd.read_csv(args.folds, sep="\t").set_index("Run_accession")["fold"].to_dict()
    X, _ = load(args.input)
    if X.shape[0] != len(accs):
        logger.error("matrix has %d sample columns but the fof lists %d", X.shape[0], len(accs))
        return 1
    logger.info("loaded %s as %s", args.input.name, X.shape)

    fold_of = np.array([folds.get(a, -1) for a in accs])
    for rank in args.ranks:
        for f in sorted(set(fold_of[fold_of >= 0])):
            kind = "pca" if args.center else "svd"
            out = args.outdir / f"unitigs.{args.tag}.{kind}{rank}.fold{f}.mat"
            if out.exists():
                logger.info("  %s exists, skipping", out.name)
                continue
            tr = fold_of != f                      # the four training folds (and any unfolded row)
            Xw = X
            if args.center:
                mu = X[tr].mean(axis=0, keepdims=True)   # means from TRAINING folds only
                Xw = X - mu
            sv = TruncatedSVD(n_components=rank, random_state=0).fit(Xw[tr])
            Z = sv.transform(Xw)                   # every row projected onto the train-only basis
            with open(out, "w") as fh:
                for j in range(rank):
                    fh.write(f"comp{j} " + " ".join(f"{v:.6g}" for v in Z[:, j]) + "\n")
            logger.info("  %s: explained var %.3f (fit on %d of %d rows)",
                        out.name, sv.explained_variance_ratio_.sum(), int(tr.sum()), len(accs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
