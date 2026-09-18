#!/usr/bin/env python3
"""Verify the S6 appended matrices before any fit consumes them.

Appending must not disturb the 110,202 fractions, the appended block must be standardised
on that fold's TRAINING folds only, and the summary must be the one the name claims. Each
of those is checked against an independent recomputation rather than trusted.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data/matrices/matrix_v9_train"
N_UNITIGS, N_APPEND = 110202, 768
fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    logger.info("%s %s", "PASS" if ok else "FAIL", msg)
    if not ok:
        fails.append(msg)


def load(path: Path) -> tuple[list[str], np.ndarray]:
    ids, rows = [], []
    with path.open() as fh:
        for line in fh:
            p = line.split()
            ids.append(p[0]); rows.append(np.asarray(p[1:], dtype=np.float32))
    return ids, np.vstack(rows).T           # samples x features


def main() -> int:
    base_ids, F = load(MAT / "unitigs.frac.mat")
    logger.info("base fraction matrix: %s", F.shape)
    E = np.load(ROOT / "data/sequences_v9/unitig_dnabert2.npy")

    accs = [l.split(" : ")[0].strip() for l in open(ROOT / "data/train_samples_v9.fof") if l.strip()]
    folds = pd.read_csv(ROOT / "data/splits_v9/dev_folds.tsv", sep="\t")
    fmap = folds.set_index("Run_accession")["fold"].to_dict()
    fold_of = np.array([fmap.get(x, -1) for x in accs])

    blocks = {}
    for how in ("max", "mean"):
        # the raw summary, recomputed here from the table and the fractions
        if how == "mean":
            w = F.sum(axis=1, keepdims=True); w[w == 0] = 1.0
            P = (F @ E) / w
        else:
            P = np.zeros((F.shape[0], E.shape[1]), dtype=np.float32)
            for s in range(F.shape[0]):
                pres = np.flatnonzero(F[s] > 0)
                if len(pres):
                    P[s] = E[pres].max(axis=0)

        for f in range(5):
            path = MAT / f"unitigs.frac.dna{how}.fold{f}.mat"
            ids, X = load(path)
            tag = f"dna{how} fold{f}"

            check(len(ids) == N_UNITIGS + N_APPEND, f"{tag}: {len(ids)} rows, expected {N_UNITIGS+N_APPEND}")
            check(X.shape[0] == F.shape[0], f"{tag}: {X.shape[0]} samples, expected {F.shape[0]}")
            check(ids[:N_UNITIGS] == base_ids, f"{tag}: unitig ids unchanged and in order")
            check(ids[N_UNITIGS] == f"dna{how}_0" and ids[-1] == f"dna{how}_{N_APPEND-1}",
                  f"{tag}: appended ids named and ordered")
            check(np.isfinite(X).all(), f"{tag}: no NaN or inf")

            # the fractions must survive untouched (6 significant figures were written)
            frac = X[:, :N_UNITIGS]
            check(np.allclose(frac, F, rtol=1e-5, atol=1e-6),
                  f"{tag}: fractions identical to the base matrix "
                  f"(max |d| {np.abs(frac - F).max():.2e})")

            # the appended block must equal an independent recomputation of the
            # standardisation, fitted on this fold's four TRAINING folds only
            tr = fold_of != f
            mu, sd = P[tr].mean(axis=0), P[tr].std(axis=0)
            sd = np.where(sd == 0, 1.0, sd)
            want = (P - mu) / sd * float(F[tr].std()) + float(F[tr].mean())
            got = X[:, N_UNITIGS:]
            check(np.allclose(got, want, rtol=1e-4, atol=1e-5),
                  f"{tag}: appended block matches recomputation "
                  f"(max |d| {np.abs(got - want).max():.2e})")

            # leakage check: standardisation must not have seen this fold
            mu_all, sd_all = P.mean(axis=0), P.std(axis=0)
            sd_all = np.where(sd_all == 0, 1.0, sd_all)
            all_data = (P - mu_all) / sd_all * float(F.std()) + float(F.mean())
            check(not np.allclose(got, all_data, rtol=1e-4, atol=1e-5),
                  f"{tag}: standardisation is fold-specific, not fitted on all data")

            blocks[(how, f)] = got

    check(not np.allclose(blocks[("max", 0)], blocks[("max", 1)]),
          "fold 0 and fold 1 appended blocks differ")
    check(not np.allclose(blocks[("max", 0)], blocks[("mean", 0)]),
          "the max and mean summaries differ")

    logger.info("%d checks, %d failed", 12 * 10 + 2, len(fails))
    for f in fails:
        logger.error("  %s", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
