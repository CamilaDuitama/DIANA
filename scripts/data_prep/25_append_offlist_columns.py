#!/usr/bin/env python3
"""S7d part 2: append the pooled off-list columns to the fraction matrix.

Input is `pooled.npy` from `24_pool_offlist.py`: one row per sample, 3,072 columns of
quantile-pooled DNABERT descriptions of the sequence the k-mer filter deleted. Output is the
matrix DIANA trains on: the 110,202 fractions unchanged, with those columns appended.

Every correction the earlier arms cost us is enforced here rather than remembered:

* the **sample order** of the pooled rows is asserted against the matrix column order, not
  assumed, and written beside each matrix as `.samples.txt`;
* standardisation is fitted on **training rows only** — the four training folds for a
  per-fold matrix, all 2,716 for the all-train matrix, and the same training statistics
  applied to held-out, never held-out's own;
* a sample with no off-list sequence would be given the training mean rather than a
  distinctive all-zero row, which in S6 became a per-sample identifier for 5 samples (none
  are in that state here, but the guard stays);
* an existing output is **never silently reused**, because a stale matrix propagates into
  every fit built from it, which is what produced S6's discarded final fit.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/matrices/matrix_v9_train"
HELD = ROOT / "data/matrices/matrix_v9_heldout"
OFF = ROOT / "data/sequences_v9/offlist"


def load_mat(path: Path):
    ids, rows = [], []
    with path.open() as fh:
        for line in fh:
            s = line.split()
            ids.append(s[0]); rows.append(np.asarray(s[1:], dtype=np.float32))
    return ids, np.vstack(rows).T


def order_of(fof: Path) -> list[str]:
    return [l.split(" : ")[0].strip() if " : " in l else l.split()[0]
            for l in fof.open() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scrambled", action="store_true")
    ap.add_argument("--heldout-only", action="store_true",
                    help="Recompute only the held-out matrix, using the all-train statistics. "
                         "Needed when held-out changes but training does not: the per-fold and "
                         "all-train matrices are the exact files the searched hyperparameters "
                         "and the frozen final models belong to, so regenerating them would "
                         "silently decouple the models from their inputs.")
    a = ap.parse_args()
    tag = "shuf" if a.scrambled else ""
    P = np.load(OFF / f"pooled{'.shuffled' if a.scrambled else ''}.npy")
    pooled_ids = [l.strip() for l in (OFF / f"pooled{'.shuffled' if a.scrambled else ''}.samples.txt").open() if l.strip()]
    pos = {s: i for i, s in enumerate(pooled_ids)}
    logger.info("pooled off-list columns: %s over %d samples", P.shape, len(pooled_ids))

    uids, Ftr = load_mat(TRAIN / "unitigs.frac.mat")
    tr_accs = order_of(TRAIN / "kmer_matrix/kmtricks.fof")
    if len(tr_accs) != Ftr.shape[0]:
        raise SystemExit(f"{len(tr_accs)} fof entries vs {Ftr.shape[0]} matrix columns")
    missing = [a_ for a_ in tr_accs if a_ not in pos]
    if missing:
        logger.warning("%d training samples have no pooled row; they get the training mean: %s",
                       len(missing), missing[:5])
    Ptr = np.vstack([P[pos[a_]] if a_ in pos else np.full(P.shape[1], np.nan) for a_ in tr_accs])

    folds = pd.read_csv(ROOT / "data/splits_v9/dev_folds.tsv", sep="\t")
    fmap = folds.set_index("Run_accession")["fold"].to_dict()
    fold_of = np.array([fmap.get(x, -1) for x in tr_accs])

    def write(out: Path, ids, F, block, sample_order):
        if out.exists():
            raise SystemExit(f"{out} exists; refusing to reuse a possibly stale matrix")
        with out.open("w") as fh:
            for i, uid in enumerate(ids):
                fh.write(f"{uid} " + " ".join(f"{v:.6g}" for v in F[:, i]) + "\n")
            for j in range(block.shape[1]):
                fh.write(f"off{tag}_{j} " + " ".join(f"{v:.6g}" for v in block[:, j]) + "\n")
        (out.parent / f"{out.name}.samples.txt").write_text("\n".join(sample_order) + "\n")
        logger.info("  %s: %d rows x %d samples", out.name, len(ids) + block.shape[1], F.shape[0])

    def standardise(Praw, fit_rows, Fref):
        good = fit_rows & ~np.isnan(Praw).any(axis=1)
        mu, sd = Praw[good].mean(0), Praw[good].std(0)
        sd = np.where(sd == 0, 1.0, sd)
        Q = np.where(np.isnan(Praw), mu, Praw)
        return (Q - mu) / sd * float(Fref[good].std()) + float(Fref[good].mean()), mu, sd

    folds = ["all"] if a.heldout_only else list(range(5)) + ["all"]
    for f in folds:
        fit = np.ones(len(tr_accs), bool) if f == "all" else (fold_of != f)
        Z, mu, sd = standardise(Ptr, fit, Ftr)
        name = (f"unitigs.frac.off{tag}.alltrain.mat" if f == "all"
                else f"unitigs.frac.off{tag}.fold{f}.mat")
        if not a.heldout_only:
            write(TRAIN / name, uids, Ftr, Z, tr_accs)
        else:
            logger.info("held-out only: training matrices left untouched, all-train "
                        "statistics recomputed for the held-out standardisation")
        if f == "all":
            hids, Fhe = load_mat(HELD / "unitigs.frac.mat")
            if hids != uids:
                raise SystemExit("held-out unitig rows differ from train")
            he_accs = order_of(HELD / "kmer_matrix/kmtricks.fof")
            Phe = np.vstack([P[pos[x]] if x in pos else np.full(P.shape[1], np.nan)
                             for x in he_accs])
            Zhe = (np.where(np.isnan(Phe), mu, Phe) - mu) / sd * float(Ftr[fit].std()) \
                  + float(Ftr[fit].mean())
            logger.info("held-out standardised with TRAIN statistics only")
            write(HELD / f"unitigs.frac.off{tag}.mat", hids, Fhe, Zhe, he_accs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
