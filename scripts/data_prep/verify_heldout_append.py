#!/usr/bin/env python3
"""Two specific defect hypotheses about the S6 held-out read, tested.

H1  The held-out appended block was scaled with held-out statistics rather than training
    ones, or its unitig rows are misaligned with the train matrix.
H2  The final model trained on fold 0's matrix, whose appended block was standardised on
    folds 1-4 (80 % of train), while the held-out matrix was standardised on all 2,716
    training runs. If so the model saw one scaling and was tested on another, which would
    show up as a mechanical collapse rather than smooth degradation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/matrices/matrix_v9_train"
HELD = ROOT / "data/matrices/matrix_v9_heldout"
N_UNITIGS = 110202


def load(p: Path):
    ids, rows = [], []
    with p.open() as fh:
        for line in fh:
            s = line.split()
            ids.append(s[0]); rows.append(np.asarray(s[1:], dtype=np.float32))
    return ids, np.vstack(rows).T


def main() -> int:
    E = np.load(ROOT / "data/sequences_v9/unitig_dnabert2.npy")
    tr_ids, Ftr = load(TRAIN / "unitigs.frac.mat")
    he_ids, Fhe = load(HELD / "unitigs.frac.mat")

    logger.info("H1a unitig rows identical train vs held-out: %s",
                "PASS" if tr_ids == he_ids else "FAIL")

    def summarise(F):
        P = np.zeros((F.shape[0], E.shape[1]), dtype=np.float32)
        for s in range(F.shape[0]):
            pres = np.flatnonzero(F[s] > 0)
            if len(pres):
                P[s] = E[pres].max(0)
        return P

    Ptr, Phe = summarise(Ftr), summarise(Fhe)

    accs = [l.split(" : ")[0].strip() for l in open(ROOT / "data/train_samples_v9.fof") if l.strip()]
    folds = pd.read_csv(ROOT / "data/splits_v9/dev_folds.tsv", sep="\t")
    fmap = folds.set_index("Run_accession")["fold"].to_dict()
    fold_of = np.array([fmap.get(x, -1) for x in accs])

    def scaled(P, fit_rows):
        mu, sd = Ptr[fit_rows].mean(0), Ptr[fit_rows].std(0)
        sd = np.where(sd == 0, 1.0, sd)
        return (P - mu) / sd * float(Ftr[fit_rows].std()) + float(Ftr[fit_rows].mean())

    all_rows = np.ones(len(accs), dtype=bool)
    f0_rows = fold_of != 0

    # H1b: does the written held-out block equal the all-train scaling?
    he_ids2, Xhe = load(HELD / "unitigs.frac.dnamax.mat")
    got = Xhe[:, N_UNITIGS:]
    want_all = scaled(Phe, all_rows)
    want_held = ((Phe - Phe.mean(0)) / np.where(Phe.std(0) == 0, 1, Phe.std(0))
                 * float(Fhe.std()) + float(Fhe.mean()))
    logger.info("H1b held-out block == ALL-TRAIN scaling: %s (max |d| %.2e)",
                "PASS" if np.allclose(got, want_all, rtol=1e-4, atol=1e-5) else "FAIL",
                np.abs(got - want_all).max())
    logger.info("H1c held-out block == HELD-OUT-OWN scaling (leakage): %s (max |d| %.2e)",
                "LEAK" if np.allclose(got, want_held, rtol=1e-4, atol=1e-5) else "clean",
                np.abs(got - want_held).max())

    # H2: how far apart are the two scalings the model straddled?
    a = scaled(Phe, f0_rows)       # what the model was trained to expect
    b = scaled(Phe, all_rows)      # what it was actually given
    d = np.abs(a - b)
    logger.info("H2 fold0-scaling vs all-train-scaling on the SAME held-out data:")
    logger.info("    mean |difference| %.4f, max %.4f", d.mean(), d.max())
    logger.info("    as a fraction of the appended block's own sd: mean %.3f, max %.3f",
                (d.mean(0) / np.clip(b.std(0), 1e-12, None)).mean(),
                (d.mean(0) / np.clip(b.std(0), 1e-12, None)).max())
    logger.info("    appended block sd %.4f, fraction block sd %.4f", b.std(), Fhe.std())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
