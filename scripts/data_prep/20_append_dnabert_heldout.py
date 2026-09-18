#!/usr/bin/env python3
"""Build the held-out appended matrix for S6, standardised on TRAIN statistics only.

The leakage trap this exists to avoid
-------------------------------------
The appended columns are standardised. If that standardisation uses the held-out samples'
own mean and standard deviation, then every held-out input carries information computed
from the held-out set, and the read is contaminated in a way no score would reveal. The
training mean and standard deviation are therefore computed from the 2,716 TRAINING samples
and applied unchanged to held-out, which is the same rule the per-fold matrices follow.

The other alignment that has to be right
----------------------------------------
Held-out and train share the vocabulary: both matrices have the same 110,202 unitig rows in
the same order, so the same description table indexes both. That is asserted here rather
than assumed, on the unitig ids of both files.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/matrices/matrix_v9_train"
HELD = ROOT / "data/matrices/matrix_v9_heldout"
TABLES = {"real": ("data/sequences_v9/unitig_dnabert2.npy", ""),
          "shuffled": ("data/sequences_v9/unitig_dnabert2.shuffled.npy", "shuf")}


def load(path: Path):
    ids, rows = [], []
    with path.open() as fh:
        for line in fh:
            p = line.split()
            ids.append(p[0]); rows.append(np.asarray(p[1:], dtype=np.float32))
    return ids, np.vstack(rows).T


def summarise(F: np.ndarray, E: np.ndarray, how: str) -> np.ndarray:
    if how == "mean":
        w = F.sum(1, keepdims=True); w[w == 0] = 1.0
        return (F @ E) / w
    P = np.zeros((F.shape[0], E.shape[1]), dtype=np.float32)
    for s in range(F.shape[0]):
        pres = np.flatnonzero(F[s] > 0)
        if len(pres):
            P[s] = E[pres].max(0)
    return P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", choices=["max", "mean"], required=True)
    ap.add_argument("--table", choices=sorted(TABLES), default="real")
    a = ap.parse_args()

    table_path, suffix = TABLES[a.table]
    E = np.load(ROOT / table_path)

    tr_ids, Ftr = load(TRAIN / "unitigs.frac.mat")
    he_ids, Fhe = load(HELD / "unitigs.frac.mat")
    if tr_ids != he_ids:
        n = sum(1 for x, y in zip(tr_ids, he_ids) if x != y)
        logger.error("unitig ids differ between train and held-out at %d rows", n)
        return 1
    logger.info("vocabulary identical across train and held-out, all %d rows", len(tr_ids))
    logger.info("train %s, held-out %s, table %s", Ftr.shape, Fhe.shape, E.shape)

    Ptr = summarise(Ftr, E, a.summary)
    Phe = summarise(Fhe, E, a.summary)

    # TRAIN statistics only, for both the standardisation and the rescale target
    mu, sd = Ptr.mean(0), Ptr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    scale, centre = float(Ftr.std()), float(Ftr.mean())
    Zhe = (Phe - mu) / sd * scale + centre
    logger.info("standardised held-out with TRAIN mean/sd; appended block mean %.4f sd %.4f "
                "(train fraction block: mean %.4f sd %.4f)",
                Zhe.mean(), Zhe.std(), centre, scale)

    out = HELD / f"unitigs.frac.dna{a.summary}{suffix}.mat"
    with out.open("w") as fh:
        for i, uid in enumerate(he_ids):
            fh.write(f"{uid} " + " ".join(f"{v:.6g}" for v in Fhe[:, i]) + "\n")
        for j in range(Zhe.shape[1]):
            fh.write(f"dna{a.summary}{suffix}_{j} " + " ".join(f"{v:.6g}" for v in Zhe[:, j]) + "\n")
    logger.info("wrote %s: %d rows x %d samples", out.name,
                len(he_ids) + Zhe.shape[1], Fhe.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
