#!/usr/bin/env python3
"""Assemble the 922 held-out runs into a matrix diana-test can read.

v9's features live in two places. The 2,716 training runs are columns of
`matrix_v9_train/unitigs.frac.mat`, which is also what defined the 110,202 unitigs
(**R3.4**: the vocabulary comes from training runs only). Every other run has a
per-sample vector under `results/v9_vectors/<acc>/`, written by the diana-predict
feature-extraction path in the same unitig order.

`diana-test` reads a matrix, not per-sample vectors, so the held-out runs have to be
materialised in the same format: features as rows, column 0 the unitig id, one
column per sample, plus a `kmer_matrix/kmtricks.fof` giving the column order, which
is where `MatrixLoader` reads sample ids from.

Why the verification at the end is not optional
-----------------------------------------------
If the column order and the fof disagree by even one position, every prediction is
scored against the wrong run's label and the result looks plausible but is wrong.
So after writing, the matrix is read back through `MatrixLoader` and each sampled
run's row is compared against its original vector file. A mismatch raises.

    ./env/bin/python scripts/data_prep/10_build_heldout_matrix_v9.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIN_MATRIX = PROJECT_ROOT / "data/matrices/matrix_v9_train/unitigs.frac.mat"
VECTORS = PROJECT_ROOT / "results/v9_vectors"
N_FEATURES = 110202

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def unitig_ids(matrix: Path) -> list[str]:
    """Column 0 of the train matrix, in row order. The held-out matrix must reuse it."""
    ids = []
    with open(matrix) as fh:
        for line in fh:
            ids.append(line[:line.index(" ")])
    return ids


def read_vector(acc: str) -> np.ndarray:
    f = VECTORS / acc / f"{acc}_unitig_fraction.txt"
    v = np.loadtxt(f, dtype=np.float32)
    if v.shape != (N_FEATURES,):
        raise ValueError(f"{acc}: vector has shape {v.shape}, expected ({N_FEATURES},)")
    if not v.any():
        # load_v9_features treats an all-zero vector as missing: the sample could not
        # be represented at k=31, rather than genuinely sharing no k-mers.
        raise ValueError(f"{acc}: vector is entirely zero, so the run is unusable")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accessions", type=Path,
                    default=PROJECT_ROOT / "data/splits_v9/test_accessions.txt")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "data/matrices/matrix_v9_heldout")
    ap.add_argument("--check", type=int, default=25,
                    help="how many runs to verify by round-trip after writing")
    args = ap.parse_args()

    accs = args.accessions.read_text().split()
    logger.info("%d accessions requested", len(accs))

    missing = [a for a in accs if not (VECTORS / a / f"{a}_unitig_fraction.txt").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} run(s) have no vector, e.g. {missing[:5]}")

    ids = unitig_ids(TRAIN_MATRIX)
    if len(ids) != N_FEATURES:
        raise SystemExit(f"{TRAIN_MATRIX} has {len(ids)} rows, expected {N_FEATURES}")
    logger.info("reusing %d unitig ids from the train matrix", len(ids))

    X = np.empty((len(accs), N_FEATURES), dtype=np.float32)
    for i, a in enumerate(accs):
        X[i] = read_vector(a)
        if (i + 1) % 200 == 0:
            logger.info("  read %d/%d", i + 1, len(accs))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "kmer_matrix").mkdir(exist_ok=True)
    # The fof fixes the column order MatrixLoader will report as sample ids, so it is
    # written from the same list, in the same order, that filled X.
    (args.output / "kmer_matrix" / "kmtricks.fof").write_text(
        "".join(f"{a} : {VECTORS / a / f'{a}_unitig_fraction.txt'}\n" for a in accs))

    out = args.output / "unitigs.frac.mat"
    logger.info("writing %s (%d features x %d samples)", out, N_FEATURES, len(accs))
    Xt = X.T                                   # (features, samples), the file's layout
    with open(out, "w") as fh:
        for i, uid in enumerate(ids):
            fh.write(uid + " " + " ".join(f"{v:.2f}" for v in Xt[i]) + "\n")
            if (i + 1) % 20000 == 0:
                logger.info("  wrote %d/%d rows", i + 1, N_FEATURES)

    # --- the part that matters: prove nothing was permuted -------------------
    from diana.data.loader import MatrixLoader
    logger.info("verifying by round-trip through MatrixLoader")
    feats, sample_ids, _ = MatrixLoader(str(out)).load()
    sample_ids = [str(s) for s in sample_ids]
    if sample_ids != accs:
        raise SystemExit("sample id order from the fof does not match the column order")
    if feats.shape != (len(accs), N_FEATURES):
        raise SystemExit(f"loaded {feats.shape}, expected {(len(accs), N_FEATURES)}")

    rng = np.random.default_rng(0)
    picks = rng.choice(len(accs), size=min(args.check, len(accs)), replace=False)
    for i in picks:
        acc = accs[i]
        orig = read_vector(acc)
        got = feats[i]
        if not np.allclose(orig, got, atol=5e-3):
            bad = int(np.argmax(np.abs(orig - got)))
            raise SystemExit(
                f"{acc} (column {i}) does not round-trip: unitig row {bad} "
                f"has {orig[bad]:.4f} in the vector but {got[bad]:.4f} in the matrix")
    logger.info("%d/%d sampled runs round-trip exactly; column order is correct",
                len(picks), len(accs))
    logger.info("done -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
