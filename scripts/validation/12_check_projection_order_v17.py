#!/usr/bin/env python3
"""Gate for the v17 held-out projection: re-project three training runs through the
diana-predict feature path and compare each vector with the run's row in the v17
training matrix. A unitig-order mismatch gives a correlation near zero; the gate
fails below 0.9. Agreement and correlation are printed for the record.

    ./env/bin/python scripts/validation/12_check_projection_order_v17.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MATRIX = ROOT / "data/matrices/matrix_v17_train/unitigs.frac.mat"
CONTROL = ROOT / "results/v17_vectors_train_control"
ACCS = ROOT / "data/project_v17_train_control.txt"
MIN_R = 0.9

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    from diana.data.loader import MatrixLoader

    X, ids, _ = MatrixLoader(str(MATRIX)).load()
    row = {str(a): i for i, a in enumerate(ids)}
    ok = True
    for acc in open(ACCS).read().split():
        v = np.loadtxt(CONTROL / acc / f"{acc}_unitig_fraction.txt")
        m = X[row[acc]].astype(float)
        if v.size != m.size:
            log.error("%s: %d projected features vs %d in the matrix", acc, v.size, m.size)
            ok = False
            continue
        r = float(np.corrcoef(v, m)[0, 1])
        log.info("%s: r = %.4f, identical entries %.3f, nonzero projected %d / matrix %d",
                 acc, r, float(np.mean(np.isclose(v, m))), int((v > 0).sum()), int((m > 0).sum()))
        ok &= r >= MIN_R
    if not ok:
        log.error("projection order check FAILED")
        return 1
    log.info("projection order check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
