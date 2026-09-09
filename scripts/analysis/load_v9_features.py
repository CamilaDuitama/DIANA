#!/usr/bin/env python
"""Assemble a feature matrix for v9 runs from both sources.

v9 spans two feature stores: matrix_v9_train, built from the 2,716 training runs
that define the vocabulary, and the per-sample vectors built for every other run
by the diana-predict feature-extraction path. Both are 110,202-dimensional in the
same unitig order, so they concatenate directly -- but nothing else in the repo
knows that, so every v9 consumer needs this.

A vector that is present but entirely zero is treated as missing: it means the
sample could not be represented at k=31 (a truncated or near-empty input), not
that it genuinely shares no k-mers with the reference.

Used by the v9 baselines and any v9 model training.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MATRIX = PROJECT_ROOT / "data/matrices/matrix_v9_train/unitigs.frac.mat"
VECTORS = PROJECT_ROOT / "results/v9_vectors"
N_FEATURES = 110202


def load_features(accessions: Iterable[str]) -> tuple:
    """Return (X, kept) for the requested accessions, in the given order."""
    from diana.data.loader import MatrixLoader

    accessions = list(accessions)
    want = set(accessions)

    X_mat, ids, _ = MatrixLoader(str(MATRIX)).load()
    idx = {str(a): i for i, a in enumerate(ids) if str(a) in want}

    rows, kept = [], []
    for acc in accessions:
        if acc in idx:
            rows.append(X_mat[idx[acc]])
            kept.append(acc)
            continue
        f = VECTORS / acc / f"{acc}_unitig_fraction.txt"
        if f.exists():
            v = np.loadtxt(f)
            if v.size == N_FEATURES and v.sum() > 0:
                rows.append(v.astype(X_mat.dtype))
                kept.append(acc)
    if not rows:
        raise ValueError("no features found for any requested accession")
    return np.vstack(rows), kept
