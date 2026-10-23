#!/usr/bin/env python3
"""Assemble the held-out feature vectors (13_project_v19.sbatch) into one matrix file.

Output: results/v19_heldout/heldout_frac.npz, heldout_ids.txt and heldout_metadata.tsv
(the runs kept, for 04_ --heldout); the npz holds `frac` (runs x 85,986 float32),
`sample_ids` and `feature_ids` (the training matrix's unitig ids, in row order), read by
MatrixLoader like any .npz matrix. Every held-out accession must have a complete vector
(85,986 values, the projection gate rule, PROTOCOLS.md v19 C5). A run whose vector is
all zero cannot be represented by the vocabulary; it is listed in unrepresentable.txt and
left out of the matrix, so every model is scored without it.

    ./env/bin/python scripts/data_prep/12_build_heldout_npz_v19.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ACCS = ROOT / "data/splits_v19/test_accessions.txt"
VECTORS = ROOT / "results/v19_vectors"
UNITIGS = ROOT / "data/matrices/matrix_v17_train/unitigs.fa"
OUT = ROOT / "results/v19_heldout"
N_FEATURES = 85986

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "heldout_frac.npz").exists():
        raise SystemExit(f"{OUT}/heldout_frac.npz exists; refusing to overwrite")
    ids = [l[1:].split()[0] for l in open(UNITIGS) if l.startswith(">")]
    if len(ids) != N_FEATURES:
        raise SystemExit(f"{UNITIGS}: {len(ids)} unitigs, expected {N_FEATURES}")
    accs = ACCS.read_text().split()
    rows, kept, empty = [], [], []
    for acc in accs:
        f = VECTORS / acc / f"{acc}_unitig_fraction.txt"
        if not f.exists():
            raise SystemExit(f"{acc}: no vector at {f}")
        v = np.loadtxt(f, dtype=np.float32)
        if v.shape != (N_FEATURES,):
            raise SystemExit(f"{acc}: {v.shape[0]} values, expected {N_FEATURES}")
        if not v.any():
            empty.append(acc)
            continue
        rows.append(v)
        kept.append(acc)
    (OUT / "unrepresentable.txt").write_text("".join(f"{a}\n" for a in empty))
    (OUT / "heldout_ids.txt").write_text("".join(f"{a}\n" for a in kept))
    meta = pd.read_csv(ROOT / "data/splits_v19/test_metadata.tsv", sep="\t", dtype=str)
    meta[meta.Run_accession.isin(kept)].to_csv(OUT / "heldout_metadata.tsv", sep="\t", index=False)
    np.savez_compressed(OUT / "heldout_frac.npz", frac=np.vstack(rows), sample_ids=np.array(kept),
                        feature_ids=np.array(ids))
    log.info("held-out matrix: %d runs x %d unitigs; %d unrepresentable (all-zero) runs listed",
             len(kept), N_FEATURES, len(empty))


if __name__ == "__main__":
    main()
