#!/usr/bin/env python3
"""The authors' dna_to_vector (run_dna_to_vector_v19.sbatch) against our database-projection
vectors (36_hypervectors_v19.py --projection database) and, for the control runs, against the
released database: identical integer vectors or not, per run, with the Jaccard estimate of the
two vectors. Writes results/hypervectors_v19/dna_to_vector/comparison.tsv.

    ./env/bin/python scripts/data_prep/40_compare_dna_to_vector_v19.py
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "results/hypervectors_v19/dna_to_vector"
OURS = ROOT / "results/hypervectors_v19/database_projection/hv_raw.npz"
DB = ROOT / "data/external/faure_hypervectors"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    spec = importlib.util.spec_from_file_location("cmp38", ROOT / "scripts/data_prep/38_compare_hypervectors_v19.py")
    C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    runs = [l.strip() for l in open(W / "runs.txt") if l.strip()]
    control = {l.strip() for l in open(W / "control_runs.txt") if l.strip()}
    z = np.load(OURS)
    ours = dict(zip(z["sample_ids"].astype(str), z["raw"].astype(np.int64)))
    names, vec, d = C.open_vectors(C.db_folder(DB))
    pos = {n: i for i, n in enumerate(names) if n in control}
    rows = []
    for r in runs:
        f = W / "vec" / f"{r}.bin"
        if not f.exists():
            raise SystemExit(f"{r}: no dna_to_vector output")
        t = np.fromfile(f, dtype="<i2").astype(np.int64)
        if t.size != d:
            raise SystemExit(f"{r}: {t.size} values, expected {d}")
        row = {"Run_accession": r, "control": r in control, "equals_ours": bool(np.array_equal(t, ours[r])),
               "jaccard_ours": C.jaccard(t, ours[r])}
        if r in control:
            v = np.asarray(vec[pos[r]], dtype=np.int64)
            row["equals_db"] = bool(np.array_equal(t, v))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(W / "comparison.tsv", sep="\t", index=False)
    for name, g in out.groupby("control"):
        log.info("%s: %d runs, dna_to_vector == ours: %d%s", "control (in database)" if name else "not in database", len(g),
                 int(g.equals_ours.sum()), f", == database: {int(g.equals_db.sum())}" if name else "")
    diff = out[~out.equals_ours]
    if len(diff):
        log.info("not identical: Jaccard(dna_to_vector, ours) %s", diff.jaccard_ours.describe().round(4).to_dict())


if __name__ == "__main__":
    main()
