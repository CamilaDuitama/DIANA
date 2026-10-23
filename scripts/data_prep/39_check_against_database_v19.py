#!/usr/bin/env python3
"""Our hypervectors against the released database of Faure et al. 2026, every run it holds.

Compares the integer sums in <dir>/hv_raw.npz (36_hypervectors_v19.py) with the released
vectors (data/external/faure_hypervectors), run by run: identical or not, and the Jaccard
estimate of their Eq. 1 between the two vectors. Writes <dir>/database_check.tsv.

    ./env/bin/python scripts/data_prep/39_check_against_database_v19.py results/hypervectors_v19/database_projection
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data/external/faure_hypervectors"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compare_module():
    spec = importlib.util.spec_from_file_location("cmp38", ROOT / "scripts/data_prep/38_compare_hypervectors_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(folder: Path) -> None:
    C = compare_module()
    z = np.load(folder / "hv_raw.npz")
    ours = dict(zip(z["sample_ids"].astype(str), z["raw"]))
    names, vec, d = C.open_vectors(C.db_folder(DB))
    if d != z["raw"].shape[1]:
        raise SystemExit(f"dimension {d} in the database, {z['raw'].shape[1]} in {folder}")
    rows = []
    for i, n in enumerate(names):
        if n in ours:
            u, v = ours[n].astype(np.int64), np.asarray(vec[i], dtype=np.int64)
            rows.append({"Run_accession": n, "identical": bool(np.array_equal(u, v)), "jaccard": C.jaccard(u, v)})
    out = pd.DataFrame(rows)
    out.to_csv(folder / "database_check.tsv", sep="\t", index=False)
    log.info("runs in the database: %d of %d; identical: %d; Jaccard of the non-identical: %s", len(out), len(ours),
             int(out.identical.sum()), out.loc[~out.identical, "jaccard"].describe().round(4).to_dict())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
