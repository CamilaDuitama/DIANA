#!/usr/bin/env python3
"""Hypervectors from three sources, run by run (PROTOCOLS.md amendment of 2026-10-15).

  tool : the authors' `project_everything` on our sketches (results/hypervectors_v19/faure_tool/db)
  ours : 36_hypervectors_v19.py on the same sketches (results/hypervectors_v19/hv_raw.npz)
  db   : the authors' released database (data/external/faure_hypervectors), runs in both only

Reported: per run, whether tool equals ours and tool equals db exactly (integer sums), and for
unequal db pairs the Jaccard estimate of Faure et al. Eq. 1 between the two vectors of the same
run. Then writes the inputs of the tests from the tool vectors only (one source for every run),
normalised as in 36_ (divided by sqrt(d), then by length): faure_tool/hv_train.npz and
faure_tool/hv_heldout.npz (MatrixLoader layout).

    ./env/bin/python scripts/data_prep/38_compare_hypervectors_v19.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HV = ROOT / "results/hypervectors_v19"
TOOL = HV / "faure_tool"
DB = ROOT / "data/external/faure_hypervectors"
SPLIT = ROOT / "data/splits_v19"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def db_folder(root: Path) -> Path:
    hits = [p.parent for p in root.rglob("vector_norms.txt")]
    if len(hits) != 1:
        raise SystemExit(f"{root}: expected one vector_norms.txt, found {len(hits)}")
    return hits[0]


def open_vectors(folder: Path) -> tuple[list[str], np.ndarray, int]:
    """Names (vector_norms.txt order) and a read-only memmap of the integer vectors."""
    d = int((folder / "dimension.txt").read_text().split()[0])
    names = [l.split()[0] for l in open(folder / "vector_norms.txt") if l.strip()]
    # the released database has no dtype.txt (its README says int32); the byte count decides
    width = (folder / "vectors.bin").stat().st_size / (len(names) * d)
    dtype = {2: "int16", 4: "int32"}.get(width)
    if dtype is None:
        raise SystemExit(f"{folder}: {width} bytes per value for {len(names)} names x {d}")
    if (folder / "dtype.txt").exists() and (folder / "dtype.txt").read_text().split()[0] != dtype:
        raise SystemExit(f"{folder}: dtype.txt disagrees with the file size ({dtype})")
    log.info("%s: %d vectors, %s", folder, len(names), dtype)
    vec = np.memmap(folder / "vectors.bin", dtype=np.dtype(dtype), mode="r")
    return names, vec.reshape(len(names), d), d


def jaccard(u: np.ndarray, v: np.ndarray) -> float:
    u, v = u.astype(np.float64), v.astype(np.float64)
    return float(u @ v / (u @ u + v @ v - u @ v))


def main() -> None:
    train = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).Run_accession.tolist()
    test = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).Run_accession.tolist()
    tnames, tvec, d = open_vectors(TOOL / "db")
    tool = dict(zip(tnames, range(len(tnames))))
    missing = sorted(set(train + test) - set(tool))
    if missing:
        raise SystemExit(f"{len(missing)} runs have no tool vector (zero vector or unread sketch), e.g. {missing[:3]}")
    z = np.load(HV / "hv_raw.npz")
    ours = dict(zip(z["sample_ids"].astype(str), z["raw"]))
    dnames, dvec, dd = open_vectors(db_folder(DB))
    if dd != d:
        raise SystemExit(f"dimension {dd} in the database, {d} in the tool output")
    db = {n: i for i, n in enumerate(dnames) if n in tool}
    rows = []
    for r in train + test:
        t = np.asarray(tvec[tool[r]], dtype=np.int64)
        row = {"Run_accession": r, "split": "train" if r in set(train) else "held-out",
               "tool_equals_ours": bool(np.array_equal(t, ours[r].astype(np.int64))), "in_db": r in db}
        if r in db:
            v = np.asarray(dvec[db[r]], dtype=np.int64)
            row["tool_equals_db"] = bool(np.array_equal(t, v))
            row["jaccard_tool_db"] = jaccard(t, v)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TOOL / "comparison.tsv", sep="\t", index=False)
    log.info("tool == ours: %d / %d runs", int(out.tool_equals_ours.sum()), len(out))
    indb = out[out.in_db]
    log.info("in the released database: %d runs; tool == db: %d; Jaccard(tool, db) for unequal runs: %s",
             len(indb), int(indb.tool_equals_db.sum()),
             indb.loc[~indb.tool_equals_db.astype(bool), "jaccard_tool_db"].describe().round(4).to_dict())
    for name, ids in (("hv_train.npz", train), ("hv_heldout.npz", test)):
        m = np.stack([np.asarray(tvec[tool[r]], dtype=np.float64) / np.sqrt(d) for r in ids])
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        np.savez_compressed(TOOL / name, frac=m.astype(np.float32),
                            feature_ids=np.array([f"hv_{i}" for i in range(d)]), sample_ids=np.array(ids))
        log.info("%s: %d runs", name, len(ids))


if __name__ == "__main__":
    main()
