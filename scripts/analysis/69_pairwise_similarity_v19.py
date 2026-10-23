#!/usr/bin/env python3
"""Pairwise similarity of all 3,638 runs, two ways (Faure et al. 2026 comparison, 2026-10-15).

  hypervector : Eq. 1 of Faure et al. on the hypervectors in their released database's projection
                (results/hypervectors_v19/database_projection/hv_raw.npz; equal to their database
                for every run it holds), J = u.v / (|u|^2 + |v|^2 - u.v)
  exact       : Jaccard of the two sketches themselves (data/sketches_v9/unitigs_split_v9.zip,
                k = 31, 1 in 1,000), which the hypervector value estimates

Writes results/pairwise_similarity_v19/similarity.npz (run ids, split, BioProject, both matrices,
float32) and summary.tsv: per kind of pair (same BioProject, different BioProject, held-out vs
training), the distribution of each measure and the share at or above 0.05 (the matrix's storage
floor); per held-out run, its highest similarity to any training run (nearest_training.tsv).

    ./env/bin/python scripts/analysis/69_pairwise_similarity_v19.py
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/pairwise_similarity_v19"
HV = ROOT / "results/hypervectors_v19/database_projection/hv_raw.npz"
SPLIT = ROOT / "data/splits_v19"
BLOCK = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def hv_jaccard(raw: np.ndarray) -> np.ndarray:
    u = raw.astype(np.float64)
    g = u @ u.T
    n = np.diag(g)
    return (g / (n[:, None] + n[None, :] - g)).astype(np.float32)


def exact_jaccard(sketches: list[np.ndarray]) -> np.ndarray:
    sizes = np.array([len(s) for s in sketches], dtype=np.int64)
    uniq, inv = np.unique(np.concatenate(sketches), return_inverse=True)
    log.info("%d hashes in total, %d distinct", int(sizes.sum()), len(uniq))
    X = sp.csr_matrix((np.ones(len(inv), dtype=np.float32), inv.astype(np.int64),
                       np.concatenate([[0], np.cumsum(sizes)])), shape=(len(sketches), len(uniq)))
    XT = X.T.tocsc()
    inter = np.zeros((len(sketches), len(sketches)), dtype=np.float64)
    for start in range(0, len(sketches), BLOCK):
        inter[start:start + BLOCK] = (X[start:start + BLOCK] @ XT).toarray()
        log.info("intersections: %d / %d rows", min(start + BLOCK, len(sketches)), len(sketches))
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def describe(v: np.ndarray) -> dict:
    return {"pairs": len(v), "median": float(np.median(v)), "p90": float(np.quantile(v, .9)),
            "p99": float(np.quantile(v, .99)), "max": float(v.max()), "share_ge_0.05": float((v >= 0.05).mean())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.concat([pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).assign(split="train"),
                      pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).assign(split="held-out")]).set_index("Run_accession")
    z = np.load(HV)
    runs = z["sample_ids"].astype(str)
    if set(runs) != set(meta.index):
        raise SystemExit("hypervector runs differ from the split")
    spec = importlib.util.spec_from_file_location("hv36", ROOT / "scripts/data_prep/36_hypervectors_v19.py")
    hv = importlib.util.module_from_spec(spec); spec.loader.exec_module(hv)
    sk = hv.read_sketches(hv.SKETCHES)
    J_hv = hv_jaccard(z["raw"])
    J_ex = exact_jaccard([sk[r] for r in runs])
    split = meta.loc[runs, "split"].to_numpy().astype(str); proj = meta.loc[runs, "archive_project"].to_numpy().astype(str)
    np.savez_compressed(OUT / "similarity.npz", runs=runs, split=split, bioproject=proj, hypervector=J_hv, exact=J_ex)
    iu = np.triu_indices(len(runs), k=1)
    same = proj[iu[0]] == proj[iu[1]]
    cross = split[iu[0]] != split[iu[1]]
    rows = []
    for kind, m in (("same BioProject", same), ("different BioProject", ~same), ("held-out vs training", cross)):
        for name, J in (("exact", J_ex), ("hypervector", J_hv)):
            rows.append({"pairs_of": kind, "measure": name, **describe(J[iu][m])})
    err = J_hv[iu] - J_ex[iu]
    rows.append({"pairs_of": "all", "measure": "hypervector minus exact", "pairs": len(err), "median": float(np.median(err)),
                 "p90": float(np.quantile(np.abs(err), .9)), "p99": float(np.quantile(np.abs(err), .99)), "max": float(np.abs(err).max())})
    pd.DataFrame(rows).to_csv(OUT / "summary.tsv", sep="\t", index=False)
    te, tr = np.flatnonzero(split == "held-out"), np.flatnonzero(split == "train")
    near = pd.DataFrame({"Run_accession": runs[te], "bioproject": proj[te],
                         "max_exact_to_training": J_ex[np.ix_(te, tr)].max(1),
                         "nearest_training_run": runs[tr][J_ex[np.ix_(te, tr)].argmax(1)],
                         "max_hypervector_to_training": J_hv[np.ix_(te, tr)].max(1)})
    near.to_csv(OUT / "nearest_training.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("held-out runs, highest exact similarity to any training run:", near.max_exact_to_training.describe().round(4).to_dict())
    print("held-out runs at >= 0.5 / >= 0.9 to some training run:", int((near.max_exact_to_training >= 0.5).sum()), int((near.max_exact_to_training >= 0.9).sum()))
    print("correlation hypervector vs exact over all pairs:", round(float(np.corrcoef(J_hv[iu], J_ex[iu])[0, 1]), 4))


if __name__ == "__main__":
    main()
