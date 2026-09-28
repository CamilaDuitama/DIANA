#!/usr/bin/env python3
"""S8.1: the off-list shared list. Which k-mers outside the vocabulary recur across studies?

Inputs (all produced by S8.0 and `scripts/data_prep/27_offlist_hashes_v9.sbatch`):
  data/sketches_v9/unitigs_split_v9.zip   one FracMinHash sketch per run, k=31, scaled=1000
  data/sketches_v9/vocab_v9.sig.zip       the same sketch of unitigs.fa: the vocabulary's
                                          hashes under the identical hash function and cut

Rule, fixed 2026-09-28 before the count (PROJECT.md section 4, S8.1): a hash is
"off-list" if it is not among the vocabulary's hashes; it is KEPT if it occurs in
>= MIN_SAMPLES of the 2,716 TRAINING samples from >= MIN_PROJECTS training
BioProjects. Held-out sketches are never read here. Only the sketch JSON is parsed
(no sourmash import), so ./env's python suffices.

Outputs, under data/sketches_v9/:
  offlist_shared_hashes.tsv   hash, n_samples, n_projects for the kept set
  offlist_summary.json        sizes at several floors, per-run statistics
  offlist_class_coverage.tsv  per eligible class: kept hashes present in >= half its runs
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SK = ROOT / "data/sketches_v9"
SPLITS = ROOT / "data/splits_v9"
MIN_SAMPLES, MIN_PROJECTS = 14, 2
FLOORS = [(14, 2), (28, 2), (14, 1), (20, 2), (272, 2)]
TASKS = ["community_type", "feature", "sample_host", "material"]


def read_sketches(zip_path: Path, ksize: int = 31) -> dict[str, np.ndarray]:
    """name -> sorted uint64 hashes, from a sourmash zip of JSON signatures."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            n = info.filename
            if not (n.endswith(".sig") or n.endswith(".sig.gz")):
                continue
            raw = z.read(info)
            if n.endswith(".gz"):
                raw = gzip.decompress(raw)
            for rec in json.load(io.BytesIO(raw)):
                for s in rec.get("signatures", []):
                    if s.get("ksize") != ksize:
                        continue
                    out[rec["name"]] = np.array(s["mins"], dtype=np.uint64)
    return out


def main() -> int:
    t0 = time.time()
    vocab = read_sketches(SK / "vocab_v9.sig.zip")
    if len(vocab) != 1:
        raise SystemExit(f"expected one vocabulary signature, found {len(vocab)}")
    V = np.sort(next(iter(vocab.values())))
    logger.info("vocabulary: %d hashes under scaled=1000", len(V))

    sk = read_sketches(SK / "unitigs_split_v9.zip")
    logger.info("read %d sketches in %.0f s", len(sk), time.time() - t0)
    train = [l.strip() for l in open(SPLITS / "train_accessions.txt") if l.strip()]
    missing = [a for a in train if a not in sk]
    if missing:
        raise SystemExit(f"{len(missing)} training runs have no sketch, e.g. {missing[:5]}")
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    projects = meta.loc[train, "archive_project"].astype(str).to_numpy()
    proj_idx = pd.factorize(projects)[0].astype(np.int32)

    # Off-list hashes per training sample, with the sample index alongside.
    H, S, per_run = [], [], []
    for i, acc in enumerate(train):
        m = sk[acc]
        off = m[~np.isin(m, V, assume_unique=True)]
        H.append(off)
        S.append(np.full(len(off), i, dtype=np.int32))
        per_run.append((acc, len(m), len(off)))
    H = np.concatenate(H)
    S = np.concatenate(S)
    logger.info("%d off-list (hash, sample) entries from %d training runs", len(H), len(train))
    pr = pd.DataFrame(per_run, columns=["Run_accession", "n_hashes", "n_offlist"])
    pr.to_csv(SK / "offlist_per_run_train.tsv", sep="\t", index=False)

    # Samples per hash.
    uniq, first, counts = np.unique(H, return_index=True, return_counts=True)
    logger.info("%d distinct off-list hashes; samples per hash: median %d, max %d",
                len(uniq), int(np.median(counts)), int(counts.max()))

    # Projects per hash, computed only for hashes that could pass any floor.
    cand = uniq[counts >= min(f for f, _ in FLOORS)]
    keep = np.isin(H, cand)
    Hc, Pc = H[keep], proj_idx[S[keep]]
    pair = np.unique(np.stack([Hc, Pc.astype(np.uint64)], axis=1), axis=0)
    ph, pcount = np.unique(pair[:, 0], return_counts=True)
    n_proj = pd.Series(pcount, index=ph)
    n_samp = pd.Series(counts, index=uniq).loc[ph]
    tbl = pd.DataFrame({"hash": ph, "n_samples": n_samp.to_numpy(), "n_projects": n_proj.to_numpy()})

    summary = {"n_training_runs": len(train), "n_vocab_hashes": int(len(V)),
               "n_offlist_entries": int(len(H)), "n_distinct_offlist_hashes": int(len(uniq)),
               "median_hashes_per_run": float(pr.n_hashes.median()),
               "median_offlist_per_run": float(pr.n_offlist.median()),
               "kept_by_floor": {f"samples>={s},projects>={p}": int(((tbl.n_samples >= s) & (tbl.n_projects >= p)).sum())
                                 for s, p in FLOORS}}
    kept = tbl[(tbl.n_samples >= MIN_SAMPLES) & (tbl.n_projects >= MIN_PROJECTS)].sort_values("n_samples", ascending=False)
    kept.to_csv(SK / "offlist_shared_hashes.tsv", sep="\t", index=False)
    summary["K"] = int(len(kept))
    logger.info("kept K=%d hashes at >=%d samples, >=%d projects; %s", len(kept), MIN_SAMPLES, MIN_PROJECTS,
                summary["kept_by_floor"])

    # Per-class coverage: kept hashes present in at least half of the class's training runs.
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    elig = elig[elig.evaluable]
    K = np.sort(kept.hash.to_numpy().astype(np.uint64))
    acc_idx = {a: i for i, a in enumerate(train)}
    rows = []
    for _, r in elig.iterrows():
        runs = [a for a in meta.index[meta[r.target].astype(str) == str(r["class"])] if a in acc_idx]
        if not runs:
            rows.append({"task": r.target, "class": r["class"], "train_runs": 0, "kept_in_half": 0, "kept_in_all": 0})
            continue
        hh = np.concatenate([sk[a][np.isin(sk[a], K, assume_unique=True)] for a in runs])
        u, c = np.unique(hh, return_counts=True)
        rows.append({"task": r.target, "class": r["class"], "train_runs": len(runs),
                     "kept_in_half": int((c >= max(1, len(runs) / 2)).sum()),
                     "kept_in_all": int((c == len(runs)).sum())})
    cov = pd.DataFrame(rows).sort_values(["task", "train_runs"])
    cov.to_csv(SK / "offlist_class_coverage.tsv", sep="\t", index=False)
    json.dump(summary, open(SK / "offlist_summary.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2))
    print(cov.to_string(index=False))
    logger.info("done in %.0f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
