#!/usr/bin/env python3
"""Are off-list sequences ever both LONG and SHARED across BioProjects?

The arm needs both. Long, because 69.6 % of what we embedded was under 50 bp and the floor
is 31 bp, one k-mer, which carries nothing beyond composition. Shared across studies,
because a sequence confined to one BioProject cannot generalise: the model would be learning
that study, not the class.

The two requirements may be in conflict. A long, well-assembled contig is more likely to be
specific to one sample's depth and coverage, while content common to many studies is exactly
what the 10-to-90 % k-mer filter already kept and put in the matrix. If so, the route closes
for a structural reason rather than an empirical one.

Measured two ways, no embedding and no GPU:

  exact      identical canonical sequence appearing in >= 2 BioProjects
  by content for each long sequence, the largest number of distinct BioProjects reached by
             any single one of its canonical 31-mers. Robust to assembly differences, which
             exact matching is not: two studies assembling the same organism rarely produce
             byte-identical contigs.
"""
from __future__ import annotations

import glob
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "data/sequences_v9/offlist"
K = 31
LONG = 200


def canon_kmers(codes: np.ndarray) -> np.ndarray:
    if len(codes) < K:
        return np.empty(0, dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(codes.astype(np.uint64), K)
    sh = (np.arange(K, dtype=np.uint64) * np.uint64(2))[::-1]
    fwd = (w << sh).sum(axis=1, dtype=np.uint64)
    rev = ((np.uint64(3) - w)[:, ::-1] << sh).sum(axis=1, dtype=np.uint64)
    return np.minimum(fwd, rev)


def main() -> None:
    meta = pd.concat([pd.read_csv(ROOT / f"data/splits_v9/{n}", sep="\t", low_memory=False)
                      for n in ("train_metadata.tsv", "test_metadata.tsv")])
    proj = meta.drop_duplicates("Run_accession").set_index("Run_accession")["archive_project"]
    pid = {p: i for i, p in enumerate(sorted(proj.dropna().unique()))}
    logger.info("%d BioProjects", len(pid))

    km_all, pr_all = [], []            # every off-list k-mer with its project
    long_seqs = []                     # (project, kmers, length, sha1) for sequences > LONG
    n_seq = n_long = 0
    for sf in sorted(glob.glob(str(OFF / "shard*.npz"))):
        z = np.load(sf, allow_pickle=True)
        bases, offs, samples, counts = z["bases"], z["offsets"], z["samples"], z["counts"]
        at = 0
        for s, c in zip(samples, counts):
            p = pid.get(proj.get(str(s)), -1)
            for j in range(at, at + c):
                codes = bases[offs[j]:offs[j + 1]]
                km = canon_kmers(codes)
                if len(km):
                    km_all.append(km)
                    pr_all.append(np.full(len(km), p, dtype=np.int16))
                n_seq += 1
                if len(codes) > LONG:
                    n_long += 1
                    long_seqs.append((p, km, len(codes),
                                      hashlib.sha1(codes.tobytes()).hexdigest()))
            at += c
        logger.info("%s: %d sequences so far, %d over %d bp", Path(sf).stem, n_seq, n_long, LONG)

    logger.info("=== totals ===")
    logger.info("off-list sequences            %d", n_seq)
    logger.info("over %d bp                   %d  (%.2f %%)", LONG, n_long, 100 * n_long / n_seq)

    # exact: identical canonical sequence in >= 2 projects
    by_hash = defaultdict(set)
    for p, _, _, h in long_seqs:
        by_hash[h].add(p)
    shared_exact = sum(1 for v in by_hash.values() if len(v) >= 2)
    logger.info("distinct long sequences       %d", len(by_hash))
    logger.info("EXACT: long and in >= 2 BioProjects  %d  (%.3f %% of distinct)",
                shared_exact, 100 * shared_exact / max(len(by_hash), 1))

    # by content: how many projects does any of a long sequence's k-mers reach?
    km = np.concatenate(km_all); pr = np.concatenate(pr_all)
    del km_all, pr_all
    order = np.argsort(km, kind="stable")
    km, pr = km[order], pr[order]
    uniq, start = np.unique(km, return_index=True)
    ends = np.append(start[1:], len(km))
    nproj = np.fromiter((len(np.unique(pr[a:b])) for a, b in zip(start, ends)),
                        dtype=np.int32, count=len(uniq))
    logger.info("distinct off-list k-mers      %d", len(uniq))
    logger.info("k-mers reaching >= 2 projects %d  (%.2f %%)",
                int((nproj >= 2).sum()), 100 * (nproj >= 2).mean())

    reach = []
    for p, k, _, _ in long_seqs:
        if len(k) == 0:
            reach.append(0); continue
        idx = np.searchsorted(uniq, k)
        idx[idx >= len(uniq)] = 0
        ok = uniq[idx] == k
        reach.append(int(nproj[idx[ok]].max()) if ok.any() else 0)
    reach = np.asarray(reach)
    logger.info("BY CONTENT, long sequences reaching >= 2 projects  %d  (%.2f %%)",
                int((reach >= 2).sum()), 100 * (reach >= 2).mean())
    for t in (2, 3, 5, 10):
        logger.info("   >= %2d projects: %d", t, int((reach >= t).sum()))
    n_samples = sum(1 for _ in long_seqs)
    logger.info("per-sample average of long AND shared (>=2 projects): %.1f",
                (reach >= 2).sum() / 3638)


if __name__ == "__main__":
    main()
