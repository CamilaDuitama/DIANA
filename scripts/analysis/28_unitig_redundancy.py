#!/usr/bin/env python3
"""Sanity check: is there any sequence redundancy among the v9 unitigs to borrow from?

The premise of a sequence-aware model is that a rare unitig can inherit signal from
sequence-similar unitigs that are better represented. That only works if look-alikes
exist. Two reasons to doubt it here, and this measures the second:

1. **Exact, and already confirmed.** ggcat builds unitigs as unbranching paths in a
   de Bruijn graph, so every 31-mer belongs to exactly one unitig. Measured on 15,000
   unitigs and 1.89 M positions: **zero** 31-mers appear in more than one unitig. The
   set is 31-mer-disjoint by construction, so similarity must be approximate.

2. **Approximate, measured here.** Two unitigs can be 95 % identical and share no exact
   31-mer if mismatches fall every ~20 bp, so this indexes shorter k-mers (default 15)
   over the WHOLE corpus and reports, per unitig, the best containment against any other
   unitig: |shared k-mers| / min(|A|, |B|).

An earlier version searched only an 8,000-unitig sample, i.e. 7 % of the corpus, so each
unitig's true best match was usually absent and the redundancy was understated. This
searches all 110,202.

Very frequent k-mers (low-complexity, adapters) would generate a quadratic blow-up in
pairs and are not evidence of homology, so k-mers occurring in more than `--max-occ`
unitigs are skipped; the count of skipped k-mers is reported.

    ./env/bin/python scripts/analysis/28_unitig_redundancy.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CODE = np.full(256, 255, dtype=np.uint8)
for i, b in enumerate("ACGT"):
    CODE[ord(b)] = i


def encode(seq: str, k: int) -> np.ndarray:
    """Canonical k-mer codes for one sequence, uint64; invalid bases dropped."""
    a = CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]
    if len(a) < k:
        return np.empty(0, dtype=np.uint64)
    ok = a != 255
    w = np.lib.stride_tricks.sliding_window_view(a, k)
    good = np.lib.stride_tricks.sliding_window_view(ok, k).all(axis=1)
    w = w[good].astype(np.uint64)
    if w.size == 0:
        return np.empty(0, dtype=np.uint64)
    mult = (np.uint64(4) ** np.arange(k - 1, -1, -1, dtype=np.uint64))
    fwd = (w * mult).sum(axis=1)
    rc = ((np.uint64(3) - w)[:, ::-1] * mult).sum(axis=1)
    return np.minimum(fwd, rc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max-occ", type=int, default=50)
    ap.add_argument("--fasta", type=Path,
                    default=ROOT / "data/matrices/matrix_v9_train/unitigs.fa")
    ap.add_argument("--out", type=Path, default=ROOT / "results/unitig_redundancy")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)

    seqs, cur = [], []
    for line in open(a.fasta):
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    n = len(seqs)
    logger.info("%d unitigs, %d bp", n, sum(map(len, seqs)))

    codes, owners, nk = [], [], np.zeros(n, dtype=np.int64)
    for i, s in enumerate(seqs):
        c = np.unique(encode(s, a.k))
        nk[i] = c.size
        codes.append(c); owners.append(np.full(c.size, i, dtype=np.int32))
    codes = np.concatenate(codes); owners = np.concatenate(owners)
    logger.info("%d k-mer occurrences, %d distinct", codes.size, np.unique(codes).size)

    order = np.argsort(codes, kind="stable")
    codes, owners = codes[order], owners[order]
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    best = np.zeros(n, dtype=np.float32)
    skipped = 0
    from collections import defaultdict
    pair = defaultdict(int)
    for s, e in zip(bounds[:-1], bounds[1:]):
        m = e - s
        if m < 2:
            continue
        grp = np.unique(owners[s:e])
        if grp.size < 2:
            continue
        if grp.size > a.max_occ:
            skipped += 1
            continue
        for x in range(grp.size):
            for y in range(x + 1, grp.size):
                pair[(int(grp[x]), int(grp[y]))] += 1
    logger.info("%d pairs with >=1 shared %d-mer; %d high-occupancy k-mers skipped",
                len(pair), a.k, skipped)
    for (x, y), c in pair.items():
        d = min(nk[x], nk[y])
        if d:
            v = c / d
            if v > best[x]: best[x] = v
            if v > best[y]: best[y] = v

    np.save(a.out / f"best_containment_k{a.k}.npy", best)
    print(f"\nall {n} unitigs, k={a.k}, best containment against ANY other unitig:")
    print(f"   share no {a.k}-mer with any other unitig : {100*(best==0).mean():5.1f} %")
    for thr in (0.1, 0.3, 0.5, 0.8, 0.95):
        print(f"   best containment > {thr:.2f} : {100*(best>thr).mean():5.2f} %  "
              f"({int((best>thr).sum())} unitigs)")
    print(f"   median = {np.median(best):.4f}   mean = {best.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
