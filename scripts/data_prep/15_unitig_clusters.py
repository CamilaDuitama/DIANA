#!/usr/bin/env python3
"""S2 step 1: cluster unitigs by approximate sequence similarity.

The borrowing hypothesis is that a unitig seen in 11 samples can inherit signal from
sequence-similar unitigs that are better represented. This builds the groups.

**Threshold fixed before looking: containment >= 0.30**, i.e. two unitigs are joined when
they share at least 30 % of the canonical 15-mers of the smaller one. Chosen from the
measured distribution over the whole corpus (`28_unitig_redundancy.py`): 45.1 % of unitigs
have a best match above 0.10, **17.5 % above 0.30**, 5.7 % above 0.50 and **none above
0.80**. So 0.30 groups a real minority without collapsing the corpus, and near-duplicates
do not exist to inflate it.

Why 15-mers and not 31: ggcat builds unitigs as unbranching paths in a de Bruijn graph, so
every 31-mer belongs to exactly one unitig and distinct unitigs share none (verified: 0 of
1.89 M positions). Similarity here must therefore be approximate.

Clusters are connected components of the containment graph. Single linkage can chain, so
the component size distribution is reported and a giant component would be a finding, not
a detail.

    ./env/bin/python scripts/data_prep/15_unitig_clusters.py
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data/matrices/matrix_v9_train"
CODE = np.full(256, 255, dtype=np.uint8)
for i, b in enumerate("ACGT"):
    CODE[ord(b)] = i
THRESHOLD = 0.30          # pre-committed, see docstring


def encode(seq: str, k: int) -> np.ndarray:
    a = CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]
    if len(a) < k:
        return np.empty(0, dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(a, k)
    good = np.lib.stride_tricks.sliding_window_view(a != 255, k).all(axis=1)
    w = w[good].astype(np.uint64)
    if w.size == 0:
        return np.empty(0, dtype=np.uint64)
    mult = np.uint64(4) ** np.arange(k - 1, -1, -1, dtype=np.uint64)
    fwd = (w * mult).sum(axis=1)
    rc = ((np.uint64(3) - w)[:, ::-1] * mult).sum(axis=1)
    return np.minimum(fwd, rc)


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--max-occ", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "results/unitig_redundancy")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    if abs(a.threshold - THRESHOLD) > 1e-9:
        logger.warning("threshold %.2f differs from the pre-committed %.2f — "
                       "this must be disclosed if any result is quoted", a.threshold, THRESHOLD)

    seqs, cur = [], []
    for line in open(MAT / "unitigs.fa"):
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur = []
        else: cur.append(line.strip())
    if cur: seqs.append("".join(cur))
    n = len(seqs)

    codes, owners, nk = [], [], np.zeros(n, dtype=np.int64)
    for i, s in enumerate(seqs):
        c = np.unique(encode(s, a.k)); nk[i] = c.size
        codes.append(c); owners.append(np.full(c.size, i, dtype=np.int32))
    codes = np.concatenate(codes); owners = np.concatenate(owners)
    order = np.argsort(codes, kind="stable")
    codes, owners = codes[order], owners[order]
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])

    shared = defaultdict(int)
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 2: continue
        grp = np.unique(owners[s:e])
        if grp.size < 2 or grp.size > a.max_occ: continue
        for x in range(grp.size):
            for y in range(x + 1, grp.size):
                shared[(int(grp[x]), int(grp[y]))] += 1

    dsu = DSU(n); joined = 0
    for (x, y), c in shared.items():
        d = min(nk[x], nk[y])
        if d and c / d >= a.threshold:
            dsu.union(x, y); joined += 1
    logger.info("%d candidate pairs, %d joined at containment >= %.2f",
                len(shared), joined, a.threshold)

    lab = np.array([dsu.find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    sizes = np.bincount(lab)
    multi = sizes[sizes > 1]
    print(f"\n{n} unitigs -> {sizes.size} clusters at containment >= {a.threshold}")
    print(f"   singletons          : {int((sizes==1).sum()):6d} clusters")
    print(f"   clusters with >= 2  : {int(multi.size):6d}  covering {int(multi.sum()):6d} unitigs "
          f"({100*multi.sum()/n:.1f} %)")
    if multi.size:
        print(f"   member counts: median {int(np.median(multi))}, mean {multi.mean():.1f}, "
              f"max {int(multi.max())}")
        big = int((sizes > 100).sum())
        print(f"   clusters larger than 100 members: {big}"
              + ("  <-- CHAINING, treat with suspicion" if big else ""))
    np.save(a.out / f"cluster_labels_k{a.k}_t{a.threshold:.2f}.npy", lab)
    logger.info("wrote cluster labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
