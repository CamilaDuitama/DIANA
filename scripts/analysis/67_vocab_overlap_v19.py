#!/usr/bin/env python3
"""How much of the v17 vocabulary is already in earlier vocabularies, and which BLAST hits carry over.

For each v17 unitig and each earlier vocabulary (v9 training vocabulary, 110,202 unitigs;
paper-era vocabulary, 107,480 unitigs, all BLASTed in January 2026):
  exact      the same sequence (either orientation);
  contained  a substring of one earlier unitig (either orientation), located through its
             first canonical 31-mer (each k-mer sits in one unitig of a compacted graph);
  kmer_share the share of its 31-mers present anywhere in the earlier vocabulary.
For the paper-era vocabulary, a hit carries over to a contained v17 unitig only if one of
the container's BLAST alignments spans the whole contained stretch.
Output: results/blast_v19/vocab_overlap_v17.tsv and a printed summary.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
K = 31
V17 = ROOT / "data/matrices/matrix_v17_train/unitigs.fa"
EARLIER = {"v9": ROOT / "data/matrices/matrix_v9_train/unitigs.fa",
           "paper": ROOT / "data/matrices/large_matrix_3070_with_frac/unitigs.fa"}
PAPER_HITS = ROOT / "results/feature_analysis/all_features_blast/blast_results.txt"
OUT = ROOT / "results/blast_v19/vocab_overlap_v17.tsv"
COMP = str.maketrans("ACGT", "TGCA")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def canon(s: str) -> str:
    r = rc(s)
    return s if s <= r else r


def read_fasta(path: Path) -> dict[str, str]:
    out, name, parts = {}, None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if name is not None:
                out[name] = "".join(parts).upper()
            name, parts = line[1:].split()[0], []
        else:
            parts.append(line)
    if name is not None:
        out[name] = "".join(parts).upper()
    return out


def main() -> None:
    v17 = read_fasta(V17)
    res = pd.DataFrame({"unitig": list(v17), "length": [len(s) for s in v17.values()]}).set_index("unitig")
    for name, path in EARLIER.items():
        old = read_fasta(path)
        exact = {canon(s) for s in old.values()}
        index = {}
        for uid, s in old.items():
            for i in range(len(s) - K + 1):
                index[canon(s[i:i + K])] = uid
        log.info("%s: %d unitigs, %d distinct 31-mers indexed", name, len(old), len(index))
        ex, cont, where, share = [], [], [], []
        for uid, s in v17.items():
            kmers = [canon(s[i:i + K]) for i in range(len(s) - K + 1)]
            share.append(float(np.mean([k in index for k in kmers])))
            ex.append(canon(s) in exact)
            host = index.get(kmers[0])
            pos = None
            if host is not None:
                h = old[host]
                p = h.find(s)
                if p >= 0:
                    pos = (host, p, p + len(s))
                else:
                    p = h.find(rc(s))
                    if p >= 0:
                        pos = (host, p, p + len(s))
            cont.append(pos is not None)
            where.append(pos)
        res[f"{name}_exact"] = ex
        res[f"{name}_contained"] = cont
        res[f"{name}_kmer_share"] = share
        if name == "paper":
            hits = pd.read_csv(PAPER_HITS, sep="\t", header=None, usecols=[0, 6, 7], dtype={0: str})
            hits.columns = ["q", "qstart", "qend"]
            spans = hits.groupby("q")[["qstart", "qend"]].apply(lambda d: list(zip(d.qstart, d.qend))).to_dict()
            carry = []
            for pos in where:
                if pos is None:
                    carry.append(False)
                    continue
                host, a, b = pos
                carry.append(any(qs <= a + 1 and qe >= b for qs, qe in spans.get(host, [])))
            res["paper_hit_carries_over"] = carry
    res.to_csv(OUT, sep="\t")
    n = len(res)
    for name in EARLIER:
        log.info("%s: exact %d (%.1f %%), contained %d (%.1f %%), all 31-mers present %d (%.1f %%)", name,
                 int(res[f"{name}_exact"].sum()), 100 * res[f"{name}_exact"].mean(),
                 int(res[f"{name}_contained"].sum()), 100 * res[f"{name}_contained"].mean(),
                 int((res[f"{name}_kmer_share"] == 1).sum()), 100 * (res[f"{name}_kmer_share"] == 1).mean())
    log.info("paper-era BLAST hit carries over (alignment spans the contained stretch): %d of %d (%.1f %%)",
             int(res.paper_hit_carries_over.sum()), n, 100 * res.paper_hit_carries_over.mean())


if __name__ == "__main__":
    main()
