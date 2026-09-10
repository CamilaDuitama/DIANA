#!/usr/bin/env python
"""
How much of the published DIANA feature vocabulary survives into v9?

The paper's model used data/matrices/large_matrix_3070_with_frac (107,480
unitigs, BLAST-annotated in results/feature_analysis). v9 uses
data/matrices/matrix_v9_train (110,202 unitigs) built from training runs only.

The two vocabularies were assembled independently by ggcat from different
sample sets, so unitig IDs are not comparable and unitig *boundaries* shift:
the same genomic sequence can be one unitig in one build and a longer or
shorter one in the other. Matching therefore happens at two levels:

  exact  -- canonical unitig sequence identical in both builds (strict; a
            boundary shift of one base makes a true match fail)
  k-mer  -- canonical 31-mers, the unit MUSET actually counts. For each paper
            unitig, the fraction of its 31-mers present anywhere in the v9
            vocabulary. This is the biologically meaningful measure.

Then: of the paper unitigs carrying a BLAST annotation, how many survive?

Usage:
    ./env/bin/python scripts/analysis/compare_paper_v9_vocabulary.py

Output: results/vocabulary_overlap/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_FA = ROOT / "data/matrices/large_matrix_3070_with_frac/unitigs.fa"
V9_FA = ROOT / "data/matrices/matrix_v9_train/unitigs.fa"
BLAST = ROOT / "results/feature_analysis/unitigs_with_blast_hits.tsv"
OUT = ROOT / "results/vocabulary_overlap"
K = 31

CODE = np.full(256, 255, dtype=np.uint8)
for b, v in zip(b"ACGT", range(4)):
    CODE[b] = v
COMP = {0: 3, 1: 2, 2: 1, 3: 0}
MASK = (1 << (2 * K)) - 1


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    ids, seqs, cur = [], [], []
    with open(path) as fh:
        for line in fh:
            if line[0] == ">":
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                ids.append(line[1:].split()[0])
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return ids, seqs


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def canon_seqs(seqs: list[str]) -> list[str]:
    return [min(s, revcomp(s)) for s in seqs]


def kmers_with_owner(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Canonical 31-mers as uint64, plus the index of the unitig each came from."""
    vals, owner = [], []
    rc_shift = 2 * (K - 1)
    for i, s in enumerate(seqs):
        c = CODE[np.frombuffer(s.encode(), dtype=np.uint8)]
        n = len(c)
        if n < K:
            continue
        f = r = 0
        run = 0
        for j in range(n):
            b = int(c[j])
            if b == 255:
                run = 0
                f = r = 0
                continue
            f = ((f << 2) | b) & MASK
            r = (r >> 2) | (COMP[b] << rc_shift)
            run += 1
            if run >= K:
                vals.append(f if f < r else r)
                owner.append(i)
    return np.asarray(vals, dtype=np.uint64), np.asarray(owner, dtype=np.int64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    p_ids, p_seqs = read_fasta(PAPER_FA)
    v_ids, v_seqs = read_fasta(V9_FA)
    print(f"paper vocabulary : {len(p_seqs):,} unitigs", flush=True)
    print(f"v9 vocabulary    : {len(v_seqs):,} unitigs", flush=True)

    # --- level 1: exact canonical sequence
    p_can, v_can = canon_seqs(p_seqs), set(canon_seqs(v_seqs))
    exact = np.array([s in v_can for s in p_can])
    print(f"exact sequence matches: {exact.sum():,} "
          f"({exact.mean()*100:.1f}% of the paper vocabulary)", flush=True)

    # --- level 2: canonical 31-mers
    pk, owner = kmers_with_owner(p_seqs)
    vk, _ = kmers_with_owner(v_seqs)
    vk_u = np.unique(vk)
    pk_u = np.unique(pk)
    inter = np.intersect1d(pk_u, vk_u, assume_unique=True)
    print(f"\ndistinct 31-mers  paper {len(pk_u):,}   v9 {len(vk_u):,}   "
          f"shared {len(inter):,} "
          f"({len(inter)/len(pk_u)*100:.1f}% of paper, {len(inter)/len(vk_u)*100:.1f}% of v9)",
          flush=True)

    present = np.isin(pk, inter, assume_unique=False)
    n_tot = np.bincount(owner, minlength=len(p_seqs))
    n_hit = np.bincount(owner, weights=present, minlength=len(p_seqs))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_tot > 0, n_hit / np.maximum(n_tot, 1), np.nan)

    # --- BLAST annotations
    bl = pd.read_csv(BLAST, sep="\t", low_memory=False)
    bl["unitig_id"] = bl["unitig_id"].astype(str)
    ann = bl.set_index("unitig_id")["has_blast_hit"].reindex(p_ids).fillna(False)
    ann = ann.astype(str).str.lower().eq("true").to_numpy()

    df = pd.DataFrame({"paper_unitig_id": p_ids, "length": [len(s) for s in p_seqs],
                       "n_kmers": n_tot, "n_kmers_in_v9": n_hit.astype(int),
                       "frac_kmers_in_v9": frac, "exact_match_in_v9": exact,
                       "has_blast_hit": ann})
    df.to_csv(OUT / "paper_unitig_retention.csv", index=False)

    rows = []
    for label, sel in [("all paper unitigs", np.ones(len(p_seqs), bool)),
                       ("BLAST-annotated", ann), ("no BLAST hit", ~ann)]:
        f = frac[sel]
        rows.append(dict(subset=label, n=int(sel.sum()),
                         exact=int(exact[sel].sum()),
                         fully_retained=int(np.nansum(f >= 0.999)),
                         mostly_retained=int(np.nansum(f >= 0.5)),
                         any_kmer=int(np.nansum(f > 0)),
                         none=int(np.nansum(f == 0)),
                         median_frac=float(np.nanmedian(f))))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "retention_summary.csv", index=False)
    print("\n" + s.to_string(index=False), flush=True)

    # top-feature annotations from the paper (401 rows, per task)
    top = pd.read_csv(ROOT / "results/feature_analysis/blast_annotations.tsv", sep="\t")
    idx = {i: n for i, n in enumerate(p_ids)}
    top["paper_unitig_id"] = top["feature_index"].map(idx)
    m = top.merge(df, on="paper_unitig_id", how="left")
    m.to_csv(OUT / "paper_top_features_retention.csv", index=False)
    g = (m.groupby("task")
          .agg(n=("paper_unitig_id", "size"),
               with_blast=("has_blast_hit_x", "sum"),
               exact=("exact_match_in_v9", "sum"),
               fully=("frac_kmers_in_v9", lambda x: int((x >= 0.999).sum())),
               any_kmer=("frac_kmers_in_v9", lambda x: int((x > 0).sum())),
               median_frac=("frac_kmers_in_v9", "median")))
    print("\npaper top-attributed features, by task:\n" + g.to_string(), flush=True)
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
