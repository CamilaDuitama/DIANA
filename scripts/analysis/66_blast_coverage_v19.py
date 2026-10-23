#!/usr/bin/env python3
"""BLAST coverage of the v17 vocabulary from earlier searches, and the unitigs still unsearched.

Earlier searches, all megablast against the cluster's nt:
  * every unitig of the paper-era vocabulary (107,480; January 2026;
    results/feature_analysis/all_features_blast/, queries data/matrices/large_matrix_3070_with_frac/unitigs.fa);
  * the attributed v9 unitigs (results/attribution_v9/blast_coverage_v9.tsv, sequences inline;
    the 568 never searched before were searched in October 2026, results/blast_v9/).
A v17 unitig counts as searched only if its sequence equals a searched sequence exactly, in
either orientation (unitig orientation is arbitrary). Output: results/blast_v19/
coverage_v17_vocabulary.tsv (per unitig: searched, hit, source) and unsearched_v17.fasta.

    ./env/bin/python scripts/analysis/66_blast_coverage_v19.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V17 = ROOT / "data/matrices/matrix_v17_train/unitigs.fa"
PAPER_FA = ROOT / "data/matrices/large_matrix_3070_with_frac/unitigs.fa"
PAPER_HITS = ROOT / "results/feature_analysis/all_features_blast/blast_results.txt"
V9_COV = ROOT / "results/attribution_v9/blast_coverage_v9.tsv"
V9_OCT_FA = ROOT / "results/blast_v9/unsearched_unitigs_v9_safeid.fasta"
V9_OCT_HITS = ROOT / "results/blast_v9/unsearched_unitigs_v9_blast_safeid.tsv"
OUT = ROOT / "results/blast_v19"
COMP = str.maketrans("ACGTacgt", "TGCAtgca")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def canonical(seq: str) -> str:
    seq = seq.upper()
    rc = seq.translate(COMP)[::-1]
    return min(seq, rc)


def read_fasta(path: Path) -> dict[str, str]:
    out, name, parts = {}, None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if name is not None:
                out[name] = "".join(parts)
            name, parts = line[1:].split()[0], []
        else:
            parts.append(line)
    if name is not None:
        out[name] = "".join(parts)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    known: dict[str, tuple[bool, str]] = {}          # canonical sequence -> (hit, source)

    paper = read_fasta(PAPER_FA)
    hit_ids = set(pd.read_csv(PAPER_HITS, sep="\t", header=None, usecols=[0], dtype=str)[0])
    if len(paper) != 107480:
        raise SystemExit(f"{PAPER_FA}: {len(paper)} sequences, the January search covered 107,480")
    for uid, seq in paper.items():
        known[canonical(seq)] = (uid in hit_ids, "paper_vocabulary_2026-01")
    log.info("paper vocabulary: %d searched, %d with a hit", len(paper), sum(u in hit_ids for u in paper))

    v9 = pd.read_csv(V9_COV, sep="\t")
    for _, r in v9[v9.searched_feb2026_blast].iterrows():
        known.setdefault(canonical(r.sequence), (bool(r.blast_hit), "attribution_v9_2026-02"))
    oct_seqs = read_fasta(V9_OCT_FA)
    oct_hits = set(pd.read_csv(V9_OCT_HITS, sep="\t", header=None, usecols=[0], dtype=str)[0])
    for sid, seq in oct_seqs.items():
        known.setdefault(canonical(seq), (sid in oct_hits, "attribution_v9_2026-10"))

    v17 = read_fasta(V17)
    rows = []
    for uid, seq in v17.items():
        hit, src = known.get(canonical(seq), (None, ""))
        rows.append({"unitig": uid, "length": len(seq), "searched": hit is not None,
                     "blast_hit": hit, "source": src})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "coverage_v17_vocabulary.tsv", sep="\t", index=False)
    todo = cov[~cov.searched]
    with open(OUT / "unsearched_v17.fasta", "w") as fh:
        for uid in todo.unitig:
            fh.write(f">{uid}\n{v17[uid]}\n")
    s = cov.searched
    log.info("v17 vocabulary: %d unitigs; searched before %d (%.1f %%), of which with a hit %d; never searched %d",
             len(cov), int(s.sum()), 100 * s.mean(), int((cov.blast_hit == True).sum()), int((~s).sum()))  # noqa: E712
    log.info("never searched, length: median %d bp, %d under 100 bp", int(todo.length.median()) if len(todo) else 0,
             int((todo.length < 100).sum()))


if __name__ == "__main__":
    main()
