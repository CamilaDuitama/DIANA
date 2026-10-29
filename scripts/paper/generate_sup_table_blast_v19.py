#!/usr/bin/env python3
"""BLAST annotation summary of the v19 vocabulary (85,986 unitigs) against NCBI nt (Supplementary Table).

Reads results/blast_v19/chunks/chunk_*.fasta (the queries) and chunk_*_blast.tsv (outfmt 6 qseqid sseqid
pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids sscinames stitle;
scripts/analysis/run_blast_v19.sbatch). Per unitig, the best hit is the one with the highest bitscore; its
identity and query coverage ((qend - qstart + 1) / unitig length) are summarised. Species: the first two words
of the title of the best hit whose title names no non-specific term (metagenome, uncultured, unidentified,
environmental, synthetic construct, vector). Writes results/blast_v19/summary_v19.json and the manuscript
table tables/sup_table_06_blast_summary.tex.

    ./env/bin/python scripts/paper/generate_sup_table_blast_v19.py
"""
from __future__ import annotations

import glob
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "results/blast_v19/chunks"
SUMMARY = ROOT / "results/blast_v19/summary_v19.json"
TABLES = Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA/tables")
COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send",
        "evalue", "bitscore", "staxids", "sscinames", "stitle"]
NON_SPECIFIC = ("metagenome", "uncultured", "unidentified", "environmental", "synthetic construct", "vector")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def query_lengths() -> dict[str, int]:
    lengths, name = {}, None
    for f in sorted(CHUNKS.glob("chunk_*.fasta")):
        for line in open(f):
            if line.startswith(">"):
                name = line[1:].split()[0]
                lengths[name] = 0
            else:
                lengths[name] += len(line.strip())
    return lengths


def species(title: str) -> str | None:
    t = str(title).replace("PREDICTED: ", "")
    if any(w in t.lower() for w in NON_SPECIFIC):
        return None
    words = t.split()
    return " ".join(words[:2]) if len(words) >= 2 else None


def summarise() -> dict:
    qlen = query_lengths()
    best, informative = [], Counter()
    for f in sorted(glob.glob(str(CHUNKS / "chunk_*_blast.tsv"))):
        d = pd.read_csv(f, sep="\t", header=None, names=COLS, dtype={"qseqid": str}, usecols=range(len(COLS)))
        d = d.sort_values(["qseqid", "bitscore"], ascending=[True, False], kind="stable")
        best.append(d.drop_duplicates("qseqid")[["qseqid", "pident", "qstart", "qend"]])
        d["sp"] = d.stitle.map(species)
        informative.update(d.dropna(subset=["sp"]).drop_duplicates("qseqid").sp)
    b = pd.concat(best)
    if b.qseqid.duplicated().any() or not set(b.qseqid) <= set(qlen):
        raise ValueError("hits are duplicated across chunks or name an unknown query")
    cov = (b.qend - b.qstart + 1) / b.qseqid.map(qlen)
    p = b.pident
    return {"total_features": len(qlen), "features_with_hit": int(len(b)), "hit_rate_percent": 100 * len(b) / len(qlen),
            "mean_identity": float(p.mean()), "mean_query_coverage_percent": float(100 * cov.mean()),
            "identity": {">=95": int((p >= 95).sum()), "90-95": int(((p >= 90) & (p < 95)).sum()),
                         "80-90": int(((p >= 80) & (p < 90)).sum()), "<80": int((p < 80).sum())},
            "top_species": dict(informative.most_common(10))}


def table(s: dict) -> str:
    rows = [f"Unitigs searched & {s['total_features']:,} \\\\", f"Unitigs with a hit & {s['features_with_hit']:,} "
            f"({s['hit_rate_percent']:.1f}\\,\\%) \\\\", f"Mean identity of the best hit & {s['mean_identity']:.1f}\\,\\% \\\\",
            f"Mean query coverage of the best hit & {s['mean_query_coverage_percent']:.1f}\\,\\% \\\\", "\\midrule",
            "\\multicolumn{2}{l}{\\textit{Identity of the best hit}} \\\\"]
    rows += [f"{lab} & {s['identity'][k]:,} \\\\" for k, lab in ((">=95", "$\\geq$95\\,\\%"), ("90-95", "90--95\\,\\%"),
                                                                     ("80-90", "80--90\\,\\%"), ("<80", "$<$80\\,\\%"))]
    rows += ["\\midrule", "\\multicolumn{2}{l}{\\textit{Most frequent species (best specific hit)}} \\\\"]
    rows += [f"\\textit{{{sp}}} & {n:,} \\\\" for sp, n in s["top_species"].items()]
    return ("\\centering\n\\caption{Megablast search of the 85,986 unitig features against NCBI nt (E-value $\\leq 10^{-5}$, up to 5 targets per unitig). Best hit: highest "
            "bitscore per unitig. Species: the best hit whose title names no non-specific term (metagenome, uncultured, unidentified, "
            "environmental, synthetic construct, vector).}\n\\label{tab:blast_summary}\n\\small\n\\begin{tabular}{lr}\n\\toprule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    s = summarise()
    SUMMARY.write_text(json.dumps(s, indent=1))
    (TABLES / "sup_table_06_blast_summary.tex").write_text(table(s))
    log.info("%d of %d unitigs with a hit (%.1f %%), mean identity %.1f, mean coverage %.1f %%", s["features_with_hit"],
             s["total_features"], s["hit_rate_percent"], s["mean_identity"], s["mean_query_coverage_percent"])


if __name__ == "__main__":
    main()
