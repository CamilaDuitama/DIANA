#!/usr/bin/env python3
"""Exact canonical 31-mer overlap between the MUSET vocabularies built on different sample sets
(v7 train+test 3,190 runs; the 3,070-run build; v9 train 2,716 runs), same filters (-a 2,
-F 0.1, -f 0.1, -l 61). Writes results/vocab_stability_v9/vocab_kmer_overlap.json.

    sbatch on edid, 32 GB: ./env/bin/python scripts/analysis/51_vocab_kmer_overlap.py
"""
import itertools, json, sys
from pathlib import Path
ROOT = Path("/pasteur/helix/scratch/cduitama/EDID/decOM-classify")
FASTAS = {"v7_3190": "data/matrices/matrix_v7_3190/unitigs.fa",
          "large_3070": "data/matrices/large_matrix_3070_with_frac/unitigs.fa",
          "v9_train_2716": "data/matrices/matrix_v9_train/unitigs.fa"}
K = 31
COMP = str.maketrans("ACGT", "TGCA")
def kmers(path):
    s = set()
    seq = []
    def flush():
        if seq:
            u = "".join(seq).upper()
            for i in range(len(u) - K + 1):
                km = u[i:i + K]
                rc = km.translate(COMP)[::-1]
                s.add(min(km, rc))
            seq.clear()
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
            else:
                seq.append(line.strip())
        flush()
    return s
sets = {n: kmers(ROOT / p) for n, p in FASTAS.items()}
out = {n: {"n_kmers": len(s)} for n, s in sets.items()}
rows = []
for a, b in itertools.combinations(sets, 2):
    inter = len(sets[a] & sets[b]); union = len(sets[a] | sets[b])
    rows.append({"a": a, "b": b, "kmers_a": len(sets[a]), "kmers_b": len(sets[b]), "shared": inter,
                 "share_of_a_in_b": inter / len(sets[a]), "share_of_b_in_a": inter / len(sets[b]), "jaccard": inter / union})
json.dump({"kmers": out, "pairs": rows}, open(ROOT / "results/vocab_stability_v9/vocab_kmer_overlap.json", "w"), indent=2)
import pandas as pd
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
