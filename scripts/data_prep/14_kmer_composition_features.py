#!/usr/bin/env python3
"""S1: per-fold matrices with k-mer composition columns concatenated to the fractions.

The hypothesis is that knowing *what the sequences are* adds something to knowing
*which unitigs are present*. Each unitig gets a fixed description -- its canonical
k-mer composition -- and each sample gets the fraction-weighted average of the
descriptions of the unitigs it contains. Those columns are **appended** to the existing
110,202 fractions.

Concatenate, never replace. Replacing the fractions with a projection is a fixed linear
map from 110,202 to d, and that family has tied 36 times (16 DIANA arms, 20 logistic
regression). Appending adds information instead of substituting it.

Counting is `sklearn.feature_extraction.text.CountVectorizer(analyzer="char")`, not a
hand-rolled loop. Counts are folded onto canonical k-mers (a k-mer and its reverse
complement are one feature, 136 of them at k=4) because DNA is double-stranded and two
homologous unitigs may be stored in opposite orientations; the unitig matrix itself was
built from canonical k-mers.

**Why one matrix per fold.** The appended columns are on a completely different scale
from the fractions and the trainer has no standardisation step, so they must be
standardised or they are either ignored or dominant. The mean and standard deviation are
computed on that fold's four TRAINING folds only. Fitting them on all of train would
leak across the fold boundary.

    ./env/bin/python scripts/data_prep/14_kmer_composition_features.py --k 4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data/matrices/matrix_v9_train"
COMP = str.maketrans("ACGT", "TGCA")


def read_unitigs(path: Path) -> list[str]:
    seqs, cur = [], []
    for line in open(path):
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def composition(seqs: list[str], k: int) -> tuple[np.ndarray, list[str]]:
    """Canonical k-mer frequencies per unitig, rows summing to 1."""
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False,
                          vocabulary=None)
    X = vec.fit_transform(seqs)                       # sparse, unitigs x observed k-mers
    names = vec.get_feature_names_out()
    keep = [i for i, w in enumerate(names) if set(w) <= set("ACGT")]
    X, names = X[:, keep], names[keep]
    # fold each k-mer onto min(kmer, revcomp) so the two strands are one feature
    canon = {}
    for i, w in enumerate(names):
        c = min(w, w.translate(COMP)[::-1])
        canon.setdefault(c, []).append(i)
    cols = sorted(canon)
    out = np.zeros((X.shape[0], len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        out[:, j] = np.asarray(X[:, canon[c]].sum(axis=1)).ravel()
    tot = out.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return out / tot, cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--base", type=Path, default=MAT / "unitigs.frac.mat")
    ap.add_argument("--tag", default="frac")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    seqs = read_unitigs(MAT / "unitigs.fa")
    C, cols = composition(seqs, a.k)
    logger.info("%d unitigs, %d canonical %d-mers", len(seqs), len(cols), a.k)

    ids, rows = [], []
    for line in open(a.base):
        p = line.split()
        if p:
            ids.append(p[0]); rows.append(np.asarray(p[1:], dtype=np.float32))
    F = np.vstack(rows).T                              # samples x unitigs
    if F.shape[1] != C.shape[0]:
        logger.error("matrix has %d unitigs, fasta has %d", F.shape[1], C.shape[0])
        return 1

    # fraction-weighted average composition per sample
    w = F.sum(axis=1, keepdims=True); w[w == 0] = 1.0
    P = (F @ C) / w
    logger.info("projected %s -> %s", F.shape, P.shape)

    accs = [l.split(" : ")[0].strip() for l in open(ROOT / "data/train_samples_v9.fof") if l.strip()]
    folds = pd.read_csv(ROOT / "data/splits_v9/dev_folds.tsv", sep="\t")
    fold_of = np.array([folds.set_index("Run_accession")["fold"].to_dict().get(x, -1) for x in accs])

    for f in sorted(set(fold_of[fold_of >= 0])):
        out = MAT / f"unitigs.{a.tag}.kmer{a.k}.fold{f}.mat"
        if out.exists():
            logger.info("  %s exists, skipping", out.name); continue
        tr = fold_of != f
        mu, sd = P[tr].mean(axis=0), P[tr].std(axis=0)
        sd[sd == 0] = 1.0
        # match the fraction matrix's spread so neither block dominates the other
        Z = (P - mu) / sd * float(F[tr].std()) + float(F[tr].mean())
        with open(out, "w") as fh:
            for i, uid in enumerate(ids):
                fh.write(f"{uid} " + " ".join(f"{v:.6g}" for v in F[:, i]) + "\n")
            for j, c in enumerate(cols):
                fh.write(f"kmer_{c} " + " ".join(f"{v:.6g}" for v in Z[:, j]) + "\n")
        logger.info("  %s: %d rows (%d unitigs + %d %d-mers), fit on %d of %d samples",
                    out.name, len(ids) + len(cols), len(ids), len(cols), a.k,
                    int(tr.sum()), len(accs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
