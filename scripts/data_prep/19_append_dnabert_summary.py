#!/usr/bin/env python3
"""S6: keep all 110,202 fractions and APPEND a sequence summary, rather than replacing them.

The distinction that makes this a different experiment
-----------------------------------------------------
S4 and S5 both **replaced** the input layer, discarding the 110,202 private per-unitig
weight vectors. Both lost, S5 by 0.208 to 0.359 on held-out, and the scrambled controls
showed the sequence content contributed nothing either way, so the loss came from the
discarding. Neither arm ever tested whether sequence helps a model that keeps what already
works. This one appends columns and changes nothing else, so it can barely lose and any
gain is attributable to the columns.

Why the summary must be non-linear
----------------------------------
With fractions ``x`` and appended columns ``z``, the first layer computes ``W1 x + W2 z``.
If ``z = E' x`` then ``W1 x + W2 E' x = (W1 + W2 E')x``, and ``W1`` is already
unconstrained, so a fraction-weighted **sum** of per-unitig embeddings adds no function the
model could not already express. Two summaries are therefore built:

  max   ``z_j = max over the unitigs PRESENT in the sample of E[u, j]``, genuinely
        non-linear, and it asks "is any unitig here at all with this sequence property"
        instead of averaging 26,600 of them into near-constancy.
  mean  ``z = (F @ E) / sum(F)``, the literal form, non-linear only through the division,
        which is the same and only escape from redundancy that S1 had.

Each summary needs a **scrambled** twin, built with ``--table shuffled``, and it is not
optional. The pre-registered condition for reporting any clearing cell is that the same cell
does **not** clear in the scrambled arm. Comparing max against mean does not test that: it
varies the pooling function while holding the DNA fixed, which is a different question.

``E`` is the frozen DNABERT-2 table already built for S5, so nothing is trained here and
the arm costs what S1 cost. The table is informative, and the raw-cosine reading that
suggested otherwise was wrong: raw cosines are dominated by the shared cone direction (norm
2.29 against a typical deviation of 1.00). Centred, different unitigs sit at -0.003 and a
unitig sits at 0.169 from its own scrambled self, so unitigs are mutually orthogonal and
scrambling destroys about 83 % of what distinguishes one. Only 30 % of the table is linearly
predictable from 4-mer composition.

Standardisation is fitted on each fold's four TRAINING folds only and rescaled to the
fraction block's spread, so neither block dominates.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data/matrices/matrix_v9_train"
# The canonical tables, not the averaged ones: averaging a unitig with its reverse
# complement costs 22 % of the description's informative magnitude (mean deviation norm
# 1.014 against 1.325), measured 2026-09-17.
TABLES = {"real":     ("data/sequences_v9/unitig_dnabert2.canon.npy", ""),
          "shuffled": ("data/sequences_v9/unitig_dnabert2.canon.shuffled.npy", "shuf")}


def read_matrix(path: Path):
    ids, rows = [], []
    with path.open() as fh:
        for line in fh:
            p = line.split()
            if p:
                ids.append(p[0]); rows.append(np.asarray(p[1:], dtype=np.float32))
    return ids, np.vstack(rows).T           # samples x unitigs


def summarise(F: np.ndarray, E: np.ndarray, how: str, chunk: int = 64) -> np.ndarray:
    if how == "mean":
        w = F.sum(axis=1, keepdims=True); w[w == 0] = 1.0
        return (F @ E) / w
    Z = np.zeros((F.shape[0], E.shape[1]), dtype=np.float32)
    for s in range(F.shape[0]):
        present = np.flatnonzero(F[s] > 0)
        if len(present):
            Z[s] = E[present].max(axis=0)
        if s % 500 == 0:
            logger.info("  max-pooled %d/%d samples", s, F.shape[0])
    return Z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", choices=["max", "mean"], required=True)
    ap.add_argument("--all-train", action="store_true",
                    help="Write ONE matrix standardised on all 2,716 training runs instead of "
                         "five fold-specific ones. Required for the final fit: the held-out "
                         "matrix is scaled on all of train, so a final model trained on a "
                         "fold matrix straddles two different scalings. Measured 2026-09-16: "
                         "that mismatch shifts the appended columns by a mean 0.09 of their "
                         "own sd. The per-fold matrices stay fold-specific, which is correct "
                         "for the dev-fold screen and is what keeps each fold honest.")
    ap.add_argument("--table", choices=sorted(TABLES), default="real",
                    help="'shuffled' builds the mandatory control: the same pooling over "
                         "descriptions of sequences whose bases were permuted within each "
                         "unitig. Without it a clearing cell is unreportable, because the "
                         "pre-registered condition is that the cell must NOT clear in the "
                         "scrambled arm. max-versus-mean does not answer that: it tests the "
                         "pooling function, not whether the DNA matters.")
    ap.add_argument("--base", type=Path, default=MAT / "unitigs.frac.mat")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    table_path, suffix = TABLES[a.table]
    E = np.load(ROOT / table_path)
    logger.info("description table: %s (%s)", table_path, a.table)
    ids, F = read_matrix(a.base)
    if F.shape[1] != E.shape[0]:
        logger.error("matrix has %d unitigs, table has %d", F.shape[1], E.shape[0])
        return 1
    logger.info("fractions %s, frozen DNABERT-2 table %s", F.shape, E.shape)
    logger.info("present unitigs per sample: mean %.0f", (F > 0).sum(axis=1).mean())

    P = summarise(F, E, a.summary)
    logger.info("summarised to %s by %s", P.shape, a.summary)

    # A sample with no present unitigs has no pooled summary to give. Left as zeros it
    # becomes one identical distinctive vector after standardisation, i.e. an identifier the
    # model can use to recognise those specific samples: 5 training samples are in this
    # state. Setting them to the TRAINING mean makes them indistinguishable from average,
    # which is the honest encoding of "no information".
    empty = np.flatnonzero((F > 0).sum(axis=1) == 0)
    if len(empty):
        logger.warning("%d samples have no present unitigs; their appended block is set to "
                       "the training mean so it carries no identifier", len(empty))

    accs = [l.split(" : ")[0].strip() for l in open(ROOT / "data/train_samples_v9.fof") if l.strip()]
    # The fold mask is built from this order and applied to rows that come from the .mat
    # column order. They agree today at all 2,716 positions, but nothing asserted it, and a
    # divergence would fit the per-fold scaler on evaluation-fold rows.
    kfof = MAT / "kmer_matrix/kmtricks.fof"
    if kfof.exists():
        mat_order = [l.split()[0] for l in kfof.open() if l.strip()]
        if mat_order != accs:
            raise SystemExit(f"sample order differs between train_samples_v9.fof and "
                             f"{kfof}: the fold mask would be applied to the wrong rows")
        logger.info("sample order asserted against %s, all %d positions", kfof.name, len(accs))
    else:
        raise SystemExit(f"{kfof} missing; cannot assert the matrix column order")
    if len(accs) != F.shape[0]:
        raise SystemExit(f"{len(accs)} accessions but {F.shape[0]} matrix columns")
    folds = pd.read_csv(ROOT / "data/splits_v9/dev_folds.tsv", sep="\t")
    fold_of = np.array([folds.set_index("Run_accession")["fold"].to_dict().get(x, -1) for x in accs])

    folds_to_write = ["all"] if a.all_train else sorted(set(fold_of[fold_of >= 0]))
    for f in folds_to_write:
        out = MAT / (f"unitigs.frac.dna{a.summary}{suffix}.alltrain.mat" if f == "all"
                     else f"unitigs.frac.dna{a.summary}{suffix}.fold{f}.mat")
        if out.exists():
            raise SystemExit(f"{out} exists. Refusing to skip: a stale matrix built from a "
                             f"different description table or pooling silently propagates "
                             f"into every fit. Move it aside or delete it deliberately.")
        tr = np.ones(len(accs), dtype=bool) if f == "all" else (fold_of != f)
        # statistics from training rows only, and never from a row with no information
        fit = tr.copy()
        fit[empty] = False
        mu, sd = P[fit].mean(axis=0), P[fit].std(axis=0)
        sd[sd == 0] = 1.0
        Pz = P.copy()
        Pz[empty] = mu                      # no information, not a distinctive vector
        Z = (Pz - mu) / sd * float(F[fit].std()) + float(F[fit].mean())
        with open(out, "w") as fh:
            for i, uid in enumerate(ids):
                fh.write(f"{uid} " + " ".join(f"{v:.6g}" for v in F[:, i]) + "\n")
            for j in range(Z.shape[1]):
                fh.write(f"dna{a.summary}{suffix}_{j} " + " ".join(f"{v:.6g}" for v in Z[:, j]) + "\n")
        (out.parent / f"{out.name}.samples.txt").write_text("\n".join(accs) + "\n")
        logger.info("  %s: %d rows (%d unitigs + %d appended), standardised on %d of %d; "
                    "sample order written to %s.samples.txt",
                    out.name, len(ids) + Z.shape[1], len(ids), Z.shape[1],
                    int(fit.sum()), len(accs), out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
