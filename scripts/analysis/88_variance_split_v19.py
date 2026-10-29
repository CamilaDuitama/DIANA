#!/usr/bin/env python3
"""Variance split of the training unitig profiles: class against BioProject within class (VARSPLIT, v19).

Rule: PROTOCOLS.md v19 amendment 2026-10-27, written before any computation. Per field, on the v19 training runs
labelled with an evaluable class: nested PERMANOVA with cells = BioProject x class,
SS_total = SS_class + SS_BioProject(class) + SS_within, for squared Euclidean distance on the unitig fractions (DIANA's
input) and Jaccard distance on presence (fraction > 0). pseudo-F = [SS_class / (c - 1)] / [SS_BioProject(class) / (b - c)];
p by permuting the class labels among whole cells (9,999 permutations, seed 0). Training runs only; descriptive.
Writes results/varsplit_v19/varsplit.tsv.

    ./env/bin/python scripts/analysis/88_variance_split_v19.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.data.loader import MatrixLoader  # noqa: E402
from diana.evaluation import scoring as S  # noqa: E402

SPLIT = ROOT / "data/splits_v19"
MATRIX = ROOT / "data/matrices/matrix_v17_train/unitigs.frac.mat"
OUT = ROOT / "results/varsplit_v19"
FIELDS = ["community_type", "feature", "sample_host", "material"]
N_PERM, SEED = 9999, 0
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def squared_distances(X: np.ndarray) -> dict[str, np.ndarray]:
    """Squared Euclidean distances on the fractions and squared Jaccard distances on presence, between all runs."""
    K = X @ X.T
    sq = np.diag(K)
    euclid = np.maximum(sq[:, None] + sq[None, :] - 2 * K, 0.0)
    B = (X > 0).astype(np.float32)
    inter = (B @ B.T).astype(np.float64)  # counts up to 85,986, exact in float32
    size = np.diag(inter)
    union = size[:, None] + size[None, :] - inter
    jaccard = np.where(union > 0, 1.0 - inter / np.where(union > 0, union, 1.0), 0.0)
    return {"euclidean": euclid, "jaccard": jaccard ** 2}


def indicator(groups: np.ndarray, n_groups: int) -> np.ndarray:
    M = np.zeros((len(groups), n_groups))
    M[np.arange(len(groups)), groups] = 1.0
    return M


def ss_within(Q: np.ndarray, sizes: np.ndarray, groups: np.ndarray, n_groups: int) -> float:
    """PERMANOVA within-group sum of squares when cells are grouped by `groups`; Q = squared distances summed per cell pair."""
    G = indicator(groups, n_groups)
    return float(np.sum(np.diag(G.T @ Q @ G) / (2 * (G.T @ sizes))))


def decompose(D2: np.ndarray, projects: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    """Nested PERMANOVA (cells = BioProject x class) with the class labels permuted among whole cells."""
    cell_of_run, cells = pd.factorize(pd.MultiIndex.from_arrays([projects, labels]))
    cell_class, classes = pd.factorize(cells.get_level_values(1))
    b, c, n = len(cells), len(classes), len(labels)
    M = indicator(cell_of_run, b)
    Q, sizes = M.T @ D2 @ M, M.sum(axis=0)
    ss_t = Q.sum() / (2 * n)
    ss_cells = float(np.sum(np.diag(Q) / (2 * sizes)))
    ss_cw = ss_within(Q, sizes, cell_class, c)

    def pseudo_f(within_class: float) -> float:
        return ((ss_t - within_class) / (c - 1)) / ((within_class - ss_cells) / (b - c)) if b > c else np.nan

    f = pseudo_f(ss_cw)
    perm = np.array([pseudo_f(ss_within(Q, sizes, rng.permutation(cell_class), c)) for _ in range(N_PERM)]) if b > c else []
    return {"runs": n, "classes": c, "cells": b, "bioprojects": len(set(projects)),
            "single_cell_classes": int(np.sum(np.bincount(cell_class) == 1)),
            "r2_class": (ss_t - ss_cw) / ss_t, "r2_bioproject_within_class": (ss_cw - ss_cells) / ss_t,
            "r2_within_cells": ss_cells / ss_t, "r2_cells": 1 - ss_cells / ss_t, "pseudo_f": f,
            "p_perm": (1 + np.sum(np.asarray(perm) >= f)) / (N_PERM + 1) if b > c else np.nan, "ss_total": ss_t}


def load() -> tuple[np.ndarray, pd.DataFrame]:
    """Training fractions (runs x unitigs, float64) and metadata in the same run order; the split is asserted."""
    X, ids, _ = MatrixLoader(MATRIX).load()
    ids = np.asarray(ids).astype(str)
    meta = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).set_index("Run_accession")
    expected = set(pd.read_csv(SPLIT / "train_accessions.txt", header=None)[0].astype(str))
    if set(ids) != expected or set(meta.index) != expected:
        raise ValueError("matrix runs, metadata and train_accessions.txt differ")
    log.info("%d training runs x %d unitigs", *X.shape)
    return np.asarray(X, dtype=np.float64), meta.loc[ids].reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, meta = load()
    distances = squared_distances(X)
    rows = []
    for field in FIELDS:
        keep = meta[field].isin(S.eligible_classes(SPLIT / "class_eligibility.tsv", field)).to_numpy()
        idx = np.flatnonzero(keep)
        Xs = X[idx]
        direct = float(np.sum((Xs - Xs.mean(axis=0)) ** 2))
        for name, D2 in distances.items():
            r = decompose(D2[np.ix_(idx, idx)], meta.archive_project.to_numpy()[idx], meta[field].to_numpy()[idx],
                          np.random.default_rng(SEED))
            if name == "euclidean" and not np.isclose(r["ss_total"], direct, rtol=1e-6):
                raise AssertionError(f"{field}: SS_total {r['ss_total']} differs from the direct sum {direct}")
            rows.append({"field": field, "distance": name, **r})
            log.info("%s %s: R2 class %.3f, BioProject(class) %.3f, within %.3f, pseudo-F %.2f, p %.4f", field, name,
                     r["r2_class"], r["r2_bioproject_within_class"], r["r2_within_cells"], r["pseudo_f"], r["p_perm"])
    pd.DataFrame(rows).to_csv(OUT / "varsplit.tsv", sep="\t", index=False)
    log.info("wrote %s", OUT / "varsplit.tsv")


if __name__ == "__main__":
    main()
