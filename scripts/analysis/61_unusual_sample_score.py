#!/usr/bin/env python3
"""X11 step 3.5: the unusual-sample score of every training run, on the grouped dev folds.

Space = the 110,202 v9 unitig fractions, distance = cosine, depth = log10 of the run's
non-zero unitig count. For a run in fold k the candidates are the training-fold runs (folds
!= k) within 0.25 log10 units of its depth, or the 5 closest in depth when fewer than 5
qualify; the score is the mean cosine distance to its 5 nearest candidates. The reference
distribution of fold k is the same score for every training-fold run against training-fold
runs from other BioProjects; the cutoff is its 95th percentile. A run with no non-zero
unitig is at distance 1 from everything and is unusual. Settings fixed in PROJECT.md §8
(X11 protocol) before this ran.

Writes results/flag_postprocessing_v9/unusual_scores.tsv (Run_accession, fold, project,
n_nonzero, score, cutoff, unusual) and prints the per-fold counts and the PRJEB42014 share.

    sbatch on edid, 4 cores, 16 GB: ./env/bin/python scripts/analysis/61_unusual_sample_score.py
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/flag_postprocessing_v9/unusual_scores.tsv"
K, DEPTH_WINDOW, PERCENTILE = 5, 0.25, 95.0
CHECK_PROJECT = "PRJEB42014"


def knn_score(D: np.ndarray, log_depth: np.ndarray, query: np.ndarray, pool: np.ndarray,
              same_project: np.ndarray | None = None) -> np.ndarray:
    """Mean cosine distance from each query run to its K nearest depth-matched pool runs.
    same_project[i] (boolean over pool, per query i) excludes a query's own study."""
    out = np.empty(len(query), dtype=np.float32)
    for n, i in enumerate(query):
        allowed = np.ones(len(pool), dtype=bool) if same_project is None else ~same_project[n]
        cand = pool[allowed]
        close = np.abs(log_depth[cand] - log_depth[i]) <= DEPTH_WINDOW
        if close.sum() >= K:
            cand = cand[close]
        else:
            cand = cand[np.argsort(np.abs(log_depth[cand] - log_depth[i]), kind="stable")[:K]]
        d = D[i, cand]
        out[n] = float(np.sort(d)[:K].mean())
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    t0 = time.time()
    with np.load(NPZ, allow_pickle=False) as z:
        ids = z["sample_ids"].astype(str); X = z["frac"].astype(np.float32)
    n_nonzero = (X > 0).sum(axis=1)
    norms = np.linalg.norm(X, axis=1); zero = norms == 0
    Xn = X / np.where(zero, 1.0, norms)[:, None]
    D = 1.0 - Xn @ Xn.T                      # cosine distance; zero rows are at distance 1 from everything
    D[zero, :] = 1.0; D[:, zero] = 1.0; np.fill_diagonal(D, 0.0)
    del X, Xn
    logger.info("distance matrix %s in %.0f s; %d runs with no non-zero unitig", D.shape, time.time() - t0, int(zero.sum()))
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids]
    fold = folds["fold"].to_numpy(); project = folds["archive_project"].astype(str).to_numpy()
    log_depth = np.log10(1.0 + n_nonzero)
    score = np.full(len(ids), np.nan, dtype=np.float32); cutoff = np.full(len(ids), np.nan, dtype=np.float32)
    for k in sorted(set(fold)):
        ev = np.flatnonzero(fold == k); tr = np.flatnonzero(fold != k)
        score[ev] = knn_score(D, log_depth, ev, tr)
        ref = knn_score(D, log_depth, tr, tr, same_project=(project[tr][:, None] == project[tr][None, :]))
        cutoff[ev] = np.percentile(ref, PERCENTILE)
        logger.info("fold %d: %d eval runs, cutoff %.4f, unusual %d (%.1f%%)", k, len(ev), cutoff[ev][0],
                    int((score[ev] > cutoff[ev]).sum()), 100 * (score[ev] > cutoff[ev]).mean())
    res = pd.DataFrame({"Run_accession": ids, "fold": fold, "project": project, "n_nonzero": n_nonzero,
                        "score": score, "cutoff": cutoff, "unusual": score > cutoff})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, sep="\t", index=False)
    chk = res[res.project == CHECK_PROJECT]
    print(f"unusual runs: {int(res.unusual.sum())} of {len(res)} ({100 * res.unusual.mean():.1f}%); "
          f"runs with no non-zero unitig: {int(zero.sum())} (all unusual: {bool(res.unusual[zero].all()) if zero.any() else 'n/a'})")
    print(f"{CHECK_PROJECT}: {int(chk.unusual.sum())} of {len(chk)} runs unusual ({100 * chk.unusual.mean():.1f}%); "
          f"score median {chk.score.median():.4f} vs cutoff {chk.cutoff.iloc[0]:.4f}")
    print(res.groupby("project").unusual.agg(["sum", "size", "mean"]).query("sum > 0").sort_values("mean", ascending=False).round(3).to_string())
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
