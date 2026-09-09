#!/usr/bin/env python3
"""Rebuild the v9 development folds for better class coverage.

The first dev folds stratified on `community_type|feature|material` and left
`sample_host` out, then took the first seed. Because a BioProject studies one host
or one environment, class membership is almost perfectly confounded with the group,
so a careless assignment holds out whole classes: fold 3 had only **16.8 %** of its
`feature` test rows in the training vocabulary, which caps that head near zero
before the model does anything.

This script stratifies on all four tasks and searches seeds for the assignment that
maximises the **worst** per-fold in-vocabulary coverage, averaged over tasks.

The objective never consults a model -- it is computed from labels and BioProject
membership alone -- so this selects an evaluation that can measure the model, not
one that flatters it. The search is still a search: record the seed.

Only `dev_folds.tsv` is rewritten. train/held-out membership is untouched, so the
pre-committed held-out thresholds stay valid.

    ./env/bin/python scripts/data_prep/08_rebuild_dev_folds_v9.py --n-seeds 200
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
GROUP_COL = "archive_project"
TASKS = ["community_type", "feature", "sample_host", "material"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def coverage(md: pd.DataFrame, assign: np.ndarray) -> tuple[dict, np.ndarray]:
    """Per task: (mean, worst) fraction of test rows whose class is in train."""
    folds = sorted(set(assign))
    out = {}
    for t in TASKS:
        cov = []
        for f in folds:
            te = md.loc[assign == f, t].dropna()
            tr = set(md.loc[assign != f, t].dropna())
            cov.append(te.isin(tr).mean() if len(te) else np.nan)
        out[t] = (float(np.nanmean(cov)), float(np.nanmin(cov)))
    sizes = np.array([int((assign == f).sum()) for f in folds])
    return out, sizes


def objective(out: dict) -> float:
    return float(np.mean([v[1] for v in out.values()]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--min-stratum", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    md = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    held = set((SPLITS / "test_accessions.txt").read_text().split())
    groups = md[GROUP_COL].astype(str).to_numpy()

    key = md[TASKS].fillna("_").agg("|".join, axis=1)
    vc = key.value_counts()
    key = key.where(key.map(vc) >= args.min_stratum, "__rare__").to_numpy()

    cur = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t")
    cur_assign = md.Run_accession.map(dict(zip(cur.Run_accession, cur.fold))).to_numpy()
    cur_out, cur_sizes = coverage(md, cur_assign)
    logger.info("current  objective=%.4f  sizes=%s", objective(cur_out), list(cur_sizes))

    best = (objective(cur_out), None, None)
    for seed in range(args.n_seeds):
        sg = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
        a = np.full(len(md), -1, dtype=int)
        for k, (_, vi) in enumerate(sg.split(md, key, groups=groups)):
            a[vi] = k
        if (a < 0).any():
            continue
        out, _ = coverage(md, a)
        if objective(out) > best[0]:
            best = (objective(out), seed, a)

    if best[1] is None:
        logger.info("No seed beat the current folds; leaving them alone.")
        return

    score, seed, assign = best
    out, sizes = coverage(md, assign)
    logger.info("best     objective=%.4f  seed=%d  sizes=%s", score, seed, list(sizes))
    for t, (m, mn) in out.items():
        logger.info("  %-16s mean %6.1f%%   worst %6.1f%%", t, 100 * m, 100 * mn)

    dev = md[["Run_accession", GROUP_COL]].copy()
    dev["fold"] = assign

    spanning = dev.groupby(GROUP_COL).fold.nunique()
    if (spanning > 1).any():
        raise AssertionError(f"{int((spanning > 1).sum())} BioProject(s) span folds.")
    if set(dev.Run_accession) & held:
        raise AssertionError("held-out runs appear in the dev folds.")
    if set(dev.Run_accession) != set(md.Run_accession):
        raise AssertionError("dev folds do not cover train exactly.")

    if args.dry_run:
        logger.info("--dry-run: nothing written.")
        return

    dev.to_csv(SPLITS / "dev_folds.tsv", sep="\t", index=False)
    cfg_path = SPLITS / "split_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["dev_folds"] = {
        "seed": int(seed),
        "n_folds": int(args.n_folds),
        "stratified_on": TASKS,
        "selected_by": "max of the mean-across-tasks worst-per-fold in-vocabulary "
                       "coverage; objective uses labels and BioProject membership "
                       "only, never a model",
        "seeds_searched": int(args.n_seeds),
        "objective": round(score, 4),
        "previous_objective": round(objective(cur_out), 4),
        "date": str(date.today()),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    logger.info("wrote dev_folds.tsv and recorded the choice in split_config.json")


if __name__ == "__main__":
    main()
