#!/usr/bin/env python3
"""W1: five within-study folds over the 2,716 v9 training runs, grouped by SAMPLE, not by study.

Purpose (PROJECT.md section 8, W protocol): the number for "a new run from a study the model
has seen", to sit beside the held-out number for "a run from a study it has never seen".
Runs of one physical sample (archive_sample_accession) stay together, so no fold tests a
run whose sister run was trained on; studies are on both sides by construction.

Stratification: the composite key community_type | feature | material with rare strata
pooled, as 06_create_bioproject_splits_v9.py did. Seed 42. Asserted in code: no sample on
both sides of any fold, every run assigned once, and the share of studies that appear in
more than one fold is reported.

Writes results/within_study_v9/folds.tsv (Run_accession, archive_project,
archive_sample_accession, fold) and fold_report.json.

    ./env/bin/python scripts/data_prep/34_within_study_folds_v9.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS, OUT = ROOT / "data/splits_v9", ROOT / "results/within_study_v9"
N_FOLDS, SEED, MIN_STRATUM = 5, 42, 5
SAMPLE = "archive_sample_accession"


def stratify_key(df: pd.DataFrame) -> np.ndarray:
    key = (df["community_type"].fillna("_").astype(str) + "|" + df["feature"].fillna("_").astype(str)
           + "|" + df["material"].fillna("_").astype(str))
    vc = key.value_counts()
    return key.where(key.map(vc) >= MIN_STRATUM, "__rare__").to_numpy()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "folds.tsv").exists():
        raise SystemExit(f"{OUT / 'folds.tsv'} exists; refusing to overwrite")
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    train = [l.strip() for l in open(SPLITS / "train_accessions.txt") if l.strip()]
    meta = meta.set_index("Run_accession").loc[train].reset_index()
    if meta[SAMPLE].isna().any():
        raise SystemExit(f"{int(meta[SAMPLE].isna().sum())} runs have no {SAMPLE}")
    y, groups = stratify_key(meta), meta[SAMPLE].astype(str)
    fold = pd.Series(-1, index=meta.index, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(sgkf.split(meta, y, groups=groups)):
        fold.iloc[te] = k
    if (fold < 0).any():
        raise AssertionError("a run was not assigned to a fold")
    for k in range(N_FOLDS):
        a, b = set(groups[fold == k]), set(groups[fold != k])
        if a & b:
            raise AssertionError(f"fold {k}: {len(a & b)} samples on both sides")
    out = pd.DataFrame({"Run_accession": meta.Run_accession, "archive_project": meta.archive_project,
                        SAMPLE: groups, "fold": fold})
    out.to_csv(OUT / "folds.tsv", sep="\t", index=False)
    proj_folds = out.groupby("archive_project").fold.nunique()
    report = {"n_runs": len(out), "n_samples": int(groups.nunique()), "n_projects": int(out.archive_project.nunique()),
              "runs_per_fold": out.fold.value_counts().sort_index().to_dict(),
              "projects_in_more_than_one_fold": int((proj_folds > 1).sum()),
              "projects_in_one_fold_only": int((proj_folds == 1).sum()),
              "runs_whose_project_is_also_in_another_fold": int(out.archive_project.isin(proj_folds[proj_folds > 1].index).sum()),
              "seed": SEED, "grouping": SAMPLE, "stratification": "community_type|feature|material, strata < 5 pooled"}
    json.dump(report, open(OUT / "fold_report.json", "w"), indent=2)
    logger.info("%s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
