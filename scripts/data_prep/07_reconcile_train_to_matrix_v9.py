#!/usr/bin/env python3
"""Reconcile the v9 training set with the feature matrix.

The v9 split listed 2,775 training runs but the matrix holds 2,716. The 59
difference is not an accident and not a build failure: those runs have no Logan
unitigs, so they never entered `data/train_samples_v9.fof`, which is what MUSET
was given. They were vectorised from raw FASTQ instead.

`MatrixLoader` keeps only runs that have a matrix column, so the trainer was
already dropping them -- silently. This script makes that exclusion explicit so
the recorded counts match what is actually trained on.

What it does NOT do: touch the held-out set. Ten of the 59 (all of PRJEB79566)
could in principle move there, but the success thresholds were pre-committed
against the 922-run held-out set, and 49 of the 59 could not move regardless --
their BioProjects also have runs in train, so moving them would put one
BioProject on both sides of the split.

Held-out files are asserted byte-identical before and after.

    ./env/bin/python scripts/data_prep/07_reconcile_train_to_matrix_v9.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data_prep"))

from importlib import import_module

_split_mod = import_module("06_create_bioproject_splits_v9")
stratify_key = _split_mod.stratify_key

SPLITS = PROJECT_ROOT / "data/splits_v9"
FOF = PROJECT_ROOT / "data/train_samples_v9.fof"
GROUP_COL = "archive_project"
SEED = 42
N_DEV_FOLDS = 5
MIN_STRATUM = 10

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    matrix_runs = {l.split(":")[0].strip()
                   for l in FOF.read_text().splitlines() if l.strip()}
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    held_before = {p.name: _md5(p) for p in
                   (SPLITS / "test_accessions.txt", SPLITS / "test_metadata.tsv")}

    excluded = train[~train.Run_accession.isin(matrix_runs)]
    kept = train[train.Run_accession.isin(matrix_runs)].reset_index(drop=True)
    logger.info("train %d -> %d (%d excluded, no matrix column)",
                len(train), len(kept), len(excluded))
    if len(excluded) == 0:
        logger.info("Already reconciled; nothing to do.")
        return
    if set(kept.Run_accession) != matrix_runs:
        raise AssertionError(
            f"kept set ({len(kept)}) does not equal the matrix fof "
            f"({len(matrix_runs)}); refusing to write a mismatched split.")

    backup = SPLITS / "pre_matrix_reconcile"
    backup.mkdir(exist_ok=True)
    for name in ("train_accessions.txt", "train_metadata.tsv", "dev_folds.tsv"):
        shutil.copy2(SPLITS / name, backup / name)
    logger.info("backed up 3 files to %s", backup)

    (SPLITS / "excluded_no_matrix.txt").write_text(
        "\n".join(sorted(excluded.Run_accession)) + "\n")
    excluded.to_csv(SPLITS / "excluded_no_matrix_metadata.tsv", sep="\t", index=False)

    (SPLITS / "train_accessions.txt").write_text("\n".join(kept.Run_accession) + "\n")
    kept.to_csv(SPLITS / "train_metadata.tsv", sep="\t", index=False)

    # Dev folds rebuilt over the reconciled train set, same method and seed as
    # 06_create_bioproject_splits_v9.py so the two stay comparable.
    groups = kept[GROUP_COL].astype(str)
    y = stratify_key(kept, MIN_STRATUM)
    sgkf = StratifiedGroupKFold(n_splits=N_DEV_FOLDS, shuffle=True, random_state=SEED)
    fold_of = pd.Series(-1, index=kept.index, dtype=int)
    for k, (_, vi) in enumerate(sgkf.split(kept, y, groups=groups)):
        fold_of.iloc[vi] = k
    if (fold_of < 0).any():
        raise AssertionError(f"{int((fold_of < 0).sum())} runs got no dev fold.")

    dev = kept.assign(fold=fold_of.values)[["Run_accession", GROUP_COL, "fold"]]

    held_acc = set((SPLITS / "test_accessions.txt").read_text().split())
    leak = set(dev.Run_accession) & held_acc
    if leak:
        raise AssertionError(f"{len(leak)} held-out runs appear in dev_folds.tsv.")
    spanning = dev.groupby(GROUP_COL).fold.nunique()
    if (spanning > 1).any():
        raise AssertionError(
            f"{int((spanning > 1).sum())} BioProject(s) span more than one dev fold.")
    if set(dev.Run_accession) != matrix_runs:
        raise AssertionError("dev folds do not cover exactly the matrix runs.")

    dev.to_csv(SPLITS / "dev_folds.tsv", sep="\t", index=False)
    logger.info("dev folds: %d runs, %d folds, sizes %s",
                len(dev), dev.fold.nunique(),
                dev.fold.value_counts().sort_index().to_list())

    cfg_path = SPLITS / "split_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["counts"]["train"] = int(len(kept))
    cfg["bioprojects"]["train"] = int(kept[GROUP_COL].nunique())
    cfg["excluded_no_matrix"] = {
        "n": int(len(excluded)),
        "reason": "no Logan unitigs, so absent from data/train_samples_v9.fof and "
                  "from the MUSET matrix; vectorised from FASTQ instead",
        "file": "excluded_no_matrix.txt",
        "date": str(date.today()),
        "held_out_untouched": True,
    }
    cfg.setdefault("assertion_log", []).append(
        f"train reconciled to matrix: {len(train)} -> {len(kept)}, "
        f"{len(excluded)} excluded, held-out unchanged")
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    held_after = {p.name: _md5(p) for p in
                  (SPLITS / "test_accessions.txt", SPLITS / "test_metadata.tsv")}
    if held_before != held_after:
        raise AssertionError("held-out files changed; they must not.")
    logger.info("held-out unchanged (md5 verified)")
    logger.info("OK: train=%d, held-out=%d", len(kept), len(held_acc))


if __name__ == "__main__":
    main()
