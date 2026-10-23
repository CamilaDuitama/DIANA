#!/usr/bin/env python3
"""Out-of-fold probability tables of the v19 networks on the dev folds.

From results/seeds_v19/<arm>/seed<s>/fold<k>/ (probs_last.npz, eval_runs.txt,
label_classes.json), per task:
  probs_Separate_<task>.tsv      the single-task arm of that task, five seeds averaged
  probs_Shared_<task>.tsv        the multi-task arm's head for that task, five seeds averaged
  probs_Separate1_<task>.tsv     the single-task arm, seed 42 only
  probs_Shared1_<task>.tsv       the multi-task arm, seed 42 only
Same layout as the baseline tables (Run_accession, fold, y_true, p_<class>), labelled runs
only, fold-absent classes exactly 0. Asserted: every seed scored the same runs, every
labelled training run has exactly one out-of-fold row, rows sum to 1.

    ./env/bin/python scripts/analysis/42_oof_probs_v19.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 1, 2, 3, 4]
TASKS = ["community_type", "feature", "sample_host", "material"]
BASE = ROOT / "results/seeds_v19"
OUT = ROOT / "results/devfold_probs_v19"
SPLIT = ROOT / "data/splits_v19"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def fold_probs(d: Path, task: str) -> np.ndarray:
    """Probabilities of one fit: at its early-stopping epoch when it recorded one (screens of 2026-10-21,
    saved with --save-probs-by-epoch), otherwise after its last epoch."""
    sel = json.load(open(d / "history.json")).get("selected_epoch")
    if sel is None:
        return np.load(d / "probs_last.npz")[task]
    return np.load(d / "probs_by_epoch.npz")[task][int(sel) - 1]


def oof(arm: str, task: str, seeds: list[int], meta: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for k in range(5):
        dirs = [BASE / arm / f"seed{s}" / f"fold{k}" for s in seeds]
        runs = [l.strip() for l in open(dirs[0] / "eval_runs.txt") if l.strip()]
        classes = json.load(open(dirs[0] / "label_classes.json"))[task]
        for d in dirs[1:]:
            if [l.strip() for l in open(d / "eval_runs.txt") if l.strip()] != runs:
                raise ValueError(f"{arm} fold {k}: seeds scored different runs")
            if json.load(open(d / "label_classes.json"))[task] != classes:
                raise ValueError(f"{arm} fold {k}: class order differs between seeds")
        P = np.mean([fold_probs(d, task) for d in dirs], axis=0)
        df = pd.DataFrame(P, columns=[f"p_{c}" for c in classes])
        df.insert(0, "fold", k)
        df.insert(0, "Run_accession", runs)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    lab = meta.set_index("Run_accession")[task]
    out.insert(2, "y_true", lab.reindex(out.Run_accession).to_numpy())
    out = out[out.y_true.notna()].reset_index(drop=True)
    want = set(lab.dropna().index)
    if set(out.Run_accession) != want or out.Run_accession.duplicated().any():
        raise ValueError(f"{arm}/{task}: out-of-fold rows do not cover the labelled runs exactly once")
    pc = [c for c in out.columns if c.startswith("p_")]
    if not np.allclose(out[pc].sum(1), 1.0, atol=1e-4):
        raise ValueError(f"{arm}/{task}: probabilities do not sum to 1")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold
    for task in TASKS:
        for name, arm in (("Separate", task), ("Shared", "multitask")):
            for suffix, seeds in (("", SEEDS), ("1", [42])):
                d = oof(arm, task, seeds, meta)
                if (d.fold.to_numpy() != folds.reindex(d.Run_accession).to_numpy()).any():
                    raise ValueError(f"{name}{suffix}/{task}: fold column disagrees with dev_folds.tsv")
                path = OUT / f"probs_{name}{suffix}_{task}.tsv"
                d.to_csv(path, sep="\t", index=False)
                log.info("%s: %d runs", path.name, len(d))


if __name__ == "__main__":
    main()
