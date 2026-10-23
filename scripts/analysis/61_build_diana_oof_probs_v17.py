#!/usr/bin/env python3
"""Assemble v17 out-of-fold probability tables for the shared network and DIANA.

Multitask = the 5-seed average of results/seeds_v17/multitask/seed<s>/fold<k>.
DIANA = 0.5 * multitask + 0.5 * the tuned logistic regression
(results/devfold_probs_v17/probs_LogisticRegression_Bal_<task>.tsv).
Output: probs_Multitask_<task>.tsv and probs_DIANA_<task>.tsv in the same
format and directory as the baselines, so every scorer treats them alike.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = ROOT / "results/seeds_v17/multitask"
PROBS_DIR = ROOT / "results/devfold_probs_v17"
SPLITS = ROOT / "data/splits_v17"
TASKS = ["community_type", "feature", "sample_host", "material"]
SEEDS = [42, 1, 2, 3, 4]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    for task in TASKS:
        parts = []
        for k in range(5):
            dirs = [SEEDS_DIR / f"seed{s}" / f"fold{k}" for s in SEEDS]
            runs = [l.strip() for l in open(dirs[0] / "eval_runs.txt") if l.strip()]
            classes = json.load(open(dirs[0] / "label_classes.json"))[task]
            for d in dirs[1:]:
                r2 = [l.strip() for l in open(d / "eval_runs.txt") if l.strip()]
                assert r2 == runs, f"{task} fold {k}: seed dirs scored different runs"
            P = np.mean([np.load(d / "probs_last.npz")[task] for d in dirs], axis=0)
            df = pd.DataFrame(P, columns=[f"p_{c}" for c in classes])
            df.insert(0, "y_true", meta.loc[runs, task].astype(object).to_numpy())
            df.insert(0, "fold", k)
            df.insert(0, "Run_accession", runs)
            parts.append(df)
        mt = pd.concat(parts, ignore_index=True)
        mt.to_csv(PROBS_DIR / f"probs_Multitask_{task}.tsv", sep="\t", index=False)

        lr = pd.read_csv(PROBS_DIR / f"probs_LogisticRegression_Bal_{task}.tsv", sep="\t")
        p_cols_mt = sorted(c for c in mt.columns if c.startswith("p_"))
        p_cols_lr = sorted(c for c in lr.columns if c.startswith("p_"))
        assert p_cols_mt == p_cols_lr, f"{task}: class columns differ between network and logistic regression"
        lr_idx = lr.set_index("Run_accession")
        # the logistic regression scores only runs labelled for the task; DIANA's
        # average exists where both components do
        common = [r for r in mt.Run_accession if r in lr_idx.index]
        mt_idx = mt.set_index("Run_accession").loc[common]
        avg = 0.5 * mt_idx[p_cols_mt].to_numpy() + 0.5 * lr_idx.loc[common, p_cols_mt].to_numpy()
        out = pd.DataFrame(avg, columns=p_cols_mt)
        out.insert(0, "y_true", mt_idx["y_true"].to_numpy())
        out.insert(0, "fold", mt_idx["fold"].to_numpy())
        out.insert(0, "Run_accession", common)
        out.to_csv(PROBS_DIR / f"probs_DIANA_{task}.tsv", sep="\t", index=False)
        log.info("%s: multitask %d runs, DIANA %d runs, %d classes",
                 task, len(mt), len(out), len(p_cols_mt))


if __name__ == "__main__":
    main()
