#!/usr/bin/env python3
"""Out-of-fold probability tables of the study-weighted shared network (v19, amendment of 2026-10-12).

From results/studyweight_v19/multitask/seed<s>/fold<k>/ with the table builder of
42_oof_probs_v19.py (five seeds averaged, fold-absent classes exactly 0), per task:
  probs_SharedW_<task>.tsv  the weighted shared network
  probs_DIANAW_<task>.tsv   the candidate: on community_type 0.5 * weighted network + 0.5 *
                            logistic regression (the unweighted one DIANA uses), elsewhere the
                            weighted network
written to results/studyweight_v19/. Asserted as in 42_ and 45_: same runs and folds as the
logistic regression table, rows summing to 1.

    ./env/bin/python scripts/analysis/48_studyweight_probs_v19.py
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
SEEDS = [42, 1, 2, 3, 4]
OUT = ROOT / "results/studyweight_v19"
LR = ROOT / "results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def table_builder():
    spec = importlib.util.spec_from_file_location("oof42", ROOT / "scripts/analysis/42_oof_probs_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BASE = OUT
    return mod


def main() -> None:
    B = table_builder()
    meta = pd.read_csv(B.SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    folds = pd.read_csv(B.SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold
    for task in TASKS:
        sw = B.oof("multitask", task, SEEDS, meta)
        if (sw.fold.to_numpy() != folds.reindex(sw.Run_accession).to_numpy()).any():
            raise ValueError(f"SharedW/{task}: fold column disagrees with dev_folds.tsv")
        sw.to_csv(OUT / f"probs_SharedW_{task}.tsv", sep="\t", index=False)
        cand = sw
        if task == "community_type":
            net = sw.set_index("Run_accession")
            lr = pd.read_csv(str(LR).format(task=task), sep="\t").set_index("Run_accession")
            if set(lr.index) != set(net.index) or (lr.loc[net.index, "fold"] != net["fold"]).any():
                raise ValueError(f"{task}: network and logistic regression differ in runs or folds")
            lr = lr.loc[net.index]
            cols = sorted({c for c in net.columns if c.startswith("p_")} | {c for c in lr.columns if c.startswith("p_")})
            avg = 0.5 * net.reindex(columns=cols, fill_value=0.0) + 0.5 * lr.reindex(columns=cols, fill_value=0.0)
            cand = pd.concat([net[["fold", "y_true"]], avg], axis=1).reset_index()
        pc = [c for c in cand.columns if c.startswith("p_")]
        if not np.allclose(cand[pc].sum(1), 1.0, atol=1e-4):
            raise ValueError(f"DIANAW/{task}: rows do not sum to 1")
        cand.to_csv(OUT / f"probs_DIANAW_{task}.tsv", sep="\t", index=False)
        log.info("%s: %d runs", task, len(cand))


if __name__ == "__main__":
    main()
