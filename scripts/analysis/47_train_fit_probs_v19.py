#!/usr/bin/env python3
"""Training-set probability tables of DIANA v19 (each run predicted by networks fitted on it).

From results/train_fit_v19/multitask/seed<s>/ (train_probs.npz, train_runs.txt,
label_classes.json), per task:
  probs_Shared_<task>.tsv   the shared network's head for that task, five seeds averaged
  probs_DIANA_<task>.tsv    DIANA as fixed on the dev folds: on community_type the average
                            0.5 * shared + 0.5 * logistic regression (its training-set table,
                            results/train_fit_v19/baselines/community_type/), elsewhere shared
Layout as the baseline tables (Run_accession, y_true, p_<class>), labelled runs only.
Asserted: every seed fitted the same runs and class order, rows sum to 1. Logged: share of
labelled training runs whose refit label equals the final fit's (results/final_v19), per seed.

    ./env/bin/python scripts/analysis/47_train_fit_probs_v19.py
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
REFIT = ROOT / "results/train_fit_v19/multitask"
FINAL = ROOT / "results/final_v19/multitask"
LR = ROOT / "results/train_fit_v19/baselines/{task}/probs_LogisticRegression_Bal.tsv"
OUT = ROOT / "results/train_fit_v19"
SPLIT = ROOT / "data/splits_v19"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def lines(path: Path) -> list[str]:
    return [l.strip() for l in open(path) if l.strip()]


def shared(task: str, lab: pd.Series) -> pd.DataFrame:
    dirs = [REFIT / f"seed{s}" for s in SEEDS]
    runs = lines(dirs[0] / "train_runs.txt")
    classes = json.load(open(dirs[0] / "label_classes.json"))[task]
    for d in dirs[1:]:
        if lines(d / "train_runs.txt") != runs or json.load(open(d / "label_classes.json"))[task] != classes:
            raise ValueError(f"{task}: seeds differ in fitted runs or class order")
    P = np.mean([np.load(d / "train_probs.npz")[task] for d in dirs], axis=0)
    out = pd.DataFrame(P, columns=[f"p_{c}" for c in classes])
    out.insert(0, "y_true", lab.reindex(runs).to_numpy())
    out.insert(0, "Run_accession", runs)
    out = out[out.y_true.notna()].reset_index(drop=True)
    if set(out.Run_accession) != set(lab.dropna().index):
        raise ValueError(f"{task}: training-set rows do not cover the labelled runs")
    for s, d in zip(SEEDS, dirs):
        f = FINAL / f"seed{s}"
        if lines(f / "train_runs.txt") != runs:
            raise ValueError(f"seed {s}: refit and final fit hold different runs")
        keep = lab.reindex(runs).notna().to_numpy()
        same = np.load(d / "train_preds.npz")[task][keep] == np.load(f / "train_preds.npz")[task][keep]
        log.info("%s seed %d: refit label = final-fit label on %.4f of labelled runs", task, s, same.mean())
    return out


def check(d: pd.DataFrame, name: str) -> None:
    pc = [c for c in d.columns if c.startswith("p_")]
    if not np.allclose(d[pc].sum(1), 1.0, atol=1e-4):
        raise ValueError(f"{name}: probabilities do not sum to 1")


def main() -> None:
    meta = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    for task in TASKS:
        sh = shared(task, meta[task])
        check(sh, f"Shared/{task}")
        sh.to_csv(OUT / f"probs_Shared_{task}.tsv", sep="\t", index=False)
        dia = sh
        if task == "community_type":
            lr = pd.read_csv(str(LR).format(task=task), sep="\t").set_index("Run_accession")
            net = sh.set_index("Run_accession")
            lr = lr.loc[net.index]
            cols = sorted(set(c for c in net.columns if c.startswith("p_")) | set(c for c in lr.columns if c.startswith("p_")))
            avg = 0.5 * net.reindex(columns=cols, fill_value=0.0) + 0.5 * lr.reindex(columns=cols, fill_value=0.0)
            dia = pd.concat([net[["y_true"]], avg], axis=1).reset_index()
        check(dia, f"DIANA/{task}")
        dia.to_csv(OUT / f"probs_DIANA_{task}.tsv", sep="\t", index=False)
        log.info("%s: %d labelled training runs", task, len(dia))


if __name__ == "__main__":
    main()
