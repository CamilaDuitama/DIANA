#!/usr/bin/env python3
"""Out-of-fold probability tables of the v19 dev-fold screens (PROTOCOLS.md, v19 amendment of 2026-10-21).

For each arm of configs/screens_v19/arms.tsv, from results/screens_v19/<arm>/seed<s>/fold<k>/ with the
table builder of 42_oof_probs_v19.py (five seeds averaged; an early-stopping fit read at its selected
epoch; fold-absent classes exactly 0), per task:
  probs_<arm>net_<task>.tsv  the changed network alone
  probs_<arm>_<task>.tsv     the candidate: on community_type 0.5 * network + 0.5 * logistic regression
                             (on the arm's own input for the input arms, else the fraction one DIANA uses),
                             elsewhere the network
written to results/screens_v19/probs/. Asserted as in 48_: same runs and folds as the logistic
regression table, rows summing to 1.

    ./env/bin/python scripts/analysis/71_screen_probs_v19.py
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
SEEDS = [42, 1, 2, 3, 4]
BASE = ROOT / "results/screens_v19"
OUT = BASE / "probs"
ARMS = ROOT / "configs/screens_v19/arms.tsv"
LR_FRACTION = ROOT / "results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv"
LR_INPUT = BASE / "lr/{arm}/{task}/oof_probs_LogisticRegression_Bal.tsv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def table_builder():
    spec = importlib.util.spec_from_file_location("oof42", ROOT / "scripts/analysis/42_oof_probs_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BASE = BASE
    return mod


def average_with_lr(net: pd.DataFrame, lr_path: Path, name: str) -> pd.DataFrame:
    net = net.set_index("Run_accession")
    lr = pd.read_csv(lr_path, sep="\t").set_index("Run_accession")
    if set(lr.index) != set(net.index) or (lr.loc[net.index, "fold"] != net["fold"]).any():
        raise ValueError(f"{name}: network and logistic regression differ in runs or folds")
    lr = lr.loc[net.index]
    cols = sorted({c for c in net.columns if c.startswith("p_")} | {c for c in lr.columns if c.startswith("p_")})
    avg = 0.5 * net.reindex(columns=cols, fill_value=0.0) + 0.5 * lr.reindex(columns=cols, fill_value=0.0)
    return pd.concat([net[["fold", "y_true"]], avg], axis=1).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", type=Path, default=ARMS, help="arm table (arms_adapted.tsv for the five adapted arms)")
    a = ap.parse_args()
    B = table_builder()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(B.SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    folds = pd.read_csv(B.SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold
    arms = pd.read_csv(a.arms, sep="\t", header=None, names=["arm", "config", "epochs", "extra"], dtype=str)
    for arm, cfg in zip(arms.arm, arms.config):
        input_arm = "screens_v19" in cfg          # its own input file, so its own logistic regression
        for task in TASKS:
            net = B.oof(arm, task, SEEDS, meta)
            if (net.fold.to_numpy() != folds.reindex(net.Run_accession).to_numpy()).any():
                raise ValueError(f"{arm}/{task}: fold column disagrees with dev_folds.tsv")
            net.to_csv(OUT / f"probs_{arm}net_{task}.tsv", sep="\t", index=False)
            cand = net
            if task == "community_type":
                lr = Path(str(LR_INPUT).format(arm=arm, task=task)) if input_arm else Path(str(LR_FRACTION).format(task=task))
                cand = average_with_lr(net, lr, f"{arm}/{task}")
            pc = [c for c in cand.columns if c.startswith("p_")]
            if not np.allclose(cand[pc].sum(1), 1.0, atol=1e-4):
                raise ValueError(f"{arm}/{task}: rows do not sum to 1")
            cand.to_csv(OUT / f"probs_{arm}_{task}.tsv", sep="\t", index=False)
        log.info("%s: four tasks written (logistic regression on %s)", arm, "its own input" if input_arm else "the fractions")


if __name__ == "__main__":
    main()
