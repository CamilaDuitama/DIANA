#!/usr/bin/env python3
"""Out-of-fold probabilities of the network + logistic-regression average (v19 composition test).

For each task and each architecture (Separate, Shared): P = 0.5 * P_network + 0.5 * P_logreg,
both five-fold out-of-fold (results/devfold_probs_v19/probs_<arch>_<task>.tsv and
results/baselines_v19/<task>/oof_probs_LogisticRegression_Bal.tsv), aligned by run accession
and class name; a class absent from one model's columns counts as probability 0 there
(both models give 0 to classes absent from a fold's fit). Writes probs_<arch>LR_<task>.tsv
in the same layout. Asserted: identical run sets and folds, rows summing to 1.

    ./env/bin/python scripts/analysis/45_composition_probs_v19.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NET = ROOT / "results/devfold_probs_v19"
LR = ROOT / "results/baselines_v19"
TASKS = ["community_type", "feature", "sample_host", "material"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def table(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(path, sep="\t").set_index("Run_accession")
    pc = [c for c in d.columns if c.startswith("p_")]
    return d[pc], d[["fold", "y_true"]]


def main() -> None:
    for task in TASKS:
        lr, lr_meta = table(LR / task / "oof_probs_LogisticRegression_Bal.tsv")
        for arch in ("Separate", "Shared"):
            net, net_meta = table(NET / f"probs_{arch}_{task}.tsv")
            if set(net.index) != set(lr.index):
                raise ValueError(f"{arch}/{task}: network and logistic regression cover different runs")
            lr_ = lr.loc[net.index]
            if (lr_meta.loc[net.index, "fold"] != net_meta["fold"]).any():
                raise ValueError(f"{arch}/{task}: folds disagree")
            cols = sorted(set(net.columns) | set(lr_.columns))
            avg = 0.5 * net.reindex(columns=cols, fill_value=0.0) + 0.5 * lr_.reindex(columns=cols, fill_value=0.0)
            if not np.allclose(avg.sum(1), 1.0, atol=1e-4):
                raise ValueError(f"{arch}/{task}: averaged rows do not sum to 1")
            out = pd.concat([net_meta, avg], axis=1).reset_index()
            path = NET / f"probs_{arch}LR_{task}.tsv"
            out.to_csv(path, sep="\t", index=False)
            log.info("%s: %d runs, %d classes", path.name, len(out), len(cols))


if __name__ == "__main__":
    main()
