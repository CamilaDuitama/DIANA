#!/usr/bin/env python3
"""Runs where every model agrees on a label other than the curated one (v19; descriptive).

For each field and each labelled run that a model can be scored on, the top class of DIANA, logistic
regression, random forest and k-NN, each from a model that never saw the run: out-of-fold dev-fold
predictions for training runs (`scored` dev runs only, so the curated class was in the fit), held-out
predictions of the released DIANA and the read-1 baselines for held-out runs (curated class among the
training classes). A run is a consensus disagreement when the four top classes are equal and differ from
the curated label. Writes results/consensus_v19/runs.tsv (every consensus disagreement) and
results/consensus_v19/summary.tsv (counts by field, curated label, consensus label and study), and prints
the summary. Nothing is refitted.

    ./env/bin/python scripts/analysis/76_consensus_disagreement_v19.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
SPLIT = ROOT / "data/splits_v19"
OUT = ROOT / "results/consensus_v19"
DEV = {"DIANA": "results/devfold_probs_v19/probs_DIANA_{task}.tsv",
       "LogReg": "results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv",
       "RandomForest": "results/baselines_v19/{task}/oof_probs_RandomForest_Bal.tsv",
       "kNN": "results/baselines_v19/{task}/oof_probs_kNN.tsv"}
HELD = {"DIANA": "results/heldout_v19/diana_refit/probs_DIANA_{task}.tsv",
        "LogReg": "results/heldout_v19/baselines/{task}/probs_LogisticRegression_Bal.tsv",
        "RandomForest": "results/heldout_v19/baselines/{task}/probs_RandomForest_Bal.tsv",
        "kNN": "results/heldout_v19/baselines/{task}/probs_kNN.tsv"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def top(path: str, task: str, runs: pd.Index) -> pd.Series:
    d = pd.read_csv(ROOT / path.format(task=task), sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
    missing = runs.difference(d.index)
    if len(missing):
        raise ValueError(f"{path.format(task=task)}: {len(missing)} runs without a prediction")
    pc = [c for c in d.columns if c.startswith("p_")]
    return d.loc[runs, pc].idxmax(axis=1).str[2:]


def side(meta: pd.DataFrame, runs: pd.Index, models: dict, task: str, split: str) -> pd.DataFrame:
    preds = pd.DataFrame({m: top(p, task, runs) for m, p in models.items()})
    out = meta.loc[runs, ["project_name", "archive_project", "material", "community_type", "sample_host"]].copy()
    out["field"], out["split"], out["curated"] = task, split, meta.loc[runs, task]
    out = pd.concat([out, preds], axis=1)
    same = preds.nunique(axis=1) == 1
    out["consensus"] = np.where(same, preds.iloc[:, 0], "")
    return out


def main() -> None:
    global OUT
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--net", action="store_true", help="DIANA without logistic regression (PROTOCOLS.md v19 amendment 2026-10-26)")
    if _p.parse_known_args()[0].net:
        DEV["DIANA"] = "results/devfold_probs_v19net/probs_DIANA_{task}.tsv"; HELD["DIANA"] = "results/heldout_v19/diana_net/probs_DIANA_{task}.tsv"
        OUT = ROOT / "results/consensus_v19net"
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    rows = []
    for task in TASKS:
        dev = S.scored_devfold(tr[task], folds)
        held = S.scored_heldout(te[task], set(tr[task].dropna()))
        rows.append(side(tr, dev[dev].index, DEV, task, "training (out-of-fold)"))
        rows.append(side(te, held[held].index, HELD, task, "held-out"))
    d = pd.concat(rows)
    dis = d[(d.consensus != "") & (d.consensus != d.curated)]
    OUT.mkdir(parents=True, exist_ok=True)
    dis.reset_index().to_csv(OUT / "runs.tsv", sep="\t", index=False)
    s = (dis.groupby(["field", "curated", "consensus", "project_name", "split"]).size().rename("runs").reset_index()
         .sort_values(["field", "runs"], ascending=[True, False]))
    s.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    tot = d.groupby("field").size()
    for task in TASKS:
        log.info("%s: %d of %d scored runs where all four models agree on another label", task, int((dis.field == task).sum()), int(tot[task]))
    pd.set_option("display.width", 220)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
