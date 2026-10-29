#!/usr/bin/env python3
"""Sequencing-depth control with tuned baselines (v19; R1.1, R3.7, R3.11).

Logistic regression and random forest given only the run's read count (log10, AncientMetagenomeDir libraries table;
results/depth_control_v19/depth.tsv from 77_), tuned exactly as the baselines on the unitig fractions
(scripts/evaluation/40_baselines_v19.py: same grids, same model arguments, same dev folds, selection on the pooled
out-of-fold macro-F1 over eligible classes, first in grid order on a tie), then fitted on every labelled training run
and evaluated on the held-out runs. Compared, on the dev folds (out-of-fold) and the held-out set, with the same
baseline on the unitig fractions (stored predictions, as 75_) and with the released DIANA, paired over BioProject
resamples. Writes results/depth_control_tuned_v19/{grid.tsv, selected.tsv, scores.tsv, paired.tsv}.

    ./env/bin/python scripts/analysis/81_depth_tuned_v19.py
"""
from __future__ import annotations

import importlib.util
import json
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
DEPTH = ROOT / "results/depth_control_v19/depth.tsv"
OUT = ROOT / "results/depth_control_tuned_v19"
MODELS = {"LogReg": "LogisticRegression_Bal", "RandomForest": "RandomForest_Bal"}
DEV = {"DIANA": "results/devfold_probs_v19/probs_DIANA_{task}.tsv",
       "LogReg": "results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv",
       "RandomForest": "results/baselines_v19/{task}/oof_probs_RandomForest_Bal.tsv"}
HELDOUT = {"DIANA": "results/heldout_v19/diana_refit/probs_DIANA_{task}.tsv",
           "LogReg": "results/heldout_v19/baselines/{task}/probs_LogisticRegression_Bal.tsv",
           "RandomForest": "results/heldout_v19/baselines/{task}/probs_RandomForest_Bal.tsv"}
METRICS = {"macro_f1": S.f1_macro_eligible, "balanced_accuracy": S.balanced_accuracy_eligible}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def tuning_module():
    spec = importlib.util.spec_from_file_location("tune", ROOT / "scripts/evaluation/40_baselines_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def argmax(path: str, task: str) -> pd.Series:
    d = pd.read_csv(ROOT / path.format(task=task), sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
    return d[[c for c in d.columns if c.startswith("p_")]].idxmax(axis=1).str[2:]


def tune(T, name: str, x: pd.Series, y: pd.Series, folds: pd.Series, scored: pd.Series, eligible: set) -> tuple[dict, pd.Series, list]:
    """Grid search on the dev folds with the baselines' rule; returns the selection and its out-of-fold predictions."""
    lab = y.notna()
    best, grid = None, []
    for i, params in enumerate(T.GRIDS[name]):
        pred = pd.Series("", index=y.index, dtype=object)
        for k in sorted(folds.unique()):
            fit, ev = lab & (folds != k), lab & (folds == k)
            m = T.build(name, params, 8).fit(x[fit].to_numpy()[:, None], y[fit].to_numpy())
            pred[ev] = m.predict(x[ev].to_numpy()[:, None])
        score = S.f1_macro_eligible(y[scored].astype(str).to_numpy(), pred[scored].astype(str).to_numpy(), eligible)
        grid.append({"model": name, "config": i, "params": json.dumps(params), "f1_macro_eligible": score})
        if best is None or score > best[0]:
            best = (score, params, pred)
    return best[1], best[2], grid


def main() -> None:
    global OUT
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--net", action="store_true", help="DIANA without logistic regression (PROTOCOLS.md v19 amendment 2026-10-26)")
    if _p.parse_known_args()[0].net:
        DEV["DIANA"] = "results/devfold_probs_v19net/probs_DIANA_{task}.tsv"; HELDOUT["DIANA"] = "results/heldout_v19/diana_net/probs_DIANA_{task}.tsv"
        OUT = ROOT / "results/depth_control_tuned_v19net"
    T = tuning_module()
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    if folds.isna().any():
        raise ValueError("training runs without a dev fold")
    reads = pd.read_csv(DEPTH, sep="\t", index_col=0).log_reads
    reads.index = reads.index.astype(str)
    if reads.reindex(tr.index.append(te.index)).isna().any():
        raise ValueError("runs without a read count")
    grids, selected, scores, pairs = [], [], [], []
    for task in TASKS:
        el = S.eligible_classes(SPLIT / "class_eligibility.tsv", task)
        y, x = tr[task], reads.reindex(tr.index)
        dev_sc = S.scored_devfold(y, folds)
        held_sc = S.scored_heldout(te[task], set(y.dropna()))
        dev_runs, held_runs = dev_sc[dev_sc].index, held_sc[held_sc].index
        dev = {m: argmax(p, task).reindex(dev_runs) for m, p in DEV.items()}
        held = {m: argmax(p, task).reindex(held_runs) for m, p in HELDOUT.items()}
        for short, name in MODELS.items():
            params, oof, grid = tune(T, name, x, y, folds, dev_sc, el)
            grids += [{"task": task, **r} for r in grid]
            selected.append({"task": task, "model": name, "params": json.dumps(params)})
            log.info("%s %s selected %s", task, name, params)
            lab = y.notna()
            m = T.build(name, params, 8).fit(x[lab].to_numpy()[:, None], y[lab].to_numpy())
            dev[f"Reads_{short}"] = oof.reindex(dev_runs)
            held[f"Reads_{short}"] = pd.Series(m.predict(reads.reindex(held_runs).to_numpy()[:, None]), index=held_runs)
        for split, runs, meta, preds in (("dev folds", dev_runs, tr, dev), ("held-out", held_runs, te, held)):
            if any(p.isna().any() for p in preds.values()):
                raise ValueError(f"{task}/{split}: scored runs without a prediction")
            yy = meta.loc[runs, task].astype(str).to_numpy()
            g = meta.loc[runs, "archive_project"].astype(str).to_numpy()
            cls = el & set(yy)
            P = {k: v.astype(str).to_numpy() for k, v in preds.items()}
            for metric, fn in METRICS.items():
                for k, p in P.items():
                    r = S.score_with_interval(yy, p, g, cls, f"depthtuned|{split}|{task}", metric=fn)
                    scores.append({"task": task, "split": split, "metric": metric, "model": k, "value": r["value"],
                                   "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
                for a, b in [(s, f"Reads_{s}") for s in MODELS] + [("DIANA", f"Reads_{s}") for s in MODELS]:
                    r = S.paired_difference(runs, runs, yy, P[a], P[b], g, cls, f"depthtunedpair|{split}|{task}", metric=fn)
                    pairs.append({"task": task, "split": split, "metric": metric, "a": a, "b": b, "delta": r["delta"],
                                  "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_boot": r["p_boot"]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grids).to_csv(OUT / "grid.tsv", sep="\t", index=False)
    pd.DataFrame(selected).to_csv(OUT / "selected.tsv", sep="\t", index=False)
    pd.DataFrame(scores).to_csv(OUT / "scores.tsv", sep="\t", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "paired.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(selected).to_string(index=False))
    s = pd.DataFrame(scores)
    print(s.pivot_table(index=["split", "metric", "model"], columns="task", values="value").round(3).to_string())
    print(pd.DataFrame(pairs).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
