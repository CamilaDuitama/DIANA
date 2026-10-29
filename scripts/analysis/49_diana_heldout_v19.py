#!/usr/bin/env python3
"""DIANA v19 on the held-out runs (ledger read 2): probability tables, no-call, reliability.

From the final fits (results/final_v19/multitask/seed<s>/probs_last.npz, eval_runs.txt,
label_classes.json; seeds 42, 1, 2, 3, 4), per task:
  probs_Shared_<task>.tsv  the shared network, five seeds averaged
  probs_DIANA_<task>.tsv   DIANA as fixed on the dev folds: on community_type 0.5 * shared +
                           0.5 * logistic regression (its read-1 held-out table), elsewhere shared
Layout as the read-1 baseline tables (Run_accession, y_true, p_<class>), all 863 runs.
Asserted: seeds agree on runs and class order, runs equal test_metadata, rows sum to 1.

Also on the scored held-out runs (true class among the training labels, as 43_ --heldout):
no-call at the dev-fold cutoffs (results/posthoc_v19/abstention_v19.json): share of runs
answered and accuracy, beside accuracy with no abstention; reliability curve (10
equal-width bins of the top probability: runs, mean confidence, accuracy) and expected
calibration error, descriptive only (amendment 2026-10-08). Written to
results/heldout_v19/diana/.

The released DIANA (read 5) is the same recipe refitted with saved weights:
--fits results/final_v19_weights/multitask --out results/heldout_v19/diana_refit
(scripts/analysis/run_refit_scoring_v19.sbatch). Its logistic regression
(results/final_v19_lr_unused/community_type/; moved out of the release when DIANA became the network alone,
PROTOCOLS.md v19 amendment 2026-10-26) gives held-out probabilities identical to the
read-1 table used here.

    ./env/bin/python scripts/analysis/49_diana_heldout_v19.py
"""
from __future__ import annotations

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
SEEDS = [42, 1, 2, 3, 4]
FITS = ROOT / "results/final_v19/multitask"
SPLIT = ROOT / "data/splits_v19"
LR = ROOT / "results/heldout_v19/baselines/{task}/probs_LogisticRegression_Bal.tsv"
CUTOFFS = ROOT / "results/posthoc_v19/abstention_v19.json"
OUT = ROOT / "results/heldout_v19/diana"
N_BINS = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def lines(path: Path) -> list[str]:
    return [l.strip() for l in open(path) if l.strip()]


def shared(task: str, test: pd.DataFrame) -> pd.DataFrame:
    dirs = [FITS / f"seed{s}" for s in SEEDS]
    runs = lines(dirs[0] / "eval_runs.txt")
    classes = json.load(open(dirs[0] / "label_classes.json"))[task]
    for d in dirs[1:]:
        if lines(d / "eval_runs.txt") != runs or json.load(open(d / "label_classes.json"))[task] != classes:
            raise ValueError(f"{task}: seeds differ in held-out runs or class order")
    if set(runs) != set(test.index) or len(runs) != len(test):
        raise ValueError(f"{task}: final-fit runs differ from test_metadata")
    P = np.mean([np.load(d / "probs_last.npz")[task] for d in dirs], axis=0)
    out = pd.DataFrame(P, columns=[f"p_{c}" for c in classes])
    out.insert(0, "y_true", test.loc[runs, task].to_numpy())
    out.insert(0, "Run_accession", runs)
    return out


def with_logreg(net: pd.DataFrame, task: str) -> pd.DataFrame:
    n = net.set_index("Run_accession")
    lr = pd.read_csv(str(LR).format(task=task), sep="\t").set_index("Run_accession")
    if set(lr.index) != set(n.index):
        raise ValueError(f"{task}: network and logistic regression cover different runs")
    lr = lr.loc[n.index]
    cols = sorted({c for c in n.columns if c.startswith("p_")} | {c for c in lr.columns if c.startswith("p_")})
    avg = 0.5 * n.reindex(columns=cols, fill_value=0.0) + 0.5 * lr.reindex(columns=cols, fill_value=0.0)
    return pd.concat([n[["y_true"]], avg], axis=1).reset_index()


def reliability(top: np.ndarray, correct: np.ndarray) -> tuple[pd.DataFrame, float]:
    edges = np.linspace(0, 1, N_BINS + 1)
    b = np.clip(np.digitize(top, edges[1:-1]), 0, N_BINS - 1)
    rows = [{"bin_low": edges[i], "bin_high": edges[i + 1], "runs": int((b == i).sum()),
             "mean_confidence": float(top[b == i].mean()) if (b == i).any() else np.nan,
             "accuracy": float(correct[b == i].mean()) if (b == i).any() else np.nan} for i in range(N_BINS)]
    t = pd.DataFrame(rows)
    ok = t.runs > 0
    ece = float((t.runs[ok] / t.runs.sum() * (t.accuracy[ok] - t.mean_confidence[ok]).abs()).sum())
    return t, ece


def main() -> None:
    import argparse
    global FITS, OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fits", type=Path, default=FITS, help="final fits folder (seed<s>/ inside)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--cutoffs", type=Path, default=CUTOFFS, help="no-call cutoffs JSON (min_confidence per task)")
    a = ap.parse_args()
    FITS, OUT = a.fits, a.out
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    test = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    cut = json.load(open(a.cutoffs))["min_confidence"]
    summary, curves = [], []
    for task in TASKS:
        sh = shared(task, test)
        dia = with_logreg(sh, task) if task == "community_type" else sh
        for name, d in (("Shared", sh), ("DIANA", dia)):
            pc = [c for c in d.columns if c.startswith("p_")]
            if not np.allclose(d[pc].sum(1), 1.0, atol=1e-4):
                raise ValueError(f"{name}/{task}: rows do not sum to 1")
            d.to_csv(OUT / f"probs_{name}_{task}.tsv", sep="\t", index=False)
        scored = S.scored_heldout(test[task], set(tr[task].dropna()))
        d = dia.set_index("Run_accession").loc[scored[scored].index]
        pc = [c for c in d.columns if c.startswith("p_")]
        top = d[pc].max(axis=1).to_numpy()
        correct = (d[pc].idxmax(axis=1).str[2:] == d.y_true.astype(str)).to_numpy()
        keep = top >= cut[task]
        t, ece = reliability(top, correct)
        t.insert(0, "task", task)
        curves.append(t)
        summary.append({"task": task, "scored_runs": int(len(d)), "cutoff": cut[task],
                        "answered_share": float(keep.mean()), "accuracy_answered": float(correct[keep].mean()),
                        "accuracy_all": float(correct.mean()), "ece": ece})
        log.info("%s: %d scored runs, cutoff %.2f answers %.3f, accuracy %.3f (all %.3f), ECE %.3f", task, len(d),
                 cut[task], keep.mean(), correct[keep].mean(), correct.mean(), ece)
    pd.DataFrame(summary).to_csv(OUT / "nocall_calibration_test.tsv", sep="\t", index=False)
    pd.concat(curves).to_csv(OUT / "reliability_test.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
