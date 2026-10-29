#!/usr/bin/env python3
"""Dev-fold probabilities with the class prior added back (v19 CAL-B2; PROTOCOLS.md v19 amendment 2026-10-26).

Per dev fold k, pi = class shares of the labelled training-fold runs (folds != k). Network (results/devfold_probs_v19/
probs_Shared_<task>.tsv): P' proportional to P * pi^0.6998; class-balanced logistic regression (results/baselines_v19/<task>/
oof_probs_LogisticRegression_Bal.tsv): P' proportional to P * pi; DIANA community type = 0.5 P'_network + 0.5 P'_LR, other
fields = P'_network. Writes results/prior_v19/probs_{DIANAprior,LogRegprior}_<task>.tsv (same layout as the inputs: run,
fold, y_true, p_*) and results/prior_v19/calibration.tsv: per field and model, original vs corrected, the expected calibration
error (10 equal-width bins of the confidence of the predicted label; label = argmax of the original P) and the mean negative
log-likelihood of the true label, with 95 % BioProject intervals (2,000 resamples), on the scored dev-fold runs; and macro-F1
if labels were taken from argmax P' (reported, not deciding).

    ./env/bin/python scripts/analysis/83_prior_correction_v19.py
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
SPLIT = ROOT / "data/splits_v19"
NET = ROOT / "results/devfold_probs_v19/probs_Shared_{task}.tsv"
LR = ROOT / "results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv"
OUT = ROOT / "results/prior_v19"
TAU = json.load(open(ROOT / "configs/v19_final/multitask.json"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def find_tau(cfg) -> float:
    if isinstance(cfg, dict):
        if "logit_adjust_tau" in cfg:
            return float(cfg["logit_adjust_tau"])
        for v in cfg.values():
            t = find_tau(v)
            if t is not None:
                return t
    return None


def read(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(path, sep="\t").set_index("Run_accession")
    return d[[c for c in d.columns if c.startswith("p_")]].astype(float), d[["fold", "y_true"]]


def add_prior(P: pd.DataFrame, folds: pd.Series, labels: pd.Series, all_folds: pd.Series, tau: float) -> pd.DataFrame:
    out = P.copy()
    classes = [c[2:] for c in P.columns]
    for k in sorted(folds.unique()):
        fit = labels[(all_folds != k) & labels.notna()].astype(str)
        pi = fit.value_counts(normalize=True).reindex(classes).fillna(0.0).to_numpy()
        rows = folds.index[folds == k]
        X = P.loc[rows].to_numpy() * pi[None, :] ** tau
        s = X.sum(axis=1, keepdims=True)
        out.loc[rows] = np.where(s > 0, X / np.where(s > 0, s, 1), P.loc[rows].to_numpy())
    return out


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def with_interval(fn, groups: np.ndarray, key: str) -> tuple[float, float, float]:
    point = fn(np.arange(len(groups)))
    lo, hi = S.interval([fn(i) for i in S.project_draws(groups, key)])
    return point, lo, hi


def main() -> None:
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", action="store_true", help="DIANA without logistic regression (PROTOCOLS.md v19 amendment 2026-10-26)")
    net_only = ap.parse_args().net
    if net_only:
        OUT = ROOT / "results/prior_v19net"
    tau = find_tau(TAU)
    if tau is None:
        raise ValueError("logit_adjust_tau not found in configs/v19_final/multitask.json")
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    all_folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in TASKS:
        net, meta = read(Path(str(NET).format(task=task)))
        lr, lr_meta = read(Path(str(LR).format(task=task)))
        lr = lr.reindex(net.index).reindex(columns=net.columns).fillna(0.0)
        if not lr_meta.reindex(net.index).fold.equals(meta.fold):
            raise ValueError(f"{task}: network and logistic-regression folds disagree")
        net_p = add_prior(net, meta.fold, tr[task], all_folds, tau)
        lr_p = add_prior(lr, meta.fold, tr[task], all_folds, 1.0)
        orig = {"DIANA": 0.5 * net + 0.5 * lr if task == "community_type" and not net_only else net, "LogReg": lr}
        corr = {"DIANA": 0.5 * net_p + 0.5 * lr_p if task == "community_type" and not net_only else net_p, "LogReg": lr_p}
        for m in corr:
            pd.concat([meta, corr[m]], axis=1).reset_index().to_csv(OUT / f"probs_{m}prior_{task}.tsv", sep="\t", index=False)
        sc = S.scored_devfold(tr[task], all_folds)
        runs = sc[sc].index.intersection(net.index)
        y = tr.loc[runs, task].astype(str).to_numpy()
        g = tr.loc[runs, "archive_project"].astype(str).to_numpy()
        el = S.eligible_classes(SPLIT / "class_eligibility.tsv", task)
        classes = [c[2:] for c in net.columns]
        yi = np.array([classes.index(c) for c in y])
        for m in corr:
            O, C = orig[m].loc[runs].to_numpy(), corr[m].loc[runs].to_numpy()
            pred = O.argmax(axis=1)
            correct = (pred == yi).astype(float)
            for name, P in (("original", O), ("prior added", C)):
                conf = P[np.arange(len(P)), pred]
                ptrue = np.clip(P[np.arange(len(P)), yi], 1e-12, 1.0)
                e = with_interval(lambda i: ece(conf[i], correct[i]), g, f"ece|{task}")
                n = with_interval(lambda i: float(-np.log(ptrue[i]).mean()), g, f"nll|{task}")
                f1 = S.f1_macro_eligible(y, np.array(classes)[P.argmax(axis=1)], el & set(y))
                rows.append({"task": task, "model": m, "probabilities": name, "ece": e[0], "ece_ci_low": e[1], "ece_ci_high": e[2],
                             "nll": n[0], "nll_ci_low": n[1], "nll_ci_high": n[2], "macro_f1_if_argmax": f1})
            log.info("%s %s: ECE %.3f -> %.3f", task, m, rows[-2]["ece"], rows[-1]["ece"])
    pd.DataFrame(rows).to_csv(OUT / "calibration.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
