#!/usr/bin/env python3
"""v19 post-hoc steps on DIANA's dev-fold out-of-fold probabilities: temperature scaling
(confidence correction) and the "no call" cutoffs.

Temperature: one T per task. Cross-fitted: for each dev fold k, T minimises the negative
log-likelihood of the true labels over the scored runs of the other four folds and is
applied to fold k (P_T proportional to P^(1/T), i.e. softmax of the log-probabilities / T,
which is exact for a softmax output and is applied to the five-seed ensemble). Writes the
cross-fitted table probs_DIANAcal_<task>.tsv for the paired flagging test (adopted for a
task only if its choosing score improves with the interval above zero; the argmax, hence
macro-F1, is unchanged), and the all-folds T per task for the test set.

No-call cutoff (rule of 2026-10-05, unchanged): the smallest tau on the 0 to 0.9 grid,
step 0.05, whose accuracy over the runs with top probability >= tau is at least 0.05 above
the accuracy with no abstention, keeping coverage >= 0.5; 0 if none. Computed on the
scored dev runs from the probabilities DIANA will use (calibrated where adopted: run with
--calibrated-tasks after the paired test).

Fixed-coverage cutoff (selective classification; Camila 2026-10-23, replaces the rule above for the
paper and diana-predict): per task, the (1 - coverage) quantile of the top probability over the
scored dev runs, i.e. the largest cutoff at which at least `coverage` of them are answered.
Writes abstention_v19_coverage<pct>.json; abstention_v19.json is kept as record.

    ./env/bin/python scripts/analysis/46_posthoc_v19.py temperature
    ./env/bin/python scripts/analysis/46_posthoc_v19.py cutoffs --calibrated-tasks feature,material
    ./env/bin/python scripts/analysis/46_posthoc_v19.py coverage-cutoffs --coverage 0.8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
SPLIT = ROOT / "data/splits_v19"
PROBS = ROOT / "results/devfold_probs_v19"
OUT = ROOT / "results/posthoc_v19"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def scale(P: np.ndarray, T: float) -> np.ndarray:
    logp = np.log(np.clip(P, 1e-12, None)) / T
    logp[P <= 0] = -np.inf                      # classes the model could not predict stay at 0
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def fit_T(P: np.ndarray, y_idx: np.ndarray) -> float:
    def nll(logT):
        Q = scale(P, float(np.exp(logT)))
        return -np.mean(np.log(np.clip(Q[np.arange(len(y_idx)), y_idx], 1e-12, None)))
    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def load(task: str):
    d = pd.read_csv(PROBS / f"probs_DIANA_{task}.tsv", sep="\t").set_index("Run_accession")
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    scored = S.scored_devfold(tr[task], folds)
    pc = [c for c in d.columns if c.startswith("p_")]
    return d, pc, scored.reindex(d.index).fillna(False).to_numpy(), tr


def temperature() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final = {}
    for task in TASKS:
        d, pc, sc, _ = load(task)
        classes = [c[2:] for c in pc]
        P = d[pc].to_numpy()
        y = d.y_true.astype(str).to_numpy()
        y_idx = np.array([classes.index(v) if v in classes else -1 for v in y])
        ok = sc & (y_idx >= 0)
        Q = P.copy()
        Ts = {}
        for k in sorted(d.fold.unique()):
            fit = ok & (d.fold.to_numpy() != k)
            Ts[int(k)] = fit_T(P[fit], y_idx[fit])
            rows = d.fold.to_numpy() == k
            Q[rows] = scale(P[rows], Ts[int(k)])
        out = d.copy()
        out[pc] = Q
        out.reset_index().to_csv(PROBS / f"probs_DIANAcal_{task}.tsv", sep="\t", index=False)
        final[task] = {"T_all_folds": fit_T(P[ok], y_idx[ok]), "T_cross_fitted": Ts}
        log.info("%s: T per held-back fold %s, all folds %.3f", task,
                 {k: round(v, 3) for k, v in Ts.items()}, final[task]["T_all_folds"])
    json.dump(final, open(OUT / "temperatures.json", "w"), indent=1)


def cutoffs(calibrated: set[str]) -> None:
    rule = ("smallest tau on the 0 to 0.9 grid (step 0.05) whose retained accuracy is at least 0.05 above "
            "the no-abstention accuracy; 0 if none reaches that with coverage >= 0.5")
    out = {"rule": rule, "min_confidence": {}, "curves": {}}
    T = json.load(open(OUT / "temperatures.json"))
    for task in TASKS:
        d, pc, sc, _ = load(task)
        P = d[pc].to_numpy()
        if task in calibrated:
            P = scale(P, T[task]["T_all_folds"])
        y = d.y_true.astype(str).to_numpy()[sc]
        top = P[sc].max(axis=1)
        pred = np.array([pc[i][2:] for i in P[sc].argmax(axis=1)])
        base = float((pred == y).mean())
        chosen, curve = 0.0, []
        for tau in np.round(np.arange(0, 0.901, 0.05), 2):
            keep = top >= tau
            cov = float(keep.mean())
            acc = float((pred[keep] == y[keep]).mean()) if keep.any() else np.nan
            curve.append({"tau": float(tau), "coverage": cov, "accuracy": acc})
            if chosen == 0.0 and tau > 0 and cov >= 0.5 and acc >= base + 0.05:
                chosen = float(tau)
        out["min_confidence"][task] = chosen
        out["curves"][task] = curve
        log.info("%s: no-abstention accuracy %.3f, cutoff %.2f", task, base, chosen)
    out["calibrated_tasks"] = sorted(calibrated)
    json.dump(out, open(OUT / "abstention_v19.json", "w"), indent=1)


def coverage_cutoffs(coverage: float, calibrated: set[str]) -> None:
    """Per task, the largest cutoff at which at least `coverage` of the scored dev runs are answered."""
    out = {"rule": f"fixed coverage: per task, the largest top-probability cutoff at which at least {coverage:.0%} of the scored "
                   "dev-fold runs (out-of-fold predictions) are answered, i.e. the (1 - coverage) quantile of the top probability",
           "coverage": coverage, "min_confidence": {}, "dev": {}}
    T = json.load(open(OUT / "temperatures.json"))
    for task in TASKS:
        d, pc, sc, _ = load(task)
        P = d[pc].to_numpy()
        if task in calibrated:
            P = scale(P, T[task]["T_all_folds"])
        y = d.y_true.astype(str).to_numpy()[sc]
        top = P[sc].max(axis=1)
        pred = np.array([pc[i][2:] for i in P[sc].argmax(axis=1)])
        cut = float(np.sort(top)[int(np.floor((1 - coverage) * len(top)))])
        keep = top >= cut
        if keep.mean() < coverage:
            raise ValueError(f"{task}: cutoff {cut} answers {keep.mean():.3f} < {coverage}")
        out["min_confidence"][task] = cut
        out["dev"][task] = {"runs": int(len(top)), "answered": float(keep.mean()), "accuracy_answered": float((pred[keep] == y[keep]).mean()),
                            "accuracy_all": float((pred == y).mean())}
        log.info("%s: cutoff %.4f answers %.3f of %d dev runs, accuracy %.3f (all %.3f)", task, cut, keep.mean(), len(top),
                 out["dev"][task]["accuracy_answered"], out["dev"][task]["accuracy_all"])
    out["calibrated_tasks"] = sorted(calibrated)
    json.dump(out, open(OUT / f"abstention_v19_coverage{round(coverage * 100)}.json", "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["temperature", "cutoffs", "coverage-cutoffs"])
    ap.add_argument("--calibrated-tasks", default="")
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--net", action="store_true", help="DIANA without logistic regression (PROTOCOLS.md v19 amendment 2026-10-26)")
    a = ap.parse_args()
    global PROBS, OUT
    if a.net:
        PROBS, OUT = ROOT / "results/devfold_probs_v19net", ROOT / "results/posthoc_v19net"
    if a.step == "temperature":
        temperature()
    elif a.step == "coverage-cutoffs":
        coverage_cutoffs(a.coverage, {t for t in a.calibrated_tasks.split(",") if t})
    else:
        cutoffs({t for t in a.calibrated_tasks.split(",") if t})


if __name__ == "__main__":
    main()
