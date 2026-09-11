#!/usr/bin/env python3
"""Fit one temperature per task on out-of-fold probabilities. (C1, R3.14)

Referee 3 (**R3.14**) notes there is no calibration analysis. The anomaly detector
also thresholds a probability, so an uncalibrated probability makes its operating
point meaningless.

Why not `01_calibrate_model.py`
------------------------------
That script fits the temperature on a random 10 % of train, stratified by
`sample_type`. v9 dropped that task, so it would raise; and a random split shares
BioProjects with the fitting data, so the model looks better calibrated than it is on
unseen studies. Same leak class as **R3.4**.

What this does instead
----------------------
`run_calibration_refits_v9.sbatch` refits the final hyperparameters on 4 of the 5
BioProject-grouped dev folds and predicts the held-back fold, 4 tasks x 5 folds. The
five held-back folds cover all 2,716 training runs and none is predicted by a model
that saw its study, so pooling them gives honest out-of-fold probabilities.

`diana-test` writes probabilities, not logits. Softmax is invariant to an additive
shift in the logits, so `log p` is a valid stand-in: softmax(log p / T) is exactly
the temperature-scaled distribution. T is fitted by minimising negative log
likelihood, which is the standard objective for temperature scaling and is better
behaved than minimising ECE directly, ECE being piecewise constant in T.

Held-out is not read here.

    ./env/bin/python scripts/calibration/02_fit_temperature_v9.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
EPS = 1e-12


def load_oof(task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled out-of-fold (log-probabilities, true class index) for one task.

    Each fold's label encoder is fitted on that fold's own training runs, so the
    folds do not share a class space: `feature` has 8, 9 or 10 outputs depending on
    the fold. Columns are therefore aligned by class NAME into one global space, and
    a class a fold has no output for gets probability 0, which is what that model
    actually assigns it. The true class is read from the name column for the same
    reason: `<task>_true_idx` is fold-local and not comparable across folds.
    """
    md = pd.read_csv(PROJECT_ROOT / "data/splits_v9/train_metadata.tsv",
                     sep="\t", low_memory=False)
    glob_cls = sorted(md[task].dropna().astype(str).unique())
    gidx = {c: i for i, c in enumerate(glob_cls)}

    logits, y, fold_of = [], [], []
    for f in range(5):
        base = PROJECT_ROOT / f"results/calibration_v9"
        enc = json.loads((base / f"fit_{task}_fold{f}/label_encoders.json").read_text())
        fold_cls = enc[task]["classes"] if isinstance(enc[task], dict) else enc[task]
        d = pd.read_csv(base / f"oof_{task}_fold{f}/test_predictions.tsv", sep="\t")
        cols = sorted((c for c in d.columns if c.startswith(f"{task}_prob_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        if len(cols) != len(fold_cls):
            raise SystemExit(f"{task} fold {f}: {len(cols)} prob columns but "
                             f"{len(fold_cls)} encoder classes")
        d = d[d[f"{task}_true"].notna()]
        d = d[d[f"{task}_true"].astype(str).isin(gidx)]
        if d.empty:
            continue
        P = np.zeros((len(d), len(glob_cls)), dtype=np.float64)
        src = d[cols].to_numpy(dtype=np.float64)
        for j, c in enumerate(fold_cls):
            if str(c) in gidx:
                P[:, gidx[str(c)]] = src[:, j]
        logits.append(np.log(np.clip(P, EPS, None)))
        y.append(d[f"{task}_true"].astype(str).map(gidx).to_numpy(dtype=int))
        fold_of.append(np.full(len(d), f, dtype=int))
    return np.vstack(logits), np.concatenate(y), np.concatenate(fold_of)


def nll(T: float, z: np.ndarray, y: np.ndarray) -> float:
    s = z / T
    s = s - s.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(s).sum(axis=1))
    return float(-(s[np.arange(len(y)), y] - logZ).mean())


def ece(z: np.ndarray, y: np.ndarray, T: float, n_bins: int = 15) -> tuple[float, list]:
    s = z / T
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s); p /= p.sum(axis=1, keepdims=True)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc, cf, w = correct[m].mean(), conf[m].mean(), m.mean()
        total += abs(acc - cf) * w
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "n": int(m.sum()),
                     "confidence": float(cf), "accuracy": float(acc)})
    return float(total), rows


def fit_T(z, y, grid) -> float:
    """T minimising ECE.

    NLL is the textbook objective but it fails on these models. Three of the four
    are UNDER-confident, because training used label smoothing and logit adjustment
    which deliberately soften the outputs, yet NLL pushed T *up* on all four and made
    ECE worse in every case (community_type 0.414 -> 0.476, feature hit the T=20
    bound). The cause is a tail of rare-class runs where the true class gets
    essentially zero probability: log of that is enormous, so NLL is dominated by it
    and flattens the whole distribution to limit the worst case. That reduces NLL and
    is not calibration. ECE is what R3.14 asks about, so ECE is what is minimised,
    by grid search since it is piecewise constant in T.
    """
    return float(grid[int(np.argmin([ece(z, y, T)[0] for T in grid]))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/calibration_v9")
    args = ap.parse_args()
    grid = np.concatenate([np.linspace(0.10, 1.0, 46), np.linspace(1.05, 8.0, 140)])

    out = {}
    print(f"{'task':<16}{'n oof':>7}{'T':>7}{'ECE raw':>9}{'ECE fitted':>11}"
          f"{'ECE LOFO':>10}{'mean conf':>11}{'accuracy':>9}")
    for task in TASKS:
        z, y, fold = load_oof(task)
        T = fit_T(z, y, grid)
        e_raw = ece(z, y, 1.0)[0]
        e_fit = ece(z, y, T)[0]

        # Leave-one-fold-out: fit T on four folds, measure ECE on the fifth. Fitting
        # and measuring on the same rows would understate ECE, even for one parameter.
        lofo, per_fold_T = [], []
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            if te.sum() < 20 or len(np.unique(y[tr])) < 2:
                continue
            Tf = fit_T(z[tr], y[tr], grid)
            per_fold_T.append(Tf)
            lofo.append(ece(z[te], y[te], Tf)[0])
        e_lofo = float(np.mean(lofo)) if lofo else float("nan")

        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        conf, acc = float(p.max(1).mean()), float((p.argmax(1) == y).mean())
        out[task] = {"temperature": T, "n_oof": int(len(y)),
                     "ece_raw": e_raw, "ece_fitted": e_fit, "ece_lofo": e_lofo,
                     "temperature_per_fold": per_fold_T,
                     "nll_raw": nll(1.0, z, y), "nll_fitted": nll(T, z, y),
                     "mean_confidence_raw": conf, "accuracy": acc,
                     "direction": "under-confident" if conf < acc else "over-confident",
                     "reliability_raw": ece(z, y, 1.0)[1],
                     "reliability_fitted": ece(z, y, T)[1]}
        print(f"{task:<16}{len(y):>7}{T:>7.2f}{e_raw:>9.4f}{e_fit:>11.4f}"
              f"{e_lofo:>10.4f}{conf:>11.3f}{acc:>9.3f}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "temperatures.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.output / 'temperatures.json'}")
    print("\nT < 1 sharpens an under-confident model; T > 1 softens an over-confident one.")
    print("ECE LOFO is the honest figure: T fitted on four folds, measured on the fifth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
