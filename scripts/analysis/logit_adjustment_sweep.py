#!/usr/bin/env python
"""
Post-hoc logit adjustment for the DIANA classification heads.

Why
---
The heads appear to have been trained in a class-balanced regime, so their
softmax approximates a *balanced* posterior. Taking a plain argmax on a
naturally imbalanced test split then systematically favours rare classes: 158 of
235 `Homo sapiens` test runs are predicted `Papio hamadryas` (5 training runs),
and the final-layer weight norms are inversely related to training frequency
(`Homo sapiens`, n=1378, has one of the smallest norms).

Adding tau * log(prior) back to the log-probabilities restores the natural
posterior (Menon et al., Long-tail learning via logit adjustment, ICLR 2021).
tau = 1 exactly inverts training under a balanced loss.

IMPORTANT
---------
tau is a hyperparameter and MUST be selected on the validation split, then
applied once to test. Selecting it on test is precisely the leakage the referees
are looking for. Pass --select-on to do this correctly; --sweep is diagnostic
only and prints a test-set curve for inspection.

Usage
-----
    ./env/bin/python scripts/analysis/logit_adjustment_sweep.py --sweep
    ./env/bin/python scripts/analysis/logit_adjustment_sweep.py \
        --select-on results/validation_predictions_bioproject_v7/predictions.tsv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "sample_host", "material"]
NULLS = {"nan", "none", "null", ""}
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def canon(v) -> str:
    """Canonical label string.

    The encoder stores the absent-label class as JSON null -> str(None) ==
    'None', while the predictions TSV and the metadata carry it as NaN ->
    'nan'. Left unmapped, the two never compare equal, the absent-label class
    scores F1 = 0 at every tau, and macro-F1 is silently wrong.
    """
    s = str(v).strip()
    return "__none__" if s.lower() in NULLS else s


def load(pred_path: Path, encoders: dict, train_meta: pd.DataFrame):
    d = pd.read_csv(pred_path, sep="\t")
    out = {}
    for t in TASKS:
        classes = [canon(c) for c in encoders[t]["classes"]]
        cols = [f"{t}_prob_{i}" for i in range(len(classes))]
        if not set(cols) <= set(d.columns):
            continue
        counts = train_meta[t].map(canon).value_counts()
        prior = np.array([max(int(counts.get(c, 0)), 1) for c in classes], float)
        out[t] = dict(classes=np.array(classes),
                      P=d[cols].values,
                      y=d[f"{t}_true"].map(canon).values,
                      prior=prior / prior.sum())
    return out


def score(task: dict, tau: float) -> dict:
    """Metrics at a given tau. Null-label runs are reported separately."""
    adj = np.log(task["P"] + 1e-12) + tau * np.log(task["prior"])
    pred = task["classes"][adj.argmax(1)]
    y = task["y"]
    real = (pd.Series(y) != "__none__").values
    seen = sorted(set(y))
    seen_real = [c for c in seen if c != "__none__"]
    return dict(tau=tau,
                accuracy_all=accuracy_score(y, pred),
                accuracy_real=accuracy_score(y[real], pred[real]),
                f1_macro_seen=f1_score(y, pred, average="macro",
                                       labels=seen, zero_division=0),
                f1_macro_seen_real=f1_score(y, pred, average="macro",
                                            labels=seen_real, zero_division=0),
                n_real=int(real.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-pred", type=Path,
                    default=ROOT / "results/test_evaluation_bioproject_v7/test_predictions.tsv")
    ap.add_argument("--encoders", type=Path,
                    default=ROOT / "results/training_bioproject_v7/label_encoders.json")
    ap.add_argument("--train-meta", type=Path,
                    default=ROOT / "data/splits_v7/train_metadata.tsv")
    ap.add_argument("--select-on", type=Path, default=None,
                    help="validation predictions TSV; tau is chosen here, then "
                         "applied once to --test-pred")
    ap.add_argument("--sweep", action="store_true",
                    help="diagnostic only: print the test-set tau curve")
    ap.add_argument("--out", type=Path, default=ROOT / "results/logit_adjustment")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    enc = json.load(open(args.encoders))
    train_meta = pd.read_csv(args.train_meta, sep="\t", low_memory=False)
    test = load(args.test_pred, enc, train_meta)

    rows = []
    if args.select_on:
        val = load(args.select_on, enc, train_meta)
        for t in test:
            if t not in val:
                continue
            curve = [score(val[t], x) for x in TAUS]
            best = max(curve, key=lambda r: r["f1_macro_seen"])["tau"]
            r = score(test[t], best)
            r.update(task=t, tau_selected_on="validation", tau=best)
            rows.append(r)
    if args.sweep or not args.select_on:
        for t in test:
            for x in TAUS:
                r = score(test[t], x)
                r.update(task=t, tau_selected_on="NONE - diagnostic sweep on test")
                rows.append(r)

    df = pd.DataFrame(rows)[["task", "tau", "accuracy_all", "accuracy_real",
                             "f1_macro_seen", "f1_macro_seen_real", "n_real",
                             "tau_selected_on"]]
    df.to_csv(args.out / "logit_adjustment.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
