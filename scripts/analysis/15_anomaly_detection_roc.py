#!/usr/bin/env python3
"""Score label-anomaly detection against planted mislabels. (R3.6)

R3.6 is the concern that "metadata validation and anomaly detection" is claimed as
DIANA's primary role but never demonstrated: no experiment flags a known mislabelled
sample, no false-flag rate is reported, no confidence threshold is established. This
produces those numbers.

The detector
------------
For a run whose metadata says class `c`, the flag score is **1 - P(c)**, the
probability the model assigns to the label on file. A model that is confident the
label is wrong scores near 1. This needs no extra training: it reuses the
probabilities `diana-test` already writes.

Scoring
-------
`plant_mislabels.py` corrupts a known fraction of labels and records which rows it
touched. Those flags are the ground truth, so detection becomes a ranking problem:
ROC-AUC and average precision over all rows, plus the operating points that matter
for use, namely the detection rate at a fixed false-flag rate.

Two cautions the numbers must be read with:

* **The corruption model sets the difficulty.** `--model uniform` relabels at
  random, which is trivially catchable and inflates the result; `empirical` and
  `mixed` sample from the 154 real merge-key mislabels, whose direction is known.
  Report the model used, and never report uniform alone.
* **The false-flag rate is what limits use.** A detector that catches 90 % of
  planted errors while firing on 30 % of clean samples is unusable for curation.
  The reported operating point is chosen by false-flag rate, not by detection rate.

    ./env/bin/python scripts/analysis/15_anomaly_detection_roc.py \\
        --predictions results/final_eval_v9/test_predictions.tsv \\
        --planted results/planted_mislabels/planted_test_mixed_r0.1.tsv \\
        --label DIANA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
# False-flag rates a curator might accept. The detection rate at these is the
# claim; the AUC is context.
TARGET_FPR = [0.01, 0.05, 0.10]


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 1:
        raise SystemExit(f"{path} parsed to one column with sep={sep!r}")
    return df


def prob_of_label(pred: pd.DataFrame, task: str, labels: pd.Series) -> np.ndarray:
    """P(stated label) per row, from diana-test's <task>_prob_<i> columns."""
    prob_cols = sorted((c for c in pred.columns if c.startswith(f"{task}_prob_")),
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    if not prob_cols:
        raise SystemExit(
            f"no {task}_prob_* columns in the predictions file. The detector needs "
            "P(stated label); re-run diana-test, which writes them.")
    P = pred[prob_cols].to_numpy()
    # column index i corresponds to encoder class i; recover the class order from
    # the (label, index) pairs diana-test also writes
    idx_col = f"{task}_true_idx"
    if idx_col in pred.columns:
        name_by_idx = (pred[[idx_col, f"{task}_true"]].dropna()
                       .drop_duplicates().set_index(idx_col)[f"{task}_true"].to_dict())
        idx_by_name = {v: int(k) for k, v in name_by_idx.items()}
    else:
        raise SystemExit(f"{idx_col} missing; cannot map class names to probability columns")
    out = np.full(len(pred), np.nan)
    for i, lab in enumerate(labels.to_numpy()):
        j = idx_by_name.get(lab)
        if j is not None and j < P.shape[1]:
            out[i] = P[i, j]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="diana-test output, with <task>_prob_* columns")
    ap.add_argument("--planted", type=Path, required=True,
                    help="plant_mislabels.py output, with <task>_planted flags")
    ap.add_argument("--label", default="DIANA")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/anomaly_detection")
    args = ap.parse_args()

    pred = read_table(args.predictions)
    plant = read_table(args.planted)
    if "Run_accession" not in pred.columns or "Run_accession" not in plant.columns:
        raise SystemExit("both files need a Run_accession column")

    rows, curves = [], {}
    for task in TASKS:
        flag_col = f"{task}_planted"
        if flag_col not in plant.columns or f"{task}_true" not in pred.columns:
            print(f"  {task}: missing {flag_col} or predictions, skipping")
            continue
        # the label the curator would see is the (possibly corrupted) one
        keep = [c for c in ("Run_accession", task, flag_col) if c in plant.columns]
        m = pred.merge(plant[keep], on="Run_accession", how="inner",
                       suffixes=("", "_planted_file"))
        stated = m[task] if task in m.columns else None
        if stated is None:
            print(f"  {task}: planted file has no {task} column, skipping")
            continue
        m = m[stated.notna() & m[flag_col].notna()]
        if m.empty:
            continue
        p_stated = prob_of_label(m, task, m[task])
        ok = ~np.isnan(p_stated)
        y = m.loc[ok, flag_col].astype(int).to_numpy()
        score = 1.0 - p_stated[ok]
        if y.sum() == 0 or y.sum() == len(y):
            print(f"  {task}: ground truth has one class only, skipping")
            continue

        auc = float(roc_auc_score(y, score))
        ap_ = float(average_precision_score(y, score))
        fpr, tpr, thr = roc_curve(y, score)
        rec = {"model": args.label, "task": task, "n_scored": int(len(y)),
               "n_planted": int(y.sum()), "roc_auc": auc, "average_precision": ap_}
        for t in TARGET_FPR:
            i = int(np.searchsorted(fpr, t, side="right") - 1)
            i = max(i, 0)
            rec[f"tpr_at_fpr{int(t*100)}"] = float(tpr[i])
            rec[f"threshold_at_fpr{int(t*100)}"] = float(thr[i])
        rows.append(rec)
        curves[task] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        print(f"  {task}: n={len(y)} planted={y.sum()} AUC={auc:.3f} AP={ap_:.3f}")

    if not rows:
        raise SystemExit("nothing scored; check that the two files share runs and tasks")

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / f"roc_{args.label}.tsv", sep="\t", index=False)
    json.dump(curves, open(args.output / f"curves_{args.label}.json", "w"))

    lines = [f"Label-anomaly detection — {args.label}  (R3.6)", "",
             f"Planted mislabels: {args.planted.name}",
             "Flag score = 1 - P(label on file). Ground truth = the planted flags.",
             "",
             f"{'task':<16}{'n':>6}{'planted':>9}{'AUC':>7}{'AP':>7}"
             f"{'TPR@1%':>9}{'TPR@5%':>9}{'TPR@10%':>9}"]
    for r in rows:
        lines.append(f"{r['task']:<16}{r['n_scored']:>6}{r['n_planted']:>9}"
                     f"{r['roc_auc']:>7.3f}{r['average_precision']:>7.3f}"
                     f"{r['tpr_at_fpr1']:>9.3f}{r['tpr_at_fpr5']:>9.3f}"
                     f"{r['tpr_at_fpr10']:>9.3f}")
    lines += ["",
              "TPR@x% is the fraction of planted mislabels caught while firing on x %",
              "of correctly-labelled runs. That false-flag rate is what limits use in",
              "curation, so quote the operating point, not the AUC.",
              f"Corruption model is set by plant_mislabels.py; uniform relabelling is",
              "easier to catch than the empirical confusions and must not be reported",
              "on its own."]
    report = "\n".join(lines)
    print("\n" + report)
    (args.output / f"summary_{args.label}.txt").write_text(report + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
