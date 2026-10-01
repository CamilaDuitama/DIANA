#!/usr/bin/env python3
"""Classification and detection by training-support regime, DIANA against the baselines.

Classification is already stratified in `results/final_eval_v9/stratified_by_shot.tsv`:
`f1_macro_eligible` computed over only the eligible classes whose *training* support
falls in each regime (few-shot < 20 runs, medium-shot 20-100, many-shot > 100; the
ImageNet-LT / OLTR split points).

Detection is not, so it is computed here on the same runs and the same regimes. The
detector is the one in §2: flag a run when `1 - P(stated label)` exceeds the task's
threshold, the threshold set once per task on the whole held-out set at a 5 % false-flag
budget, exactly as reported. A run is assigned to the regime of its *true* class, so the
question each cell answers is "of the planted mislabels on runs that really belong to a
class this rare, what share does the detector catch". `recall@5%` is that share, with the
number of planted errors in the cell alongside, because some cells rest on very few.

Models without `predict_proba` (LinearSVM) have no detector and are absent from the
detection half; k-NN has one but its tuned n_neighbors=1 makes every probability 0 or 1.

    ./env/bin/python scripts/analysis/23_regime_stratified_table.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The classification half's reproducible producer (the rule reverse-engineered on
# 2026-09-16), so another arm's DIANA rows can be built instead of read from the
# writer-less stratified_by_shot.tsv.
sys.path.insert(0, str(Path(__file__).resolve().parent))
_regime = import_module("30_heldout_regime_table")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL = PROJECT_ROOT / "results/final_eval_v9"
SPLITS = PROJECT_ROOT / "data/splits_v9"
PLANTED = PROJECT_ROOT / "results/planted_mislabels_v9/planted_test_mixed_r0.1.tsv"
BASE_PROB = PROJECT_ROOT / "results/baseline_predictions_v9"
OUT = PROJECT_ROOT / "results/final_eval_v9/regime_table.tsv"

TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot (<20)", 0, 20), ("medium-shot (20-100)", 20, 100),
           ("many-shot (>100)", 100, np.inf)]
FALSE_FLAG_BUDGET = 0.05


def support() -> pd.DataFrame:
    """Training runs per class, and whether the class is eligible (>= 2 BioProjects)."""
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t")
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    rows = []
    for task in TASKS:
        n = train[task].value_counts()
        for cls, cnt in n.items():
            rows.append({"task": task, "class": cls, "n_train": int(cnt)})
    sup = pd.DataFrame(rows)
    return sup.merge(elig[["target", "class", "evaluable"]].rename(columns={"target": "task"}),
                     on=["task", "class"], how="left")


def regime_of(n: float) -> str | None:
    for name, lo, hi in REGIMES:
        if lo < n <= hi if lo else 0 < n <= hi:
            return name
    return None


def p_stated_diana(task: str, stated: pd.Series, pred: pd.DataFrame,
                   classes: list | None = None) -> np.ndarray:
    """P(stated label) from diana-test's probability columns.

    With `classes` (the model's label_encoders.json) every training class maps to its
    column. Without it the map is recovered from the true labels seen on held-out, which
    silently drops any planted row whose stated label is a class with no true held-out
    run; see 15_anomaly_detection_roc.py for the counts that cost.
    """
    cols = sorted((c for c in pred.columns if c.startswith(f"{task}_prob_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    P = pred[cols].to_numpy()
    if classes is not None:
        if len(classes) != P.shape[1]:
            raise SystemExit(f"{task}: {len(classes)} encoder classes, {P.shape[1]} probability columns")
        idx = {c: i for i, c in enumerate(classes)}
    else:
        name_by_idx = (pred[[f"{task}_true_idx", f"{task}_true"]].dropna().drop_duplicates()
                       .set_index(f"{task}_true_idx")[f"{task}_true"].to_dict())
        idx = {v: int(k) for k, v in name_by_idx.items()}
    out = np.full(len(pred), np.nan)
    for i, lab in enumerate(stated.to_numpy()):
        j = idx.get(lab)
        if j is not None and j < P.shape[1]:
            out[i] = P[i, j]
    return out


def detection_by_regime(task: str, p_stated: np.ndarray, planted: np.ndarray,
                        true_regime: np.ndarray) -> list[dict]:
    """Recall per regime at one task-level threshold set by the false-flag budget."""
    ok = ~np.isnan(p_stated)
    score = 1.0 - p_stated[ok]
    y = planted[ok].astype(int)
    reg = true_regime[ok]
    clean = score[y == 0]
    if len(clean) == 0 or y.sum() == 0:
        return []
    # the threshold fires on FALSE_FLAG_BUDGET of the correctly labelled runs
    thr = float(np.quantile(clean, 1.0 - FALSE_FLAG_BUDGET))
    rows = []
    for name, _, _ in REGIMES:
        sel = (reg == name) & (y == 1)
        n = int(sel.sum())
        rows.append({"task": task, "stratum": name, "n_planted": n,
                     "recall_at_5pct": float((score[sel] > thr).mean()) if n else np.nan,
                     "threshold": thr})
    return rows


def main() -> int:
    global SPLITS, PLANTED, BASE_PROB
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # X4 (2026-09-25): the same table for another set of final models, without touching
    # the read-1 outputs. With --encoders-pattern the detection rows are like-for-like
    # with the baselines; without it the historical (row-dropping) mapping is kept.
    ap = argparse.ArgumentParser(description="Table 2, classification and detection by regime")
    ap.add_argument("--eval-dir", type=Path, default=EVAL)
    ap.add_argument("--single-pattern", default="heldout_single_{task}")
    ap.add_argument("--encoders-pattern", default=None,
                    help="label_encoders.json of the single-task model, with {task}")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--splits", type=Path, default=SPLITS, help="split directory; v13 = data/splits_v13")
    ap.add_argument("--planted", type=Path, default=PLANTED)
    ap.add_argument("--baseline-dir", type=Path, default=BASE_PROB, help="baseline held-out probabilities directory")
    a = ap.parse_args()
    SPLITS, PLANTED, BASE_PROB = a.splits, a.planted, a.baseline_dir
    sup = support()
    sup["regime"] = sup.n_train.map(regime_of)
    plant = pd.read_csv(PLANTED, sep="\t")

    det = []
    for task in TASKS:
        reg_by_class = (sup[sup.task == task].set_index("class")["regime"].to_dict())
        # the true label is the one in test_metadata; the planted file overwrites it
        truth = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t")[["Run_accession", task]]
        truth = truth.rename(columns={task: "true_label"})
        keep = plant[["Run_accession", task, f"{task}_planted"]]

        # DIANA: the single-task net, which is what §2 reports
        dp = pd.read_csv(a.eval_dir / a.single_pattern.format(task=task) / "test_predictions.tsv",
                         sep="\t")
        classes = (json.load(open(a.encoders_pattern.format(task=task)))[task]["classes"]
                   if a.encoders_pattern else None)
        m = dp.merge(keep, on="Run_accession").merge(truth, on="Run_accession")
        m = m[m[task].notna() & m[f"{task}_planted"].notna() & m.true_label.notna()]
        if m.empty:
            continue
        tr = m.true_label.map(reg_by_class).to_numpy()
        for r in detection_by_regime(task, p_stated_diana(task, m[task], m, classes),
                                     m[f"{task}_planted"].to_numpy(), tr):
            det.append({**r, "model": "DIANA single"})

        # baselines: wide p_<class> columns, one block per model
        bp = pd.read_csv(BASE_PROB / f"heldout_probabilities_{task}.tsv", sep="\t")
        for model, blk in bp.groupby("model"):
            if model == "MajorityClass":
                continue
            mb = blk.merge(keep, on="Run_accession").merge(truth, on="Run_accession")
            mb = mb[mb[task].notna() & mb[f"{task}_planted"].notna() & mb.true_label.notna()]
            if mb.empty:
                continue
            ps = np.full(len(mb), np.nan)
            for i, lab in enumerate(mb[task].to_numpy()):
                col = f"p_{lab}"
                if col in mb.columns:
                    ps[i] = mb.iloc[i][col]
            trb = mb.true_label.map(reg_by_class).to_numpy()
            for r in detection_by_regime(task, ps, mb[f"{task}_planted"].to_numpy(), trb):
                det.append({**r, "model": model})

    d = pd.DataFrame(det)
    clf = pd.read_csv(EVAL / "stratified_by_shot.tsv", sep="\t")
    if a.eval_dir != EVAL or a.single_pattern != "heldout_single_{task}":
        # Another arm: its DIANA rows come from the reproducible rule, the baseline rows
        # (same baselines, same held-out runs) are unchanged.
        arm = _regime.arm_table(a.eval_dir, _regime.training_support(), a.single_pattern)
        arm["model"] = "DIANA single"
        clf = pd.concat([clf[clf.model != "DIANA single"], arm[clf.columns]], ignore_index=True)
    clf = clf.rename(columns={"f1": "f1_macro_eligible"})
    out = clf.merge(d[["task", "stratum", "model", "n_planted", "recall_at_5pct"]],
                    on=["task", "stratum", "model"], how="outer")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)
    logger.info("wrote %s (%d rows)", a.out, len(out))

    for task in TASKS:
        print(f"\n=== {task} ===")
        sub = out[out.task == task]
        for _, name in enumerate(r[0] for r in REGIMES):
            s = sub[sub.stratum == name]
            if s.empty:
                continue
            nc = s.n_classes.dropna()
            nr = s.n_runs.dropna()
            print(f"  {name}  classes={int(nc.iloc[0]) if len(nc) else 0} "
                  f"runs={int(nr.iloc[0]) if len(nr) else 0} "
                  f"planted={int(s.n_planted.dropna().iloc[0]) if s.n_planted.notna().any() else 0}")
            for _, r in s.sort_values("model").iterrows():
                f1 = "  .  " if pd.isna(r.f1_macro_eligible) else f"{r.f1_macro_eligible:.3f}"
                rc = "  .  " if pd.isna(r.recall_at_5pct) else f"{r.recall_at_5pct:.3f}"
                print(f"     {str(r.model):26s} F1={f1}  recall@5%={rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
