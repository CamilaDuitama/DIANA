#!/usr/bin/env python3
"""Every model's held-out `f1_macro_eligible` with its BioProject-level interval (R1.10).

`results/final_eval_v9/heldout_all_models.tsv` was committed on 2026-09-11 without a
producer. This rebuilds the same six columns (task, model, f1, lo, hi, bal) from the
artefacts that carry the numbers: `diana-test`'s `test_metrics.json` for the DIANA arms
and `09_baselines_v9.py`'s `summary.csv` (split == test) for the baselines. No
bootstrap is re-run here, so the intervals are the ones those artefacts hold.

    ./env/bin/python scripts/evaluation/40_heldout_all_models.py \\
        --eval-dir results/final_eval_budget_v9 --single-pattern "heldout_{task}" \\
        --output results/final_eval_budget_v9/heldout_all_models.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
BASELINE_ORDER = ["MajorityClass", "LogisticRegression_Bal", "LinearSVM_Bal",
                  "RandomForest", "RandomForest_Bal", "kNN"]


def diana_row(metrics: Path, task: str, label: str) -> dict:
    m = json.load(open(metrics))[task]
    return {"task": task, "model": label, "f1": m["f1_macro_eligible"],
            "lo": m["f1_macro_eligible_ci_low"], "hi": m["f1_macro_eligible_ci_high"],
            "bal": m["balanced_accuracy"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "results/final_eval_v9")
    ap.add_argument("--single-pattern", default="heldout_single_{task}")
    ap.add_argument("--baselines", type=Path,
                    default=ROOT / "results/baseline_predictions_v9/summary.csv")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    s = pd.read_csv(a.baselines)
    s = s[s.split == "test"]
    rows = []
    for task in TASKS:
        for model in BASELINE_ORDER:
            r = s[(s.task == task) & (s.model == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"task": task, "model": model, "f1": r.f1_macro_eligible,
                         "lo": r.f1_macro_eligible_ci_low, "hi": r.f1_macro_eligible_ci_high,
                         "bal": r.balanced_accuracy})
        rows.append(diana_row(a.eval_dir / "heldout_multitask/test_metrics.json", task,
                              "DIANA multi-task"))
        rows.append(diana_row(a.eval_dir / a.single_pattern.format(task=task) / "test_metrics.json",
                              task, "DIANA single-task"))
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
