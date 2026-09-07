#!/usr/bin/env python
"""Fill the empty DIANA rows in the v7 baseline comparison table.

Question answered
-----------------
How does DIANA compare with the baselines on the *same* BioProject-disjoint v7
test split, under the *same* metric definitions? (PROJECT.md P0, referees R1.3 /
R3.9.)

Why a separate script
---------------------
The baselines in ``results/baseline_comparison_v7/`` are expensive to refit and
CLAUDE.md forbids overwriting a ``results/*_v7/`` directory. This recomputes only
the DIANA rows and writes a new table alongside the existing one, leaving the
baseline numbers byte-identical.

Metric definitions are imported from the baseline script itself rather than
reimplemented, so ``f1_macro_seen`` (macro-F1 over classes present in the split),
the bootstrap seed and the CI percentiles cannot drift apart.

Headline numbers are recomputed from the raw predictions TSV, never from the
summary JSON (PROJECT.md working rules).

Inputs
------
results/test_evaluation_bioproject_v7/test_predictions.tsv
results/baseline_comparison_v7/metrics.json
results/baseline_comparison_v7/summary.csv

Outputs
-------
results/baseline_comparison_v7/metrics_with_diana.json
results/baseline_comparison_v7/summary_with_diana.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "evaluation" / "08_test_set_baseline_comparison.py"


def load_baseline_module():
    """Import the baseline script so metric definitions are shared, not copied."""
    spec = importlib.util.spec_from_file_location("baseline_comparison", BASELINE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diana_classification_metrics(df: pd.DataFrame, task: str, bc) -> Dict[str, float]:
    """Metrics for one classification task, from the raw predictions TSV."""
    y_true = df[f"{task}_true"].astype(str).to_numpy()
    y_pred = df[f"{task}_pred"].astype(str).to_numpy()

    # 'UNSEEN' marks a test label absent from the training label space. The
    # baselines drop those rows too, so drop them here for a like-for-like split.
    keep = y_pred != "UNSEEN"
    dropped = int((~keep).sum())
    if dropped:
        logger.warning("%s: %d rows with out-of-vocabulary labels excluded", task, dropped)

    metrics = bc.evaluate_with_ci(y_true[keep], y_pred[keep])
    metrics["n_samples"] = int(keep.sum())
    return metrics


def diana_regression_metrics(df: pd.DataFrame, task: str, bc) -> Dict[str, float]:
    """Metrics for one regression task, from the raw predictions TSV (original units)."""
    y_true = df[f"{task}_true"].to_numpy(dtype=float)
    y_pred = df[f"{task}_pred"].to_numpy(dtype=float)

    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    metrics = bc.evaluate_regression_with_ci(y_true[valid], y_pred[valid])
    metrics["n_samples"] = int(valid.sum())
    return metrics


def build_rows(diana: Dict[str, dict], task_types: Dict[str, str]) -> List[dict]:
    rows = []
    for task, metrics in diana.items():
        rows.append({
            "model": "DIANA",
            "split": "test",
            "task": task,
            "task_type": task_types.get(task, "classification"),
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "f1_macro_seen": metrics.get("f1_macro_seen"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "r2": metrics.get("r2"),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path,
                        default=PROJECT_ROOT / "results/test_evaluation_bioproject_v7/test_predictions.tsv")
    parser.add_argument("--baseline-dir", type=Path,
                        default=PROJECT_ROOT / "results/baseline_comparison_v7")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite metrics.json/summary.csv instead of writing "
                             "*_with_diana copies. CLAUDE.md forbids overwriting a "
                             "results/*_v7 directory, so this is opt-in.")
    args = parser.parse_args()

    if not args.predictions.exists():
        logger.error("Predictions not found: %s", args.predictions)
        logger.error("Run diana-test for v7 first (see PROJECT.md P0).")
        return 1

    metrics_path = args.baseline_dir / "metrics.json"
    summary_path = args.baseline_dir / "summary.csv"
    for path in (metrics_path, summary_path):
        if not path.exists():
            logger.error("Baseline artefact not found: %s", path)
            return 1

    bc = load_baseline_module()
    df = pd.read_csv(args.predictions, sep="\t")
    logger.info("Loaded %d DIANA test predictions from %s", len(df), args.predictions)

    with open(metrics_path) as fh:
        metrics = json.load(fh)
    task_types = metrics.get("metadata", {}).get("task_types", {})
    if not task_types:
        logger.error("metrics.json has no metadata.task_types; cannot classify tasks")
        return 1

    diana: Dict[str, dict] = {}
    for task, kind in task_types.items():
        if f"{task}_pred" not in df.columns:
            logger.warning("%s: no predictions column, skipping", task)
            continue
        if kind == "regression":
            diana[task] = diana_regression_metrics(df, task, bc)
        else:
            diana[task] = diana_classification_metrics(df, task, bc)
        logger.info("%s [%s]: %s", task, kind,
                    {k: round(v, 4) for k, v in diana[task].items()
                     if isinstance(v, float) and not k.endswith(("_ci_low", "_ci_high"))})

    metrics["diana"] = diana
    metrics.setdefault("metadata", {})["diana_source"] = str(args.predictions)

    # Replace the placeholder DIANA rows, keeping every baseline row untouched.
    summary = pd.read_csv(summary_path)
    baselines_only = summary[summary["model"] != "DIANA"]
    diana_rows = pd.DataFrame(build_rows(diana, task_types))
    combined = pd.concat([diana_rows, baselines_only], ignore_index=True)[summary.columns.tolist()]

    if args.in_place:
        out_metrics, out_summary = metrics_path, summary_path
    else:
        out_metrics = args.baseline_dir / "metrics_with_diana.json"
        out_summary = args.baseline_dir / "summary_with_diana.csv"

    with open(out_metrics, "w") as fh:
        json.dump(metrics, fh, indent=2)
    combined.to_csv(out_summary, index=False)

    logger.info("Wrote %s", out_metrics)
    logger.info("Wrote %s", out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
