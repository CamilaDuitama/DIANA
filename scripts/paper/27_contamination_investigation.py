#!/usr/bin/env python3
"""
27_contamination_investigation.py

INVESTIGATION SCRIPT — NOT FOR PAPER

Purpose:
    Identify BioProjects that may contain contaminated or mislabelled samples
    by finding ALL DIANA errors (predicted != true_label), with optional
    confidence filtering.

    Covers:
        - Validation set  (987 samples, via results/validation_predictions/)
        - Test set        (461 samples, via results/test_evaluation/test_predictions.tsv)
        - Training set    — NOT available at per-sample prediction level

    "Genuine confusion" = the true label WAS seen during training.
    These are the primary contamination/mislabelling candidates because:
        - DIANA saw that class in training
        - DIANA still predicts a *different* class
        → Possible that the AMD metadata is wrong, not DIANA

Output:
    Printed to stdout (not written to file — this is an investigation tool)

Usage:
    python scripts/paper/27_contamination_investigation.py
    python scripts/paper/27_contamination_investigation.py --threshold 0.9  # optional confidence filter
    python scripts/paper/27_contamination_investigation.py --genuine-only
"""

import sys
import json
import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "paper"))
from config import PATHS

TEST_PREDICTIONS   = REPO / "results" / "test_evaluation" / "test_predictions.tsv"
LABEL_ENCODERS     = REPO / "results" / "training" / "label_encoders.json"
VAL_METADATA       = REPO / "paper" / "metadata" / "validation_metadata.tsv"
TEST_METADATA      = REPO / "paper" / "metadata" / "test_metadata.tsv"
TRAIN_METADATA     = REPO / "paper" / "metadata" / "train_metadata.tsv"

TASKS = ["sample_type", "community_type", "sample_host", "material"]
META_COLS = ["Run_accession", "BioProject", "project_name", "publication_year",
             "sample_type", "community_type", "sample_host", "material"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_label_encoders() -> dict[str, list[str]]:
    with open(LABEL_ENCODERS) as f:
        raw = json.load(f)
    return {task: raw[task]["classes"] for task in TASKS}


def load_meta(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    cols = [c for c in META_COLS if c in df.columns]
    df = df[cols].copy()
    df["split"] = split
    return df


# ---------------------------------------------------------------------------
# Load validation errors (pre-computed by script 24)
# ---------------------------------------------------------------------------

def load_validation_errors(threshold: float | None,
                            le: dict[str, list[str]]) -> pd.DataFrame:
    """
    Load ALL validation misclassifications (predicted != true_label).
    Uses the raw per-sample JSON predictions, not the pre-filtered TSV,
    so we get every error regardless of confidence.
    Optional confidence threshold applied if provided.
    """
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "validation"))
    from load_validation_data import load_validation_predictions

    df = load_validation_predictions(quiet=True)
    # Keep only errors
    df = df[~df["is_correct"]].copy()
    if threshold is not None:
        df = df[df["confidence"] >= threshold]

    df = df.rename(columns={"sample_id": "Run_accession", "pred_label": "predicted"})

    # Merge BioProject and project_name from validation metadata
    val_meta = pd.read_csv(VAL_METADATA, sep="\t", low_memory=False)
    meta_slim = val_meta[["Run_accession", "BioProject", "project_name"]].drop_duplicates()
    df = df.merge(meta_slim, on="Run_accession", how="left")
    df["BioProject"]   = df["BioProject"].fillna("Unknown")
    df["project_name"] = df["project_name"].fillna("")
    df["split"] = "validation"

    # is_seen: true label was in training classes
    df["is_seen"] = False
    for task in TASKS:
        mask = df["task"] == task
        training_classes = set(le[task])
        df.loc[mask, "is_seen"] = df.loc[mask, "true_label"].isin(training_classes)

    return df[["Run_accession", "BioProject", "project_name", "task",
               "true_label", "predicted", "confidence", "is_seen", "split"]]


# ---------------------------------------------------------------------------
# Load test errors (from test_predictions.tsv — wide format → long)
# ---------------------------------------------------------------------------

def load_test_errors(threshold: float | None,
                     le: dict[str, list[str]],
                     test_meta: pd.DataFrame) -> pd.DataFrame:
    """Load ALL test misclassifications (predicted != true_label), optionally filtered by confidence."""
    preds = pd.read_csv(TEST_PREDICTIONS, sep="\t", low_memory=False)

    rows = []
    for task in TASKS:
        pred_col = f"{task}_pred"
        true_col = f"{task}_true"
        if pred_col not in preds.columns or true_col not in preds.columns:
            continue

        # Confidence = max probability across task classes
        prob_cols = [c for c in preds.columns if c.startswith(f"{task}_prob_")]
        if prob_cols:
            conf = preds[prob_cols].max(axis=1)
        else:
            conf = pd.Series([float("nan")] * len(preds))

        task_df = pd.DataFrame({
            "Run_accession": preds["Run_accession"],
            "task":          task,
            "true_label":    preds[true_col],
            "predicted":     preds[pred_col],
            "confidence":    conf,
        })

        # Keep only errors (predicted != true); optionally filter by confidence
        mask = task_df["true_label"] != task_df["predicted"]
        if threshold is not None:
            mask = mask & (task_df["confidence"] >= threshold)
        task_df = task_df[mask].copy()

        training_classes = set(le[task])
        task_df["is_seen"] = task_df["true_label"].isin(training_classes)
        rows.append(task_df)

    if not rows:
        return pd.DataFrame()

    long = pd.concat(rows, ignore_index=True)

    # Merge BioProject from test metadata
    meta_slim = test_meta[["Run_accession", "BioProject", "project_name"]].drop_duplicates()
    long = long.merge(meta_slim, on="Run_accession", how="left")
    long["split"] = "test"
    long["BioProject"] = long["BioProject"].fillna("Unknown")
    long["project_name"] = long["project_name"].fillna("")

    return long[["Run_accession", "BioProject", "project_name", "task",
                 "true_label", "predicted", "confidence", "is_seen", "split"]]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def sep(char="-", width=100):
    print(char * width)


def print_summary_table(df: pd.DataFrame, title: str):
    """Print BioProject-level summary sorted by misclassified sample count."""
    sep("=")
    print(f"  {title}")
    sep("=")

    # Count distinct *samples* (not run×task pairs) with ≥1 confident error
    sample_counts = (
        df.groupby(["BioProject", "project_name", "split"])
        .agg(
            n_samples_with_errors=("Run_accession", "nunique"),
            n_run_task_errors=("Run_accession", "count"),
            tasks_affected=("task", lambda x: ", ".join(sorted(x.unique()))),
            genuine_confusion=("is_seen", "sum"),
        )
        .reset_index()
        .sort_values("n_samples_with_errors", ascending=False)
    )

    print(f"\n{'BioProject':<18} {'Study':<22} {'Split':<12} "
          f"{'Samples w/ err':>14} {'Run×task errs':>13} "
          f"{'Genuine (seen)':>14} {'Tasks'}")
    sep()
    for _, row in sample_counts.iterrows():
        print(f"{row['BioProject']:<18} {str(row['project_name'])[:20]:<22} "
              f"{row['split']:<12} {row['n_samples_with_errors']:>14} "
              f"{row['n_run_task_errors']:>13} {int(row['genuine_confusion']):>14} "
              f"{row['tasks_affected']}")
    print()


def print_detail_table(df: pd.DataFrame, title: str, only_genuine: bool = False):
    """Print per-sample detail rows."""
    sep("=")
    print(f"  {title}")
    if only_genuine:
        print("  Filtered to: true label IS in training set (genuine confusion = contamination candidates)")
    sep("=")

    if only_genuine:
        df = df[df["is_seen"]].copy()

    if df.empty:
        print("  No records match the filter.\n")
        return

    # Group and display per BioProject
    for bp, bdf in df.sort_values(
        ["BioProject", "Run_accession", "task"]
    ).groupby("BioProject", sort=False):
        study = bdf["project_name"].iloc[0]
        split = bdf["split"].iloc[0]
        samples = bdf["Run_accession"].nunique()
        print(f"\n  BioProject: {bp}  |  Study: {study}  |  Split: {split}  |  {samples} unique runs with errors")
        sep("-", 80)
        print(f"    {'Run_accession':<16} {'Task':<18} {'True label':<35} {'Predicted':<35} {'Conf':>6} {'Seen?'}")
        sep("-", 80)
        for _, row in bdf.sort_values(["Run_accession", "task"]).iterrows():
            seen_flag = "YES" if row["is_seen"] else "no"
            print(f"    {row['Run_accession']:<16} {row['task']:<18} "
                  f"{str(row['true_label'])[:33]:<35} {str(row['predicted'])[:33]:<35} "
                  f"{row['confidence']:>6.3f} {seen_flag}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional confidence threshold (e.g. 0.9). Default: show ALL errors")
    parser.add_argument("--genuine-only", action="store_true",
                        help="In detail tables, show only genuine confusion (seen label)")
    args = parser.parse_args()

    le = load_label_encoders()
    test_meta  = load_meta(TEST_METADATA,  "test")
    val_meta   = load_meta(VAL_METADATA,   "validation")
    train_meta = load_meta(TRAIN_METADATA, "train")

    thresh_str = f">= {args.threshold}" if args.threshold is not None else "ALL errors (no threshold)"
    print(f"\nDIANA Contamination / Mislabelling Investigation")
    print(f"Confidence filter: {thresh_str}")
    print(f"Training set: NO per-sample predictions stored — cannot assess train errors\n")

    # --- Load errors ---
    val_err  = load_validation_errors(args.threshold, le)
    test_err = load_test_errors(args.threshold, le, test_meta)

    all_err = pd.concat([val_err, test_err], ignore_index=True)

    print(f"Total misclassifications loaded:")
    print(f"  Validation : {len(val_err)} run×task pairs across {val_err['Run_accession'].nunique()} samples")
    if not test_err.empty:
        print(f"  Test       : {len(test_err)} run×task pairs across {test_err['Run_accession'].nunique()} samples")
    else:
        print(f"  Test       : 0 errors")
    print(f"  TOTAL      : {len(all_err)} run×task pairs across {all_err['Run_accession'].nunique()} samples\n")

    # --- Summary table: all errors ---
    print_summary_table(all_err, "ALL ERRORS — by BioProject (sorted by samples with errors)")

    # --- Summary table: genuine confusion only ---
    genuine = all_err[all_err["is_seen"]].copy()
    print_summary_table(genuine,
                        f"GENUINE CONFUSION ONLY (true label in training set) — "
                        f"CONTAMINATION CANDIDATES")

    # --- Detail: genuine confusion in TEST set ---
    if not test_err.empty:
        print_detail_table(test_err,
                           "DETAIL — TEST SET errors",
                           only_genuine=args.genuine_only)

    # --- Detail: genuine confusion in VALIDATION set ---
    print_detail_table(val_err,
                       "DETAIL — VALIDATION SET errors",
                       only_genuine=args.genuine_only)

    # --- Per-task breakdown of genuine confusion ---
    sep("=")
    print("  GENUINE CONFUSION per task — what is DIANA confusing?")
    sep("=")
    confusion_pairs = (
        genuine.groupby(["split", "task", "true_label", "predicted"])
        .size()
        .reset_index(name="n")
        .sort_values(["task", "n"], ascending=[True, False])
    )
    for task, tdf in confusion_pairs.groupby("task"):
        print(f"\n  Task: {task}")
        print(f"    {'True label':<35} {'Predicted':<35} {'Count':>6} {'Split'}")
        print("    " + "-" * 85)
        for _, row in tdf.iterrows():
            print(f"    {str(row['true_label'])[:33]:<35} {str(row['predicted'])[:33]:<35} "
                  f"{row['n']:>6}  {row['split']}")


if __name__ == "__main__":
    main()
