#!/usr/bin/env python3
"""
23_confident_error_mining.py

Diagnose whether confident errors are concentrated in specific subsets.

For each task, this script:
1. Extracts all misclassified validation samples with confidence >= threshold
2. Enriches them with sequencing metadata (BioProject, Platform, Instrument,
   LibraryLayout, Avg_read_len, Avg_num_reads) and label context
3. Computes margin = top_prob - second_prob (a tighter measure than raw confidence)
4. Asks:
   - Are confident errors dominated by a few BioProjects/studies?
   - Are they dominated by specific (true → predicted) label pairs?
   - Do they cluster by platform, library layout, or read length?

Outputs (all in paper/tables/final/):
  - confident_errors_raw.tsv        — full enriched table of all errors
  - confident_errors_by_bioproject.tsv — error counts and mean confidence by BioProject
  - confident_errors_by_labelpair.tsv  — error counts by (task, true→pred) pair

Usage:
    cd <repo_root>
    python scripts/paper/23_confident_error_mining.py [--threshold 0.9] [--verbose]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add validation and paper directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))
sys.path.insert(0, str(Path(__file__).parent))

from load_validation_data import load_validation_predictions
from config import PATHS, TASKS, SAMPLE_TYPE_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_probabilities() -> pd.DataFrame:
    """
    Load raw per-class probabilities from every *_predictions.json file so we
    can compute the confidence margin (top − second).

    Handles both probability formats:
    - Class-name keyed: {"ancient": 0.95, "modern": 0.05}
    - Numeric-index keyed: {"0": 0.95, "1": 0.05}  (legacy format)

    Returns a DataFrame with columns: sample_id, task, confidence, margin
    """
    predictions_dir = Path(PATHS["predictions_dir"])

    records = []
    for pred_file in predictions_dir.rglob("*_predictions.json"):
        sample_id = pred_file.parent.name
        with open(pred_file) as f:
            pred = json.load(f)

        for task in TASKS:
            if task not in pred.get("predictions", {}):
                continue
            task_pred = pred["predictions"][task]

            probs_raw = task_pred.get("probabilities", {})
            if not probs_raw:
                continue

            # Extract numeric probability values regardless of key format
            probs = np.array([float(v) for v in probs_raw.values()])

            if len(probs) < 2:
                margin = float(probs[0]) if len(probs) == 1 else 0.0
            else:
                sorted_probs = np.sort(probs)[::-1]
                margin = float(sorted_probs[0] - sorted_probs[1])

            records.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "confidence": float(task_pred.get("confidence", 0.0)),
                    "margin": margin,
                }
            )

    return pd.DataFrame(records)


def load_metadata_cols() -> pd.DataFrame:
    """
    Load validation metadata enriched with archive_project (BioProject ID, e.g. PRJNA…)
    from the AncientMetagenomeDir TSVs.

    Join path:
        validation_metadata.Run_accession
        → validation_metadata.archive_accession  (SRS/ERS sample accession)
        → AncientMetagenomeDir.archive_accession  (exploded, comma-separated)
        → AncientMetagenomeDir.archive_project    (PRJNA…/PRJEB… BioProject ID)
    """
    meta = pd.read_csv(PATHS["validation_metadata"], sep="\t", low_memory=False)

    # Build archive_project lookup from AncientMetagenomeDir cached files
    amd_env  = Path("data/metadata/ancientmetagenome-environmental_samples.tsv")
    amd_host = Path("data/metadata/ancientmetagenome-hostassociated_samples.tsv")
    amd_frames = []
    for amd_path in [amd_env, amd_host]:
        if amd_path.exists():
            df = pd.read_csv(amd_path, sep="\t", low_memory=False,
                             usecols=["archive_accession", "archive_project"])
            amd_frames.append(df)

    if amd_frames:
        amd = pd.concat(amd_frames, ignore_index=True)
        # Explode comma-separated accessions (e.g. "SRS001,SRS002")
        amd = (
            amd.assign(archive_accession=amd["archive_accession"].str.split(","))
            .explode("archive_accession")
        )
        amd["archive_accession"] = amd["archive_accession"].str.strip()
        amd = amd.drop_duplicates(subset="archive_accession")

        # Join onto validation metadata via sample-level accession
        meta = meta.merge(amd, on="archive_accession", how="left")
    else:
        meta["archive_project"] = pd.NA

    keep = [
        "Run_accession",
        "archive_project",   # BioProject ID (PRJNA…/PRJEB…)
        "project_name",      # Human-readable study name (e.g. Velsko2024)
        "publication_year",
        "Platform",
        "Instrument",
        "LibraryLayout",
        "Avg_read_len",
        "Avg_num_reads",
        "geo_loc_name",
    ]
    keep = [c for c in keep if c in meta.columns]
    return meta[keep].copy()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_confident_errors(df_preds: pd.DataFrame,
                           df_margins: pd.DataFrame,
                           meta: pd.DataFrame,
                           threshold: float) -> pd.DataFrame:
    """
    Merge all sources into a single enriched table of misclassified,
    high-confidence samples.
    """
    wrong = df_preds[~df_preds["is_correct"]].copy()

    # Attach margins
    wrong = wrong.merge(
        df_margins[["sample_id", "task", "margin"]],
        on=["sample_id", "task"],
        how="left",
    )

    # Filter by confidence threshold
    high_conf = wrong[wrong["confidence"] >= threshold].copy()

    # Attach sequencing metadata
    meta_renamed = meta.rename(columns={"Run_accession": "sample_id"})
    high_conf = high_conf.merge(meta_renamed, on="sample_id", how="left")

    # Nice label-pair column
    high_conf["error_pair"] = (
        high_conf["true_label"] + " → " + high_conf["pred_label"]
    )

    return high_conf


def bioproject_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Errors grouped by archive_project (BioProject ID) × task."""
    grp_cols = [c for c in ["task", "archive_project", "project_name"] if c in df.columns]
    grp = (
        df.groupby(grp_cols, dropna=False)
        .agg(
            n_errors=("sample_id", "count"),
            mean_confidence=("confidence", "mean"),
            mean_margin=("margin", "mean"),
            dominant_pair=("error_pair", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
        .sort_values(["task", "n_errors"], ascending=[True, False])
    )
    return grp


def label_pair_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Errors grouped by task × (true → predicted) pair."""
    grp = (
        df.groupby(["task", "true_label", "pred_label", "error_pair"])
        .agg(
            n_errors=("sample_id", "count"),
            mean_confidence=("confidence", "mean"),
            mean_margin=("margin", "mean"),
        )
        .reset_index()
        .sort_values(["task", "n_errors"], ascending=[True, False])
    )
    return grp


def platform_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Errors grouped by task × Platform × LibraryLayout."""
    cols = [c for c in ["task", "Platform", "LibraryLayout"] if c in df.columns]
    grp = (
        df.groupby(cols)
        .agg(
            n_errors=("sample_id", "count"),
            mean_confidence=("confidence", "mean"),
            mean_margin=("margin", "mean"),
        )
        .reset_index()
        .sort_values(["task", "n_errors"], ascending=[True, False])
    )
    return grp


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(high_conf: pd.DataFrame,
                 bp: pd.DataFrame,
                 lp: pd.DataFrame,
                 plat: pd.DataFrame,
                 threshold: float,
                 verbose: bool) -> None:
    n_total = len(high_conf)
    n_samples = high_conf["sample_id"].nunique()
    print(f"\n{'='*70}")
    print(f"CONFIDENT ERROR MINING  (confidence ≥ {threshold:.0%})")
    print(f"{'='*70}")
    print(f"Total confident errors : {n_total}  ({n_samples} unique samples)")
    print()

    for task in TASKS:
        t = high_conf[high_conf["task"] == task]
        if t.empty:
            continue
        print(f"── Task: {task} ({'n=' + str(len(t))})")

        # Top label pairs
        pairs = lp[lp["task"] == task].head(8)
        if not pairs.empty:
            print("  Top error pairs (true → predicted):")
            for _, row in pairs.iterrows():
                print(
                    f"    {row['error_pair']:50s}  n={int(row['n_errors']):4d}"
                    f"  conf={row['mean_confidence']:.3f}"
                    f"  margin={row['mean_margin']:.3f}"
                )

        # Top BioProjects
        bps = bp[bp["task"] == task].head(5)
        if not bps.empty:
            print("  Top BioProjects contributing errors:")
            for _, row in bps.iterrows():
                prj = str(row.get("archive_project", "?"))
                name = str(row.get("project_name", ""))
                label = f"{prj} ({name})" if name else prj
                print(
                    f"    {label:40s}  n={int(row['n_errors']):4d}"
                    f"  conf={row['mean_confidence']:.3f}"
                    f"  dominant=({row['dominant_pair']})"
                )

        # Platform breakdown
        pf = plat[plat["task"] == task] if "Platform" in plat.columns else pd.DataFrame()
        if not pf.empty:
            print("  By Platform:")
            for _, row in pf.iterrows():
                layout = row.get("LibraryLayout", "?")
                print(
                    f"    {str(row['Platform']):20s} [{layout}]  n={int(row['n_errors']):4d}"
                    f"  conf={row['mean_confidence']:.3f}"
                )
        print()

    if verbose:
        print("\nFull confident-error sample list:")
        cols = [
            "task", "sample_id", "true_label", "pred_label", "confidence",
            "margin", "error_pair", "BioProject", "Platform",
            "Avg_read_len", "Avg_num_reads",
        ]
        cols = [c for c in cols if c in high_conf.columns]
        pd.set_option("display.max_rows", 200)
        print(high_conf[cols].sort_values(["task", "confidence"], ascending=[True, False]).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Confidence threshold for 'high-confidence' (default: 0.9)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full per-sample list to stdout")
    args = parser.parse_args()

    out_dir = Path(PATHS["tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading validation predictions...")
    df_preds = load_validation_predictions(quiet=False)

    print("\n[2/5] Loading probability margins from raw JSON files...")
    df_margins = load_all_probabilities()

    print("\n[3/5] Loading sequencing metadata...")
    meta = load_metadata_cols()

    print("\n[4/5] Building confident error table...")
    high_conf = build_confident_errors(df_preds, df_margins, meta, args.threshold)

    bp   = bioproject_breakdown(high_conf)
    lp   = label_pair_breakdown(high_conf)
    plat = platform_breakdown(high_conf)

    print("\n[5/5] Saving outputs...")
    raw_path  = out_dir / "confident_errors_raw.tsv"
    bp_path   = out_dir / "confident_errors_by_bioproject.tsv"
    lp_path   = out_dir / "confident_errors_by_labelpair.tsv"
    plat_path = out_dir / "confident_errors_by_platform.tsv"

    high_conf.to_csv(raw_path,  sep="\t", index=False)
    bp.to_csv(bp_path,          sep="\t", index=False)
    lp.to_csv(lp_path,          sep="\t", index=False)
    plat.to_csv(plat_path,      sep="\t", index=False)

    for p in [raw_path, bp_path, lp_path, plat_path]:
        print(f"  ✓ {p}")

    print_report(high_conf, bp, lp, plat, args.threshold, args.verbose)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
