#!/usr/bin/env python3
"""
24_bioproject_error_analysis.py

Five-question diagnostic of whether DIANA validation errors concentrate in
specific BioProjects or other batch variables.

Questions addressed
-------------------
1. Per-BioProject: N_total, N_wrong, error_rate
2. Per-BioProject: N_conf_wrong, conf_wrong_rate  (confidence >= threshold)
3. Confusion-pair breakdown by BioProject
4. Confidence distribution (correct vs wrong) per BioProject
5. Same analyses for sequencing platform, library layout, instrument

Outputs (paper/tables/final/)
------------------------------
  bioproject_error_rates.tsv          – Q1 & Q2
  confusion_pair_by_bioproject.tsv    – Q3
  confidence_distribution_by_bioproject.tsv – Q4
  batch_error_rates.tsv               – Q5 (platform / layout / instrument)

Usage
-----
    python scripts/paper/24_bioproject_error_analysis.py [--threshold 0.9]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR  = REPO_ROOT / "results" / "validation_predictions"
META_PATH = REPO_ROOT / "paper" / "metadata" / "validation_metadata.tsv"
LE_PATH   = REPO_ROOT / "results" / "training" / "label_encoders.json"
OUT_DIR   = REPO_ROOT / "paper" / "tables" / "final"

TASKS = ["sample_type", "community_type", "sample_host", "material"]

# True-label columns in validation_metadata (same order as TASKS)
TRUE_COLS = {
    "sample_type":    "sample_type",
    "community_type": "community_type",
    "sample_host":    "sample_host",
    "material":       "material",
}

# ── metadata columns we want from validation_metadata
META_COLS = [
    "Run_accession", "BioProject", "Platform", "Instrument",
    "LibraryLayout", "Avg_read_len", "Avg_num_reads",
    "project_name",  # study name
    "sample_type", "community_type", "sample_host", "material",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_label_encoders() -> dict[str, list[str]]:
    with open(LE_PATH) as f:
        raw = json.load(f)
    return {task: raw[task]["classes"] for task in TASKS}


def load_predictions(le: dict[str, list[str]]) -> pd.DataFrame:
    """
    Walk every *_predictions.json under PRED_DIR and return a flat DataFrame
    with one row per (sample, task).

    Columns: run_accession, task, true_label (str), predicted (str),
             confidence, margin, is_correct
    """
    rows = []
    for sample_dir in sorted(PRED_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue
        pred_files = list(sample_dir.glob("*_predictions.json"))
        if not pred_files:
            continue
        with open(pred_files[0]) as f:
            data = json.load(f)

        run = data.get("sample_id", sample_dir.name)
        preds = data.get("predictions", {})

        for task in TASKS:
            if task not in preds:
                continue
            tp = preds[task]
            classes   = le[task]
            probs_raw = tp.get("probabilities", {})
            # probabilities are keyed by index (string) or class name
            if all(k.isdigit() for k in probs_raw):
                probs = [probs_raw.get(str(i), 0.0) for i in range(len(classes))]
            else:
                probs = [probs_raw.get(c, 0.0) for c in classes]

            sorted_probs = sorted(probs, reverse=True)
            confidence = sorted_probs[0]
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

            pred_class = tp.get("predicted_class")
            pred_idx   = tp.get("class_index")
            if pred_class and pred_class in classes:
                predicted = pred_class
            elif pred_idx is not None and pred_idx < len(classes):
                predicted = classes[pred_idx]
            else:
                # fall back: class with max prob
                best_key = max(probs_raw, key=probs_raw.__getitem__)
                predicted = classes[int(best_key)] if best_key.isdigit() else best_key

            rows.append({
                "run_accession": run,
                "task": task,
                "predicted": predicted,
                "confidence": confidence,
                "margin": margin,
            })

    return pd.DataFrame(rows)


def load_metadata() -> pd.DataFrame:
    available = pd.read_csv(META_PATH, sep="\t", low_memory=False)
    # Normalise column presence
    cols = [c for c in META_COLS if c in available.columns]
    meta = available[cols].copy()
    meta = meta.rename(columns={"Run_accession": "run_accession"})
    return meta


def build_master(preds: pd.DataFrame, meta: pd.DataFrame,
                 le: dict[str, list[str]]) -> pd.DataFrame:
    """Merge predictions with metadata and add true label + is_correct."""
    df = preds.merge(meta, on="run_accession", how="left")

    # Attach true label for each task row
    true_labels = []
    for _, row in df.iterrows():
        task = row["task"]
        col  = TRUE_COLS.get(task, task)
        true_labels.append(row.get(col, pd.NA))
    df["true_label"] = true_labels

    df["is_correct"] = df["predicted"] == df["true_label"]

    # Sanitize BioProject: keep only PRJ* values, else "Unknown"
    if "BioProject" in df.columns:
        df["BioProject"] = df["BioProject"].astype(str).str.strip()
        df.loc[~df["BioProject"].str.match(r"PRJ[A-Z]+\d+"), "BioProject"] = "Unknown"
    else:
        df["BioProject"] = "Unknown"

    return df


# ---------------------------------------------------------------------------
# Q1 & Q2: Per-BioProject error rates
# ---------------------------------------------------------------------------

def bioproject_error_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    For each (task, BioProject): N_total, N_wrong, error_rate,
    N_conf_wrong (conf >= threshold), conf_wrong_rate.
    """
    rows = []
    for task, gdf in df.groupby("task"):
        for bp, bdf in gdf.groupby("BioProject"):
            n_total      = len(bdf)
            n_wrong      = (~bdf["is_correct"]).sum()
            error_rate   = n_wrong / n_total if n_total else float("nan")
            conf_wrong   = bdf[~bdf["is_correct"] & (bdf["confidence"] >= threshold)]
            n_conf_wrong = len(conf_wrong)
            conf_wrong_rate = n_conf_wrong / n_total if n_total else float("nan")
            mean_conf_wrong = conf_wrong["confidence"].mean() if n_conf_wrong else float("nan")

            rows.append({
                "task": task,
                "BioProject": bp,
                "project_name": bdf["project_name"].mode()[0] if "project_name" in bdf and bdf["project_name"].notna().any() else "",
                "N_total": n_total,
                "N_wrong": n_wrong,
                "error_rate": round(error_rate, 4),
                "N_conf_wrong": n_conf_wrong,
                "conf_wrong_rate": round(conf_wrong_rate, 4),
                "mean_confidence_on_errors": round(mean_conf_wrong, 4) if not np.isnan(mean_conf_wrong) else float("nan"),
            })

    out = pd.DataFrame(rows)
    out = out.sort_values(["task", "conf_wrong_rate"], ascending=[True, False])
    return out


# ---------------------------------------------------------------------------
# Q3: Confusion pair × BioProject
# ---------------------------------------------------------------------------

def confusion_by_bioproject(df: pd.DataFrame, threshold: float,
                             min_count: int = 3) -> pd.DataFrame:
    """
    For each (task, true_label → predicted) confusion pair, list the top
    contributing BioProjects.
    """
    errors = df[~df["is_correct"] & (df["confidence"] >= threshold)].copy()
    errors["pair"] = errors["true_label"].astype(str) + " → " + errors["predicted"].astype(str)

    rows = []
    for (task, pair), gdf in errors.groupby(["task", "pair"]):
        pair_total = len(gdf)
        for bp, bdf in gdf.groupby("BioProject"):
            rows.append({
                "task": task,
                "confusion_pair": pair,
                "pair_total_conf_errors": pair_total,
                "BioProject": bp,
                "project_name": bdf["project_name"].mode()[0] if "project_name" in bdf and bdf["project_name"].notna().any() else "",
                "n_from_bioproject": len(bdf),
                "frac_of_pair": round(len(bdf) / pair_total, 3),
                "mean_confidence": round(bdf["confidence"].mean(), 4),
                "mean_margin": round(bdf["margin"].mean(), 4),
            })

    out = pd.DataFrame(rows)
    out = out[out["pair_total_conf_errors"] >= min_count]
    out = out.sort_values(["task", "pair_total_conf_errors", "n_from_bioproject"],
                          ascending=[True, False, False])
    return out


# ---------------------------------------------------------------------------
# Q4: Confidence distribution (correct vs wrong) per BioProject
# ---------------------------------------------------------------------------

def confidence_distribution(df: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    rows = []
    for (task, bp), gdf in df.groupby(["task", "BioProject"]):
        if len(gdf) < min_samples:
            continue
        correct = gdf[gdf["is_correct"]]["confidence"]
        wrong   = gdf[~gdf["is_correct"]]["confidence"]
        rows.append({
            "task": task,
            "BioProject": bp,
            "project_name": gdf["project_name"].mode()[0] if "project_name" in gdf and gdf["project_name"].notna().any() else "",
            "N_total": len(gdf),
            "N_correct": len(correct),
            "N_wrong": len(wrong),
            "mean_conf_correct": round(correct.mean(), 4) if len(correct) else float("nan"),
            "median_conf_correct": round(correct.median(), 4) if len(correct) else float("nan"),
            "mean_conf_wrong": round(wrong.mean(), 4) if len(wrong) else float("nan"),
            "median_conf_wrong": round(wrong.median(), 4) if len(wrong) else float("nan"),
            # High-confidence errors: those are the dangerous ones
            "n_conf_wrong_90": int((wrong >= 0.9).sum()),
            "n_conf_wrong_95": int((wrong >= 0.95).sum()),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["task", "n_conf_wrong_90"], ascending=[True, False])
    return out


# ---------------------------------------------------------------------------
# Q5: Same analysis for platform / layout / instrument
# ---------------------------------------------------------------------------

def batch_error_rates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    batch_cols = [c for c in ["Platform", "LibraryLayout", "Instrument"] if c in df.columns]
    rows = []
    for col in batch_cols:
        for (task, val), gdf in df.groupby(["task", col]):
            val_str = str(val).strip()
            if not val_str or val_str in ("nan", "Not applicable", ""):
                continue
            n_total      = len(gdf)
            n_wrong      = (~gdf["is_correct"]).sum()
            error_rate   = n_wrong / n_total if n_total else float("nan")
            n_conf_wrong = (~gdf["is_correct"] & (gdf["confidence"] >= threshold)).sum()
            conf_wrong_rate = n_conf_wrong / n_total if n_total else float("nan")
            rows.append({
                "batch_variable": col,
                "batch_value": val_str,
                "task": task,
                "N_total": n_total,
                "N_wrong": n_wrong,
                "error_rate": round(error_rate, 4),
                "N_conf_wrong": n_conf_wrong,
                "conf_wrong_rate": round(conf_wrong_rate, 4),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["batch_variable", "task", "conf_wrong_rate"],
                          ascending=[True, True, False])
    return out


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_q1q2(t: pd.DataFrame, threshold: float) -> None:
    print(f"\n{'='*70}")
    print(f"Q1/Q2  Per-BioProject error rates  (conf threshold={threshold})")
    print(f"{'='*70}")
    for task, tdf in t.groupby("task"):
        # show top offenders (highest conf_wrong_rate, min 2 samples)
        sig = tdf[tdf["N_total"] >= 5].nlargest(8, "conf_wrong_rate")
        if sig.empty:
            continue
        print(f"\n  Task: {task}")
        print(f"  {'BioProject':<16}  {'Project':<28}  {'N':>5}  {'err%':>6}  {'conf_err%':>9}")
        print(f"  {'-'*16}  {'-'*28}  {'-'*5}  {'-'*6}  {'-'*9}")
        for _, r in sig.iterrows():
            proj = str(r.get("project_name",""))[:27]
            print(f"  {r['BioProject']:<16}  {proj:<28}  {r['N_total']:>5}  "
                  f"{r['error_rate']*100:>5.1f}%  {r['conf_wrong_rate']*100:>8.1f}%")


def print_q3(t: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"Q3  Confusion pairs × BioProject")
    print(f"{'='*70}")
    for task, tdf in t.groupby("task"):
        print(f"\n  Task: {task}")
        for pair, pdf in tdf.nlargest(10, "pair_total_conf_errors").groupby("confusion_pair", sort=False):
            total = pdf["pair_total_conf_errors"].iloc[0]
            print(f"    '{pair}'  (total={total})")
            for _, r in pdf.nlargest(3, "n_from_bioproject").iterrows():
                proj = str(r.get("project_name",""))[:25]
                print(f"        {r['BioProject']:<16}  {proj:<25}  "
                      f"n={r['n_from_bioproject']}  ({r['frac_of_pair']*100:.0f}%)")


def print_q4(t: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"Q4  Confidence distribution (correct vs wrong) per BioProject")
    print(f"{'='*70}")
    for task, tdf in t.groupby("task"):
        sig = tdf[tdf["n_conf_wrong_90"] >= 3].nlargest(8, "n_conf_wrong_90")
        if sig.empty:
            continue
        print(f"\n  Task: {task}")
        print(f"  {'BioProject':<16}  {'Project':<25}  {'N':>5}  "
              f"{'conf_ok':>8}  {'conf_err':>8}  {'≥90%_err':>8}")
        print(f"  {'-'*16}  {'-'*25}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
        for _, r in sig.iterrows():
            proj = str(r.get("project_name",""))[:24]
            print(f"  {r['BioProject']:<16}  {proj:<25}  {r['N_total']:>5}  "
                  f"{r['mean_conf_correct']:>8.3f}  {r['mean_conf_wrong']:>8.3f}  "
                  f"{r['n_conf_wrong_90']:>8}")


def print_q5(t: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"Q5  Error rates by sequencing batch variable")
    print(f"{'='*70}")
    if t.empty:
        print("  (no Platform/LibraryLayout/Instrument data available in metadata)")
        return
    for var, vdf in t.groupby("batch_variable"):
        print(f"\n  Variable: {var}")
        for task, tdf in vdf.groupby("task"):
            sig = tdf[tdf["N_total"] >= 5]
            if sig.empty:
                continue
            print(f"    Task: {task}")
            print(f"    {'Value':<32}  {'N':>5}  {'err%':>6}  {'conf_err%':>9}")
            for _, r in sig.iterrows():
                print(f"    {str(r['batch_value']):<32}  {r['N_total']:>5}  "
                      f"{r['error_rate']*100:>5.1f}%  {r['conf_wrong_rate']*100:>8.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="Confidence threshold for 'confident wrong' (default: 0.9)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading label encoders...")
    le = load_label_encoders()

    print("Loading predictions...")
    preds = load_predictions(le)
    print(f"  {len(preds)} (sample, task) rows loaded")

    print("Loading metadata...")
    meta = load_metadata()
    print(f"  {len(meta)} metadata rows")

    print("Building master table...")
    df = build_master(preds, meta, le)
    total_samples = df["run_accession"].nunique()
    print(f"  {total_samples} unique samples, {len(df)} rows")

    # Q1 & Q2
    print("\nComputing per-BioProject error rates (Q1/Q2)...")
    t_bp = bioproject_error_table(df, args.threshold)
    out1 = OUT_DIR / "bioproject_error_rates.tsv"
    t_bp.to_csv(out1, sep="\t", index=False)

    # Q3
    print("Computing confusion pair × BioProject (Q3)...")
    t_conf = confusion_by_bioproject(df, args.threshold)
    out3 = OUT_DIR / "confusion_pair_by_bioproject.tsv"
    t_conf.to_csv(out3, sep="\t", index=False)

    # Q4
    print("Computing confidence distributions (Q4)...")
    t_dist = confidence_distribution(df)
    out4 = OUT_DIR / "confidence_distribution_by_bioproject.tsv"
    t_dist.to_csv(out4, sep="\t", index=False)

    # Q5
    print("Computing batch variable error rates (Q5)...")
    t_batch = batch_error_rates(df, args.threshold)
    out5 = OUT_DIR / "batch_error_rates.tsv"
    t_batch.to_csv(out5, sep="\t", index=False)

    # Print summaries
    print_q1q2(t_bp, args.threshold)
    print_q3(t_conf)
    print_q4(t_dist)
    print_q5(t_batch)

    print(f"\n{'='*70}")
    print("Output files:")
    for p in [out1, out3, out4, out5]:
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
