#!/usr/bin/env python3
"""
Confidence Calibration Analysis for DIANA
==========================================
Investigates whether DIANA's per-prediction confidence scores are useful
for flagging uncertain or likely-mislabeled samples.

Three questions:
  1. Do incorrect predictions have lower confidence than correct ones?
     → Distribution plots + Mann-Whitney U test
  2. What precision/recall can be achieved at different confidence thresholds?
     → Precision-Recall curves for "flag as uncertain" (low confidence)
  3. Is confidence well-calibrated (actual accuracy ≈ predicted confidence)?
     → Calibration curves (reliability diagrams) + ECE/MCE

Evaluated on:
  - Test set  (n=523, BioProject-disjoint held-out)
  - Validation set (n=360, external)

DATA SOURCES:
  Test:       results/test_evaluation_bioproject/test_predictions.tsv
              (columns: Run_accession, {task}_pred, {task}_true, {task}_prob_0 … {task}_prob_N)
  Validation: results/validation_predictions_bioproject/<sid>/<sid>_predictions.json
              (per-sample JSON with predicted_class + full probabilities dict)
  Label encoders: results/training_bioproject/label_encoders.json

OUTPUTS:
  paper/figures/final/sup_calibration_A_confidence_distributions.png/.html
  paper/figures/final/sup_calibration_B_precision_recall_flagging.png/.html
  paper/figures/final/sup_calibration_C_reliability_diagrams.png/.html
  results/calibration_analysis/calibration_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, PLOT_CONFIG

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────

TEST_TSV        = Path("results/test_evaluation_bioproject/test_predictions.tsv")
VAL_PRED_DIR    = Path("results/validation_predictions_bioproject")
VAL_META        = Path("data/splits_bioproject/validation_metadata.tsv")
LABEL_ENCODERS  = Path(PATHS["label_encoders"])
OUTPUT_DIR      = Path("results/calibration_analysis")
FIGURES_DIR     = Path(PATHS["figures_dir"])

TASKS = ["sample_type", "community_type", "sample_host", "material"]
TASK_LABELS = {
    "sample_type":    "Sample Type",
    "community_type": "Community Type",
    "sample_host":    "Sample Host",
    "material":       "Material",
}

N_BINS = 10   # calibration bins


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_test_predictions() -> pd.DataFrame:
    """
    Load test predictions TSV. For each task, compute:
      - confidence = max over all class probability columns
      - correct = (pred == true)
    Returns one row per (sample, task).
    """
    df = pd.read_csv(TEST_TSV, sep="\t")
    rows = []
    for task in TASKS:
        prob_cols = [c for c in df.columns if c.startswith(f"{task}_prob_")]
        conf = df[prob_cols].max(axis=1).values
        pred = df[f"{task}_pred"].values
        true = df[f"{task}_true"].values
        correct = (pred == true).astype(int)
        for i, sid in enumerate(df["Run_accession"]):
            rows.append({
                "split":      "test",
                "sample_id":  sid,
                "task":       task,
                "confidence": float(conf[i]),
                "correct":    int(correct[i]),
                "pred":       pred[i],
                "true":       true[i],
            })
    return pd.DataFrame(rows)


def load_val_predictions() -> pd.DataFrame:
    """
    Load per-sample validation prediction JSONs. For each task, compute
    confidence = max probability over all classes.
    """
    val_meta = pd.read_csv(VAL_META, sep="\t")
    rows = []
    for _, meta_row in val_meta.iterrows():
        sid = meta_row["Run_accession"]
        fpath = VAL_PRED_DIR / sid / f"{sid}_predictions.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            d = json.load(f)
        preds = d.get("predictions", {})
        for task in TASKS:
            if task not in preds:
                continue
            task_pred = preds[task]
            pred_class = task_pred["predicted_class"]
            probs = task_pred.get("probabilities", {})
            conf = max(probs.values()) if probs else task_pred.get("confidence", float("nan"))
            true_class = str(meta_row.get(task, ""))
            correct = int(pred_class == true_class)
            rows.append({
                "split":      "val",
                "sample_id":  sid,
                "task":       task,
                "confidence": float(conf),
                "correct":    correct,
                "pred":       pred_class,
                "true":       true_class,
            })
    return pd.DataFrame(rows)


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def precision_recall_at_thresholds(df_task: pd.DataFrame, n_steps: int = 100) -> dict:
    """
    For the "flagging" task: mark low-confidence samples as 'uncertain'.
    At each threshold t, samples with confidence < t are flagged.
    Among flagged samples:
      - precision = fraction that are actually incorrect (errors caught correctly)
      - recall    = fraction of all errors that were flagged
    Also compute the opposite: high-confidence predictions.
      - At threshold t, keep only predictions with confidence >= t
      - Report accuracy among retained samples + fraction retained
    Returns dict with arrays for plotting.
    """
    confs   = df_task["confidence"].values
    correct = df_task["correct"].values
    errors  = (correct == 0).astype(int)
    n_total_errors = errors.sum()

    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    flag_precision, flag_recall, flag_frac = [], [], []
    keep_accuracy, keep_frac = [], []

    for t in thresholds:
        flagged = confs < t
        n_flagged = flagged.sum()
        flag_frac.append(n_flagged / len(confs))
        if n_flagged == 0:
            flag_precision.append(np.nan)
            flag_recall.append(0.0)
        else:
            flag_precision.append(errors[flagged].mean())
            flag_recall.append(errors[flagged].sum() / max(n_total_errors, 1))

        kept = confs >= t
        n_kept = kept.sum()
        keep_frac.append(n_kept / len(confs))
        if n_kept == 0:
            keep_accuracy.append(np.nan)
        else:
            keep_accuracy.append(correct[kept].mean())

    return {
        "thresholds":      thresholds.tolist(),
        "flag_precision":  flag_precision,
        "flag_recall":     flag_recall,
        "flag_frac":       flag_frac,
        "keep_accuracy":   keep_accuracy,
        "keep_frac":       keep_frac,
        "base_accuracy":   float(correct.mean()),
        "n_errors":        int(n_total_errors),
        "n_total":         len(confs),
    }


def calibration_stats(df_task: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    """
    Compute reliability diagram data (actual accuracy per confidence bin).
    Returns bin centres, mean confidence, observed accuracy, and counts.
    Also returns ECE (Expected Calibration Error) and MCE (Maximum CE).
    """
    confs   = df_task["confidence"].values
    correct = df_task["correct"].values

    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres, mean_conf, obs_acc, counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 else (confs >= lo) & (confs <= hi)
        n = mask.sum()
        counts.append(int(n))
        if n == 0:
            bin_centres.append((lo + hi) / 2)
            mean_conf.append((lo + hi) / 2)
            obs_acc.append(np.nan)
        else:
            bin_centres.append((lo + hi) / 2)
            mean_conf.append(float(confs[mask].mean()))
            obs_acc.append(float(correct[mask].mean()))

    mean_conf_arr = np.array(mean_conf)
    obs_acc_arr   = np.array(obs_acc)
    counts_arr    = np.array(counts)
    valid = ~np.isnan(obs_acc_arr)

    n_total = len(confs)
    ece = float(np.sum(counts_arr[valid] / n_total * np.abs(obs_acc_arr[valid] - mean_conf_arr[valid])))
    mce = float(np.max(np.abs(obs_acc_arr[valid] - mean_conf_arr[valid]))) if valid.any() else float("nan")

    return {
        "bin_centres":  bin_centres,
        "mean_conf":    mean_conf,
        "obs_acc":      obs_acc,
        "counts":       counts,
        "ece":          ece,
        "mce":          mce,
    }


def mannwhitney_test(df_task: pd.DataFrame) -> dict:
    """Mann-Whitney U: are correct predictions higher confidence than incorrect ones?"""
    correct   = df_task[df_task["correct"] == 1]["confidence"].values
    incorrect = df_task[df_task["correct"] == 0]["confidence"].values
    if len(incorrect) < 2:
        return {"u_stat": None, "p_value": None, "n_correct": len(correct), "n_incorrect": len(incorrect)}
    stat, p = mannwhitneyu(correct, incorrect, alternative="greater")
    return {
        "u_stat": float(stat),
        "p_value": float(p),
        "mean_conf_correct":   float(correct.mean()),
        "mean_conf_incorrect": float(incorrect.mean()),
        "n_correct":   len(correct),
        "n_incorrect": len(incorrect),
    }


# ─── Plot builders ────────────────────────────────────────────────────────────

TASK_COLORS = dict(zip(TASKS, px.colors.qualitative.Vivid))
SPLIT_DASH = {"test": "solid", "val": "dash"}

def plot_A_distributions(all_data: pd.DataFrame) -> go.Figure:
    """Violin/box + strip: confidence distribution for correct vs incorrect, per task × split."""
    n_tasks = len(TASKS)
    fig = make_subplots(
        rows=2, cols=n_tasks,
        subplot_titles=[f"{TASK_LABELS[t]}<br><sup>Test</sup>" for t in TASKS] +
                       [f"{TASK_LABELS[t]}<br><sup>Validation</sup>" for t in TASKS],
        vertical_spacing=0.14, horizontal_spacing=0.06,
    )

    for row_idx, split in enumerate(["test", "val"], start=1):
        for col_idx, task in enumerate(TASKS, start=1):
            sub = all_data[(all_data["split"] == split) & (all_data["task"] == task)]
            for label, color, name in [(1, "#2196F3", "Correct"), (0, "#F44336", "Incorrect")]:
                vals = sub[sub["correct"] == label]["confidence"].values
                fig.add_trace(go.Violin(
                    y=vals, name=name,
                    side="positive" if label == 1 else "negative",
                    fillcolor=color, line_color=color,
                    opacity=0.65, meanline_visible=True,
                    points=False,
                    showlegend=(row_idx == 1 and col_idx == 1),
                    legendgroup=name,
                    x0=name,
                    bandwidth=0.04,
                ), row=row_idx, col=col_idx)

    fig.update_yaxes(title_text="Confidence", range=[-0.02, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Confidence", range=[-0.02, 1.05], row=2, col=1)
    for row in [1, 2]:
        for col in range(1, n_tasks + 1):
            fig.update_yaxes(range=[-0.02, 1.05], row=row, col=col)
            fig.update_xaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(
        template=PLOT_CONFIG["template"],
        title=dict(
            text="Confidence Score Distributions: Correct vs Incorrect Predictions",
            font=dict(size=13), x=0.5, xanchor="center",
        ),
        violinmode="overlay",
        font=dict(size=PLOT_CONFIG["font_size"]),
        width=1300, height=580,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06),
        margin=dict(l=55, r=30, t=80, b=60),
    )
    return fig


def plot_B_flagging(all_pr: dict) -> go.Figure:
    """
    Two-panel per task: (left) flagging PR curve, (right) accuracy vs. fraction retained.
    Rows = tasks, shared x axis = threshold.
    """
    n_tasks = len(TASKS)
    fig = make_subplots(
        rows=n_tasks, cols=2,
        subplot_titles=[f"{TASK_LABELS[t]} — Flagging (low conf → error)" for t in TASKS for _ in range(2)],
        column_titles=["Precision / Recall of error flagging",
                       "Accuracy vs. fraction of predictions retained"],
        vertical_spacing=0.08, horizontal_spacing=0.10,
    )

    split_colors = {"test": "#1565C0", "val": "#B71C1C"}

    for row_idx, task in enumerate(TASKS, start=1):
        for split in ["test", "val"]:
            pr = all_pr[split][task]
            t  = pr["thresholds"]

            # Panel A: Precision + Recall of flagging
            fig.add_trace(go.Scatter(
                x=t, y=pr["flag_precision"],
                name=f"Precision ({split})", legendgroup=f"prec_{split}",
                showlegend=(row_idx == 1),
                line=dict(color=split_colors[split], dash="solid", width=2),
                hovertemplate="t=%{x:.2f} precision=%{y:.2f}<extra></extra>",
            ), row=row_idx, col=1)
            fig.add_trace(go.Scatter(
                x=t, y=pr["flag_recall"],
                name=f"Recall ({split})", legendgroup=f"rec_{split}",
                showlegend=(row_idx == 1),
                line=dict(color=split_colors[split], dash="dot", width=2),
                hovertemplate="t=%{x:.2f} recall=%{y:.2f}<extra></extra>",
            ), row=row_idx, col=1)

            # Panel B: Accuracy of retained (high-confidence) predictions
            fig.add_trace(go.Scatter(
                x=t, y=pr["keep_accuracy"],
                name=f"Accuracy ({split})", legendgroup=f"acc_{split}",
                showlegend=(row_idx == 1),
                line=dict(color=split_colors[split], dash=SPLIT_DASH[split], width=2),
                hovertemplate="t=%{x:.2f} acc=%{y:.2f} keep=%{customdata:.0%}<extra></extra>",
                customdata=pr["keep_frac"],
            ), row=row_idx, col=2)
            # Baseline accuracy (dashed horizontal)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[pr["base_accuracy"]] * 2,
                name=f"Baseline ({split})", legendgroup=f"base_{split}",
                showlegend=False,
                line=dict(color=split_colors[split], dash="dash", width=1),
            ), row=row_idx, col=2)

        # Y labels
        fig.update_yaxes(title_text="Score", range=[0, 1.05], row=row_idx, col=1)
        fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=row_idx, col=2)
        fig.update_xaxes(title_text="Confidence threshold" if row_idx == n_tasks else "",
                         range=[0, 1], row=row_idx, col=1)
        fig.update_xaxes(title_text="Confidence threshold" if row_idx == n_tasks else "",
                         range=[0, 1], row=row_idx, col=2)

    fig.update_layout(
        template=PLOT_CONFIG["template"],
        title=dict(
            text="Confidence Threshold Analysis: Error Flagging & Selective Prediction",
            font=dict(size=13), x=0.5, xanchor="center",
        ),
        font=dict(size=PLOT_CONFIG["font_size"]),
        width=1000, height=280 * n_tasks,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02),
        margin=dict(l=70, r=30, t=80, b=60),
    )
    return fig


def plot_C_reliability(all_cal: dict) -> go.Figure:
    """Reliability diagrams (calibration curves) per task × split."""
    n_tasks = len(TASKS)
    fig = make_subplots(
        rows=2, cols=n_tasks,
        subplot_titles=[f"{TASK_LABELS[t]}<br><sup>Test</sup>" for t in TASKS] +
                       [f"{TASK_LABELS[t]}<br><sup>Validation</sup>" for t in TASKS],
        vertical_spacing=0.14, horizontal_spacing=0.06,
    )

    for row_idx, split in enumerate(["test", "val"], start=1):
        for col_idx, task in enumerate(TASKS, start=1):
            cal = all_cal[split][task]
            mean_c = np.array(cal["mean_conf"])
            obs_a  = np.array(cal["obs_acc"])
            counts = np.array(cal["counts"])
            ece    = cal["ece"]

            valid = ~np.isnan(obs_a)
            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#aaaaaa", dash="dash", width=1),
                showlegend=(row_idx == 1 and col_idx == 1),
                name="Perfect calibration",
            ), row=row_idx, col=col_idx)

            # Calibration bars (gap = miscalibration)
            fig.add_trace(go.Bar(
                x=mean_c[valid], y=obs_a[valid],
                width=0.09,
                marker_color=TASK_COLORS[task],
                marker_opacity=0.7,
                name=TASK_LABELS[task],
                showlegend=False,
                customdata=counts[valid],
                hovertemplate="conf=%{x:.2f}<br>acc=%{y:.2f}<br>n=%{customdata}<extra></extra>",
            ), row=row_idx, col=col_idx)

            # ECE annotation
            fig.add_annotation(
                x=0.05, y=0.92, xref=f"x{(row_idx - 1) * n_tasks + col_idx}",
                yref=f"y{(row_idx - 1) * n_tasks + col_idx}",
                text=f"ECE={ece:.3f}",
                showarrow=False, font=dict(size=9),
                bgcolor="white", bordercolor="#cccccc",
            )

    for row in [1, 2]:
        for col in range(1, n_tasks + 1):
            fig.update_xaxes(range=[0, 1], title_text="Mean confidence" if row == 2 else "",
                             row=row, col=col)
            fig.update_yaxes(range=[0, 1], title_text="Observed accuracy" if col == 1 else "",
                             row=row, col=col)

    fig.update_layout(
        template=PLOT_CONFIG["template"],
        title=dict(
            text="Reliability Diagrams (Confidence Calibration)",
            font=dict(size=13), x=0.5, xanchor="center",
        ),
        font=dict(size=PLOT_CONFIG["font_size"]),
        width=1300, height=580,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06),
        margin=dict(l=60, r=30, t=80, b=60),
    )
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    df_test = load_test_predictions()
    df_val  = load_val_predictions()
    all_data = pd.concat([df_test, df_val], ignore_index=True)
    print(f"  Test: {len(df_test)} task-sample rows  ({len(df_test)//len(TASKS)} samples × {len(TASKS)} tasks)")
    print(f"  Val:  {len(df_val)} task-sample rows  ({len(df_val)//len(TASKS)} samples × {len(TASKS)} tasks)")

    # ── Per-task analysis ─────────────────────────────────────────────────────
    results = {}
    print("\n=== CONFIDENCE ANALYSIS SUMMARY ===")
    print(f"{'':30} {'Split':6} {'n_err':>6} {'conf_corr':>9} {'conf_err':>9} {'p_val':>10}")
    print("-" * 75)

    all_pr  = {"test": {}, "val": {}}
    all_cal = {"test": {}, "val": {}}

    for split, df in [("test", df_test), ("val", df_val)]:
        results[split] = {}
        for task in TASKS:
            sub = df[df["task"] == task].copy()
            mw  = mannwhitney_test(sub)
            pr  = precision_recall_at_thresholds(sub)
            cal = calibration_stats(sub)

            all_pr[split][task]  = pr
            all_cal[split][task] = cal
            results[split][task] = {"mannwhitney": mw, "calibration": cal}

            p_str = f"{mw['p_value']:.2e}" if mw["p_value"] is not None else "N/A"
            print(f"  {TASK_LABELS[task]:28} {split:6} {mw['n_incorrect']:>6} "
                  f"{mw['mean_conf_correct']:>9.3f} {mw['mean_conf_incorrect']:>9.3f} "
                  f"{p_str:>10}")

            # Find threshold where accuracy ≥ base + 5pp with ≥ 50% retained
            for i, (t, acc, keep) in enumerate(zip(pr["thresholds"], pr["keep_accuracy"], pr["keep_frac"])):
                if not np.isnan(acc) and keep >= 0.50 and acc >= pr["base_accuracy"] + 0.05:
                    results[split][task]["threshold_50pct"] = {
                        "threshold": float(t),
                        "accuracy":  float(acc),
                        "fraction_retained": float(keep),
                    }
                    break

    print()
    for split in ["test", "val"]:
        for task in TASKS:
            t50 = results[split][task].get("threshold_50pct")
            cal = results[split][task]["calibration"]
            print(f"  [{split}] {TASK_LABELS[task]}: ECE={cal['ece']:.3f}  MCE={cal['mce']:.3f}", end="")
            if t50:
                print(f"  | threshold={t50['threshold']:.2f} → acc={t50['accuracy']:.1%} "
                      f"(keep {t50['fraction_retained']:.0%})", end="")
            print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "calibration_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nMetrics saved to {OUTPUT_DIR}/calibration_metrics.json")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    fig_A = plot_A_distributions(all_data)
    fig_A.write_html(str(FIGURES_DIR / "sup_calibration_A_confidence_distributions.html"))
    fig_A.write_image(str(FIGURES_DIR / "sup_calibration_A_confidence_distributions.png"), scale=2)
    print("  ✓ sup_calibration_A_confidence_distributions.png")

    fig_B = plot_B_flagging(all_pr)
    fig_B.write_html(str(FIGURES_DIR / "sup_calibration_B_precision_recall_flagging.html"))
    fig_B.write_image(str(FIGURES_DIR / "sup_calibration_B_precision_recall_flagging.png"), scale=2)
    print("  ✓ sup_calibration_B_precision_recall_flagging.png")

    fig_C = plot_C_reliability(all_cal)
    fig_C.write_html(str(FIGURES_DIR / "sup_calibration_C_reliability_diagrams.html"))
    fig_C.write_image(str(FIGURES_DIR / "sup_calibration_C_reliability_diagrams.png"), scale=2)
    print("  ✓ sup_calibration_C_reliability_diagrams.png")

    # ── Console: high-confidence filter benefit table ──────────────────────────
    print("\n=== SELECTIVE PREDICTION @ conf ≥ 0.90 ===")
    print(f"{'Task':28} {'Split':6} {'Retained':>9} {'Acc(all)':>9} {'Acc(≥0.9)':>10} {'Δacc':>7}")
    print("-" * 75)
    for split, df in [("test", df_test), ("val", df_val)]:
        for task in TASKS:
            sub = df[df["task"] == task]
            base_acc = sub["correct"].mean()
            kept = sub[sub["confidence"] >= 0.90]
            if len(kept) == 0:
                continue
            kept_acc = kept["correct"].mean()
            frac = len(kept) / len(sub)
            print(f"  {TASK_LABELS[task]:28} {split:6} {frac:>9.1%} "
                  f"{base_acc:>9.1%} {kept_acc:>10.1%} {kept_acc-base_acc:>+7.1%}")

    print(f"\nAll outputs in {FIGURES_DIR}/ and {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
