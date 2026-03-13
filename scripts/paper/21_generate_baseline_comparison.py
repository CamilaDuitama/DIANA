#!/usr/bin/env python3
"""
Generate Baseline Comparison Figure (Supplementary Figure)

PURPOSE:
    Compare DIANA's 5-fold cross-validation performance against three linear
    baseline classifiers (Logistic Regression, Linear SVM, Ridge Classifier)
    across all four classification tasks, using balanced accuracy as the
    primary metric (appropriate for class-imbalanced tasks).

    DIANA solves all four tasks jointly with a single multi-task neural
    network; baselines train four independent single-task models.

    One variant per baseline family is shown (class-balanced weighting,
    appropriate for the imbalanced label distributions in this dataset).

INPUTS:
    - results/training/cv_results/aggregated_results.json  (DIANA 5-fold CV)
    - results/baseline_comparison/aggregated_metrics.json  (baseline 5-fold CV)

OUTPUTS:
    - paper/figures/final/sup_06_baseline_comparison.html
    - paper/figures/final/sup_06_baseline_comparison.png

DEPENDENCIES:
    - pandas, plotly
    - config.py (same directory)

USAGE:
    python scripts/paper/21_generate_baseline_comparison.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, PLOT_CONFIG

# ============================================================================
# CONSTANTS
# ============================================================================

CV_RESULTS_DIANA    = Path("results/training/cv_results/aggregated_results.json")
CV_RESULTS_BASELINE = Path("results/baseline_comparison/aggregated_metrics.json")
OUTPUT_DIR          = Path(PATHS["figures_dir"])

METRIC       = "balanced_accuracy"
METRIC_LABEL = "Balanced Accuracy"

TASKS = ["sample_type", "community_type", "sample_host", "material"]
TASK_LABELS = {
    "sample_type":    "Sample Type",
    "community_type": "Community Type",
    "sample_host":    "Sample Host",
    "material":       "Material",
}

# One variant per family — keep the class-balanced versions for baselines
# since all tasks are class-imbalanced, consistent with DIANA's weighted loss
MODELS = [
    "DIANA",
    "LogisticRegression_Bal",
    "LinearSVM_Bal",
    "RidgeClassifier_Bal",
]
MODEL_LABELS = {
    "DIANA":                  "DIANA (multi-task)",
    "LogisticRegression_Bal": "Logistic Regression",
    "LinearSVM_Bal":          "Linear SVM",
    "RidgeClassifier_Bal":    "Ridge Classifier",
}

VIVID = px.colors.qualitative.Vivid
MODEL_COLORS = {
    "DIANA":                  VIVID[0],   # blue
    "LogisticRegression_Bal": VIVID[1],   # orange
    "LinearSVM_Bal":          VIVID[2],   # green
    "RidgeClassifier_Bal":    VIVID[3],   # red
}
MARKER_SIZE = 14

# ============================================================================
# LOAD DATA
# ============================================================================

def get_mean_std(v):
    if isinstance(v, dict):
        return float(v.get("mean", list(v.values())[0])), float(v.get("std", 0.0))
    return float(v), 0.0


def load_results():
    with open(CV_RESULTS_DIANA) as f:
        diana_raw = json.load(f)
    with open(CV_RESULTS_BASELINE) as f:
        baseline_raw = json.load(f)

    rows = []

    diana_agg = diana_raw["aggregated_metrics"]
    for task in TASKS:
        mean, std = get_mean_std(diana_agg[task][METRIC])
        rows.append({"model": "DIANA", "task": task, "mean": mean, "std": std})

    for model in MODELS[1:]:
        for task in TASKS:
            mean, std = get_mean_std(baseline_raw[model][task][METRIC])
            rows.append({"model": model, "task": task, "mean": mean, "std": std})

    return pd.DataFrame(rows)


# ============================================================================
# BUILD FIGURE
# ============================================================================

def build_figure(df):
    fig = go.Figure()

    border  = PLOT_CONFIG["border_color"]
    opacity = min(PLOT_CONFIG["fill_opacity"] + 0.10, 1.0)

    # Order tasks by DIANA performance descending
    diana_means = (
        df[df["model"] == "DIANA"]
        .set_index("task")["mean"]
        .reindex(TASKS)
    )
    sorted_tasks   = diana_means.sort_values(ascending=False).index.tolist()
    sorted_x_labels = [TASK_LABELS[t] for t in sorted_tasks]

    for idx, model in enumerate(MODELS):
        sub = df[df["model"] == model].set_index("task")

        x_vals, y_vals, y_err = [], [], []
        for task in sorted_tasks:
            if task in sub.index:
                x_vals.append(TASK_LABELS[task])
                y_vals.append(sub.loc[task, "mean"])
                y_err.append(sub.loc[task, "std"])

        fig.add_trace(go.Scatter(
            name=MODEL_LABELS[model],
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                symbol="circle",
                size=MARKER_SIZE,
                color=MODEL_COLORS[model],
                opacity=opacity,
                line=dict(color=border, width=1.5),
            ),
            error_y=dict(
                type="data",
                array=y_err,
                visible=True,
                color=MODEL_COLORS[model],
                thickness=2,
                width=8,
            ),
            offsetgroup=str(idx),
        ))

    fig.update_layout(
        template=PLOT_CONFIG["template"],
        scattermode="group",
        scattergap=0.60,
        font=dict(size=PLOT_CONFIG["font_size"]),
        width=880,
        height=500,
        xaxis=dict(
            categoryorder="array",
            categoryarray=sorted_x_labels,
            title="",
            linecolor=border,
            linewidth=1,
        ),
        yaxis=dict(
            title=METRIC_LABEL,
            range=[0.60, 1.02],
            tickformat=".2f",
            gridcolor="#e8e8e8",
            linecolor=border,
            linewidth=1,
        ),
        legend=dict(
            title="",
            orientation="v",
            x=1.01,
            xanchor="left",
            y=1.0,
            font=dict(size=11),
        ),
        margin=dict(l=65, r=190, t=30, b=55),
    )

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()

    print(f"=== 5-fold CV mean {METRIC} ===")
    for task in TASKS:
        print(f"\n  [{TASK_LABELS[task]}]")
        sub = df[df["task"] == task].sort_values("mean", ascending=False)
        for _, row in sub.iterrows():
            flag = "  <--" if row["model"] == "DIANA" else ""
            print(f"    {MODEL_LABELS[row['model']]:<30} {row['mean']:.3f} +/- {row['std']:.3f}{flag}")

    fig = build_figure(df)

    html_path = OUTPUT_DIR / "sup_06_baseline_comparison.html"
    png_path  = OUTPUT_DIR / "sup_06_baseline_comparison.png"

    fig.write_html(str(html_path))
    print(f"\nSaved HTML -> {html_path}")

    fig.write_image(str(png_path), scale=2)
    print(f"Saved PNG  -> {png_path}")


if __name__ == "__main__":
    main()
