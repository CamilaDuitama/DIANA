#!/usr/bin/env python3
"""
Generate Generalisation Gap Slopegraphs (Test → Validation)

PURPOSE:
    Slopegraphs showing how each model's performance changes from the held-out
    test set to the external validation set.  One panel per classification task.
    Each line is one model; steep drops make overfitting immediately visible.

    Two figures are generated:
      A — Balanced accuracy slopegraph
      B — F1† (seen-class macro F1) slopegraph

INPUTS:
    results/baseline_comparison_bioproject/metrics.json   (from script 08)

OUTPUTS:
    paper/figures/final/sup_06b_generalisation_gap_balanced_accuracy.html/png
    paper/figures/final/sup_06b_generalisation_gap_f1.html/png

USAGE:
    python scripts/paper/34_generate_generalisation_gap_plots.py
"""

import json
import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, PLOT_CONFIG

# ─── Paths ────────────────────────────────────────────────────────────────────

BIOPROJECT_METRICS = Path("results/baseline_comparison_bioproject/metrics.json")
OUTPUT_DIR = Path(PATHS["figures_dir"])

# ─── Constants ────────────────────────────────────────────────────────────────

TASKS = ["sample_type", "community_type", "sample_host", "material"]
TASK_LABELS = {
    "sample_type":    "Sample Type",
    "community_type": "Community Type",
    "sample_host":    "Sample Host",
    "material":       "Material",
}

MODELS = [
    "DIANA",
    "MajorityClass",
    "LogisticRegression_Bal",
    "LinearSVM_Bal",
    "RandomForest_Bal",
]
DISPLAY_NAMES = {
    "DIANA":                  "DIANA",
    "MajorityClass":          "Majority Class",
    "LogisticRegression_Bal": "Logistic Regression",
    "LinearSVM_Bal":          "Linear SVM",
    "RandomForest_Bal":       "Random Forest",
}

VIVID = px.colors.qualitative.Vivid
MODEL_COLORS = {
    "DIANA":                  VIVID[0],
    "MajorityClass":          "#aaaaaa",
    "LogisticRegression_Bal": VIVID[1],
    "LinearSVM_Bal":          VIVID[2],
    "RandomForest_Bal":       VIVID[4],
}
MODEL_WIDTHS = {
    "DIANA":                  3.5,
    "MajorityClass":          1.2,
    "LogisticRegression_Bal": 1.8,
    "LinearSVM_Bal":          1.8,
    "RandomForest_Bal":       1.8,
}
MODEL_OPACITY = {
    "DIANA":                  1.0,
    "MajorityClass":          0.45,
    "LogisticRegression_Bal": 0.65,
    "LinearSVM_Bal":          0.65,
    "RandomForest_Bal":       0.65,
}


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_slopegraph(
    raw: dict,
    metric_key: str,
    title: str,
    ylabel: str,
) -> go.Figure:
    """Build a 4-panel slopegraph (one panel per task) for *metric_key*.

    metric_key must match a key in the per-task dicts inside metrics.json
    (e.g. 'balanced_accuracy' or 'f1_macro_seen').
    """
    # Collect values and confidence intervals
    test_v: dict = {}
    val_v:  dict = {}
    test_ci_low: dict = {}
    test_ci_high: dict = {}
    val_ci_low: dict = {}
    val_ci_high: dict = {}
    
    ci_key_low = f"{metric_key}_ci_low"
    ci_key_high = f"{metric_key}_ci_high"
    
    for task in TASKS:
        test_v[("DIANA", task)] = raw["diana"][task].get(metric_key, float("nan")) * 100
        val_v[("DIANA",  task)] = raw["diana_val"][task].get(metric_key, float("nan")) * 100
        test_ci_low[("DIANA", task)] = raw["diana"][task].get(ci_key_low, float("nan")) * 100
        test_ci_high[("DIANA", task)] = raw["diana"][task].get(ci_key_high, float("nan")) * 100
        val_ci_low[("DIANA", task)] = raw["diana_val"][task].get(ci_key_low, float("nan")) * 100
        val_ci_high[("DIANA", task)] = raw["diana_val"][task].get(ci_key_high, float("nan")) * 100
    
    for model in MODELS[1:]:
        for task in TASKS:
            test_v[(model, task)] = (
                raw["baselines_test"].get(model, {}).get(task, {}).get(metric_key, float("nan")) * 100
            )
            val_v[(model, task)] = (
                raw["baselines_val"].get(model, {}).get(task, {}).get(metric_key, float("nan")) * 100
            )
            test_ci_low[(model, task)] = (
                raw["baselines_test"].get(model, {}).get(task, {}).get(ci_key_low, float("nan")) * 100
            )
            test_ci_high[(model, task)] = (
                raw["baselines_test"].get(model, {}).get(task, {}).get(ci_key_high, float("nan")) * 100
            )
            val_ci_low[(model, task)] = (
                raw["baselines_val"].get(model, {}).get(task, {}).get(ci_key_low, float("nan")) * 100
            )
            val_ci_high[(model, task)] = (
                raw["baselines_val"].get(model, {}).get(task, {}).get(ci_key_high, float("nan")) * 100
            )

    border = PLOT_CONFIG["border_color"]
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[TASK_LABELS[t] for t in TASKS],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    shown: set = set()
    for col, task in enumerate(TASKS, start=1):
        for model in MODELS:
            t_val = test_v.get((model, task), float("nan"))
            v_val = val_v.get((model, task), float("nan"))
            
            # Compute error bars (asymmetric)
            t_ci_low = test_ci_low.get((model, task), float("nan"))
            t_ci_high = test_ci_high.get((model, task), float("nan"))
            v_ci_low = val_ci_low.get((model, task), float("nan"))
            v_ci_high = val_ci_high.get((model, task), float("nan"))
            
            # Error arrays: distance from point to error bar
            error_y_array = [t_ci_high - t_val, v_ci_high - v_val]
            error_y_arrayminus = [t_val - t_ci_low, v_val - v_ci_low]
            
            is_diana   = model == "DIANA"
            do_legend  = model not in shown
            if do_legend:
                shown.add(model)

            fig.add_trace(
                go.Scatter(
                    x=["Test", "Val"],
                    y=[t_val, v_val],
                    mode="lines+markers",
                    line=dict(color=MODEL_COLORS[model], width=MODEL_WIDTHS[model]),
                    marker=dict(
                        size=9 if is_diana else 6,
                        color=MODEL_COLORS[model],
                        line=dict(color=border, width=1.5 if is_diana else 0.5),
                    ),
                    error_y=dict(
                        type='data',
                        array=error_y_array,
                        arrayminus=error_y_arrayminus,
                        color=MODEL_COLORS[model],
                        thickness=1.5 if is_diana else 1.0,
                        width=4 if is_diana else 3,
                    ),
                    opacity=MODEL_OPACITY[model],
                    name=DISPLAY_NAMES[model],
                    showlegend=do_legend,
                    legendgroup=model,
                    customdata=[[t_ci_low, t_ci_high], [v_ci_low, v_ci_high]],
                    hovertemplate=(
                        f"<b>{DISPLAY_NAMES[model]}</b><br>"
                        "%{x}: %{y:.1f}%<br>"
                        "95% CI: [%{customdata[0]:.1f}, %{customdata[1]:.1f}]%<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )

        fig.update_xaxes(linecolor=border, linewidth=1, row=1, col=col)
        # Dotted reference at chance (50%)
        fig.add_hline(y=50, line_dash="dot", line_color="#cccccc", line_width=1, row=1, col=col)

    fig.update_yaxes(
        title_text=ylabel, range=[0, 100],
        ticksuffix="%", gridcolor="#e8e8e8",
        linecolor=border, linewidth=1,
        row=1, col=1,
    )
    for col in range(2, 5):
        fig.update_yaxes(range=[0, 100], row=1, col=col)

    fig.update_layout(
        template=PLOT_CONFIG["template"],
        font=dict(size=PLOT_CONFIG["font_size"]),
        width=1100, height=430,
        legend=dict(
            title=dict(
                text="<i>Baselines trained with<br>class-balanced weighting</i>",
                font=dict(size=9),
            ),
            orientation="v",
            x=1.01, xanchor="left", y=1.0,
            font=dict(size=10),
        ),
        margin=dict(l=65, r=170, t=50, b=50),
        title=dict(text=title, font=dict(size=13), x=0.44, xanchor="center"),
    )
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not BIOPROJECT_METRICS.exists():
        print(f"ERROR: {BIOPROJECT_METRICS} not found — run 08_test_set_baseline_comparison.py first")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BIOPROJECT_METRICS) as fh:
        raw = json.load(fh)

    # ── Balanced accuracy ──────────────────────────────────────────────────────
    fig_ba = build_slopegraph(
        raw,
        metric_key="balanced_accuracy",
        title="Generalisation: Test \u2192 Validation  (Balanced Accuracy)",
        ylabel="Balanced Accuracy (%)",
    )
    stem = "sup_06b_generalisation_gap_balanced_accuracy"
    fig_ba.write_html(str(OUTPUT_DIR / f"{stem}.html"))
    fig_ba.write_image(str(OUTPUT_DIR / f"{stem}.png"), scale=2)
    print(f"  \u2713 {stem}.png")

    # ── F1† ───────────────────────────────────────────────────────────────────
    fig_f1 = build_slopegraph(
        raw,
        metric_key="f1_macro_seen",
        title="Generalisation: Test \u2192 Validation  (F1\u2020 seen-class macro F1)",
        ylabel="F1\u2020 (%)",
    )
    stem = "sup_06b_generalisation_gap_f1"
    fig_f1.write_html(str(OUTPUT_DIR / f"{stem}.html"))
    fig_f1.write_image(str(OUTPUT_DIR / f"{stem}.png"), scale=2)
    print(f"  \u2713 {stem}.png")

    print(f"\nAll outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
