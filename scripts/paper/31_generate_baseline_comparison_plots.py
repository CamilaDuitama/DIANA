#!/usr/bin/env python3
"""
Generate Baseline Comparison Figures — 3 plots (Training CV / Test / Validation)

PURPOSE:
    Three side-by-side dot-plots comparing DIANA vs baseline classifiers using
    balanced accuracy and F1† (seen-class macro F1) as primary metrics.

    Plot A — Training (5-fold CV, old script data): shows within-training-set
              performance, kept for reference / reviewer context.
    Plot B — Test set (BioProject-disjoint held-out, n=523): main comparison.
    Plot C — Validation set (external, n=360): generalization comparison.

    The DIANA test/validation points come from diana-test evaluation.
    Baseline test/validation points come from 08_test_set_baseline_comparison.py.

INPUTS:
    - results/baseline_comparison_bioproject/metrics.json   (baselines + DIANA test)
    - results/baseline_comparison/aggregated_metrics.json   (old CV baselines)
    - results/training/cv_results/aggregated_results.json   (DIANA 5-fold CV)

OUTPUTS:
    - paper/figures/final/sup_06_baseline_comparison_test.html/png
    - paper/figures/final/sup_06_baseline_comparison_val.html/png
    - paper/figures/final/sup_06_baseline_comparison_cv.html/png
    - paper/figures/final/sup_06_baseline_comparison_combined.html/png

USAGE:
    python scripts/paper/31_generate_baseline_comparison_plots.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, PLOT_CONFIG

# ─── Paths ───────────────────────────────────────────────────────────────────

BIOPROJECT_METRICS = Path("results/baseline_comparison_bioproject/metrics.json")
CV_BASELINE        = Path("results/baseline_comparison/aggregated_metrics.json")
CV_DIANA           = Path("results/training/cv_results/aggregated_results.json")
VAL_META           = Path("data/splits_bioproject/validation_metadata.tsv")
VAL_PRED_DIR       = Path("results/validation_predictions_bioproject")
OUTPUT_DIR         = Path(PATHS["figures_dir"])

# ─── Constants ───────────────────────────────────────────────────────────────

TASKS = ["sample_type", "community_type", "sample_host", "material"]
TASK_LABELS = {
    "sample_type":    "Sample Type",
    "community_type": "Community Type",
    "sample_host":    "Sample Host",
    "material":       "Material",
}

# One representative per model family (balanced where applicable) + DIANA + trivial baseline
MODELS_EVAL = [
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
# DIANA uses the paper's primary colour; baselines use Vivid colours at reduced opacity.
MODEL_COLORS = {
    "DIANA":                  VIVID[0],
    "MajorityClass":          "#aaaaaa",
    "LogisticRegression_Bal": VIVID[1],
    "LinearSVM_Bal":          VIVID[2],
    "RandomForest_Bal":       VIVID[4],
}
# Opacity: DIANA stands out at full opacity; baselines are dimmed.
MODEL_OPACITY = {
    "DIANA":                  0.90,
    "MajorityClass":          0.50,
    "LogisticRegression_Bal": 0.55,
    "LinearSVM_Bal":          0.55,
    "RandomForest_Bal":       0.55,
}

METRIC = "balanced_accuracy"


# ─── Loaders ─────────────────────────────────────────────────────────────────

def compute_diana_val_metrics() -> dict:
    """
    Compute DIANA validation metrics from per-sample prediction JSONs.
    Returns {task: {balanced_accuracy, f1_macro_seen, accuracy}} matching
    the format of the 'diana' key in the bioproject metrics.json.
    """
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

    val_meta = pd.read_csv(VAL_META, sep="\t")
    task_preds = {t: {"y_true": [], "y_pred": []} for t in TASKS}

    for _, row in val_meta.iterrows():
        sid = row["Run_accession"]
        fpath = VAL_PRED_DIR / sid / f"{sid}_predictions.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            d = json.load(f)
        for task in TASKS:
            if task not in d.get("predictions", {}):
                continue
            pred = d["predictions"][task]["predicted_class"]
            true = str(row.get(task, ""))
            task_preds[task]["y_true"].append(true)
            task_preds[task]["y_pred"].append(pred)

    result = {}
    for task in TASKS:
        y_true = np.array(task_preds[task]["y_true"])
        y_pred = np.array(task_preds[task]["y_pred"])
        seen   = np.unique(y_true)
        result[task] = {
            "accuracy":          float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro_seen":     float(f1_score(y_true, y_pred, labels=seen,
                                                average="macro", zero_division=0)),
        }
    return result


def load_eval_data() -> dict[str, pd.DataFrame]:
    """Load test & validation results. DIANA val comes from metrics.json['diana_val']
    (populated by script 08 with bootstrap CI) or falls back to per-sample JSONs."""
    with open(BIOPROJECT_METRICS) as f:
        raw = json.load(f)

    diana_test = raw["diana"]
    # Prefer pre-computed val metrics with CI from script 08; fall back to recomputing
    diana_val = raw.get("diana_val") or compute_diana_val_metrics()

    frames = {}
    for split in ("test", "val"):
        rows = []
        diana_src = diana_test if split == "test" else diana_val
        for task in TASKS:
            d = diana_src.get(task, {})
            rows.append({
                "model":       "DIANA", "task": task,
                "mean":        d.get(METRIC, float("nan")),
                "f1_seen":     d.get("f1_macro_seen", float("nan")),
                "ci_low":      d.get(f"{METRIC}_ci_low",      float("nan")),
                "ci_high":     d.get(f"{METRIC}_ci_high",     float("nan")),
                "f1_ci_low":   d.get("f1_macro_seen_ci_low",  float("nan")),
                "f1_ci_high":  d.get("f1_macro_seen_ci_high", float("nan")),
            })
        key = f"baselines_{split}"
        if key not in raw:
            key = "baselines"
        for model, task_dict in raw.get(key, {}).items():
            if model not in MODELS_EVAL:
                continue
            for task in TASKS:
                m = task_dict.get(task, {})
                rows.append({
                    "model":      model, "task": task,
                    "mean":       m.get(METRIC, float("nan")),
                    "f1_seen":    m.get("f1_macro_seen", float("nan")),
                    "ci_low":     m.get(f"{METRIC}_ci_low",      float("nan")),
                    "ci_high":    m.get(f"{METRIC}_ci_high",     float("nan")),
                    "f1_ci_low":  m.get("f1_macro_seen_ci_low",  float("nan")),
                    "f1_ci_high": m.get("f1_macro_seen_ci_high", float("nan")),
                })
        frames[split] = pd.DataFrame(rows)
    return frames


def load_cv_data() -> pd.DataFrame:
    """Load 5-fold CV data (DIANA + old linear baselines)."""
    rows = []

    if CV_DIANA.exists():
        with open(CV_DIANA) as f:
            diana_raw = json.load(f)
        agg = diana_raw.get("aggregated_metrics", {})
        for task in TASKS:
            v = agg.get(task, {}).get(METRIC, {})
            mean = float(v["mean"]) if isinstance(v, dict) else float(v)
            std  = float(v.get("std", 0.0)) if isinstance(v, dict) else 0.0
            rows.append({"model": "DIANA", "task": task, "mean": mean, "std": std, "f1_seen": float("nan")})

    if CV_BASELINE.exists():
        with open(CV_BASELINE) as f:
            bl_raw = json.load(f)
        for model in MODELS_CV[1:]:
            if model not in bl_raw:
                continue
            for task in TASKS:
                v = bl_raw[model].get(task, {}).get(METRIC, {})
                mean = float(v["mean"]) if isinstance(v, dict) else float(v)
                std  = float(v.get("std", 0.0)) if isinstance(v, dict) else 0.0
                rows.append({"model": model, "task": task, "mean": mean, "std": std, "f1_seen": float("nan")})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["model", "task", "mean", "std", "f1_seen"]
    )


# ─── Plot builder ─────────────────────────────────────────────────────────────

def build_bar_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    models: list[str],
    row: int,
    col: int,
    y_col: str = "mean",           # "mean" (bal_acc) or "f1_seen" (F1†)
    y_label: str = "Balanced Accuracy",
    show_legend: bool = True,
    shown_in_legend: set | None = None,
) -> None:
    """Add grouped bar traces for one split panel into an existing subplot figure.

    DIANA bars are rendered at full opacity and with a heavier border to stand
    out from the baseline bars, which are drawn at reduced opacity.
    """
    if shown_in_legend is None:
        shown_in_legend = set()
    border = PLOT_CONFIG["border_color"]

    task_labels = [TASK_LABELS[t] for t in TASKS]

    for model in models:
        sub = df[df["model"] == model].set_index("task")
        y_vals, err_low, err_high, hover = [], [], [], []
        for task in TASKS:
            if task not in sub.index:
                y_vals.append(0.0); err_low.append(0.0); err_high.append(0.0)
                hover.append("")
                continue
            r = sub.loc[task]
            v_bal = r["mean"]    * 100 if pd.notna(r.get("mean"))     else float("nan")
            v_f1  = r["f1_seen"] * 100 if pd.notna(r.get("f1_seen")) else float("nan")
            y_val = v_bal if y_col == "mean" else v_f1
            y_vals.append(y_val if not np.isnan(y_val) else 0.0)

            # Error bar: asymmetric CI (high - mean, mean - low)
            ci_lo_col = "ci_low"   if y_col == "mean" else "f1_ci_low"
            ci_hi_col = "ci_high"  if y_col == "mean" else "f1_ci_high"
            ci_lo = r.get(ci_lo_col, float("nan"))
            ci_hi = r.get(ci_hi_col, float("nan"))
            if pd.notna(ci_lo) and pd.notna(ci_hi) and not np.isnan(ci_lo):
                base = v_bal if y_col == "mean" else v_f1
                err_low.append(max(base - ci_lo * 100, 0.0))
                err_high.append(max(ci_hi * 100 - base, 0.0))
            else:
                err_low.append(0.0)
                err_high.append(0.0)

            ci_str = (
                f" [{ci_lo*100:.1f}–{ci_hi*100:.1f}%]"
                if pd.notna(ci_lo) and not np.isnan(ci_lo) else ""
            )
            hover.append(
                f"<b>{DISPLAY_NAMES.get(model, model)}</b><br>"
                f"Bal. Acc: {v_bal:.1f}%{ci_str}<br>F1\u2020: {v_f1:.1f}%"
            )

        is_diana  = model == "DIANA"
        opacity   = MODEL_OPACITY.get(model, 0.55)
        lw        = 2.0 if is_diana else 0.5

        do_show = show_legend and (model not in shown_in_legend)
        if do_show:
            shown_in_legend.add(model)

        has_ci = any(v > 0.0 for v in err_high)

        fig.add_trace(
            go.Bar(
                name=DISPLAY_NAMES.get(model, model),
                x=task_labels,
                y=y_vals,
                marker_color=MODEL_COLORS.get(model, "#888888"),
                marker_line=dict(color=border, width=lw),
                opacity=opacity,
                showlegend=do_show,
                legendgroup=model,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_high,
                    arrayminus=err_low,
                    visible=has_ci,
                    color=border,
                    thickness=1.2,
                    width=4,
                ) if has_ci else dict(visible=False),
            ),
            row=row, col=col,
        )


def build_combined_figure(
    frames: dict,
    y_col: str = "mean",
    y_label: str = "Balanced Accuracy (%)",
) -> go.Figure:
    """2-panel grouped bar chart: Test set | Validation set.

    Parameters
    ----------
    y_col    : column to use for bar height — 'mean' (balanced accuracy) or 'f1_seen' (F1†)
    y_label  : y-axis title
    """
    panels = [
        ("Test set (n\u202f=\u202f523)",       "test"),
        ("Validation set (n\u202f=\u202f360)",  "val"),
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[p[0] for p in panels],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    border = PLOT_CONFIG["border_color"]
    shown: set = set()

    for col_idx, (_, split) in enumerate(panels, start=1):
        build_bar_panel(
            fig, frames[split], MODELS_EVAL,
            row=1, col=col_idx,
            y_col=y_col, y_label=y_label,
            show_legend=True, shown_in_legend=shown,
        )
        fig.update_xaxes(linecolor=border, linewidth=1, row=1, col=col_idx)

    fig.update_yaxes(
        title_text=y_label, range=[0, 105],
        ticksuffix="%", gridcolor="#e8e8e8",
        linecolor=border, linewidth=1,
        row=1, col=1,
    )
    fig.update_yaxes(range=[0, 105], row=1, col=2)

    fig.update_layout(
        template=PLOT_CONFIG["template"],
        barmode="group",
        bargap=0.18,
        bargroupgap=0.06,
        font=dict(size=PLOT_CONFIG["font_size"]),
        width=1100, height=480,
        legend=dict(
            title="", orientation="v", x=1.01, xanchor="left", y=1.0,
            font=dict(size=10),
        ),
        margin=dict(l=65, r=180, t=55, b=55),
    )
    return fig


# ─── LaTeX table ──────────────────────────────────────────────────────────────

def _fmt(val: float) -> str:
    return f"{val*100:.1f}" if not np.isnan(val) else "--"


def generate_latex_table(frames: dict, out_path: Path) -> None:
    """Write sup_table_11_baseline_comparison.tex.

    Rows = models, columns = tasks × {bal_acc, F1†}.
    Two midrule-separated sections: Test | Validation.
    """
    tasks_order = TASKS
    task_short = {
        "sample_type":    "Sample Type",
        "community_type": "Community",
        "sample_host":    "Sample Host",
        "material":       "Material",
    }
    n_cols = len(tasks_order) * 2 + 1   # model col + 2 metrics per task

    header_tasks = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{task_short[t]}}}" for t in tasks_order
    )
    header_metrics = " & ".join(["Bal.\\,Acc & F1$^\\dagger$"] * len(tasks_order))
    cmidrule_str   = " ".join(
        f"\\cmidrule(lr){{{2+i*2}-{3+i*2}}}" for i in range(len(tasks_order))
    )
    col_spec = "l" + "rr" * len(tasks_order)

    lines = [
        r"\centering",
        r"\caption{Comparison of DIANA against baseline classifiers on the held-out"
        r" test set ($n=523$) and the external validation set ($n=360$)."
        r" Bal.\,Acc\ = balanced accuracy; F1$^\dagger$ = macro-averaged F1 restricted"
        r" to classes observed in the respective split."
        r" Bold values indicate the best score per column and split.}",
        r"\label{tab:baseline_comparison}",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Model & {header_tasks} \\\\",
        cmidrule_str,
        f"& {header_metrics} \\\\",
        r"\midrule",
    ]

    for split_name, split_key in [("Test set", "test"), ("Validation set", "val")]:
        lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{split_name}}}}} \\\\")
        df = frames[split_key]

        # Collect values to find per-column bests
        vals: dict[tuple, float] = {}
        for model in MODELS_EVAL:
            sub = df[df["model"] == model].set_index("task")
            for task in tasks_order:
                v = sub.loc[task, "mean"]    if task in sub.index else float("nan")
                f = sub.loc[task, "f1_seen"] if task in sub.index else float("nan")
                vals[(model, task, "bal")] = v
                vals[(model, task, "f1")]  = f

        best_bal = {t: max((vals.get((m, t, "bal"), float("-inf")) for m in MODELS_EVAL), default=float("nan"))
                    for t in tasks_order}
        best_f1  = {t: max((vals.get((m, t, "f1"),  float("-inf")) for m in MODELS_EVAL), default=float("nan"))
                    for t in tasks_order}

        for model in MODELS_EVAL:
            sub = df[df["model"] == model].set_index("task")
            cells = []
            for task in tasks_order:
                v  = vals.get((model, task, "bal"), float("nan"))
                f  = vals.get((model, task, "f1"),  float("nan"))
                v_lo = sub.loc[task, "ci_low"]    * 100 if task in sub.index and pd.notna(sub.loc[task].get("ci_low"))    and not np.isnan(sub.loc[task].get("ci_low", float("nan")))    else None
                v_hi = sub.loc[task, "ci_high"]   * 100 if task in sub.index and pd.notna(sub.loc[task].get("ci_high"))   and not np.isnan(sub.loc[task].get("ci_high", float("nan")))   else None
                f_lo = sub.loc[task, "f1_ci_low"]  * 100 if task in sub.index and pd.notna(sub.loc[task].get("f1_ci_low"))  and not np.isnan(sub.loc[task].get("f1_ci_low", float("nan")))  else None
                f_hi = sub.loc[task, "f1_ci_high"] * 100 if task in sub.index and pd.notna(sub.loc[task].get("f1_ci_high")) and not np.isnan(sub.loc[task].get("f1_ci_high", float("nan"))) else None

                def _bold(val, best):
                    return f"\\mathbf{{{_fmt(val)}}}" if abs(val - best) < 1e-9 else _fmt(val)

                vs_val = _bold(v, best_bal[task])
                fs_val = _bold(f, best_f1[task])

                if v_lo is not None and v_hi is not None:
                    vs = f"${vs_val}_{{[{v_lo:.1f},{v_hi:.1f}]}}$"
                else:
                    vs = f"${vs_val}$"
                if f_lo is not None and f_hi is not None:
                    fs = f"${fs_val}_{{[{f_lo:.1f},{f_hi:.1f}]}}$"
                else:
                    fs = f"${fs_val}$"
                cells += [vs, fs]
            dname = DISPLAY_NAMES.get(model, model).replace("_", "\_")
            lines.append(f"{dname} & " + " & ".join(cells) + " \\\\")

        if split_key == "test":
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  ✓ {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BIOPROJECT_METRICS.exists():
        print(f"ERROR: {BIOPROJECT_METRICS} not found — run 08_test_set_baseline_comparison.py first")
        sys.exit(1)

    TABLES_DIR = Path(PATHS["tables_dir"])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    frames = load_eval_data()

    # ── Balanced accuracy figure ──────────────────────────────────────────────
    fig_bal = build_combined_figure(
        frames, y_col="mean", y_label="Balanced Accuracy (%)"
    )
    fig_bal.write_html(str(OUTPUT_DIR / "sup_06_baseline_comparison_balanced_accuracy.html"))
    fig_bal.write_image(str(OUTPUT_DIR / "sup_06_baseline_comparison_balanced_accuracy.png"), scale=2)
    print("  ✓ sup_06_baseline_comparison_balanced_accuracy.png")

    # ── F1† figure ────────────────────────────────────────────────────────────
    fig_f1 = build_combined_figure(
        frames, y_col="f1_seen", y_label="F1† — seen-class macro F1 (%)"
    )
    fig_f1.write_html(str(OUTPUT_DIR / "sup_06_baseline_comparison_f1.html"))
    fig_f1.write_image(str(OUTPUT_DIR / "sup_06_baseline_comparison_f1.png"), scale=2)
    print("  ✓ sup_06_baseline_comparison_f1.png")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    generate_latex_table(frames, TABLES_DIR / "sup_table_11_baseline_comparison.tex")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== BALANCED ACCURACY SUMMARY ===")
    for split, label in [("test", "TEST"), ("val", "VALIDATION")]:
        df = frames[split]
        print(f"\n  [{label}]")
        print(f"  {'Model':<35} " + "  ".join(f"{TASK_LABELS[t]:<16}" for t in TASKS))
        print("  " + "-" * 100)
        for model in MODELS_EVAL:
            sub = df[df["model"] == model].set_index("task")
            vals = []
            for task in TASKS:
                v = sub.loc[task, "mean"] if task in sub.index else float("nan")
                f = sub.loc[task, "f1_seen"] if task in sub.index else float("nan")
                vals.append(f"{v*100:.1f}/{f*100:.1f}%" if not np.isnan(v) else "  N/A")
            print(f"  {DISPLAY_NAMES.get(model, model):<35} " +
                  "  ".join(f"{v:<16}" for v in vals))
    print("\n(format: bal_acc/f1† %)")
    print(f"\nAll outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
