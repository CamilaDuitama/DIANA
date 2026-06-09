#!/usr/bin/env python3
"""
Generate Class Imbalance Overview Figure

PURPOSE:
    Grouped bar chart showing the absolute number of samples per class for
    each of the 4 classification tasks, broken down by Train / Test / Validation
    split.  Bars within each task are sorted descending by train count.

INPUTS:
    - data/splits_bioproject/train_metadata.tsv
    - data/splits_bioproject/test_metadata.tsv
    - data/splits_bioproject/validation_metadata.tsv

OUTPUTS:
    - paper/figures/final/sup_09_class_imbalance_overview.html
    - paper/figures/final/sup_09_class_imbalance_overview.png

USAGE:
    python scripts/paper/35_generate_class_imbalance_figure.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, PLOT_CONFIG

# ─── Paths ───────────────────────────────────────────────────────────────────

SPLITS_DIR = Path("data/splits_bioproject")
OUTPUT_DIR = Path(PATHS["figures_dir"])

# ─── Constants ───────────────────────────────────────────────────────────────

TASKS = ["sample_type", "community_type", "sample_host", "material"]
TASK_LABELS = {
    "sample_type":    "Sample Type",
    "community_type": "Community Type",
    "sample_host":    "Sample Host",
    "material":       "Material",
}

SPLITS = [
    ("train", "Train"),
    ("test",  "Test"),
    ("val",   "Validation"),
]

SPLIT_COLORS = {
    "train": "#E07B39",
    "test":  "#6B7DB3",
    "val":   "#7DB87A",
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_counts() -> dict[str, dict[str, pd.Series]]:
    """Return {split_key: {task: pd.Series(class -> count)}}."""
    meta = {
        "train": pd.read_csv(SPLITS_DIR / "train_metadata.tsv",      sep="\t"),
        "test":  pd.read_csv(SPLITS_DIR / "test_metadata.tsv",        sep="\t"),
        "val":   pd.read_csv(SPLITS_DIR / "validation_metadata.tsv",  sep="\t"),
    }
    counts: dict[str, dict[str, pd.Series]] = {}
    for split_key, df in meta.items():
        counts[split_key] = {}
        for task in TASKS:
            if task in df.columns:
                counts[split_key][task] = df[task].dropna().astype(str).value_counts()
            else:
                counts[split_key][task] = pd.Series(dtype=int)
    return counts


# ─── Figure builder ───────────────────────────────────────────────────────────

def build_figure(counts: dict) -> go.Figure:
    border = PLOT_CONFIG["border_color"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[TASK_LABELS[t] for t in TASKS],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    shown_in_legend: set = set()

    for idx, task in enumerate(TASKS):
        row = idx // 2 + 1
        col = idx % 2  + 1

        # Sort classes by train count descending
        train_counts = counts["train"].get(task, pd.Series(dtype=int))
        all_classes  = sorted(train_counts.index.tolist(), key=lambda c: -train_counts.get(c, 0))

        # Include any classes that appear in test/val but not train
        for split_key in ("test", "val"):
            for cls in counts[split_key].get(task, pd.Series(dtype=int)).index:
                if cls not in all_classes:
                    all_classes.append(cls)

        for split_key, split_label in SPLITS:
            series  = counts[split_key].get(task, pd.Series(dtype=int))
            y_vals  = [int(series.get(cls, 0)) for cls in all_classes]
            do_show = split_key not in shown_in_legend
            if do_show:
                shown_in_legend.add(split_key)

            fig.add_trace(
                go.Bar(
                    name=split_label,
                    x=all_classes,
                    y=y_vals,
                    marker_color=SPLIT_COLORS[split_key],
                    marker_line=dict(color=border, width=0.5),
                    opacity=0.85,
                    showlegend=do_show,
                    legendgroup=split_key,
                    hovertemplate=f"<b>{split_label}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
                ),
                row=row, col=col,
            )

        # Style axes
        fig.update_xaxes(
            tickangle=-40,
            linecolor=border,
            linewidth=1,
            row=row, col=col,
        )
        fig.update_yaxes(
            title_text="Number of samples" if col == 1 else "",
            gridcolor="#e8e8e8",
            linecolor=border,
            linewidth=1,
            row=row, col=col,
        )

    fig.update_layout(
        template=PLOT_CONFIG["template"],
        barmode="group",
        bargap=0.20,
        bargroupgap=0.05,
        font=dict(size=PLOT_CONFIG["font_size"]),
        width=1300, height=750,
        legend=dict(
            title=dict(text="Split", font=dict(size=10)),
            orientation="v",
            x=1.01, xanchor="left", y=1.0,
            font=dict(size=11),
        ),
        margin=dict(l=70, r=120, t=60, b=90),
    )
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading split metadata...")
    counts = load_counts()

    for split_key, split_label in SPLITS:
        for task in TASKS:
            n = counts[split_key].get(task, pd.Series(dtype=int)).sum()
            print(f"  [{split_label:>12}] {TASK_LABELS[task]:<18} → {n:,} samples, "
                  f"{len(counts[split_key].get(task, pd.Series(dtype=int)))} classes")

    print("\nBuilding figure...")
    fig = build_figure(counts)

    html_path = OUTPUT_DIR / "sup_09_class_imbalance_overview.html"
    png_path  = OUTPUT_DIR / "sup_09_class_imbalance_overview.png"

    fig.write_html(str(html_path))
    print(f"  ✓ {html_path}")

    fig.write_image(str(png_path), scale=2)
    print(f"  ✓ {png_path}")


if __name__ == "__main__":
    main()
