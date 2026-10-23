"""Supplementary balance figures (current split; material "unknown" counted as unlabelled): label distribution train vs held-out, and per dev fold.

Plotly with the manuscript's colours and template (scripts/paper/config.py PLOT_CONFIG: train and
test colours, Vivid palette), per CLAUDE.md. Five files: per task, the class share of labelled runs in
train and held-out as paired horizontal bars; and one figure of each class's training runs across the
five dev folds (stacked shares, one panel per field, classes with at least two training BioProjects).
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from config import PLOT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_v2"
MANUSCRIPT = Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA/figures")

TASKS = ["community_type", "feature", "sample_host", "material"]
COLORS = PLOT_CONFIG["colors"]
TRAIN_C, HELD_C = COLORS["train"], COLORS["test"]   # the manuscript's train / test colours
TITLES = {"community_type": "Community type", "feature": "Feature", "sample_host": "Sample host", "material": "Material"}
INK = "#333333"


def shares(d: pd.DataFrame, task: str) -> pd.Series:
    return d[task].replace("", pd.NA).dropna().value_counts(normalize=True)


def save(fig: go.Figure, name: str, height: int, width: int) -> None:
    fig.update_layout(template=PLOT_CONFIG["template"], font=dict(color=INK),
                      height=height, width=width, margin=dict(l=10, r=10, t=50, b=10))
    for d_ in (OUT, MANUSCRIPT):
        fig.write_image(d_ / name, scale=2)


def split_figure(task: str, tr: pd.DataFrame, te: pd.DataFrame) -> None:
    a, b = shares(tr, task), shares(te, task)
    classes = a.sort_values(ascending=True).index.tolist()
    classes += [c for c in b.index if c not in classes]
    n_tr = int(tr[task].replace("", pd.NA).notna().sum())
    n_te = int(te[task].replace("", pd.NA).notna().sum())
    fig = go.Figure()
    fig.add_bar(y=classes, x=[b.get(c, 0) * 100 for c in classes], orientation="h",
                name=f"held-out (n={n_te})", marker_color=HELD_C)
    fig.add_bar(y=classes, x=[a.get(c, 0) * 100 for c in classes], orientation="h",
                name=f"train (n={n_tr})", marker_color=TRAIN_C)
    fig.update_layout(barmode="group", bargap=0.25,
                      title=f"{task}: class shares in train and held-out",
                      xaxis_title="share of the split's labelled runs (%)",
                      legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98))
    save(fig, f"sup_balance_split_{task}.png",
         height=max(360, 30 * len(classes) + 140), width=760)


def folds_figure(tr: pd.DataFrame, folds: pd.DataFrame) -> None:
    """Share of each class's training runs in each dev fold, one panel per field; classes from a single
    training BioProject are left out (they cannot be split); fold colours are the palette's first five."""
    d = tr.merge(folds[["Run_accession", "fold"]], on="Run_accession")
    panels = []
    for t in TASKS:
        lab = d[d[t] != ""]
        st, n = lab.groupby(t).archive_project.nunique(), lab[t].value_counts()
        order = sorted(st[st >= 2].index, key=lambda c: (-st[c], -n[c]))
        panels.append((t, lab, order, st, n))
    fig = make_subplots(rows=len(TASKS), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[TITLES[t] for t in TASKS], row_heights=[len(p[2]) for p in panels])
    for r, (t, lab, order, st, n) in enumerate(panels, start=1):
        names = [f"{c} ({st[c]} studies, {n[c]} runs)" for c in order]
        for k in range(5):
            share = [100 * int(((lab[t] == c) & (lab.fold == k)).sum()) / n[c] for c in order]
            fig.add_bar(y=names, x=share, orientation="h", name=f"fold {k}", marker_color=COLORS["palette"][k],
                        legendgroup=f"fold {k}", showlegend=r == 1, row=r, col=1)
        fig.update_yaxes(autorange="reversed", row=r, col=1)
    fig.update_xaxes(range=[0, 100])
    fig.update_xaxes(title_text="share of the class's runs (%)", row=len(TASKS), col=1)
    fig.update_layout(barmode="stack", bargap=0.25, title="Distribution of each class's training runs across the five dev folds",
                      legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5))
    save(fig, "sup_balance_folds_stacked.png", height=40 * sum(len(p[2]) for p in panels) + 260, width=980)


def main() -> None:
    tr = pd.read_csv(ROOT / "data/splits_v19/train_metadata.tsv", sep="\t", dtype=str).fillna("")
    te = pd.read_csv(ROOT / "data/splits_v19/test_metadata.tsv", sep="\t", dtype=str).fillna("")
    folds = pd.read_csv(ROOT / "data/splits_v19/dev_folds.tsv", sep="\t")
    for t in TASKS:
        split_figure(t, tr, te)
    folds_figure(tr, folds)
    print("wrote 5 plotly figures to", OUT, "and", MANUSCRIPT)


if __name__ == "__main__":
    main()
