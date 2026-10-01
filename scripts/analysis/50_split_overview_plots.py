#!/usr/bin/env python3
"""Plots of the v9 split: runs per BioProject by split, and runs per class by split with
the training-support regimes.

Reads data/splits_v9/{train_metadata,test_metadata,class_eligibility}.tsv only. One plot
per file, results/split_overview_v9/:
  runs_per_bioproject.png            one bar per BioProject, coloured train / held-out
  runs_per_class_<task>.png          per class: training runs and held-out runs, with the
                                     regime cut-offs (20 and 100 training runs) and the
                                     eligibility rule (>= 2 BioProjects) marked
  regime_summary.tsv                 classes and runs per task and regime

    ./env/bin/python scripts/analysis/50_split_overview_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/split_overview_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
TRAIN, HELD = "#2a78d6", "#eb6834"   # dataviz reference palette, slots 1 and 2 (validated)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e2"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_projects(train: pd.DataFrame, test: pd.DataFrame) -> None:
    a = train.groupby("archive_project").size().rename("runs").reset_index().assign(split="train")
    b = test.groupby("archive_project").size().rename("runs").reset_index().assign(split="held-out")
    d = pd.concat([a, b]).sort_values("runs", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    colours = d.split.map({"train": TRAIN, "held-out": HELD})
    ax.bar(np.arange(len(d)), d.runs, color=colours, width=0.85)
    ax.set_yscale("log")
    ax.set_xlim(-1, len(d))
    ax.set_xticks([])
    ax.set_xlabel(f"BioProject, sorted by number of runs ({len(d)} projects)")
    ax.set_ylabel("runs")
    ax.set_title("Runs per BioProject, train and held-out (v9 split)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TRAIN, label=f"train: {a.shape[0]} BioProjects, {a.runs.sum():,} runs"),
                       Patch(color=HELD, label=f"held-out: {b.shape[0]} BioProjects, {b.runs.sum():,} runs")],
              frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "runs_per_bioproject.png", dpi=200)
    plt.close(fig)


def plot_classes(task: str, train: pd.DataFrame, test: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    tr = train[task].dropna().astype(str).value_counts().rename("train_runs")
    te = test[task].dropna().astype(str).value_counts().rename("heldout_runs")
    ok = set(elig[(elig.target == task) & elig.evaluable]["class"].astype(str))
    d = pd.concat([tr, te], axis=1).fillna(0).astype(int)
    d["train_projects"] = [train.loc[train[task].astype(str) == c, "archive_project"].nunique() for c in d.index]
    d["heldout_projects"] = [test.loc[test[task].astype(str) == c, "archive_project"].nunique() for c in d.index]
    d["eligible"] = [c in ok for c in d.index]
    d = d.sort_values("train_runs", ascending=False)
    d["regime"] = [next((n for n, lo, hi in REGIMES if lo < v <= hi), "no training runs") for v in d.train_runs]

    n = len(d)
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * n + 2), 4.8))
    x = np.arange(n)
    ax.bar(x - 0.2, d.train_runs.clip(lower=0.5), width=0.4, color=TRAIN, label="training runs")
    ax.bar(x + 0.2, d.heldout_runs.clip(lower=0.5), width=0.4, color=HELD, label="held-out runs")
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(d.train_runs.max(), d.heldout_runs.max()) * 3)
    for cut, name in ((20, "20 training runs"), (100, "100 training runs")):
        ax.axhline(cut, color=MUTED, lw=0.9, ls="--")
        ax.text(n - 0.5, cut * 1.12, name, ha="right", va="bottom", fontsize=8, color=MUTED)
    labels = [f"{c}{'' if e else ' *'}" for c, e in zip(d.index, d.eligible)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    for lab, e in zip(ax.get_xticklabels(), d.eligible):
        lab.set_color(INK if e else MUTED)
    ax.set_ylabel("runs (log scale)")
    ax.set_title(f"Runs per class by split, {task}")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlabel("class   (* = not eligible for scoring: fewer than 2 BioProjects)", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"runs_per_class_{task}.png", dpi=200)
    plt.close(fig)
    d.index.name = "class"
    return d.reset_index().assign(task=task)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    test = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t", low_memory=False)
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    plot_projects(train, test)
    per_class = pd.concat([plot_classes(t, train, test, elig) for t in TASKS])
    per_class.to_csv(OUT / "runs_per_class.tsv", sep="\t", index=False)
    summ = (per_class[per_class.eligible].groupby(["task", "regime"])
            .agg(classes=("class", "size"), train_runs=("train_runs", "sum"), heldout_runs=("heldout_runs", "sum"))
            .reset_index())
    summ.to_csv(OUT / "regime_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print("eligible classes and runs per task and regime (regime = training runs of the class):")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
