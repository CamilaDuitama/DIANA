#!/usr/bin/env python3
"""Two manuscript assets: (1) the pipeline diagram, redrawn programmatically for the corrected
workflow (training builds the matrix; new samples are projected by diana-predict); (2) the
dev-fold version of the per-class support figure (out-of-fold, 38 eligible classes), beside the
held-out version for Camila to choose.

    ./env/bin/python scripts/paper/generate_pipeline_and_support_v2.py --manuscript <repo>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
F = __import__("59_devfold_flagging_score")
from diana.evaluation.metrics import load_eligible_classes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_v2"
TASKS = ["community_type", "feature", "sample_host", "material"]
PRETTY = {"community_type": "Community Type", "feature": "Feature", "sample_host": "Sample Host", "material": "Material"}

INK, SUB, EDGE = "#1c1b18", "#5a574f", "#8a877d"
BLUE, ORANGE, GREEN, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#f0ede6"


def box(ax, x, y, w, h, text, fc=GREY, ec=EDGE, fs=8.2, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, text=None, fs=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=SUB, lw=1.2))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.018, text, ha="center", va="bottom", fontsize=fs, color=SUB)


def pipeline(manuscript):
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 7.6), height_ratios=[1.05, 1.0])
    for ax in (a, b):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # ---------------- panel a: training
    a.text(0.0, 0.97, "a  Training (done once)", fontsize=11, weight="bold", color=INK, va="top")
    box(a, 0.00, 0.52, 0.20, 0.26, "AncientMetagenomeDir\n+ Logan unitigs\n2,716 runs, 68 studies\n(training corpus)", fc="#e8f0fb")
    box(a, 0.27, 0.56, 0.22, 0.18, "MUSET\nunitig matrix\n110,202 unitigs ×\n2,716 runs (fractions)")
    box(a, 0.27, 0.30, 0.22, 0.16, "curated labels\ncommunity type, host,\nmaterial, feature", fc="#fdf3e4")
    box(a, 0.56, 0.60, 0.19, 0.16, "4 per-task networks\n(5 seeds, averaged)", fc="#e8f0fb")
    box(a, 0.56, 0.36, 0.19, 0.14, "logistic regression", fc="#fdeee6")
    box(a, 0.82, 0.46, 0.17, 0.20, "DIANA\nhalf-and-half\nprobability average", fc="#e6f5ee", weight="bold")
    arrow(a, 0.20, 0.655, 0.27, 0.655)
    arrow(a, 0.49, 0.67, 0.56, 0.68)
    arrow(a, 0.49, 0.62, 0.56, 0.45)
    arrow(a, 0.38, 0.46, 0.38, 0.52)
    arrow(a, 0.75, 0.665, 0.82, 0.60)
    arrow(a, 0.75, 0.43, 0.82, 0.50)
    a.text(0.0, 0.16, "Model selection, thresholds and the no-call cutoffs are fixed on five study-grouped\ncross-validation folds inside the training corpus; the held-out studies are never touched.",
           fontsize=7.6, color=SUB, va="top")
    # ---------------- panel b: inference
    b.text(0.0, 0.97, "b  Inference on a new sample (diana-predict)", fontsize=11, weight="bold", color=INK, va="top")
    box(b, 0.00, 0.56, 0.17, 0.20, "new sample\nFASTQ reads or\nLogan unitigs", fc="#e8f0fb")
    box(b, 0.24, 0.56, 0.24, 0.20, "project onto the training\nvocabulary: count reference\nk-mers (back_to_sequences),\naggregate to unitig fractions")
    box(b, 0.55, 0.56, 0.17, 0.20, "DIANA\n(frozen)", fc="#e6f5ee", weight="bold")
    box(b, 0.79, 0.70, 0.21, 0.17, "predicted labels\nwith confidence;\n“no call” below the\nper-task cutoff", fc="#fdf3e4")
    box(b, 0.79, 0.46, 0.21, 0.17, "flag score for a label\non file: 1 − P(label);\n“cannot assess” outside\nthe vocabulary", fc="#fdeee6")
    arrow(b, 0.17, 0.66, 0.24, 0.66)
    arrow(b, 0.48, 0.66, 0.55, 0.66)
    arrow(b, 0.72, 0.70, 0.79, 0.78)
    arrow(b, 0.72, 0.62, 0.79, 0.545)
    b.text(0.0, 0.30, "Evaluation in this paper follows the same direction: the 922 held-out runs from 34 unseen\nstudies are projected onto the training vocabulary and scored once, under a logged protocol.",
           fontsize=7.6, color=SUB, va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Pipeline_v2.{ext}", dpi=220)
    plt.close(fig)
    if manuscript:
        shutil.copy(OUT / "Pipeline_v2.png", manuscript / "figures/Pipeline_v2.png")


def devfold_support(manuscript):
    tr = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    rows = []
    for t in TASKS:
        D = F.reference_probs(t, ROOT / ("results/step4_2_studyweight_v9" if t == "feature" else "results/reference_v9"))
        L = F.baseline_probs("LogisticRegression_Bal", t)
        cols = sorted(set(D.columns) | set(L.columns)); idx = D.index.intersection(L.index)
        P = (D.reindex(index=idx, columns=cols).fillna(0) + L.reindex(index=idx, columns=cols).fillna(0)) / 2
        lab = tr[t].reindex(P.index); k = lab.notna()
        y, p = lab[k].astype(str), P[k.to_numpy()].idxmax(axis=1).astype(str)
        elig = load_eligible_classes(ROOT / "data/splits_v9/class_eligibility.tsv", t)
        nproj = tr.groupby(t).archive_project.nunique()
        for c in sorted(set(y) & elig):
            rows.append({"task": t, "class": c, "train_studies": int(nproj.get(c, 0)), "n_runs": int((y == c).sum()),
                         "F1": f1_score(y, p, labels=[c], average="macro", zero_division=0)})
    d = pd.DataFrame(rows); d = d[d.train_studies > 0]
    d.to_csv(OUT / "supp_class_support_f1_devfold.tsv", sep="\t", index=False)
    colors = {"community_type": BLUE, "feature": ORANGE, "sample_host": GREEN, "material": YELLOW}
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for t in TASKS:
        g = d[d.task == t]
        ax.scatter(g.train_studies, g.F1, s=30, color=colors[t], label=PRETTY[t], edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(6, color="#bbb", lw=1, ls="--", zorder=1)
    ax.text(6.3, 0.02, "6 studies", fontsize=7, color="#777")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 25]); ax.set_xticklabels([1, 2, 5, 10, 25])
    ax.set_xlabel("training BioProjects carrying the class")
    ax.set_ylabel("out-of-fold F1 of the class")
    ax.set_title("Per-class F1 by training support (dev folds)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"main_03_class_support_f1_devfold.{ext}", dpi=200)
    plt.close(fig)
    if manuscript:
        shutil.copy(OUT / "main_03_class_support_f1_devfold.png", manuscript / "figures/main_03_class_support_f1_devfold.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", type=Path, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline(a.manuscript); print("pipeline diagram written")
    devfold_support(a.manuscript); print("dev-fold support figure written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
