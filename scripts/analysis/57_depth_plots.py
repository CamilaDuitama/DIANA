#!/usr/bin/env python3
"""Two views of the depth diagnostic (56_), from the same dev-fold out-of-fold predictions.

1. acc_vs_depth_<task>.png: per class (at least 20 labelled runs), out-of-fold accuracy in
   five within-class depth bins (quintiles of log10 non-zero unitigs), DIANA single-task and
   logistic regression side by side on the same axes, one panel per class. Requested as a
   side-by-side comparison, so the figure has one panel per class.
2. shallow_oral_predictions_<model>.png and shallow_oral_predictions.tsv: what the oral runs
   below the class median depth are predicted as, against the deep half.

    ./env/bin/python scripts/analysis/57_depth_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPLITS, OUT = ROOT / "data/splits_v9", ROOT / "results/depth_vs_correctness_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
MIN_RUNS, N_BINS = 20, 5
DIANA_C, LOGREG_C, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e2"


def predictions(task: str) -> dict[str, pd.DataFrame]:
    d = pd.read_csv(ROOT / f"results/epoch_budget_v9/oof_{task}_{task}.tsv", sep="\t").dropna(subset=[f"{task}_true"])
    out = {"DIANA": pd.DataFrame({"y": d[f"{task}_true"].astype(str).to_numpy(), "p": d[f"{task}_pred"].astype(str).to_numpy()},
                                 index=d.Run_accession.to_numpy())}
    sel = pd.read_csv(ROOT / "results/baselines_sig_v9/selected_configs_sig_vs_fraction.tsv", sep="\t")
    cfg = sel[(sel.task == task) & (sel.model == "LogisticRegression_Bal") & (sel.rep == "fraction")].cfg.iloc[0]
    b = pd.read_csv(ROOT / f"results/baselines_sig_v9/pred_fraction_{task}_LogisticRegression_Bal.tsv", sep="\t")
    b = b[b.cfg == cfg]
    out["logistic regression"] = pd.DataFrame({"y": b.y_true.astype(str).to_numpy(), "p": b.y_pred.astype(str).to_numpy()},
                                              index=b.Run_accession.to_numpy())
    return out


def main() -> int:
    depth = pd.read_csv(OUT / "nonzero_counts.tsv", sep="\t").set_index("Run_accession")
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t"); elig = elig[elig.evaluable]
    tables = {}
    for task in TASKS:
        eligible = set(elig[elig.target == task]["class"].astype(str))
        preds = predictions(task)
        frames = {}
        for model, pr in preds.items():
            df = pr.join(depth[["n_nonzero"]])
            df = df[df.y.isin(eligible) & (df.n_nonzero > 0)].copy()
            df["correct"] = (df.y == df.p).astype(int); df["log_depth"] = np.log10(df.n_nonzero)
            frames[model] = df
        counts = frames["DIANA"].y.value_counts()
        classes = [c for c in counts.index if counts[c] >= MIN_RUNS]
        n = len(classes); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False, sharey=True)
        for ax, c in zip(axes.ravel(), classes):
            ref = frames["DIANA"][frames["DIANA"].y == c].log_depth
            edges = np.quantile(ref, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
            for model, colour, marker in (("DIANA", DIANA_C, "o"), ("logistic regression", LOGREG_C, "s")):
                g = frames[model][frames[model].y == c]
                b = np.clip(np.digitize(g.log_depth, edges[1:-1]), 0, N_BINS - 1)
                acc = [g.correct[b == k].mean() if (b == k).any() else np.nan for k in range(N_BINS)]
                mid = [g.log_depth[b == k].median() if (b == k).any() else np.nan for k in range(N_BINS)]
                ax.plot(mid, acc, marker=marker, color=colour, lw=1.4, ms=5, label=model)
            ax.set_title(f"{c} (n={counts[c]})", fontsize=9)
            ax.set_ylim(-0.03, 1.03); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("log10 non-zero unitigs")
        for ax in axes[:, 0]:
            ax.set_ylabel("out-of-fold accuracy")
        axes.ravel()[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Accuracy by depth within class, {task}: DIANA and logistic regression, dev folds", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"acc_vs_depth_{task}.png", dpi=200)
        plt.close(fig)
        if task == "community_type":
            for model, df in frames.items():
                g = df[df.y == "oral"].copy()
                med = g.log_depth.median()
                g["half"] = np.where(g.log_depth <= med, "shallow half", "deep half")
                tab = g.groupby(["half", "p"]).size().unstack(fill_value=0)
                share = tab.div(tab.sum(axis=1), axis=0)
                tables[model] = (tab, share)
    rows = []
    for model, (tab, share) in tables.items():
        for half in tab.index:
            for cls in tab.columns:
                rows.append({"model": model, "half": half, "predicted_as": cls, "n": int(tab.loc[half, cls]), "share": float(share.loc[half, cls])})
    pd.DataFrame(rows).to_csv(OUT / "shallow_oral_predictions.tsv", sep="\t", index=False)
    for model, (tab, share) in tables.items():
        cols = list(share.columns)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(cols)); w = 0.38
        ax.bar(x - w / 2, [share.loc["shallow half", c] if "shallow half" in share.index else 0 for c in cols], w, color=LOGREG_C,
               label=f"shallow half (n={int(tab.loc['shallow half'].sum())})")
        ax.bar(x + w / 2, [share.loc["deep half", c] if "deep half" in share.index else 0 for c in cols], w, color=DIANA_C,
               label=f"deep half (n={int(tab.loc['deep half'].sum())})")
        ax.set_xticks(x); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("share of oral runs"); ax.set_ylim(0, 1.02)
        ax.set_title(f"Predicted class of oral runs by depth half, {model}, dev folds", fontsize=10)
        ax.grid(True, axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"shallow_oral_predictions_{model.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).pivot_table(index=["model", "half"], columns="predicted_as", values="share").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
