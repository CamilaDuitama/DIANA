#!/usr/bin/env python
"""
DIANA v7 vs. baselines on the BioProject-disjoint test split.

Reads the baseline sweep (results/baseline_comparison_v7/summary.csv) and the
DIANA v7 test predictions (results/test_evaluation_bioproject_v7/
test_predictions.tsv), recomputes DIANA's metrics from the raw predictions so
both sides use identical definitions, and plots accuracy and macro-F1 per task.

Output: results/test_evaluation_bioproject_v7/v7_grouped_split_comparison.{png,csv}
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/baseline_comparison_v7/summary.csv"
PRED = ROOT / "results/test_evaluation_bioproject_v7/test_predictions.tsv"
OUT = ROOT / "results/test_evaluation_bioproject_v7"

TASKS = ["community_type", "sample_host", "material"]
NICE = {"community_type": "Community type", "sample_host": "Sample host",
        "material": "Material"}

# ---------------------------------------------------------------- collect rows
pred = pd.read_csv(PRED, sep="\t")
n_test = len(pred)

rows = []
for t in TASKS:
    y = pred[f"{t}_true"].astype(str)
    p = pred[f"{t}_pred"].astype(str)
    rows.append(dict(model="DIANA", task=t,
                     accuracy=accuracy_score(y, p),
                     f1_macro=f1_score(y, p, average="macro", zero_division=0)))

b = pd.read_csv(BASE)
b = b[(b.split == "test") & (b.task_type == "classification")]
keep = {"MajorityClass": "Majority class",
        "LogisticRegression_Bal": "Logistic regression",
        "RandomForest": "Random forest"}
for _, r in b[b.model.isin(keep)].iterrows():
    rows.append(dict(model=keep[r.model], task=r.task,
                     accuracy=r.accuracy, f1_macro=r.f1_macro_seen))

df = pd.DataFrame(rows)
df.to_csv(OUT / "v7_grouped_split_comparison.csv", index=False)

# ---------------------------------------------------------------------- style
BASE_SZ, ANN_SZ, TICK_SZ = 8, 7, 6
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": BASE_SZ,
    "axes.titlesize": BASE_SZ, "axes.labelsize": BASE_SZ,
    "xtick.labelsize": TICK_SZ, "ytick.labelsize": TICK_SZ,
    "legend.fontsize": ANN_SZ, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300,
})
COL = {"DIANA": "#1F4E79", "Random forest": "#3F9C8E",
       "Logistic regression": "#9AA0A6", "Majority class": "#BDBDBD"}
ORDER = ["Majority class", "Logistic regression", "Random forest", "DIANA"]
DY = {"Majority class": 0.24, "Logistic regression": 0.08,
      "Random forest": -0.08, "DIANA": -0.24}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, metric, xlab in zip(axes, ["accuracy", "f1_macro"],
                            ["Accuracy", "Macro-F1"]):
    for i, t in enumerate(TASKS):
        for m in ORDER:
            v = df[(df.task == t) & (df.model == m)][metric]
            if v.empty:
                continue
            v = float(v.iloc[0])
            y = i + DY[m]
            open_marker = (m == "Majority class")
            ax.plot([0, v], [y, y], color=COL[m], lw=0.8,
                    alpha=0.45 if open_marker else 0.7, zorder=1)
            ax.plot([v], [y], marker="o", ms=5.2,
                    mfc="white" if open_marker else COL[m],
                    mec=COL[m], mew=1.1, zorder=3,
                    clip_on=False)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([NICE[t] for t in TASKS])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, len(TASKS) - 0.45)
    ax.set_xlabel(xlab)
    ax.grid(axis="x", lw=0.4, color="0.9", zorder=0)
    ax.set_axisbelow(True)

# headline annotation: the sample_host inversion
d_sh = float(df[(df.task == "sample_host") & (df.model == "DIANA")].accuracy.iloc[0])
m_sh = float(df[(df.task == "sample_host") & (df.model == "Majority class")].accuracy.iloc[0])
axes[0].annotate(f"{d_sh:.2f} — below the\nmajority-class rate ({m_sh:.2f})",
                 xy=(d_sh, 1 + DY["DIANA"]), xytext=(0.04, 0.44),
                 fontsize=ANN_SZ, color=COL["DIANA"], ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=COL["DIANA"],
                                 shrinkA=0, shrinkB=3))

handles = [plt.Line2D([], [], marker="o", ls="-", lw=0.8, ms=5.2,
                      color=COL[m], mfc="white" if m == "Majority class" else COL[m],
                      mec=COL[m], mew=1.1, label=m) for m in ORDER]
axes[1].legend(handles=handles, loc="lower right", frameon=False,
               handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
axes[1].text(0.99, 1.02, "higher = better", transform=axes[1].transAxes,
             ha="right", va="bottom", fontsize=ANN_SZ, color="0.35")

fig.suptitle("On BioProject-disjoint splits, DIANA v7 trails a random forest on all three tasks",
             x=0.012, y=1.05, ha="left", fontsize=BASE_SZ)
fig.text(0.012, -0.10,
         f"Held-out test split, n = {n_test} runs from 19 BioProjects disjoint from training. "
         "Macro-F1 over all classes present in the split.",
         ha="left", va="top", fontsize=ANN_SZ, color="0.35")
fig.tight_layout()
fig.savefig(OUT / "v7_grouped_split_comparison.png", bbox_inches="tight")

# ------------------------------------------------------------------- bbox check
r = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
ticklabels = {ax: set(ax.get_xticklabels() + ax.get_yticklabels()) for ax in fig.axes}
ov = [(a.get_text()[:20], b.get_text()[:20])
      for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
print("text overlaps:", ov)
print(df.pivot(index="task", columns="model",
               values=["accuracy", "f1_macro"]).round(3).to_string())
