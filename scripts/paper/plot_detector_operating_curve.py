#!/usr/bin/env python3
"""What a curator actually gets: mislabels caught against clean runs reviewed.

The R3.6 deliverable is not an AUC, it is a decision: review this fraction of the
corpus and find that fraction of the errors. This plots exactly that, per task, with
logistic regression for comparison because Table 2 shows it ties DIANA and a reader
will ask.

x is the false-flag rate, the share of *correctly* labelled runs the threshold fires
on. y is the share of planted mislabels caught. Both axes are fractions of their own
denominator, so the two models are directly comparable at any budget.

    ./env/bin/python scripts/paper/plot_detector_operating_curve.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
# Slots 1 to 4 of the validated categorical palette, assigned in fixed order.
COLOUR = dict(zip(TASKS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))


def logreg_curve(task: str):
    prob = pd.read_csv(PROJECT_ROOT /
        f"results/baseline_predictions_v9/heldout_probabilities_{task}.tsv", sep="\t")
    g = prob[prob.model == "LogisticRegression_Bal"]
    plant = pd.read_csv(PROJECT_ROOT /
        "results/planted_mislabels_v9/planted_test_mixed_r0.1.tsv", sep="\t")
    pl = plant[["Run_accession", task, f"{task}_planted"]].dropna()
    m = g.merge(pl, on="Run_accession", suffixes=("", "_st"))
    cols = {c[2:]: c for c in m.columns if c.startswith("p_")}
    stated = m[task].astype(str).to_numpy()
    p = np.array([m[cols[s]].iloc[i] if s in cols else np.nan
                  for i, s in enumerate(stated)], dtype=float)
    ok = ~np.isnan(p)
    y = m.loc[ok, f"{task}_planted"].astype(int).to_numpy()
    fpr, tpr, _ = roc_curve(y, 1 - p[ok])
    return fpr, tpr


def main() -> int:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for task in TASKS:
        c = json.loads((PROJECT_ROOT /
            f"results/anomaly_detection/curves_DIANA_{task}.json").read_text())[task]
        ax.plot(c["fpr"], c["tpr"], color=COLOUR[task], lw=2.2, zorder=3,
                label=f"{task} — DIANA")
        f2, t2 = logreg_curve(task)
        ax.plot(f2, t2, color=COLOUR[task], lw=1.4, ls=(0, (4, 2)), alpha=0.75, zorder=2)
    ax.plot([0, 1], [0, 1], color="#b5b5b5", lw=1.0, ls=":", zorder=1)

    for b in (0.05, 0.10):
        ax.axvline(b, color="#4a4a4a", lw=0.9, alpha=0.5, zorder=1)
        ax.annotate(f"{int(b*100)} % budget", (b, 1.005), ha="center", va="bottom",
                    fontsize=8.5, color="#4a4a4a")
    ax.set_xlim(0, 0.30); ax.set_ylim(0, 1.03)
    ax.set_xlabel("fraction of CORRECTLY labelled runs flagged\n"
                  "(the review a curator pays for nothing)", fontsize=10)
    ax.set_ylabel("fraction of planted mislabels caught", fontsize=10)
    ax.set_title("Detector operating curve, held-out", fontsize=12, pad=16)
    ax.grid(True, color="#ededed", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    handles = [plt.Line2D([], [], color=COLOUR[t], lw=2.2, label=t) for t in TASKS]
    handles += [plt.Line2D([], [], color="#4a4a4a", lw=2.2, label="DIANA (solid)"),
                plt.Line2D([], [], color="#4a4a4a", lw=1.4, ls=(0, (4, 2)),
                           label="logistic regression (dashed)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)

    out = PROJECT_ROOT / "results/paper/detector_operating_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
