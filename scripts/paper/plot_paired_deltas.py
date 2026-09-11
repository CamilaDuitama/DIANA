#!/usr/bin/env python3
"""The eight paired differences, with intervals, against zero.

Absolute-performance plots are the wrong picture for this result: each model's own
interval is about +/- 0.18 because the 34 held-out BioProjects differ enormously, so
every bar overlaps every other and nothing is legible. The paired difference is the
quantity that decides anything, because both models are scored on the same resampled
projects and the shared difficulty cancels.

One row per (task, capability). A difference is established only where the interval
clears zero, so the zero line is the whole point of the figure.

    ./env/bin/python scripts/paper/plot_paired_deltas.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
# Slots 1 and 2 of the validated categorical palette, in order.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a8a"

# Detection deltas come from the paired AUC bootstrap in the same run as the table;
# sign is flipped to DIANA - baseline so both columns read the same way.
DETECTION = {"community_type": (-0.036, -0.071, +0.021),
             "feature":        (-0.079, -0.245, +0.028),
             "sample_host":    (-0.019, -0.069, +0.010),
             "material":       (+0.007, -0.048, +0.069)}


def main() -> int:
    pdl = pd.read_csv(PROJECT_ROOT / "results/paired_superiority_v9/paired_deltas.tsv", sep="\t")
    b = pd.read_csv(PROJECT_ROOT / "results/baseline_predictions_v9/summary.csv")
    b = b[(b.split == "test") & (b.model != "MajorityClass")]

    rows = []
    for t in TASKS:
        best = b[b.task == t].sort_values("f1_macro_eligible", ascending=False).iloc[0].model
        r = pdl[(pdl.task == t) & (pdl.model_a == "DIANA single-task")
                & (pdl.model_b == best)].iloc[0]
        rows.append((t, "classification", r.delta, r.ci_low, r.ci_high))
        d, lo, hi = DETECTION[t]
        rows.append((t, "detection", d, lo, hi))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.axvline(0, color="#4a4a4a", lw=1.2, zorder=1)
    ticks, labels = [], []
    y = 0.0
    for t in TASKS:
        for cap, colour in (("classification", BLUE), ("detection", ORANGE)):
            _, _, d, lo, hi = next(r for r in rows if r[0] == t and r[1] == cap)
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=colour, lw=2.4 if sig else 1.8,
                    alpha=1.0 if sig else 0.62, solid_capstyle="butt", zorder=2)
            ax.plot([d], [y], marker="o", ms=10 if sig else 8, color=colour,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            if sig:
                ax.annotate("established", (hi, y), textcoords="offset points",
                            xytext=(10, 0), va="center", fontsize=8.5,
                            fontweight="bold", color="#1a1a1a")
            ticks.append(y); labels.append(cap)
            y -= 1.0
        y -= 0.7

    for i in range(len(TASKS)):
        y0 = ticks[2 * i]; y1 = ticks[2 * i + 1]
        ax.text(-0.30, (y0 + y1) / 2, f"`{TASKS[i]}`".strip("`"),
                transform=ax.get_yaxis_transform(), rotation=90, va="center",
                ha="center", fontsize=10, fontweight="bold", fontfamily="monospace")
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Δ = DIANA − strongest baseline, paired over the 34 held-out BioProjects\n"
                  "(macro-F1 for classification, ROC-AUC for detection; bar = 95 % CI)",
                  fontsize=9.5)
    ax.set_title("Paired differences between DIANA and the strongest baseline",
                 fontsize=12, pad=14)
    ax.set_xlim(-0.30, 0.30)
    ax.xaxis.grid(True, color="#ededed", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], color=BLUE, marker="o", ms=9, lw=2.2,
                          markeredgecolor="white", label="classification (macro-F1)"),
               plt.Line2D([], [], color=ORANGE, marker="o", ms=9, lw=2.2,
                          markeredgecolor="white", label="detection (ROC-AUC)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)

    out = PROJECT_ROOT / "results/paper/paired_deltas.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
