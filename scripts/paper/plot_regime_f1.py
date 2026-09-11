#!/usr/bin/env python3
"""One figure per task: F1 by data regime, every model.

The long-tail reporting protocol groups classes by training support into few-shot
(< 20), medium-shot (20-100) and many-shot (> 100) and scores each group separately.
These are the ImageNet-LT / OLTR split points, not choices made here.

Drawn as lines across the three regimes because the question is *which model wins
where*, and a line makes a crossing visible where grouped bars would not. One file
per task, so each figure carries one plot.

DIANA's two architectures take the two validated categorical hues; the baselines are
a neutral grey, labelled at the right edge rather than given eight competing colours.

    ./env/bin/python scripts/paper/plot_regime_f1.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
BINS = ["few-shot (<20)", "medium-shot (20-100)", "many-shot (>100)"]
SHORT = ["few-shot\n< 20", "medium-shot\n20–100", "many-shot\n> 100"]
# Slots 1 and 2 of the validated categorical palette; grey is a neutral for the
# reference group, not a third identity.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a8a"
COLOUR = {"DIANA single": BLUE, "DIANA multi": ORANGE}


def main() -> int:
    df = pd.read_csv(PROJECT_ROOT / "results/final_eval_v9/stratified_by_shot.tsv", sep="\t")
    for task in TASKS:
        sub = df[df.task == task]
        present = [(i, b) for i, b in enumerate(BINS)
                   if sub[sub.stratum == b].iloc[0].n_classes > 0]
        if len(present) < 2:
            print(f"  {task}: only {len(present)} non-empty regime, skipping")
            continue
        xs = [i for i, _ in present]

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ends = []
        for model in sub.model.unique():
            ys = [sub[(sub.stratum == b) & (sub.model == model)].f1.iloc[0] for _, b in present]
            is_d = model in COLOUR
            c = COLOUR.get(model, GREY)
            ax.plot(xs, ys, marker="o", ms=8 if is_d else 6, lw=2.0 if is_d else 1.4,
                    color=c, alpha=1.0 if is_d else 0.5,
                    markeredgecolor="white", markeredgewidth=1.2,
                    zorder=3 if is_d else 2)
            ends.append([ys[-1], model, is_d, c])

        # Eight lines converge at the right edge, so the labels must be pushed apart
        # or they overprint. Sort by value and enforce a minimum gap, keeping order.
        ends.sort(key=lambda e: e[0])
        GAP = 0.048
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < GAP:
                ends[i][0] = ends[i - 1][0] + GAP
        for y_lab, model, is_d, c in ends:
            y_true_ = sub[(sub.stratum == present[-1][1]) & (sub.model == model)].f1.iloc[0]
            ax.annotate(model, xy=(xs[-1], y_true_), xytext=(xs[-1] + 0.22, y_lab),
                        va="center", fontsize=8.5,
                        color="#1a1a1a" if is_d else "#767676",
                        fontweight="bold" if is_d else "normal",
                        arrowprops=dict(arrowstyle="-", color="#d5d5d5", lw=0.7,
                                        shrinkA=0, shrinkB=3))

        labs = []
        for i, b in present:
            r = sub[sub.stratum == b].iloc[0]
            labs.append(f"{SHORT[i]}\n{int(r.n_classes)} classes, {int(r.n_runs)} runs")
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=9)
        ax.set_ylim(-0.04, 1.05)
        ax.set_xlim(min(xs) - 0.3, max(xs) + 1.15)
        ax.set_ylabel("macro-F1 within the regime", fontsize=10)
        ax.set_title(f"Held-out macro-F1 by training support — {task}", fontsize=11.5, pad=12)
        ax.yaxis.grid(True, color="#ededed", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cccccc")

        out = PROJECT_ROOT / f"results/paper/regime_f1_{task}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
