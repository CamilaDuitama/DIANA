#!/usr/bin/env python3
"""One figure: held-out f1_macro_eligible for every model, on all four tasks.

A dot-and-interval plot rather than bars, because the interval is the point. Each
model's own 95 % CI spans roughly 0.35, so a bar chart of the point estimates would
imply a precision the data does not have. Whether one model is better than another is
not readable off this figure at all; that is the paired test (Table 4).

Model order is fixed across tasks so the same row means the same model everywhere,
and colour follows the model rather than its rank.

    ./env/bin/python scripts/paper/plot_heldout_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
# Fixed model order, top to bottom within every task block.
# DIANA is the single-task network: the shared trunk was tested and dropped, so the
# multi-task arm is not a model this paper reports. Its numbers stay in PROJECT.md's
# Read-first item 0, which is the evidence for dropping it.
ORDER = ["DIANA", "LinearSVM_Bal", "LogisticRegression_Bal", "RandomForest_Bal",
         "RandomForest", "kNN", "MajorityClass"]
# Slots 1 and 2 of the validated reference palette, used in order; baselines take a
# neutral ink so the two DIANA variants are the only hues on the figure.
# Slots 1 and 2 of the validated categorical palette, used in order. Checked with
# the palette validator (node, module nodejs/26.8.1): both hues pass all six checks,
# CVD dE 24.7 protan / 32.7 tritan, normal-vision dE 33.6. GREY is deliberately a
# neutral, not a third categorical hue -- the baselines are the reference group, not
# an identity to tell apart -- so it is exempt from the chroma floor and was checked
# only for contrast against the surface, which it passes.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a8a"
COLOUR = {"DIANA": BLUE}


def collect() -> pd.DataFrame:
    b = pd.read_csv(PROJECT_ROOT / "results/baseline_predictions_v9/summary.csv")
    b = b[b.split == "test"]
    rows = []
    for t in TASKS:
        for _, r in b[b.task == t].iterrows():
            rows.append({"task": t, "model": r.model, "f1": r.f1_macro_eligible,
                         "lo": r.f1_macro_eligible_ci_low,
                         "hi": r.f1_macro_eligible_ci_high})
        p = PROJECT_ROOT / f"results/final_eval_v9/heldout_single_{t}/test_metrics.json"
        m = json.loads(p.read_text()).get(t)
        if isinstance(m, dict):
            rows.append({"task": t, "model": "DIANA",
                         "f1": m["f1_macro_eligible"],
                         "lo": m["f1_macro_eligible_ci_low"],
                         "hi": m["f1_macro_eligible_ci_high"]})
    return pd.DataFrame(rows)


def main() -> int:
    df = collect()
    missing = set(df.model) - set(ORDER)
    if missing:
        raise SystemExit(f"models not in the fixed order: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(8.2, 9.2))
    y, ticks, labels, boundaries = 0.0, [], [], []
    for ti, task in enumerate(TASKS):
        sub = df[df.task == task].set_index("model")
        cand = sub.drop(index=[m for m in COLOUR if m in sub.index], errors="ignore")
        best_base = cand.f1.idxmax() if len(cand) else None
        for model in ORDER:
            if model not in sub.index:
                continue
            r = sub.loc[model]
            c = COLOUR.get(model, GREY)
            is_diana = model in COLOUR
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=2.0,
                    alpha=1.0 if is_diana else 0.55,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r.f1], [y], marker="o", ms=9 if is_diana else 7,
                    color=c, markeredgecolor="white",
                    markeredgewidth=1.4 if is_diana else 1.0, zorder=3)
            # A number on every point is clutter; the full values are in Table 8.
            # Label the two DIANA rows and the strongest baseline, which is the
            # comparison the figure exists to show.
            if is_diana or model == best_base:
                ax.annotate(f"{r.f1:.3f}", (r.hi, y), textcoords="offset points",
                            xytext=(7, 0), va="center", fontsize=8.5,
                            color="#333333" if is_diana else "#767676")
            ticks.append(y)
            labels.append(model.replace("_Bal", " (balanced)"))
            y -= 1.0
        y -= 0.9
        if ti < len(TASKS) - 1:
            boundaries.append(y + 0.45)

    for yb in boundaries:
        ax.axhline(yb, color="#e2e2e2", lw=0.9, zorder=1)
    # task names, placed against the left edge of each block
    blocks, start = [], 0
    for task in TASKS:
        n = int((df.task == task).sum())
        blocks.append((task, ticks[start], ticks[start + n - 1]))
        start += n
    for task, y0, y1 in blocks:
        ax.text(-0.45, (y0 + y1) / 2, task, transform=ax.get_yaxis_transform(),
                rotation=90, va="center", ha="center", fontsize=11, fontweight="bold",
                fontfamily="monospace", color="#1a1a1a")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("f1_macro_eligible on the 922 held-out runs\n"
                  "(point estimate, bar = 95 % CI over 1,000 resamples of the 34 BioProjects,\n"
                  "whole projects drawn with replacement)",
                  fontsize=9.5)
    ax.set_title("Held-out f1_macro_eligible by task and model", fontsize=12, pad=14)
    ax.xaxis.grid(True, color="#ededed", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="y", length=0)

    handles = [plt.Line2D([], [], color=BLUE, marker="o", ms=9, lw=2.0,
                          markeredgecolor="white", label="DIANA (single-task)"),
               plt.Line2D([], [], color=GREY, marker="o", ms=7, lw=2.0, alpha=0.55,
                          markeredgecolor="white", label="tuned baseline")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)

    out = PROJECT_ROOT / "results/paper/heldout_model_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
