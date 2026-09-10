#!/usr/bin/env python
"""
DIANA v9 architecture, as selected by the BioProject-grouped search.

The figure carries structure only. All numeric settings belong in the caption:

  Caption. DIANA v9. A shared trunk of three fully-connected blocks
  (128, 192, 192 units; GELU; dropout 0.32; no batch normalisation) maps the
  110,202-dimensional unitig-fraction vector, standardised to mean 0 and SD 1
  with statistics fit on the training folds, to four task-specific heads. Each
  head is Linear(192 -> 96) -> GELU -> dropout -> Linear(96 -> classes). The
  objective is a weighted sum of per-head cross-entropies with logit
  adjustment at tau = 0.27; absent labels are masked rather than encoded as a
  class. Per-head loss weights are 1.40 (community type), 1.74 (feature),
  1.49 (sample host) and 1.27 (material), with label smoothing 0.02, 0.04,
  0.07 and 0.11 respectively. (b) The single-task control: the same recipe
  fitted independently per target, with no shared trunk.

Provenance:
  Searched -- results/search_v9_multitask/cv_results/best_hyperparameters.json
  and final_training_config.json: n_layers, trunk widths, dropout, activation,
  use_batch_norm, logit_adjust_tau, per-head task_weight and label smoothing.
  Fixed in the model definition, NOT searched -- src/diana/models/
  multitask_mlp.py builds every head as Linear(hidden_dims[-1],
  hidden_dims[-1] // 2), so the 192 -> 96 head width follows from the searched
  trunk width by a hard-coded rule.
  From the data -- data/splits_v9/train_metadata.tsv class counts;
  data/matrices/matrix_v9_train vocabulary size.

Output: results/paper/v9_architecture.png
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper"
OUT.mkdir(parents=True, exist_ok=True)

BASE, ANN, TICK = 8, 7, 6
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": BASE,
    "axes.titlesize": BASE, "figure.dpi": 300, "savefig.dpi": 300,
})

HEADS = [
    ("community type", 5,  "#1F4E79"),
    ("feature",        10, "#3F9C8E"),
    ("sample host",    24, "#C77F00"),
    ("material",       16, "#7B5AA6"),
]
GREY, INK, TRUNK = "#9AA0A6", "#202124", "#5F6368"
FILL = "#DDE3E9"


def box(ax, x, y, w, h, fc, ec, txt="", size=ANN, tc=INK, lw=0.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        fc=fc, ec=ec, lw=lw, zorder=2))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=size, color=tc, zorder=3, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=0.9):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
        lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=1))


fig = plt.figure(figsize=(7.0, 4.8))
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.50)
axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axa, axb):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ------------------------------------------------------------------ panel a
box(axa, 0.005, 0.24, 0.088, 0.52, "#EEF1F4", TRUNK,
    "110,202\nunitig\nfractions", ANN)

trunk_x, trunk_h = [0.155, 0.262, 0.369], [0.42, 0.58, 0.58]
for i, (x, h) in enumerate(zip(trunk_x, trunk_h)):
    box(axa, x, 0.5 - h / 2, 0.082, h, FILL, TRUNK,
        f"Linear\n{[128, 192, 192][i]}", ANN, TRUNK)
    arrow(axa, 0.093 if i == 0 else trunk_x[i - 1] + 0.082, 0.5, x, 0.5)

axa.text(0.278, 0.87, "shared trunk", ha="center", va="bottom",
         fontsize=ANN, color=TRUNK)
axa.text(0.278, 0.08, "GELU · dropout 0.32 · no batch norm",
         ha="center", va="top", fontsize=TICK, color=TRUNK)

ys = [0.855, 0.620, 0.385, 0.150]
for (name, k, col), y in zip(HEADS, ys):
    arrow(axa, 0.451, 0.5, 0.495, y, color=col, lw=1.0)
    box(axa, 0.495, y - 0.065, 0.100, 0.13, "white", col, "192 → 96", TICK, col)
    arrow(axa, 0.595, y, 0.630, y, color=col, lw=1.0)
    box(axa, 0.630, y - 0.075, 0.365, 0.15, col, col,
        f"{name}\n{k} classes", ANN, "white")

axa.text(0.545, 0.08, "task head", ha="center", va="top",
         fontsize=TICK, color=TRUNK)
axa.text(0.0, -0.12,
         "loss = Σ (per-head weight × smoothed cross-entropy), "
         "logit-adjusted at τ = 0.27;   absent labels masked",
         transform=axa.transAxes, ha="left", va="top", fontsize=TICK, color=INK)

# ------------------------------------------------------------------ panel b
for (name, k, col), x in zip(HEADS, [0.020, 0.265, 0.510, 0.755]):
    box(axb, x, 0.68, 0.225, 0.22, "#EEF1F4", TRUNK, "input", TICK, TRUNK)
    box(axb, x, 0.37, 0.225, 0.22, FILL, TRUNK, "own\ntrunk", TICK, TRUNK)
    box(axb, x, 0.06, 0.225, 0.21, col, col, f"{name}\n{k} classes", TICK, "white")
    arrow(axb, x + 0.1125, 0.68, x + 0.1125, 0.600, color=TRUNK)
    arrow(axb, x + 0.1125, 0.37, x + 0.1125, 0.280, color=col)

axa.set_title("a   DIANA v9 — shared trunk, four task heads", loc="left", pad=5)
axb.set_title("b   Single-task control — no shared trunk", loc="left", pad=4)

fig.savefig(OUT / "v9_architecture.png", bbox_inches="tight")

r = fig.canvas.get_renderer()
tick = {t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()}
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible() and t not in tick]
print("n text objects:", len(texts))
print("overlaps:", [(a.get_text()[:22], b.get_text()[:22])
                    for i, (a, ba) in enumerate(texts)
                    for b, bb in texts[i + 1:] if ba.overlaps(bb)])
