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
import json
from pathlib import Path

import matplotlib as mpl
import pandas as pd
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

TASKS = ["community_type", "feature", "sample_host", "material"]
HEAD_LABEL = {"community_type": "community type", "feature": "feature",
              "sample_host": "sample host", "material": "material"}
HEAD_COLOUR = {"community_type": "#1F4E79", "feature": "#3F9C8E",
               "sample_host": "#C77F00", "material": "#7B5AA6"}

# Values used only if no search output exists yet, so the figure still renders
# while the search runs. Every one is overwritten as soon as it does.
FALLBACK = {"hidden_dims": [128, 192, 192], "dropout": 0.32, "activation": "gelu",
            "batch_norm": False, "tau": 0.27,
            "weights": {"community_type": 1.40, "feature": 1.74,
                        "sample_host": 1.49, "material": 1.27},
            "smoothing": {"community_type": 0.02, "feature": 0.04,
                          "sample_host": 0.07, "material": 0.11}}

SEARCH_DIRS = [ROOT / "results/search_v9_multitask", ROOT / "results/search_v9_final"]


def load_searched() -> dict:
    """Read the hyperparameters the grouped search selected.

    Tried in order, first hit wins:
      1. <search>/final_training_config.json          (resolved hidden_dims list)
      2. <search>/cv_results/best_hyperparameters.json (hidden_dim_<i>, may be float)
      3. <search>/**/search_all_train_best_params.json (single all-train search)

    Optuna returns hidden_dim_<i> as a float; aggregate_cv_results.py converts
    with int(round(...)) and this mirrors that, so the figure shows exactly the
    widths the model was built with. Falls back to FALLBACK with a printed
    warning, never silently.
    """
    cands = []
    for d in SEARCH_DIRS:
        cands += [d / "final_training_config.json",
                  d / "cv_results" / "best_hyperparameters.json"]
        cands += sorted(d.rglob("search_all_train_best_params.json"))
    src = next((c for c in cands if c.exists()), None)
    if src is None:
        print(f"WARNING: no search output under {[str(d) for d in SEARCH_DIRS]} — "
              "figure drawn from FALLBACK values, do NOT use in the paper")
        return {**FALLBACK, "source": "FALLBACK (search not finished)"}

    raw = json.loads(src.read_text())
    p = raw.get("model", raw) if "hidden_dims" in raw.get("model", {}) else raw
    p = {**raw, **(raw.get("model") or {}), **(raw.get("best_hyperparameters") or {})}

    if isinstance(p.get("hidden_dims"), list):
        dims = [int(round(v)) for v in p["hidden_dims"]]
    else:
        n = int(p.get("n_layers", 3))
        dims = [int(round(p[f"hidden_dim_{i}"])) for i in range(n)]

    hp = {
        "hidden_dims": dims,
        "dropout": float(p["dropout"]),
        "activation": str(p.get("activation", "gelu")),
        "batch_norm": bool(round(float(p.get("use_batch_norm", 0)))),
        "tau": float(p.get("logit_adjust_tau", 0.0) or 0.0),
        "weights": {t: float(p[f"task_weight_{t}"]) for t in TASKS
                    if f"task_weight_{t}" in p},
        "smoothing": {t: float(p[f"ls_{t}"]) for t in TASKS if f"ls_{t}" in p},
        "source": str(src.relative_to(ROOT)),
    }
    for k in ("weights", "smoothing"):
        if not hp[k]:
            print(f"WARNING: {k} absent from {hp['source']} — using FALLBACK for it")
            hp[k] = FALLBACK[k]
    return hp


def class_counts() -> dict:
    """Head sizes as they are in the split the model was trained on."""
    md = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t",
                     low_memory=False)
    out = {}
    for t in TASKS:
        v = md[t].astype(str).str.strip()
        out[t] = int(v[~v.str.lower().isin(["nan", "none", "null", ""])].nunique())
    return out


HP = load_searched()
NCLS = class_counts()
HEADS = [(HEAD_LABEL[t], NCLS[t], HEAD_COLOUR[t]) for t in TASKS]
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

dims = HP["hidden_dims"]
bw, gap, x0 = 0.082, 0.025, 0.155
trunk_x = [x0 + i * (bw + gap) for i in range(len(dims))]
trunk_h = [0.30 + 0.28 * (d / max(dims)) for d in dims]
for i, (x, h, d) in enumerate(zip(trunk_x, trunk_h, dims)):
    box(axa, x, 0.5 - h / 2, bw, h, FILL, TRUNK, f"Linear\n{d}", ANN, TRUNK)
    arrow(axa, 0.093 if i == 0 else trunk_x[i - 1] + bw, 0.5, x, 0.5)
trunk_mid = (trunk_x[0] + trunk_x[-1] + bw) / 2
head_x = trunk_x[-1] + bw + 0.044

axa.text(trunk_mid, 0.87, "shared trunk", ha="center", va="bottom",
         fontsize=ANN, color=TRUNK)
axa.text(trunk_mid, 0.08,
         f"{HP['activation'].upper()} · dropout {HP['dropout']:.2f} · "
         f"{'batch norm' if HP['batch_norm'] else 'no batch norm'}",
         ha="center", va="top", fontsize=TICK, color=TRUNK)

ys = [0.855, 0.620, 0.385, 0.150]
for (name, k, col), y in zip(HEADS, ys):
    arrow(axa, trunk_x[-1] + bw, 0.5, head_x, y, color=col, lw=1.0)
    box(axa, head_x, y - 0.065, 0.100, 0.13, "white", col,
        f"{dims[-1]} → {dims[-1] // 2}", TICK, col)
    arrow(axa, head_x + 0.100, y, 0.630, y, color=col, lw=1.0)
    box(axa, 0.630, y - 0.075, 0.365, 0.15, col, col,
        f"{name}\n{k} classes", ANN, "white")

axa.text(head_x + 0.050, 0.08, "task head", ha="center", va="top",
         fontsize=TICK, color=TRUNK)
axa.text(0.0, -0.12,
         "loss = Σ (per-head weight × smoothed cross-entropy), "
         f"logit-adjusted at τ = {HP['tau']:.2f};   absent labels masked",
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

w = HP["weights"]; ls = HP["smoothing"]
print(f"hyperparameters read from: {HP['source']}")
print("CAPTION (paste into the paper / PROJECT.md):")
print(f"  DIANA v9. A shared trunk of {len(dims)} fully-connected blocks "
      f"({', '.join(str(d) for d in dims)} units; {HP['activation'].upper()}; "
      f"dropout {HP['dropout']:.2f}; "
      f"{'batch normalisation' if HP['batch_norm'] else 'no batch normalisation'}) "
      f"maps the 110,202-dimensional unitig-fraction vector, standardised with "
      f"statistics fit on the training folds, to {len(TASKS)} task-specific heads. "
      f"Each head is Linear({dims[-1]} -> {dims[-1] // 2}) -> "
      f"{HP['activation'].upper()} -> dropout -> Linear. The objective is a "
      f"weighted sum of per-head cross-entropies with logit adjustment at "
      f"tau = {HP['tau']:.2f}; absent labels are masked rather than encoded as a "
      f"class. Per-head loss weights are "
      + ", ".join(f"{w[t]:.2f} ({HEAD_LABEL[t]})" for t in TASKS if t in w)
      + ", with label smoothing "
      + ", ".join(f"{ls[t]:.2f}" for t in TASKS if t in ls)
      + " respectively. Head sizes are "
      + ", ".join(f"{NCLS[t]} ({HEAD_LABEL[t]})" for t in TASKS)
      + " classes. (b) The single-task control: the same recipe fitted "
        "independently per target, with no shared trunk.")

r = fig.canvas.get_renderer()
tick = {t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()}
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible() and t not in tick]
print("n text objects:", len(texts))
print("overlaps:", [(a.get_text()[:22], b.get_text()[:22])
                    for i, (a, ba) in enumerate(texts)
                    for b, bb in texts[i + 1:] if ba.overlaps(bb)])
