#!/usr/bin/env python
"""
How the resampling is done — a schematic of the confidence-interval procedure.

This figure explains the *mechanic*, not the result. It is deliberately a
schematic: 6 BioProjects holding 23 runs, so a reader can count the dots. The
real held-out set is 922 runs in 34 BioProjects, stated in the caption.

  panel a  runs are nested inside BioProjects, and the projects differ wildly
           in size
  panel b  resampling RUNS: pick 23 runs one at a time with replacement. Every
           project comes back at roughly its original share, so the resample is
           a near-copy of the original and the interval is too narrow.
  panel c  resampling BIOPROJECTS: pick 6 projects with replacement, each kept
           whole. Some projects vanish, others arrive twice, so which studies
           are present genuinely changes.

Both draws are simulated with seed 42, not hand-arranged.

Output: results/paper/resampling_unit.png
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper"
OUT.mkdir(parents=True, exist_ok=True)

# schematic sizes, chosen to mirror the real skew (one project ~1/3 of the set)
NAMES = ["P1", "P2", "P3", "P4", "P5", "P6"]
SIZES = [8, 2, 5, 3, 1, 4]
COLS = ["#1F4E79", "#C77F00", "#3F9C8E", "#7B5AA6", "#8C564B", "#6B7280"]
N_RUNS, N_PROJ = sum(SIZES), len(NAMES)
SEED = 42

BASE, ANN, TICK = 8, 7, 6
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": BASE,
    "axes.titlesize": BASE, "figure.dpi": 300, "savefig.dpi": 300,
})
INK, GREY = "#202124", "#5F6368"

rng = np.random.default_rng(SEED)
run_origin = np.repeat(np.arange(N_PROJ), SIZES)
run_draw = rng.choice(run_origin, size=N_RUNS, replace=True)
proj_draw = rng.choice(np.arange(N_PROJ), size=N_PROJ, replace=True)


def project_box(ax, x, y, n, colour, label, h=0.30, show_label=True):
    """A BioProject: a rounded box whose width scales with its run count,
    holding one dot per run."""
    w = 0.030 * n + 0.024
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc="white", ec=colour, lw=1.1, zorder=2))
    for j in range(n):
        ax.plot(x + 0.024 + 0.030 * j, y + h * 0.52, marker="o", ms=3.6,
                color=colour, zorder=3)
    if show_label:
        ax.text(x + w / 2, y - 0.045, label, ha="center", va="top",
                fontsize=TICK, color=colour)
    return w


fig = plt.figure(figsize=(7.0, 5.0))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.45)
axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))
for ax in (axa, axb, axc):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ------------------------------------------------------------------ panel a
x = 0.012
for i, (n, c, nm) in enumerate(zip(SIZES, COLS, NAMES)):
    x += project_box(axa, x, 0.42, n, c, f"{nm} · {n} run" + ("s" if n > 1 else "")) + 0.016
axa.text(0.012, 0.93,
         "one dot = one sequencing run   ·   one box = one BioProject (a study)",
         ha="left", va="top", fontsize=TICK, color=GREY)
axa.text(0.012, 0.16, "Runs inside a study share lab, protocol and platform, so they are "
         "near-duplicates — not independent observations.",
         ha="left", va="top", fontsize=TICK, color=INK)

# ------------------------------------------------------------------ panel b
axb.text(0.012, 0.96, f"Pick {N_RUNS} runs one at a time, with replacement:",
         ha="left", va="top", fontsize=TICK, color=INK)
for j, o in enumerate(run_draw):
    axb.plot(0.030 + 0.0335 * j, 0.60, marker="o", ms=4.6, color=COLS[o], zorder=3)
counts_b = np.bincount(run_draw, minlength=N_PROJ)
axb.text(0.012, 0.40,
         "→ " + "   ".join(f"{NAMES[i]}×{counts_b[i]}" for i in range(N_PROJ))
         + f"   (originally {'  '.join(str(s) for s in SIZES)})",
         ha="left", va="top", fontsize=TICK, color=GREY)
axb.text(0.012, 0.14,
         "All 6 studies survive — drawing runs can never remove one. At the real scale "
         "of 922 runs their shares\nbarely shift either, so every resample is a near-copy "
         "of the original and the interval comes out too narrow.",
         ha="left", va="top", fontsize=TICK, color="#B23A2E")

# ------------------------------------------------------------------ panel c
axc.text(0.012, 0.96, f"Pick {N_PROJ} projects with replacement, keeping each one whole:",
         ha="left", va="top", fontsize=TICK, color=INK)
x = 0.012
for i in proj_draw:
    x += project_box(axc, x, 0.42, SIZES[i], COLS[i], NAMES[i], h=0.26) + 0.016
missing = [NAMES[i] for i in range(N_PROJ) if i not in set(proj_draw.tolist())]
twice = [NAMES[i] for i in range(N_PROJ) if (proj_draw == i).sum() > 1]
axc.text(0.012, 0.14,
         f"{', '.join(missing)} gone · {', '.join(twice)} drawn twice — which studies are "
         "present really changes,\nand so does the difficulty. This is the honest interval.",
         ha="left", va="top", fontsize=TICK, color="#1F6F43")

# ------------------------------------------------------------------ arrows + footer
axa.set_title("a   The held-out set — runs nested inside studies", loc="left", pad=4)
axb.set_title("b   Resampling runs — what not to do", loc="left", pad=4)
axc.set_title("c   Resampling BioProjects — what we do", loc="left", pad=4)

fig.text(0.005, -0.05,
         "Then: score the resample, repeat 1,000 times, and take the 2.5th and 97.5th "
         "percentiles of those 1,000 scores as the 95 % interval.\n"
         "Schematic: 6 projects, 23 runs, drawn at seed 42. The real held-out set is "
         "922 runs in 34 BioProjects, one of which holds 238 runs.",
         ha="left", va="top", fontsize=TICK, color=GREY)

fig.savefig(OUT / "resampling_unit.png", bbox_inches="tight")

r = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
print(f"run draw counts={counts_b.tolist()} (orig {SIZES}) | "
      f"projects drawn={[NAMES[i] for i in proj_draw]} | missing={missing} twice={twice}")
print("overlaps:", [(a.get_text()[:24], b.get_text()[:24])
                    for i, (a, ba) in enumerate(texts)
                    for b, bb in texts[i + 1:] if ba.overlaps(bb)])
