"""Pipeline figure v3: same visual language as the submitted Pipeline.png.

Icons are cropped from the original figure (results/paper_v2/pipeline_icons/,
extracted by connected components); box borders, arrows and greys reuse the
original palette. Content updated to the v9 design: fraction matrix, BioProject-
grouped split, per-field networks + logistic regression averaged, abstention.
"""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

BLUE = "#50b0e0"
PINK = "#c771a0"
ORANGE = "#e09000"
TEAL = "#0b3d3d"
GREY = "#606060"
INK = "#111111"

ICON_DIR = "results/paper_v2/pipeline_icons"
OUT = "results/paper_v2/Pipeline_v3.png"

BOX_LW = 2.2


def box(ax, cx, cy, w, h, color):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.018",
        linewidth=BOX_LW, edgecolor=color, facecolor="white", zorder=3,
    )
    ax.add_patch(p)


def icon(ax, name, cx, cy, h):
    img = np.array(Image.open(f"{ICON_DIR}/{name}.png"))
    zoom = (h * FIG_H_PX) / img.shape[0] / (DPI / 72.0) / 72.0 * 72.0
    zoom = (h * FIG_H_PX) / img.shape[0] * (72.0 / DPI)
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (cx, cy), frameon=False, zorder=4)
    ax.add_artist(ab)


def text_block(ax, x, cy, title, sub, ha="left", title_size=12.5, sub_size=10.5):
    lines_t = title.split("\n")
    lines_s = sub.split("\n") if sub else []
    n = len(lines_t) + len(lines_s)
    dy = 0.0165
    y = cy + dy * (n - 1) / 2
    for ln in lines_t:
        ax.text(x, y, ln, ha=ha, va="center", fontsize=title_size,
                fontweight="bold", color=INK, zorder=5)
        y -= dy
    for ln in lines_s:
        ax.text(x, y, ln, ha=ha, va="center", fontsize=sub_size, color=GREY, zorder=5)
        y -= dy


def arrow(ax, p0, p1, rad=0.0):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16, linewidth=2.0,
        color=TEAL, connectionstyle=f"arc3,rad={rad}", zorder=2,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


DPI = 300
FIG_W_IN, FIG_H_IN = 12.0, 12.5
FIG_H_PX = FIG_H_IN * DPI


def main():
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cxa, cxb = 0.225, 0.775
    wa = 0.40

    ax.text(0.03, 0.985, "a. Model training", fontsize=17, fontweight="bold",
            style="italic", ha="left", va="center", color=INK)
    ax.text(0.57, 0.985, "b. Inference on new sample", fontsize=17,
            fontweight="bold", style="italic", ha="left", va="center", color=INK)

    # --- column a ---
    y1, h1 = 0.905, 0.095
    box(ax, cxa, y1, wa, h1, BLUE)
    icon(ax, "database", cxa - 0.145, y1, 0.055)
    text_block(ax, cxa - 0.095, y1, "Sample unitigs\nfrom Logan",
               "n = 3,638 runs from\nAncientMetagenomeDir")

    y2, h2 = 0.745, 0.095
    box(ax, cxa, y2, wa, h2, BLUE)
    icon(ax, "grid", cxa - 0.145, y2, 0.050)
    text_block(ax, cxa - 0.095, y2, "Build unitig\nfraction matrix",
               "MUSET\n[3,638 × 110,202]")
    arrow(ax, (cxa, y1 - h1 / 2 - 0.006), (cxa, y2 + h2 / 2 + 0.006))

    y3, h3 = 0.585, 0.085
    wsplit = 0.19
    box(ax, cxa - 0.105, y3, wsplit, h3, PINK)
    text_block(ax, cxa - 0.105, y3, "Training set", "2,716 runs\n68 BioProjects",
               ha="center", title_size=12, sub_size=10)
    box(ax, cxa + 0.105, y3, wsplit, h3, PINK)
    text_block(ax, cxa + 0.105, y3, "Held-out\ntest set", "922 runs\n34 BioProjects",
               ha="center", title_size=12, sub_size=10)
    arrow(ax, (cxa - 0.04, y2 - h2 / 2 - 0.006), (cxa - 0.105, y3 + h3 / 2 + 0.006))
    arrow(ax, (cxa + 0.04, y2 - h2 / 2 - 0.006), (cxa + 0.105, y3 + h3 / 2 + 0.006))
    ax.text(cxa, y3 + h3 / 2 + 0.042, "split by\nBioProject", fontsize=9.5,
            color=GREY, ha="center", va="center", style="italic")

    y4, h4 = 0.425, 0.095
    box(ax, cxa, y4, wa, h4, BLUE)
    icon(ax, "robot", cxa - 0.145, y4, 0.055)
    text_block(ax, cxa - 0.095, y4, "Per-field\nmodel training",
               "4 networks + logistic\nregression, averaged")
    arrow(ax, (cxa - 0.105, y3 - h3 / 2 - 0.006), (cxa, y4 + h4 / 2 + 0.006))

    # --- column b ---
    box(ax, cxb, y1, wa, h1, PINK)
    icon(ax, "document", cxb - 0.145, y1, 0.055)
    text_block(ax, cxb - 0.095, y1, "New sample", "(FASTA / FASTQ)")

    box(ax, cxb, y2, wa, h2, BLUE)
    icon(ax, "counter", cxb - 0.145, y2, 0.050)
    text_block(ax, cxb - 0.095, y2, "Count reference\nk-mers in sample",
               "back_to_sequences")
    arrow(ax, (cxb, y1 - h1 / 2 - 0.006), (cxb, y2 + h2 / 2 + 0.006))

    yb3 = 0.585
    box(ax, cxb, yb3, wa, h2, BLUE)
    icon(ax, "sigma", cxb - 0.145, yb3, 0.050)
    text_block(ax, cxb - 0.095, yb3, "Aggregate to\nunitig fractions",
               "kmat_tools (Equation 1)")
    arrow(ax, (cxb, y2 - h2 / 2 - 0.006), (cxb, yb3 + h2 / 2 + 0.006))

    yb4, hb4 = 0.435, 0.075
    box(ax, cxb, yb4, 0.30, hb4, PINK)
    text_block(ax, cxb, yb4, "Feature vector", "[1 × 110,202]", ha="center")
    arrow(ax, (cxb, yb3 - h2 / 2 - 0.006), (cxb, yb4 + hb4 / 2 + 0.006))

    # --- reference files between the columns ---
    rk_y = y2
    wref, href = 0.125, 0.055
    arrow(ax, (cxa + wa / 2 + 0.004, rk_y), (cxb - wa / 2 - 0.004, rk_y))
    box(ax, 0.50, rk_y, wref, href, PINK)
    text_block(ax, 0.50, rk_y, "Ref k-mers", "", ha="center", title_size=11)
    p0 = (cxa + wa / 2 + 0.004, y2 - 0.03)
    p1 = (cxb - wa / 2 - 0.004, yb3)
    arrow(ax, p0, p1)
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    box(ax, mid[0], mid[1], wref, href, PINK)
    text_block(ax, mid[0], mid[1], "Ref unitigs", "", ha="center", title_size=11)

    # --- trained model ---
    yt, ht = 0.265, 0.115
    box(ax, 0.50, yt, 0.26, ht, ORANGE)
    icon(ax, "target", 0.50, yt + 0.022, 0.050)
    ax.text(0.50, yt - 0.033, "Trained\nDIANA model", fontsize=12.5,
            fontweight="bold", ha="center", va="center", color=INK, zorder=5)
    arrow(ax, (cxa, y4 - h4 / 2 - 0.006), (0.405, yt + 0.025), rad=0.25)
    arrow(ax, (cxb, yb4 - hb4 / 2 - 0.006), (0.595, yt + 0.025), rad=-0.25)

    # --- output ---
    yo, ho = 0.095, 0.15
    box(ax, 0.50, yo, 0.44, ho, PINK)
    text_block(
        ax, 0.50, yo + 0.030,
        "Predicted metadata + confidence",
        "Community type: oral     Sample host: $\\it{Homo\\ sapiens}$\n"
        "Material: dental calculus     Feature: tooth",
        ha="center", title_size=12.5, sub_size=10.5,
    )
    ax.text(0.50, yo - 0.038,
            'low confidence → “no call” · stated label checked '
            '· unseen label → “cannot assess”',
            fontsize=9.5, color=GREY, ha="center", va="center", style="italic",
            zorder=5)
    arrow(ax, (0.50, yt - ht / 2 - 0.006), (0.50, yo + ho / 2 + 0.006))

    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
