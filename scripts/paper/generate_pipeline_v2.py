"""Pipeline figure v2 (main Figure 1), plotly, with the manuscript's Vivid palette and template
(scripts/paper/config.py), per CLAUDE.md; panels a/b read downwards.

Panel a: training corpus -> unitig fractions -> DIANA (one network shared by the four fields). Panel b: a new sample -> projection onto the
training vocabulary -> DIANA -> label or "no call", flag or "cannot assess". Icons from the submitted
Pipeline.png (results/paper_v2/pipeline_icons/). Same layout as the earlier matplotlib render.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import PLOT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ICON_DIR = ROOT / "results/paper_v2/pipeline_icons"
OUT = [ROOT / "results/paper_v2/Pipeline_v2.png",
       Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA/figures/Pipeline_v2.png")]
V = PLOT_CONFIG["colors"]["palette"]
# (edge, fill): palette edges, white fills
BLUE, INDIGO = (V[7], "white"), (V[1], "white")
SAGE = (V[5], "white")    # data boxes (dark teal; grey sage was indistinguishable from teal under deuteranopia)
ORANGE, TEAL, LIME, RED = (V[0], "white"), (V[2], "white"), (V[3], "white"), (V[9], "white")
INK, GREYT, ARROW = "#333333", "#555555", "#444444"
W, H = 1300, 920                     # pixels; written at scale 3
XMAX, YMAX = 2.08, 1.06              # data range: panel a in x 0..1, panel b in x 1.08..2.08
B = 1.08                             # x offset of panel b
PX = 1.39                            # matplotlib points -> plotly pixels at this size


def box(fig, cx, cy, w, h, colors, r=0.02):
    edge, fill = colors
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    ry = r * (W / XMAX) / (H / YMAX)          # same corner radius in pixels on both axes
    path = (f"M {x0 + r},{y0} L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - r},{y1} "
            f"L {x0 + r},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + r},{y0} Z")
    fig.add_shape(type="path", path=path, line=dict(color=edge, width=6), fillcolor=fill, layer="below")


def label(fig, cx, cy, lines, size=14, bold=False, color=INK, dy=0.052):
    top = cy + dy * (len(lines) - 1) / 2
    for i, text in enumerate(lines):
        fig.add_annotation(x=cx, y=top - i * dy, text=f"<b>{text}</b>" if bold else text, showarrow=False,
                           font=dict(size=size * PX, color=color), xanchor="center", yanchor="middle")


def icon(fig, name, cx, cy, h):
    img = np.array(Image.open(ICON_DIR / f"{name}.png").convert("RGBA"))
    img[(img[:, :, :3] > 240).all(axis=2), 3] = 0          # white background transparent
    px_x, px_y = (W - 20) / XMAX, (H - 20) / YMAX       # plot-area pixels per data unit (margins 10 px)
    w = h * img.shape[1] / img.shape[0] * px_y / px_x   # icon width in x units at its own aspect ratio
    fig.add_layout_image(source=Image.fromarray(img), x=cx, y=cy, xref="x", yref="y", sizex=w, sizey=h,
                         sizing="stretch", xanchor="center", yanchor="middle", layer="above")


def arrow(fig, p0, p1):
    fig.add_annotation(x=p1[0], y=p1[1], ax=p0[0], ay=p0[1], xref="x", yref="y", axref="x", ayref="y", text="",
                       showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=3, arrowcolor=ARROW)


def icon_box(fig, cx, cy, w, h, colors, name, icon_h, lines):
    """A box with its icon at a fixed inset from the left edge and the text centred in the remaining width."""
    box(fig, cx, cy, w, h, colors)
    left, inset = cx - w / 2, 0.09
    icon(fig, name, left + inset, cy, icon_h)
    label(fig, (left + 2 * inset + cx + w / 2) / 2, cy, lines, size=15, bold=True, dy=0.055)


def panel_a(fig):
    label(fig, 0.02, 1.00, ["a  Training, once"], size=17, bold=True)
    fig.layout.annotations[-1].xanchor = "left"
    icon_box(fig, 0.52, 0.88, 0.72, 0.13, INDIGO, "database", 0.055, ["Training corpus", "2,775 runs, 67 BioProjects"])
    arrow(fig, (0.52, 0.815), (0.52, 0.755))
    icon_box(fig, 0.52, 0.665, 0.86, 0.14, SAGE, "grid", 0.055, ["Unitig fractions of 85,986 unitigs", "+ curated labels"])
    arrow(fig, (0.52, 0.59), (0.52, 0.525))
    box(fig, 0.52, 0.35, 0.62, 0.30, TEAL)                   # the DIANA box holding the network
    label(fig, 0.46, 0.445, ["DIANA"], size=19, bold=True)
    icon(fig, "robot", 0.61, 0.445, 0.05)
    box(fig, 0.52, 0.31, 0.46, 0.14, BLUE)
    label(fig, 0.52, 0.31, ["Shared network", "4 fields, 5 seeds"], size=15, bold=True, dy=0.05)
    label(fig, 0.52, 0.015, ["All tuning on BioProject-grouped folds", "inside the training corpus"], size=13, color=GREYT, dy=0.03)


def panel_b(fig):
    label(fig, B + 0.02, 1.00, ["b  A new sample (diana-predict)"], size=17, bold=True)
    fig.layout.annotations[-1].xanchor = "left"
    icon_box(fig, B + 0.52, 0.88, 0.76, 0.13, INDIGO, "document", 0.055, ["New sample or held-out run", "Reads or unitigs"])
    arrow(fig, (B + 0.52, 0.815), (B + 0.52, 0.755))
    icon_box(fig, B + 0.52, 0.665, 0.62, 0.14, SAGE, "sigma", 0.05, ["Projection onto the", "training vocabulary"])
    arrow(fig, (B + 0.52, 0.59), (B + 0.52, 0.525))
    box(fig, B + 0.52, 0.45, 0.34, 0.14, TEAL)
    label(fig, B + 0.46, 0.45, ["DIANA"], size=19, bold=True)
    icon(fig, "target", B + 0.615, 0.45, 0.05)
    arrow(fig, (B + 0.44, 0.375), (B + 0.27, 0.27))
    arrow(fig, (B + 0.60, 0.375), (B + 0.76, 0.27))
    box(fig, B + 0.26, 0.175, 0.44, 0.16, LIME)
    label(fig, B + 0.26, 0.175, ["Label + confidence", "or “no call”"], size=16, bold=True, dy=0.055)
    box(fig, B + 0.765, 0.175, 0.44, 0.16, RED)
    label(fig, B + 0.765, 0.175, ["Flag = 1 − P(label)", "or “cannot assess”"], size=16, bold=True, dy=0.055)
    label(fig, B + 0.52, 0.015, ["The 863 held-out runs from 35 unseen BioProjects", "follow this path"],
          size=13, color=GREYT, dy=0.03)


def main():
    fig = go.Figure()
    panel_a(fig)
    panel_b(fig)
    fig.update_xaxes(visible=False, range=[0, XMAX])
    fig.update_yaxes(visible=False, range=[-0.03, YMAX - 0.03])
    fig.update_layout(template=PLOT_CONFIG["template"], width=W, height=H, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(color=INK))
    for p in OUT:
        fig.write_image(p, scale=3)
    print("wrote", *OUT)


if __name__ == "__main__":
    main()
