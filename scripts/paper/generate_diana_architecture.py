"""Supplementary architecture diagram of DIANA (plotly, Vivid palette, per CLAUDE.md).

Unitig fractions -> one 5-seed multilayer perceptron with hidden layers shared by the four fields and
one output head per field; the community-type head is averaged 0.5/0.5 with a logistic regression;
-> class probabilities per field (label / no call / flag).
"""

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
OUT = [ROOT / "results/paper_v2/diana_architecture.png",
       Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA/figures/diana_architecture.png")]

V = px.colors.qualitative.Vivid
BLUE, ORANGE, INDIGO, TEAL, GOLD = V[7], V[0], V[1], V[2], V[6]
INK, GREY = "#333333", "#666666"
FILL = {BLUE: "#e3eef7", INDIGO: "#e7e9f5", ORANGE: "#fcecd8", TEAL: "#e7f5f0", GOLD: "#faf2dd"}
TITLE_SIZE = 17


def rbox(fig, x0, y0, x1, y1, color, r=0.018, dash=None):
    p = (f"M {x0+r},{y0} L {x1-r},{y0} Q {x1},{y0} {x1},{y0+r} L {x1},{y1-r} "
         f"Q {x1},{y1} {x1-r},{y1} L {x0+r},{y1} Q {x0},{y1} {x0},{y1-r} "
         f"L {x0},{y0+r} Q {x0},{y0} {x0+r},{y0} Z")
    fig.add_shape(type="path", path=p, line=dict(color=color, width=2.4, dash=dash),
                  fillcolor="rgba(0,0,0,0)" if dash else FILL[color], layer="below")


def txt(fig, x, y, text, size=13, color=INK, bold=False):
    fig.add_annotation(x=x, y=y, text=f"<b>{text}</b>" if bold else text, showarrow=False,
                       font=dict(size=size, color=color), align="center")


def seg(fig, pts, arrow=True):
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        fig.add_shape(type="line", x0=xa, y0=ya, x1=xb, y1=yb,
                      line=dict(color="#555555", width=2))
    if arrow:
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        fig.add_annotation(x=xb, y=yb, ax=xa, ay=ya, xref="x", yref="y", axref="x",
                           ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2,
                           arrowwidth=2, arrowcolor="#555555", text="")


def main() -> None:
    fig = go.Figure()
    rbox(fig, 0.02, 0.38, 0.19, 0.62, GOLD)
    txt(fig, 0.105, 0.545, "unitig fractions", TITLE_SIZE, bold=True)
    txt(fig, 0.105, 0.43, "1 × 85,986", 12, GREY)

    for off in (0.028, 0.014, 0.0):
        rbox(fig, 0.27 + off, 0.46 + off, 0.52 + off, 0.86 + off, INDIGO)
    txt(fig, 0.395, 0.775, "multilayer perceptron", TITLE_SIZE, bold=True)
    txt(fig, 0.395, 0.665, "hidden layers shared by the four fields", 11, GREY)
    txt(fig, 0.395, 0.565, "5 seeds, probabilities averaged", 11, GREY)

    rbox(fig, 0.27, 0.06, 0.52, 0.30, ORANGE)
    txt(fig, 0.395, 0.215, "logistic regression", TITLE_SIZE, bold=True)
    txt(fig, 0.395, 0.12, "class-balanced, community type only", 11, GREY)

    heads = {"feature": 0.85, "sample host": 0.69, "material": 0.53, "community type": 0.37}
    for name, y in heads.items():
        rbox(fig, 0.60, y - 0.055, 0.76, y + 0.055, INDIGO, r=0.012)
        txt(fig, 0.68, y, f"{name} head", 12)
        seg(fig, [(0.565, y), (0.60, y)])
    seg(fig, [(0.52, 0.66), (0.565, 0.66)], arrow=False)
    seg(fig, [(0.565, heads["community type"]), (0.565, heads["feature"])], arrow=False)

    rbox(fig, 0.82, 0.10, 0.93, 0.42, TEAL)
    txt(fig, 0.875, 0.31, "average", TITLE_SIZE, bold=True)
    txt(fig, 0.875, 0.20, "½  +  ½", 12, GREY)

    rbox(fig, 0.985, 0.10, 1.20, 0.92, BLUE)
    txt(fig, 1.0925, 0.70, "class probabilities", TITLE_SIZE, bold=True)
    txt(fig, 1.0925, 0.62, "per field", 12, GREY)
    txt(fig, 1.0925, 0.46, "label + confidence · “no call”", 11, GREY)
    txt(fig, 1.0925, 0.36, "flag = 1 − P(stated label)", 11, GREY)

    seg(fig, [(0.19, 0.50), (0.225, 0.50)], arrow=False)
    seg(fig, [(0.225, 0.50), (0.225, 0.66), (0.27, 0.66)])
    seg(fig, [(0.225, 0.50), (0.225, 0.18), (0.27, 0.18)])
    for name in ("feature", "sample host", "material"):
        seg(fig, [(0.76, heads[name]), (0.985, heads[name])])
    seg(fig, [(0.76, heads["community type"]), (0.79, heads["community type"]), (0.79, 0.31), (0.82, 0.31)])
    seg(fig, [(0.52, 0.18), (0.79, 0.18), (0.79, 0.21), (0.82, 0.21)])
    seg(fig, [(0.93, 0.26), (0.985, 0.26)])

    fig.update_xaxes(visible=False, range=[0, 1.22])
    fig.update_yaxes(visible=False, range=[0, 1.0])
    fig.update_layout(template="plotly_white", width=1150, height=420,
                      margin=dict(l=8, r=8, t=8, b=8), font=dict(color=INK),
                      plot_bgcolor="white")
    for p in OUT:
        fig.write_image(p, scale=2)
    print("wrote", *OUT)


if __name__ == "__main__":
    main()
