"""Held-out paired differences, DIANA minus each baseline, per field (candidate figure, v19).

Two plots: macro-F1 difference (results/heldout_v19/scores_read2/classification_paired.tsv) and
difference in the share of planted mislabels caught, mean over the budgets both models can operate
at (results/rescore_v19/read2_flag_single/flagging_paired.tsv). 95 % intervals from 2,000 paired
resamples of held-out BioProjects. Colours as in 34_generate_generalisation_gap_plots.py.

    ./env/bin/python scripts/paper/generate_paired_v19.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from config import PLOT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_v19"
V = PLOT_CONFIG["colors"]["palette"]
C = PLOT_CONFIG["colors"]["models"]
BASE = {"LogReg": ("vs logistic regression", C["LogReg"], 0.15), "RandomForest": ("vs random forest", C["RandomForest"], -0.15)}
FIELDS = {"community_type": "Community type", "feature": "Feature", "sample_host": "Sample host", "material": "Material"}
PLOTS = {"classification": ("results/heldout_v19/scores_read2/classification_paired.tsv", "delta", "DIANA minus baseline, macro-F1"),
         "detection": ("results/rescore_v19/read2_flag_single/flagging_paired.tsv", "delta_choosing",
                       "DIANA minus baseline, planted mislabels caught (mean over 5, 10, 20 %)")}


REGIME = {"classification": ("f1_delta", "f1_ci_low", "f1_ci_high", "DIANA minus baseline, macro-F1"),
          "detection": ("recall5_delta", "recall5_ci_low", "recall5_ci_high", "DIANA minus baseline, planted mislabels caught (5 %)")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regime", choices=["few", "medium", "many"], default=None, help="one regime instead of the whole field")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ypos = {f: i for i, f in enumerate(reversed(list(FIELDS)))}
    for name, (path, col, xtitle) in PLOTS.items():
        if a.regime is None:
            d = pd.read_csv(ROOT / path, sep="\t")
        else:   # regime pairs: recall at 5 % (68_), renamed to the whole-field column names
            col_, lo, hi, xtitle = REGIME[name]
            d = pd.read_csv(ROOT / "results/rescore_v19/read2_regime_single_paired.tsv", sep="\t")
            d = d[d.regime == a.regime].rename(columns={col_: col, lo: "ci_low", hi: "ci_high"})
        d = d[d.a == "DIANA"]
        lim = max(0.4, float(np.ceil(max(d.ci_high.abs().max(), d.ci_low.abs().max()) * 10 + 0.5) / 10))   # every interval inside the axis
        fig = go.Figure()
        fig.add_vline(x=0, line=dict(color="#888888", width=1, dash="dash"))
        for b, (label, colour, off) in BASE.items():
            s = d[d.b == b].set_index("task").loc[list(FIELDS)]
            fig.add_scatter(x=s[col], y=[ypos[t] + off for t in s.index], mode="markers", name=label,
                            marker=dict(color=colour, size=12, line=dict(color="white", width=1)),
                            error_x=dict(type="data", symmetric=False, array=s.ci_high - s[col], arrayminus=s[col] - s.ci_low,
                                         color=colour, thickness=1.8))
        fig.update_layout(template=PLOT_CONFIG["template"], width=720, height=420, font=dict(size=14, color="#333333"),
                          xaxis=dict(title=xtitle, range=[-lim, lim], zeroline=False),
                          yaxis=dict(tickvals=list(ypos.values()), ticktext=[FIELDS[f] for f in ypos], range=[-0.5, 3.5]),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                          margin=dict(l=130, r=20, t=50, b=70))
        if a.regime is not None:
            fig.update_layout(title=f"{a.regime}-shot classes", margin=dict(t=80))
        fig.write_image(OUT / (f"paired_{name}.png" if a.regime is None else f"paired_{a.regime}_{name}.png"), scale=2)
    print("wrote", *sorted(OUT.glob("paired_*.png")))


if __name__ == "__main__":
    main()
