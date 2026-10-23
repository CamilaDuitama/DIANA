"""Held-out classification against mislabel detection, one plot per field (candidate main figure, v19).

x: macro-F1 over the evaluable classes (results/heldout_v19/scores_read2/classification.tsv);
y: share of planted mislabels caught at a 5 % false-flag budget, single-label benchmark
(results/rescore_v19/read2_flag_single/flagging_scores.tsv). Error bars: 95 % intervals over
held-out BioProjects. Models that produce flag scores at 5 %: DIANA, logistic regression, random
forest (k-NN and the majority class cannot operate at 5 % and have no y value). Colours as in
34_generate_generalisation_gap_plots.py; template from config.py. Writes results/paper_v19/plane_<field>.png, or plane_<regime>_<field>.png with --regime
(results/rescore_v19/read2_regime_single.tsv).

    ./env/bin/python scripts/paper/generate_plane_v19.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from config import PLOT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_v19"
V = PLOT_CONFIG["colors"]["palette"]
C, N = PLOT_CONFIG["colors"]["models"], PLOT_CONFIG["colors"]["model_names"]
MODELS = {m: (N[m], C[m]) for m in ("DIANA", "LogReg", "RandomForest")}
TITLES = {"community_type": "Community type", "feature": "Feature", "sample_host": "Sample host", "material": "Material"}


def values(regime: str | None):
    """(task, model) -> (F1, low, high, recall at 5 %, low, high), whole field or one regime."""
    if regime is None:
        c = pd.read_csv(ROOT / "results/heldout_v19/scores_read2/classification.tsv", sep="\t").set_index(["task", "model"])
        f = pd.read_csv(ROOT / "results/rescore_v19/read2_flag_single/flagging_scores.tsv", sep="\t").set_index(["task", "model"])
        return lambda t, m: (*c.loc[(t, m), ["f1_macro_eligible", "ci_low", "ci_high"]],
                             *f.loc[(t, m), ["recall_at_5pct", "recall_at_5pct_ci_low", "recall_at_5pct_ci_high"]])
    r = pd.read_csv(ROOT / "results/rescore_v19/read2_regime_single.tsv", sep="\t")
    r = r[r.regime == regime].set_index(["task", "model"])
    return lambda t, m: tuple(r.loc[(t, m), ["macro_f1", "f1_ci_low", "f1_ci_high", "recall_5pct", "recall_5pct_ci_low", "recall_5pct_ci_high"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regime", choices=["few", "medium", "many"], default=None, help="one regime instead of the whole field")
    a = ap.parse_args()
    get = values(a.regime)
    OUT.mkdir(parents=True, exist_ok=True)
    for task, title in TITLES.items():
        fig = go.Figure()
        for m, (name, colour) in MODELS.items():
            x, xl, xh, y, yl, yh = get(task, m)
            fig.add_scatter(x=[x], y=[y], mode="markers", name=name, marker=dict(color=colour, size=13, line=dict(color="white", width=1)),
                            error_x=dict(type="data", symmetric=False, array=[xh - x], arrayminus=[x - xl], color=colour, thickness=1.5),
                            error_y=dict(type="data", symmetric=False, array=[yh - y], arrayminus=[y - yl], color=colour, thickness=1.5))
        fig.update_layout(template=PLOT_CONFIG["template"], title=title if a.regime is None else f"{title}, {a.regime}-shot classes", width=520, height=480, font=dict(size=14, color="#333333"),
                          xaxis=dict(title="macro-F1", range=[0, 1]),
                          yaxis=dict(title="planted mislabels caught (5 % false flags)", range=[0, 1]),
                          legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0), margin=dict(l=70, r=20, t=90, b=60))
        fig.write_image(OUT / (f"plane_{task}.png" if a.regime is None else f"plane_{a.regime}_{task}.png"), scale=2)
    print("wrote", *sorted(OUT.glob("plane_*.png")))


if __name__ == "__main__":
    main()
