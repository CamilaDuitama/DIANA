"""Results figures of the v19 held-out evaluation (plotly; colours and template from config.py).

  results_classification_grid.png  per field and class group (many-, medium-, few-shot), DIANA against each
                                   baseline, macro-F1 and balanced accuracy: the cell takes the colour of the model that is
                                   better after Benjamini-Hochberg over all cells of the figure (grey: no significant difference) and shows DIANA minus the
                                   baseline (results/multiplicity_v19/net_classification_heldout/paired.tsv; released DIANA, read 5); no row is hatched:
                                   the five-BioProject rule was dropped with the permutation test (PROTOCOLS.md 2026-10-26)
  results_detection_grid.png       the same for the share of planted mislabels caught at 5 % false flags
                                   (results/multiplicity_v19/net_regime_single_paired.tsv)
  results_reliability_<field>.png      DIANA's reliability curve on the held-out set
                                       (results/heldout_v19/diana_net_prior/reliability_test.tsv; confidence with the class prior added back)
  results_performance.png          DIANA minus each baseline in held-out balanced accuracy per field, with the band of differences
                                   the BioProject permutation test does not call significant after Benjamini-Hochberg
                                   (results/multiplicity_v19/net_classification_heldout/figure3_bands.tsv, 87_)
  results_real_mislabels.png       real corrections replayed (results/heldout_v19/real_mislabels_net_prior/replay.tsv): per model,
                                   corrections flagged at 5 / 10 / 20 % false flags (of those whose superseded label is a training
                                   class) and corrections whose predicted label is the corrected one; counts, no interval (two studies)
Intervals: 95 % over 2,000 paired resamples of held-out BioProjects. Written to results/paper_v19/.

    ./env/bin/python scripts/paper/generate_results_figures_v19.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from config import PLOT_CONFIG  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from diana.evaluation import scoring as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_v19"
C, N = PLOT_CONFIG["colors"]["models"], PLOT_CONFIG["colors"]["model_names"]
FIELDS = {"community_type": "Community type", "feature": "Feature", "sample_host": "Sample host", "material": "Material"}
REGIMES = ["many", "medium", "few"]
FONT = dict(size=14, color="#333333")


def lighten(colour: str, w: float = 0.65) -> str:
    """Blend a colour (#rrggbb or rgb(r, g, b)) with white; w is the white share."""
    c = colour.strip()
    rgb = [int(c[i:i + 2], 16) for i in (1, 3, 5)] if c.startswith("#") else [int(x) for x in c[c.index("(") + 1:-1].split(",")]
    return "rgb({}, {}, {})".format(*[round(v + (255 - v) * w) for v in rgb])


GROUPS = {"many": "many-shot", "medium": "medium-shot", "few": "few-shot"}   # whole-field values are in Tables 1 and 2
TIE = "#e6e6e6"


def heldout_studies() -> dict:
    """(field, class group) -> number of held-out BioProjects whose runs are scored in that group."""
    tr = pd.read_csv(ROOT / "data/splits_v19/train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    te = pd.read_csv(ROOT / "data/splits_v19/test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    out = {}
    for f in FIELDS:
        el = S.eligible_classes(ROOT / "data/splits_v19/class_eligibility.tsv", f)
        groups = S.regime_classes(tr.set_index("Run_accession")[f], el)
        for g, cls in groups.items():
            out[(f, g)] = te[te[f].isin(cls & set(tr[f].dropna()))].archive_project.nunique()
    return out


def winner_grid(d: pd.DataFrame, cols: list[tuple[str, str, str]], out: Path, studies: dict, min_studies: int = 0) -> None:
    """Rows: field x class group; columns: (metric, baseline, header). Benjamini-Hochberg over every cell of the figure;
    a cell is coloured with the better model when its adjusted p-value is below 0.05 (grey otherwise) and shows DIANA
    minus the baseline. Cells of rows with fewer than `min_studies` held-out studies are hatched and left grey (no verdict);
    they stay in the Benjamini-Hochberg family."""
    rows = [(t, g) for t in FIELDS for g in GROUPS if ((d.task == t) & (d.regime == g)).any()]
    order = ["tie", "DIANA", "LogReg", "RandomForest", "kNN"]
    colours = [TIE] + [C[m] for m in order[1:]]
    colours += [lighten(c) for c in colours]       # light versions for rows resting on fewer than min_studies studies
    scale = [[k / len(colours), c] for i, c in enumerate(colours) for k in (i, i + 1)]
    cells = {}
    for t, g in rows:
        for metric, b, _ in cols:
            r = d[(d.task == t) & (d.regime == g) & (d.metric == metric) & (d.b == b)]
            if not r.empty and pd.notna(r.delta.iloc[0]):
                cells[(t, g, metric, b)] = r.iloc[0]
    keys = list(cells)
    q = dict(zip(keys, S.bh_adjust([cells[k].p for k in keys])))
    z, text = [], []
    for t, g in rows:
        zr, tr = [], []
        for metric, b, _ in cols:
            k = (t, g, metric, b)
            if k not in cells:
                zr.append(None); tr.append("")
                continue
            r = cells[k]
            win = "tie" if not q[k] < 0.05 or studies[(t, g)] < min_studies else ("DIANA" if r.delta > 0 else b)   # hatched rows: no verdict
            v = f"{r.delta:+.2f}"
            light = len(order) if studies[(t, g)] < min_studies else 0
            zr.append(order.index(win) + light); tr.append("0.00" if v in ("+0.00", "-0.00") else v)
        z.append(zr); text.append(tr)
    labels = [f"{FIELDS[t]}, {GROUPS[g]} ({studies[(t, g)]})" for t, g in rows]
    xs, ys = list(range(len(cols))), list(range(len(rows)))
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, colorscale=scale, zmin=-0.5, zmax=len(colours) - 0.5, showscale=False,
                               xgap=3, ygap=3, hoverinfo="skip"))
    few = [i for i, (t, g) in enumerate(rows) if studies[(t, g)] < min_studies]
    hatch = dict(shape="/", fgcolor="#a6a6a6", bgcolor="rgba(0,0,0,0)", size=9, solidity=0.12)
    px, py = [], []                            # diagonal hatching on rows resting on fewer than min_studies held-out studies
    for i in few:
        for j in xs:
            px += [j - 0.49, j + 0.49, j + 0.49, j - 0.49, j - 0.49, None]
            py += [i - 0.45, i - 0.45, i + 0.45, i + 0.45, i - 0.45, None]
    if few:
        fig.add_scatter(x=px, y=py, mode="lines", fill="toself", line=dict(width=0), fillcolor="rgba(0,0,0,0)",
                        fillpattern=hatch, showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[j for _ in ys for j in xs], y=[i for i in ys for _ in xs], mode="text", text=[v for r in text for v in r],
                    textfont=dict(size=13, color="#222222"), showlegend=False, hoverinfo="skip")
    for m, label in [("DIANA", "DIANA better"), *[(b, f"{N[b]} better") for b in dict.fromkeys(c[1] for c in cols)], ("tie", "no significant difference")]:
        fig.add_scatter(x=[None], y=[None], mode="markers", name=label,
                        marker=dict(symbol="square", size=14, color=TIE if m == "tie" else C[m]))
    if few:
        fig.add_scatter(x=[-5, -4, -4, -5, -5], y=[-5, -5, -4, -4, -5], mode="lines", fill="toself", line=dict(width=0),
                        fillcolor=lighten(TIE), fillpattern=dict(hatch, bgcolor=lighten(TIE)), name=f"fewer than {min_studies} held-out BioProjects")
    fig.update_layout(template=PLOT_CONFIG["template"], width=290 + 150 * len(cols), height=150 + 34 * len(rows), font=FONT,
                      xaxis=dict(side="top", showgrid=False, zeroline=False, tickangle=0, tickvals=xs, ticktext=[c[2] for c in cols],
                                 range=[-0.5, len(cols) - 0.5]),
                      yaxis=dict(showgrid=False, zeroline=False, tickvals=ys, ticktext=labels, range=[len(rows) - 0.5, -0.5]),
                      legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0), margin=dict(l=270, r=20, t=110, b=90))
    fig.write_image(out, scale=2)
    n_sig = sum(1 for k in keys if q[k] < 0.05)
    print(f"{out.name}: {len(keys)} cells, {sum(1 for k in keys if cells[k].p < 0.05)} with p < 0.05, {n_sig} with BH q < 0.05")


def reliability(rel: pd.DataFrame) -> None:
    for task, title in FIELDS.items():
        r = rel[(rel.task == task) & (rel.runs > 0)]
        mid = (r.bin_low + r.bin_high) / 2
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#888888", dash="dash", width=1), showlegend=False)
        fig.add_scatter(x=r.mean_confidence, y=r.accuracy, mode="lines+markers", name="DIANA",
                        marker=dict(color=C["DIANA"], size=[max(6, min(18, 3 * np.sqrt(n))) for n in r.runs]),
                        line=dict(color=C["DIANA"], width=2), text=[f"{int(n)} runs" for n in r.runs], showlegend=False)
        fig.update_layout(template=PLOT_CONFIG["template"], title=title, width=420, height=420, font=FONT,
                          xaxis=dict(title="mean confidence (predicted label)", range=[0, 1]),
                          yaxis=dict(title="observed accuracy", range=[0, 1]), margin=dict(l=70, r=20, t=60, b=60))
        fig.write_image(OUT / f"results_reliability_{task}.png", scale=2)


def real_corrections(rep: pd.DataFrame) -> None:
    """Grouped bars: per model, real corrections flagged at each budget and corrections predicted correctly (counts)."""
    models = ["DIANA", "LogReg", "RandomForest"]
    ok = rep[rep.verdict != "cannot assess"]
    n_ok, n_all = ok.groupby("model").size(), rep.groupby("model").size()
    if n_ok.nunique() != 1 or n_all.nunique() != 1:
        raise ValueError("models differ in the corrections they are scored on")
    n_ok, n_all = int(n_ok.iloc[0]), int(n_all.iloc[0])
    cats = [(f"flagged at {b} %<br>(of {n_ok})", lambda d, b=b: int(d[d.verdict != "cannot assess"][f"flagged_{b}pct"].sum()))
            for b in (5, 10, 20)]
    cats.append((f"predicted the<br>corrected label<br>(of {n_all})", lambda d: int(d.prediction_is_corrected_label.sum())))
    fig = go.Figure()
    for m in models:
        d = rep[rep.model == m]
        y = [f(d) for _, f in cats]
        fig.add_bar(x=[c for c, _ in cats], y=y, name=N[m], marker_color=C[m], text=y, textposition="outside")
    fig.update_layout(template=PLOT_CONFIG["template"], barmode="group", width=720, height=440, font=FONT,
                      yaxis=dict(title="corrections", range=[0, n_all + 2]), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      margin=dict(l=70, r=20, t=50, b=90))
    fig.write_image(OUT / "results_real_mislabels.png", scale=2)


def performance(bands: pd.DataFrame) -> None:
    """DIANA minus each baseline in held-out balanced accuracy per field (whole field). Bar: the differences the
    BioProject permutation test does not call significant at the Benjamini-Hochberg threshold of the figure's 16
    comparisons (87_); filled marker outside its bar = significant, open marker = not significant."""
    baselines = ["LogReg", "RandomForest", "kNN", "Majority"]
    d = bands.set_index(["task", "b"])
    finite = d.halfwidth[np.isfinite(d.halfwidth)]
    top = 1.1 * max(finite.max() if len(finite) else 0, d.delta.abs().max())
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#888888", dash="dash", width=1))
    for i, b in enumerate(baselines):
        x = [j + (i - 1.5) * 0.17 for j in range(len(FIELDS))]
        r = d.loc[[(t, b) for t in FIELDS]]
        for xi, h in zip(x, r.halfwidth):
            h = min(h, top)
            fig.add_scatter(x=[xi, xi], y=[-h, h], mode="lines", showlegend=False, opacity=0.35, line=dict(color=C[b], width=7))
        fig.add_scatter(x=[None], y=[None], mode="markers", name=f"vs {N[b]}", marker=dict(color=C[b], size=11))
        fig.add_scatter(x=x, y=r.delta, mode="markers", showlegend=False,
                        marker=dict(color=[C[b] if v else "white" for v in r.significant], size=11, line=dict(color=C[b], width=2)))
    for label, fill in (("significant", "#888888"), ("not significant", "white")):
        fig.add_scatter(x=[None], y=[None], mode="markers", name=label, marker=dict(color=fill, size=11, line=dict(color="#888888", width=2)))
    fig.add_scatter(x=[None], y=[None], mode="lines", name="not significant if inside", opacity=0.35, line=dict(color="#888888", width=7))
    fig.update_layout(template=PLOT_CONFIG["template"], width=820, height=460, font=FONT,
                      xaxis=dict(tickvals=list(range(len(FIELDS))), ticktext=list(FIELDS.values()), showgrid=False, zeroline=False),
                      yaxis=dict(title="DIANA minus baseline, balanced accuracy", zeroline=False, range=[-top, top]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(l=70, r=20, t=50, b=50))
    fig.write_image(OUT / "results_performance.png", scale=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    studies = heldout_studies()
    M = ROOT / "results/multiplicity_v19"
    cls = pd.read_csv(M / "net_classification_heldout/paired.tsv", sep="\t").rename(columns={"p_perm": "p"})   # permutation test (PROTOCOLS.md 2026-10-26)
    short = {"LogReg": "logistic<br>regression", "RandomForest": "random<br>forest", "kNN": "k-NN"}
    cols = [(met, b, f"{lab}<br>vs {short[b]}") for met, lab in (("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy"))
            for b in ("LogReg", "RandomForest", "kNN")]
    winner_grid(cls, cols, OUT / "results_classification_grid.png", studies)
    det = pd.read_csv(M / "net_regime_single_paired.tsv", sep="\t").drop(columns=["recall5_p_boot"]).rename(columns={"recall5_p_perm": "recall5_p_boot"})
    det = det[det.a == "DIANA"].rename(columns={"recall5_delta": "delta", "recall5_ci_low": "ci_low", "recall5_ci_high": "ci_high",
                                                "recall5_p_boot": "p"}).assign(metric="recall5")
    winner_grid(det, [("recall5", b, f"vs {short[b]}") for b in ("LogReg", "RandomForest")], OUT / "results_detection_grid.png", studies)
    reliability(pd.read_csv(ROOT / "results/heldout_v19/diana_net_prior/reliability_test.tsv", sep="\t"))
    performance(pd.read_csv(M / "net_classification_heldout/figure3_bands.tsv", sep="\t"))
    real_corrections(pd.read_csv(ROOT / "results/heldout_v19/real_mislabels_net_prior/replay.tsv", sep="\t"))


if __name__ == "__main__":
    main()
