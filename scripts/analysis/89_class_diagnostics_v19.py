#!/usr/bin/env python3
"""Per-class held-out F1 against training coverage, and training runs whose label every model contradicts (v19, descriptive).

Writes three plots (one per file) and their tables to results/diagnostics_v19/:
  f1_vs_training_runs.png         DIANA held-out F1 per eligible class against its training runs
  f1_vs_training_bioprojects.png  the same against its training BioProjects
  all_models_disagree_by_material.png  share of training runs (out-of-fold) where DIANA, logistic regression,
                                  random forest and k-NN agree on one label that differs from the curated one, in any
                                  field, by material (materials with at least 30 training runs)
Inputs: results/heldout_v19/diana_net/probs_DIANA_<task>.tsv (network-only DIANA, read 5) and
results/consensus_v19net/runs.tsv. No new prediction.

    ./env/bin/python scripts/analysis/89_class_diagnostics_v19.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

SPLIT = ROOT / "data/splits_v19"
OUT = ROOT / "results/diagnostics_v19"
FIELDS = {"community_type": "Community type", "feature": "Feature", "sample_host": "Sample host", "material": "Material"}
V = px.colors.qualitative.Vivid
COLORS = {"community_type": V[1], "feature": V[4], "sample_host": V[2], "material": V[3]}
LABEL_MIN_RUNS = 100


def per_class_f1() -> pd.DataFrame:
    """DIANA held-out one-vs-rest F1 per eligible class, with the class's training runs and BioProjects."""
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str)
    rows = []
    for f in FIELDS:
        el = S.eligible_classes(SPLIT / "class_eligibility.tsv", f)
        d = pd.read_csv(ROOT / f"results/heldout_v19/diana_net/probs_DIANA_{f}.tsv", sep="\t", dtype={"y_true": str})
        d = d[d.y_true.isin(el)]
        pred = d[[c for c in d.columns if c.startswith("p_")]].idxmax(axis=1).str[2:]
        for c in sorted(set(d.y_true)):
            rows.append({"field": f, "class": c, "training_runs": int((tr[f] == c).sum()),
                         "training_bioprojects": int(tr.loc[tr[f] == c, "archive_project"].nunique()),
                         "heldout_runs": int((d.y_true == c).sum()), "f1": f1_score(d.y_true == c, pred == c, zero_division=0)})
    return pd.DataFrame(rows)


def scatter(t: pd.DataFrame, x: str, xtitle: str, title: str, path: Path, xrange: tuple[float, float]) -> None:
    fig = go.Figure()
    for f, name in FIELDS.items():
        s = t[t.field == f]
        lab = np.where(s.training_runs >= LABEL_MIN_RUNS, s["class"], "")
        fig.add_trace(go.Scatter(x=s[x], y=s.f1, mode="markers+text", name=name, text=lab, textposition="bottom right" if f == "feature" else "middle right",
                                 textfont=dict(size=10), marker=dict(size=10, color=COLORS[f], line=dict(width=0.5, color="black"))))
    fig.update_layout(template="plotly_white", title=title, width=900, height=600, legend_title_text="Field",
                      xaxis=dict(title=xtitle, type="log", range=[np.log10(xrange[0]), np.log10(xrange[1])]), yaxis=dict(title="DIANA held-out F1", range=[-0.03, 1.03]))
    fig.write_image(path, scale=2)


def disagreement_by_material() -> pd.DataFrame:
    r = pd.read_csv(ROOT / "results/consensus_v19net/runs.tsv", sep="\t", dtype=str)
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str)
    tr["all_models_disagree"] = tr.Run_accession.isin(set(r.loc[r.split == "training (out-of-fold)", "Run_accession"]))
    g = tr.groupby("material").agg(training_runs=("Run_accession", "size"), all_models_disagree=("all_models_disagree", "sum"),
                                   training_bioprojects=("archive_project", "nunique")).reset_index()
    g["share"] = g.all_models_disagree / g.training_runs
    return g[g.training_runs >= 30].sort_values("share", ascending=False)


def bars(g: pd.DataFrame, path: Path) -> None:
    fig = go.Figure(go.Bar(x=g.material, y=g.share, marker_color=V[7],
                           text=[f"{a}/{n}" for a, n in zip(g.all_models_disagree, g.training_runs)], textposition="outside"))
    fig.update_layout(template="plotly_white", width=900, height=550,
                      title="Training runs where all four models agree on a label other than the curated one, by material",
                      xaxis_title="Material", yaxis=dict(title="Share of training runs (out-of-fold)", range=[0, 1]))
    fig.write_image(path, scale=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = per_class_f1()
    t.to_csv(OUT / "heldout_f1_per_class.tsv", sep="\t", index=False)
    scatter(t, "training_runs", "Training runs of the class (log scale)", "DIANA held-out F1 per class vs training runs",
            OUT / "f1_vs_training_runs.png", (3, 8000))
    scatter(t, "training_bioprojects", "Training BioProjects of the class (log scale)",
            "DIANA held-out F1 per class vs training BioProjects", OUT / "f1_vs_training_bioprojects.png", (0.8, 80))
    g = disagreement_by_material()
    g.to_csv(OUT / "all_models_disagree_by_material.tsv", sep="\t", index=False)
    bars(g, OUT / "all_models_disagree_by_material.png")


if __name__ == "__main__":
    main()
