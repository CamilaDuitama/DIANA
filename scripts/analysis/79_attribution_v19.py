#!/usr/bin/env python3
"""Which unitigs drive the released DIANA's predictions? (v19; R1.1, R3.11, R3.20)

Integrated gradients (zero baseline, which for unitig fractions means "no sequence observed"; 32 steps) of the true-class
logit of each of the five released networks (results/final_v19_weights/multitask/seed*/model.pt), on the training runs each
network classifies correctly. Per network and
field: attribution averaged within each class, then over classes (class-balanced). Ranking: mean over the five networks;
stability: in how many networks' own top 15 each unitig appears. Top unitigs are annotated with their best NCBI nt hit
(results/blast_v19/chunks; highest bitscore, species from the first specific hit title) or "no NCBI hit".
Writes results/attribution_v19/<field>_top.tsv (top 50) and one 2 x 2 figure of the four fields,
results/paper_v19/sup_feature_importance.png: per field, the top unitigs that have an NCBI hit (unitigs without one are
left out until their LexicMap search; the stability counts still refer to each network's full top 15).
--plot-only redraws the figure from the stored tables.

    ./env/bin/python scripts/analysis/79_attribution_v19.py [--plot-only]
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper"))
from config import PLOT_CONFIG  # noqa: E402
from diana.data.loader import MatrixLoader  # noqa: E402
from diana.models.multitask_mlp import MultiTaskMLP  # noqa: E402
from diana.training.active_classes import masked_softmax  # noqa: E402

TASKS = {"community_type": "Community type", "feature": "Feature", "sample_host": "Sample host", "material": "Material"}
SEEDS = [42, 1, 2, 3, 4]
FITS = ROOT / "results/final_v19_weights/multitask"
MATRIX = ROOT / "data/matrices/matrix_v17_train"
SPLIT = ROOT / "data/splits_v19"
BLAST = ROOT / "results/blast_v19/chunks"
OUT, FIG = ROOT / "results/attribution_v19", ROOT / "results/paper_v19"
STEPS, BATCH, TOP = 32, 128, 15
NON_SPECIFIC = ("metagenome", "uncultured", "unidentified", "environmental", "synthetic construct", "vector")
COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore",
        "staxids", "sscinames", "stitle"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def unitig_ids() -> list[str]:
    return [l[1:].split()[0] for l in open(MATRIX / "unitigs.fa") if l.startswith(">")]


def integrated_gradients(net: MultiTaskMLP, X: torch.Tensor, task: str, target: torch.Tensor) -> torch.Tensor:
    total = torch.zeros_like(X)
    for a in (np.arange(STEPS) + 0.5) / STEPS:
        x = (X * float(a)).requires_grad_(True)
        logit = net(x)[task].gather(1, target[:, None]).sum()
        total += torch.autograd.grad(logit, x)[0]
    return X * total / STEPS


def attribute(X: np.ndarray, runs: list[str], labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """field -> (seeds x features) class-balanced mean attribution."""
    out = {t: [] for t in TASKS}
    for s in SEEDS:
        m = torch.load(FITS / f"seed{s}/model.pt", map_location="cpu", weights_only=False)
        net = MultiTaskMLP(input_dim=m["input_dim"], num_classes=m["num_classes"], **m["model_params"]).eval()
        net.load_state_dict(m["state_dict"])
        for p in net.parameters():
            p.requires_grad_(False)
        for t in TASKS:
            classes = m["classes"][t]
            act = torch.tensor([c in m["active_classes"][t] for c in classes])
            y = labels.loc[runs, t]
            keep = np.flatnonzero(y.isin(classes).to_numpy())
            idx = torch.tensor([classes.index(c) for c in y.iloc[keep]])
            sums, counts = {}, {}
            for b in range(0, len(keep), BATCH):
                xb = torch.from_numpy(X[keep[b:b + BATCH]])
                tb = idx[b:b + BATCH]
                with torch.no_grad():
                    pred = masked_softmax(net(xb)[t], act).argmax(1)
                ok = pred == tb
                if not ok.any():
                    continue
                ig = integrated_gradients(net, xb[ok], t, tb[ok]).numpy()
                for c in tb[ok].unique().tolist():
                    sel = (tb[ok] == c).numpy()
                    sums[c] = sums.get(c, 0) + ig[sel].sum(0); counts[c] = counts.get(c, 0) + int(sel.sum())
            out[t].append(np.mean([sums[c] / counts[c] for c in sums], axis=0))
            log.info("seed %d %s: %d classes attributed", s, t, len(sums))
    return {t: np.vstack(v) for t, v in out.items()}


def blast_annotation(wanted: set[str]) -> dict[str, str]:
    ann = {}
    for f in sorted(glob.glob(str(BLAST / "chunk_*_blast.tsv"))):
        d = pd.read_csv(f, sep="\t", header=None, names=COLS, dtype={"qseqid": str}, usecols=range(len(COLS)))
        d = d[d.qseqid.isin(wanted)].sort_values(["qseqid", "bitscore"], ascending=[True, False], kind="stable")
        for q, g in d.groupby("qseqid", sort=False):
            spec = [t for t in g.stitle.astype(str).str.replace("PREDICTED: ", "") if not any(w in t.lower() for w in NON_SPECIFIC)]
            ann[q] = " ".join(spec[0].split()[:2]) if spec else "non-specific hit"
    return ann


def plot(tops: dict[str, pd.DataFrame], out: Path) -> None:
    """2 x 2 grid, one panel per field: the top unitigs with an NCBI hit, bars = mean over the five networks, error bars = SD."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[TASKS[t] for t in tops], horizontal_spacing=0.36, vertical_spacing=0.1)
    for i, (t, top) in enumerate(tops.items()):
        d = top[top.annotation != "no NCBI hit"].head(TOP).iloc[::-1]
        lab = [f"{u}: {a} ({k}/5)" for u, a, k in zip(d.unitig, d.annotation, d.seeds_in_top15)]
        fig.add_bar(x=d.importance, y=lab, orientation="h", marker_color=PLOT_CONFIG["colors"]["models"]["DIANA"], showlegend=False,
                    error_x=dict(type="data", array=d.importance_sd, color="#555555", thickness=1), row=i // 2 + 1, col=i % 2 + 1)
    fig.update_xaxes(title_text="mean attribution to the correct class", row=2)
    fig.update_yaxes(tickfont=dict(size=13))
    fig.update_layout(template=PLOT_CONFIG["template"], width=1700, height=1150, font=dict(size=14, color="#333333"),
                      margin=dict(l=330, r=20, t=50, b=70))
    fig.write_image(out, scale=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/attribution_v19/<field>_top.tsv")
    if ap.parse_args().plot_only:
        tops = {t: pd.read_csv(OUT / f"{t}_top.tsv", sep="\t", dtype={"unitig": str}) for t in TASKS}
        plot(tops, FIG / "sup_feature_importance.png")
        return
    X, runs, _ = MatrixLoader(MATRIX / "unitigs.frac.mat").load()
    X = np.ascontiguousarray(X, dtype=np.float32)
    runs = [str(r) for r in runs]
    ids = unitig_ids()
    if len(ids) != X.shape[1]:
        raise ValueError("unitigs.fa and the matrix disagree on the number of unitigs")
    labels = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).set_index("Run_accession")
    A = attribute(X, runs, labels)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    tops = {}
    for t, M in A.items():
        mean, sd = M.mean(0), M.std(0)
        order = np.argsort(-mean)[:50]
        per_seed = [set(np.argsort(-row)[:TOP]) for row in M]
        tops[t] = pd.DataFrame({"unitig": [ids[i] for i in order], "importance": mean[order], "importance_sd": sd[order],
                                "seeds_in_top15": [sum(i in s for s in per_seed) for i in order]})
    ann = blast_annotation({u for d in tops.values() for u in d.unitig})
    for t, d in tops.items():
        d["annotation"] = d.unitig.map(ann).fillna("no NCBI hit")
        d.to_csv(OUT / f"{t}_top.tsv", sep="\t", index=False)
        log.info("%s top %d: %s", t, TOP, "; ".join(f"{u} {a} ({k}/5)" for u, a, k in d.head(TOP)[["unitig", "annotation", "seeds_in_top15"]].itertuples(index=False)))
    plot(tops, FIG / "sup_feature_importance.png")


if __name__ == "__main__":
    main()
