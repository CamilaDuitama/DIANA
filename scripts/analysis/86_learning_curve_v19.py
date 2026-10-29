#!/usr/bin/env python3
"""Learning curve over the number of training BioProjects, dev folds (v19; PROTOCOLS.md amendment 2026-10-26).

  subsets            for each dev fold k, fraction f in {0.25, 0.5, 0.75} and repeat r in {0, 1, 2}: keep round(f x n) of the
                     training-fold BioProjects (random, seeded by f, r, k) with all their runs, plus every run of fold k.
                     Writes results/learning_curve_v19/subsets/f<f>_r<r>/fold<k>/{train_ids.txt, dev_folds.tsv, config.json}
                     (the shared network's selected config with train_ids_path pointing at the subset) and jobs.tsv.
  baselines --job i  logistic regression and random forest at their selected settings (results/baselines_v19/<field>/best_*.json)
                     fitted on the labelled kept runs of job i, probabilities for the runs of fold k ->
                     results/learning_curve_v19/baselines/f<f>_r<r>/fold<k>/<model>_<field>.tsv
  collect            pools each (model, f, r) over the five folds, scores the scored dev-fold runs of the full split
                     (macro-F1, balanced accuracy over the eligible classes present), writes curve.tsv and one plot per field
                     (results/paper_v19/learning_curve_<field>.png). f = 1: the existing seed-42 network fits
                     (results/seeds_v19/multitask/seed42/) and the baselines' out-of-fold tables (results/baselines_v19/).
  plot               redraws the plots from curve_summary.tsv.
The network fits themselves run through scripts/training/04_epoch_budget_dev_folds.py (run_learning_curve_v19.sbatch).

    ./env/bin/python scripts/analysis/86_learning_curve_v19.py subsets
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
SPLIT = ROOT / "data/splits_v19"
OUT = ROOT / "results/learning_curve_v19"
CONFIG = ROOT / "configs/v19_final/multitask.json"
FRACTIONS, REPEATS, FOLDS = (0.25, 0.5, 0.75), (0, 1, 2), range(5)
BASELINES = {"LogReg": "LogisticRegression_Bal", "RandomForest": "RandomForest_Bal"}
METRICS = {"macro_f1": S.f1_macro_eligible, "balanced_accuracy": S.balanced_accuracy_eligible}
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def tag(f: float, r: int) -> str:
    return f"f{int(f * 100)}_r{r}"


def meta() -> tuple[pd.DataFrame, pd.Series]:
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    return tr, folds


def subsets() -> None:
    tr, folds = meta()
    base = json.load(open(CONFIG))
    jobs = []
    for f in FRACTIONS:
        for r in REPEATS:
            for k in FOLDS:
                projects = sorted(tr.loc[folds != k, "archive_project"].unique())
                seed = int(hashlib.sha256(f"lc|{f}|{r}|{k}".encode()).hexdigest()[:8], 16)
                keep = set(np.random.default_rng(seed).choice(projects, size=max(1, round(f * len(projects))), replace=False))
                runs = tr.index[(folds == k) | tr.archive_project.isin(keep)]
                d = OUT / "subsets" / tag(f, r) / f"fold{k}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "train_ids.txt").write_text("\n".join(runs) + "\n")
                pd.DataFrame({"Run_accession": runs, "fold": folds.loc[runs].astype(int).to_numpy()}).to_csv(d / "dev_folds.tsv", sep="\t", index=False)
                cfg = dict(base, train_ids_path=str((d / "train_ids.txt").relative_to(ROOT)))
                json.dump(cfg, open(d / "config.json", "w"), indent=1)
                jobs.append({"job": len(jobs), "fraction": f, "repeat": r, "fold": k, "dir": str(d.relative_to(ROOT)),
                             "train_bioprojects": len(keep), "train_runs": int((folds.loc[runs] != k).sum()), "all_train_bioprojects": len(projects)})
    pd.DataFrame(jobs).to_csv(OUT / "jobs.tsv", sep="\t", index=False)
    log.info("%d jobs written to %s", len(jobs), OUT / "jobs.tsv")


def baselines(job: int, n_jobs: int) -> None:
    from diana.data.loader import MatrixLoader
    spec = importlib.util.spec_from_file_location("b40", ROOT / "scripts/evaluation/40_baselines_v19.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    j = pd.read_csv(OUT / "jobs.tsv", sep="\t").set_index("job").loc[job]
    tr, folds = meta()
    X, ids, _ = MatrixLoader(ROOT / "data/matrices/matrix_v17_train/unitigs.frac.mat").load()
    pos = pd.Series(np.arange(len(ids)), index=[str(i) for i in ids])
    runs = pd.Index([l.strip() for l in open(ROOT / j.dir / "train_ids.txt") if l.strip()])
    fit_runs, ev_runs = runs[folds.loc[runs] != j.fold], runs[folds.loc[runs] == j.fold]
    out = ROOT / str(j.dir).replace("subsets", "baselines")
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        lab = fit_runs[tr.loc[fit_runs, task].notna().to_numpy()]
        ys = tr.loc[lab, task].astype(str).to_numpy()
        for short, name in BASELINES.items():
            if len(set(ys)) < 2:   # a subset left with one class: every model predicts it
                pd.DataFrame(1.0, index=ev_runs, columns=[f"p_{ys[0]}"]).rename_axis("Run_accession").to_csv(out / f"{short}_{task}.tsv", sep="\t")
                log.info("job %d %s %s: one class (%s) in %d fit runs, constant prediction", job, task, short, ys[0], len(lab))
                continue
            params = json.load(open(ROOT / f"results/baselines_v19/{task}/best_{name}.json"))["params"]
            m = B.build(name, params, n_jobs).fit(X[pos[lab].to_numpy()].astype(np.float32), ys)
            p = m.predict_proba(X[pos[ev_runs].to_numpy()].astype(np.float32))
            pd.DataFrame(p, index=ev_runs, columns=[f"p_{c}" for c in m.classes_]).rename_axis("Run_accession").to_csv(out / f"{short}_{task}.tsv", sep="\t")
            log.info("job %d %s %s: %d fit runs", job, task, short, len(lab))


def network_probs(d: Path, task: str) -> pd.DataFrame:
    runs = [l.strip() for l in open(d / "eval_runs.txt") if l.strip()]
    classes = json.load(open(d / "label_classes.json"))[task]
    return pd.DataFrame(np.load(d / "probs_last.npz")[task], index=runs, columns=[f"p_{c}" for c in classes])


def argmax(P: pd.DataFrame) -> pd.Series:
    return P.idxmax(axis=1).str[2:]


def collect() -> None:
    tr, folds = meta()
    jobs = pd.read_csv(OUT / "jobs.tsv", sep="\t")
    rows = []
    for task in TASKS:
        el = S.eligible_classes(SPLIT / "class_eligibility.tsv", task)
        sc = S.scored_devfold(tr[task], folds)
        runs = sc[sc].index
        y = tr.loc[runs, task].astype(str).to_numpy()
        cls = el & set(y)
        settings = [(f, r) for f in FRACTIONS for r in REPEATS] + [(1.0, 0)]
        for f, r in settings:
            preds = {}
            if f == 1.0:
                preds["Network"] = pd.concat([argmax(network_probs(ROOT / f"results/seeds_v19/multitask/seed42/fold{k}", task)) for k in FOLDS])
                for short, name in BASELINES.items():
                    d = pd.read_csv(ROOT / f"results/baselines_v19/{task}/oof_probs_{name}.tsv", sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
                    preds[short] = argmax(d[[c for c in d.columns if c.startswith("p_")]])
                n_bp = float(jobs.all_train_bioprojects.drop_duplicates().mean())
            else:
                sel = jobs[(jobs.fraction == f) & (jobs.repeat == r)]
                preds["Network"] = pd.concat([argmax(network_probs(ROOT / str(d).replace("subsets", "network"), task)) for d in sel.dir])
                for short in BASELINES:
                    preds[short] = pd.concat([argmax(pd.read_csv(ROOT / str(d).replace("subsets", "baselines") / f"{short}_{task}.tsv", sep="\t",
                                                                dtype={"Run_accession": str}).set_index("Run_accession")) for d in sel.dir])
                n_bp = float(sel.train_bioprojects.mean())
            for m, p in preds.items():
                if runs.difference(p.index).size:
                    raise ValueError(f"{task} {m} f={f} r={r}: scored runs without a prediction")
                for metric, fn in METRICS.items():
                    rows.append({"task": task, "model": m, "fraction": f, "repeat": r, "train_bioprojects": n_bp, "metric": metric,
                                 "value": fn(y, p.loc[runs].astype(str).to_numpy(), cls)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "curve.tsv", sep="\t", index=False)
    summary = d.groupby(["task", "metric", "model", "fraction"]).agg(mean=("value", "mean"), low=("value", "min"), high=("value", "max"),
                                                                        bioprojects=("train_bioprojects", "mean")).reset_index()
    summary.to_csv(OUT / "curve_summary.tsv", sep="\t", index=False)
    plot(summary)
    pd.set_option("display.width", 200)
    print(summary[summary.metric == "macro_f1"].pivot_table(index=["task", "fraction"], columns="model", values="mean").round(3).to_string())


def plot(summary: pd.DataFrame) -> None:
    import plotly.graph_objects as go
    from config import PLOT_CONFIG
    C, N = PLOT_CONFIG["colors"]["models"], PLOT_CONFIG["colors"]["model_names"]
    # DIANA is the shared network alone (2026-10-26); these fits use seed 42 only
    names = {"Network": ("DIANA, one seed", C["DIANA"]), "LogReg": (N["LogReg"], C["LogReg"]),
             "RandomForest": (N["RandomForest"], C["RandomForest"])}
    for task in TASKS:
        s = summary[(summary.task == task) & (summary.metric == "macro_f1")]
        fig = go.Figure()
        for m, (label, colour) in names.items():
            d = s[s.model == m].sort_values("bioprojects")
            fig.add_scatter(x=d.bioprojects, y=d["mean"], mode="lines+markers", name=label, line=dict(color=colour, width=2),
                            marker=dict(color=colour, size=9),
                            error_y=dict(type="data", symmetric=False, array=d["high"] - d["mean"], arrayminus=d["mean"] - d["low"], color=colour, width=4))
        fig.update_layout(template=PLOT_CONFIG["template"], width=640, height=500, font=dict(size=14, color="#333333"),
                          xaxis=dict(title="training BioProjects per dev fold"), yaxis=dict(title="macro-F1 (dev folds, out-of-fold)"),
                          legend=dict(orientation="v", yanchor="top", y=-0.18, x=0), margin=dict(l=70, r=20, t=20, b=150))
        (ROOT / "results/paper_v19").mkdir(parents=True, exist_ok=True)
        fig.write_image(ROOT / f"results/paper_v19/learning_curve_{task}.png", scale=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["subsets", "baselines", "collect", "plot"])
    ap.add_argument("--job", type=int)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    if a.step == "subsets":
        subsets()
    elif a.step == "baselines":
        baselines(a.job, a.n_jobs)
    elif a.step == "plot":
        plot(pd.read_csv(OUT / "curve_summary.tsv", sep="\t"))
    else:
        collect()


if __name__ == "__main__":
    main()
