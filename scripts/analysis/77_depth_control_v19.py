#!/usr/bin/env python3
"""Sequencing-depth control for the released DIANA (v19; R1.1, R3.7, R3.11).

(A) Can depth alone predict the labels? Logistic regression (balanced, standardised inputs) and random forest
(balanced) given only three depth measures: log10 read count (AncientMetagenomeDir libraries table), log10 number of
non-zero unitig fractions and mean unitig fraction (from the training matrix and the held-out vectors). Dev folds:
out-of-fold, fitted on the other four folds; held-out: fitted on all labelled training runs. Scored like DIANA (macro-F1
and balanced accuracy over the eligible classes present, scored-run rules of the scoring module) and paired with
DIANA (dev-fold out-of-fold probabilities; released DIANA on held-out) over BioProject resamples.
(B) Does DIANA fail more on shallow runs? Per field, runs of classes with at least 10 scored runs are split at the
class median of log10 non-zero unitigs; accuracy on the deep half minus the shallow half, pooled over classes, with a
BioProject interval.

Writes results/depth_control_v19/{depth.tsv, scores.tsv, paired.tsv, within_class.tsv}.

    ./env/bin/python scripts/analysis/77_depth_control_v19.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.data.loader import MatrixLoader  # noqa: E402
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
SPLIT = ROOT / "data/splits_v19"
AMD = ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0"
TRAIN_FRAC = ROOT / "data/matrices/matrix_v17_train/unitigs.frac.mat"
HELD_FRAC = ROOT / "results/v19_heldout/heldout_frac.npz"
DIANA_DEV = ROOT / "results/devfold_probs_v19/probs_DIANA_{task}.tsv"
DIANA_HELD = ROOT / "results/heldout_v19/diana_refit/probs_DIANA_{task}.tsv"
OUT = ROOT / "results/depth_control_v19"
FEATS = ["log_reads", "log_nonzero", "mean_fraction"]
METRICS = {"macro_f1": S.f1_macro_eligible, "balanced_accuracy": S.balanced_accuracy_eligible}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def depth(runs: pd.Index) -> pd.DataFrame:
    lib = pd.concat([pd.read_csv(AMD / f"ancientmetagenome-{k}_libraries.tsv", sep="\t", low_memory=False)
                     for k in ("hostassociated", "environmental")], ignore_index=True).drop_duplicates("archive_data_accession")
    reads = pd.to_numeric(lib.set_index("archive_data_accession").read_count, errors="coerce")
    X, ids, _ = MatrixLoader(TRAIN_FRAC).load()
    rows = {str(a): ((x > 0).sum(), x.mean()) for a, x in zip(ids, X)}
    del X
    z = np.load(HELD_FRAC, allow_pickle=True)
    rows.update({str(a): ((x > 0).sum(), x.mean()) for a, x in zip(z["sample_ids"], z["frac"])})
    d = pd.DataFrame.from_dict(rows, orient="index", columns=["nonzero", "mean_fraction"]).reindex(runs)
    if d.nonzero.isna().any():
        raise ValueError(f"{int(d.nonzero.isna().sum())} runs without a fraction vector")
    d["log_nonzero"] = np.log10(d.nonzero.clip(lower=1))
    d["log_reads"] = np.log10(reads.reindex(runs).clip(lower=1))
    log.info("read count missing for %d of %d runs (filled with the training median when fitting)", int(d.log_reads.isna().sum()), len(d))
    return d


def models(prefix: str = "Depth") -> dict:
    return {f"{prefix}_LogReg": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
            f"{prefix}_RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=5,
                                                         n_jobs=-1, random_state=42)}


def fit_predict(Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame) -> dict[str, np.ndarray]:
    """Depth_* on all three measures; Reads_* on read count alone (added after the first result, to separate sequencing
    depth from the two unitig-based measures, which also reflect content)."""
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    out = {}
    for prefix, cols in (("Depth", FEATS), ("Reads", ["log_reads"])):
        out.update({name: m.fit(Xtr[cols].to_numpy(), ytr.astype(str).to_numpy()).predict(Xte[cols].to_numpy())
                    for name, m in models(prefix).items()})
    return out


def diana_top(path: Path, task: str, runs: pd.Index) -> np.ndarray:
    p = pd.read_csv(str(path).format(task=task), sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
    return p.loc[runs, [c for c in p.columns if c.startswith("p_")]].idxmax(axis=1).str[2:].to_numpy()


def within_class(y: np.ndarray, pred: np.ndarray, d: np.ndarray, g: np.ndarray, key: str) -> dict:
    """Deep-half minus shallow-half accuracy, pooled over classes with >= 10 runs; BioProject interval."""
    def diff(idx: np.ndarray) -> float:
        yy, pp, dd = y[idx], pred[idx], d[idx]
        deep, shallow = [], []
        for c in np.unique(yy):
            m = yy == c
            if m.sum() < 10:
                continue
            hi = dd[m] > np.median(dd[m])
            ok = (pp[m] == c)
            deep.append(ok[hi]); shallow.append(ok[~hi])
        if not deep:
            return np.nan
        a, b = np.concatenate(deep), np.concatenate(shallow)
        return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
    point = diff(np.arange(len(y)))
    boot = [diff(i) for i in S.project_draws(g, key)]
    lo, hi = S.interval(boot)
    return {"deep_minus_shallow": point, "ci_low": lo, "ci_high": hi}


def main() -> None:
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    D = depth(tr.index.append(te.index))
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "depth.tsv", sep="\t")
    scores, pairs, within = [], [], []
    for task in TASKS:
        el = S.eligible_classes(SPLIT / "class_eligibility.tsv", task)
        for split in ("dev folds", "held-out"):
            if split == "dev folds":
                sc = S.scored_devfold(tr[task], folds); runs = sc[sc].index
                y, g = tr.loc[runs, task].astype(str).to_numpy(), tr.loc[runs, "archive_project"].astype(str).to_numpy()
                pred = {}
                for k in sorted(folds.dropna().unique()):
                    fit = tr.index[(folds != k) & tr[task].notna()]
                    sel = np.flatnonzero(folds.loc[runs].to_numpy() == k)
                    for m, p in fit_predict(D.loc[fit, FEATS], tr.loc[fit, task], D.loc[runs[sel], FEATS]).items():
                        pred.setdefault(m, np.empty(len(runs), dtype=object))[sel] = p
                pred["DIANA"] = diana_top(DIANA_DEV, task, runs)
            else:
                sc = S.scored_heldout(te[task], set(tr[task].dropna())); runs = sc[sc].index
                y, g = te.loc[runs, task].astype(str).to_numpy(), te.loc[runs, "archive_project"].astype(str).to_numpy()
                fit = tr.index[tr[task].notna()]
                pred = fit_predict(D.loc[fit, FEATS], tr.loc[fit, task], D.loc[runs, FEATS])
                pred["DIANA"] = diana_top(DIANA_HELD, task, runs)
            cls = el & set(y)
            for metric, fn in METRICS.items():
                for m, p in pred.items():
                    r = S.score_with_interval(y, p.astype(str), g, cls, f"depth|{split}|{task}", metric=fn)
                    scores.append({"task": task, "split": split, "metric": metric, "model": m, "value": r["value"],
                                   "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
                for m in ("Depth_LogReg", "Depth_RandomForest", "Reads_LogReg", "Reads_RandomForest"):
                    r = S.paired_difference(runs, runs, y, pred["DIANA"].astype(str), pred[m].astype(str), g, cls,
                                            f"depthpair|{split}|{task}", metric=fn)
                    pairs.append({"task": task, "split": split, "metric": metric, "a": "DIANA", "b": m, "delta": r["delta"],
                                  "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_boot": r["p_boot"]})
            within.append({"task": task, "split": split, **within_class(y, pred["DIANA"].astype(str), D.loc[runs, "log_nonzero"].to_numpy(), g,
                                                                         f"within|{split}|{task}")})
    pd.DataFrame(scores).to_csv(OUT / "scores.tsv", sep="\t", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "paired.tsv", sep="\t", index=False)
    pd.DataFrame(within).to_csv(OUT / "within_class.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    s = pd.DataFrame(scores)
    print(s.pivot_table(index=["split", "metric", "model"], columns="task", values="value").round(3).to_string())
    print(pd.DataFrame(pairs).round(3).to_string(index=False))
    print(pd.DataFrame(within).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
