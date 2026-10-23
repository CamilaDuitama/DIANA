#!/usr/bin/env python3
"""v19 Phase 1: tune one baseline for one task on the dev folds, keep its out-of-fold output.

Every configuration of the model's pre-registered grid is fitted on the fit rows of each
dev fold and predicts the labelled runs of the held-back fold; the out-of-fold predictions
of the five folds are pooled and scored once with the shared definition (C2: scored runs
only, macro-F1 over eligible classes). The best configuration (first in grid order on a
tie) is the model's selection, the same rule for every baseline (C2e). Its out-of-fold
probabilities (classes absent from a fold's fit get 0) and predictions are written for
the flagging benchmark and the composition tests, so no refit can diverge from the tuning.

Models: the standard baseline set, logistic regression, linear SVM, random forest, k-NN and
majority class, with the v9 grids (12_tune_baselines_v9.py); the untuned unbalanced random
forest is not a v19 model (C7). --study-weights (logistic regression only) fits with the
network's equal-weight-per-study row weights (PROTOCOLS.md amendment of 2026-10-12).

    ./env/bin/python scripts/evaluation/40_baselines_v19.py --task feature --model kNN \\
        --split-dir data/splits_v19 --matrix data/matrices/matrix_v17_train/unitigs.frac.mat \\
        --out results/baselines_v19
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.data.loader import MatrixLoader  # noqa: E402
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
SEED = 42
GRIDS = {
    "LogisticRegression_Bal": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "LinearSVM_Bal": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)],
    "RandomForest_Bal": [{"n_estimators": n, "max_features": f, "min_samples_leaf": l}
                         for n in (300, 800) for f in ("sqrt", 0.05) for l in (1, 3)],
    "kNN": [{"n_neighbors": k, "weights": w} for k in (1, 3, 5, 10, 20) for w in ("uniform", "distance")],
    "MajorityClass": [{}],
}
NO_PROBA = {"LinearSVM_Bal"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build(name: str, params: dict, n_jobs: int):
    if name == "LogisticRegression_Bal":
        return LogisticRegression(max_iter=3000, class_weight="balanced", **params)
    if name == "LinearSVM_Bal":
        return LinearSVC(max_iter=8000, class_weight="balanced", **params)
    if name == "RandomForest_Bal":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=n_jobs, random_state=SEED, **params)
    if name == "kNN":
        return KNeighborsClassifier(n_jobs=n_jobs, **params)
    if name == "MajorityClass":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(name)


def load(split_dir: Path, matrix: Path) -> tuple[np.ndarray, pd.DataFrame]:
    X, ids, _ = MatrixLoader(str(matrix)).load()
    meta = pd.read_csv(split_dir / "train_metadata.tsv", sep="\t", dtype=str)
    folds = pd.read_csv(split_dir / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold
    S.check_same_split(meta, [str(i) for i in ids], "training matrix vs metadata")
    meta = meta.set_index("Run_accession").loc[[str(i) for i in ids]].reset_index()
    meta["fold"] = meta.Run_accession.map(folds).astype(int)
    return X.astype(np.float32), meta


def network_study_weights():
    """The study weights of the network fits (04_epoch_budget_dev_folds.py), so both use one rule."""
    spec = importlib.util.spec_from_file_location("fold_fit", ROOT / "scripts/training/04_epoch_budget_dev_folds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.study_weights


def oof(name: str, params: dict, X: np.ndarray, meta: pd.DataFrame, task: str, classes: list[str],
        n_jobs: int, weigh=None) -> tuple[np.ndarray, np.ndarray | None]:
    """Out-of-fold predictions (and probabilities over `classes`) for the labelled runs;
    `weigh` (groups, labels) -> row weights, applied to each fold's fit rows."""
    lab = meta[task].notna().to_numpy()
    pred = np.full(len(meta), "", dtype=object)
    prob = None if name in NO_PROBA else np.zeros((len(meta), len(classes)), dtype=np.float32)
    for k in sorted(meta.fold.unique()):
        Xk = X(k) if callable(X) else X   # per-fold input (--matrix with {fold})
        fit = lab & (meta.fold != k).to_numpy()
        ev = lab & (meta.fold == k).to_numpy()
        kw = {} if weigh is None else {"sample_weight": weigh(meta.archive_project.to_numpy()[fit],
                                                               np.zeros(int(fit.sum()), dtype=np.int64))}
        m = build(name, params, n_jobs).fit(Xk[fit], meta.loc[fit, task].to_numpy(), **kw)
        pred[ev] = m.predict(Xk[ev])
        if prob is not None:
            p = m.predict_proba(Xk[ev])
            for j, c in enumerate(m.classes_):
                prob[np.flatnonzero(ev), classes.index(c)] = p[:, j]
    return pred, prob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--model", choices=list(GRIDS), required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--study-weights", action="store_true", help="equal weight per study (logistic regression only)")
    a = ap.parse_args()
    if a.study_weights and a.model != "LogisticRegression_Bal":
        raise SystemExit("--study-weights is defined for logistic regression only")
    weigh = network_study_weights() if a.study_weights else None
    out = a.out / a.task
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"best_{a.model}.json").exists():
        raise SystemExit(f"{out}/best_{a.model}.json exists; refusing to overwrite")
    if "{fold}" in str(a.matrix):
        per_fold = {k: load(a.split_dir, Path(str(a.matrix).format(fold=k))) for k in range(5)}
        meta = per_fold[0][1]
        for k, (_, m_k) in per_fold.items():
            if not m_k.Run_accession.equals(meta.Run_accession):
                raise ValueError(f"fold {k} matrix has a different run order")
        X = lambda k: per_fold[k][0]  # noqa: E731
    else:
        X, meta = load(a.split_dir, a.matrix)
    meta[a.task] = meta[a.task].replace("", np.nan)
    eligible = S.eligible_classes(a.split_dir / "class_eligibility.tsv", a.task)
    labels = pd.Series(meta[a.task].to_numpy(), index=meta.Run_accession.to_numpy())
    folds = pd.Series(meta.fold.to_numpy(), index=labels.index)
    scored = S.scored_devfold(labels, folds).to_numpy()
    classes = sorted(meta[a.task].dropna().unique())
    y = meta[a.task].astype(str).to_numpy()
    log.info("%s / %s: %d labelled, %d scored runs, %d eligible classes, %d grid points", a.task, a.model,
             int(labels.notna().sum()), int(scored.sum()), len(eligible), len(GRIDS[a.model]))
    rows, best = [], None
    for i, params in enumerate(GRIDS[a.model]):
        pred, prob = oof(a.model, params, X, meta, a.task, classes, a.n_jobs, weigh)
        score = S.f1_macro_eligible(y[scored], pred[scored].astype(str), eligible)
        if not np.isfinite(score):
            raise ValueError(f"{a.model} {params}: non-finite score")
        rows.append({"config": i, "params": json.dumps(params), "f1_macro_eligible": score})
        log.info("%s %s: %.4f", a.model, params, score)
        if best is None or score > best[0]:
            best = (score, i, params, pred, prob)
    pd.DataFrame(rows).to_csv(out / f"grid_{a.model}.tsv", sep="\t", index=False)
    score, i, params, pred, prob = best
    base = pd.DataFrame({"Run_accession": meta.Run_accession, "fold": meta.fold, "y_true": meta[a.task],
                         "scored": scored, "y_pred": pred})
    lab = meta[a.task].notna().to_numpy()
    base[lab].to_csv(out / f"oof_pred_{a.model}.tsv", sep="\t", index=False)
    if prob is not None:
        pd.concat([base[lab].reset_index(drop=True),
                   pd.DataFrame(prob[lab], columns=[f"p_{c}" for c in classes])], axis=1).to_csv(
            out / f"oof_probs_{a.model}.tsv", sep="\t", index=False)
    json.dump({"task": a.task, "model": a.model, "config": i, "params": params, "f1_macro_eligible": score,
               "n_scored": int(scored.sum()), "rule": "PROTOCOLS.md v19 C2e", "study_weights": bool(a.study_weights)},
              open(out / f"best_{a.model}.json", "w"), indent=1)
    log.info("%s / %s: selected %s (%.4f)", a.task, a.model, params, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
