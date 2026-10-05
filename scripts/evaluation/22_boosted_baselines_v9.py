#!/usr/bin/env python3
"""Boosted-tree baseline screen (PROTOCOLS.md, 2026-10-02): HistGradientBoosting, AdaBoost and
ExtraTrees on the v9 fractions, grouped dev folds, grids as pre-registered; writes per-config
out-of-fold predictions and, for the selected configuration, probabilities in the devfold_probs
layout (probs_<model>_<task>.tsv in results/boosted_baselines_v9/).

    ./env/bin/python scripts/evaluation/22_boosted_baselines_v9.py --task material --model HGB
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
NPZ = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
OUT = ROOT / "results/boosted_baselines_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
SEED = 42
GRIDS = {
    "HGB": [{"learning_rate": lr, "max_iter": mi} for lr in (0.1, 0.05) for mi in (200, 500)],
    "AdaBoost": [{"n_estimators": n} for n in (200, 500)],
    "ExtraTrees": [{"n_estimators": 500, "max_features": mf} for mf in ("sqrt", 0.05)],
}


def build(model: str, cfg: dict, n_jobs: int):
    if model == "HGB":
        return HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced", **cfg)
    if model == "AdaBoost":
        return AdaBoostClassifier(random_state=SEED, **cfg)
    return ExtraTreesClassifier(random_state=SEED, class_weight="balanced_subsample", n_jobs=n_jobs, **cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--model", choices=list(GRIDS), required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"probs_{a.model}_{a.task}.tsv"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with np.load(NPZ, allow_pickle=False) as z:
        ids = np.asarray(z["sample_ids"].astype(str)); X = z["frac"].astype(np.float32)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    lab = meta[a.task].notna().to_numpy(); y = meta[a.task].astype(object).to_numpy()
    classes = sorted(set(y[lab].astype(str)))
    scores, oof_probs = [], {}
    for ci, cfg in enumerate(GRIDS[a.model]):
        t0 = time.time()
        pred = pd.Series(index=ids[lab], dtype=object)
        probs = np.zeros((lab.sum(), len(classes)), dtype=np.float32)
        pos = {r: i for i, r in enumerate(ids[lab])}
        for f in sorted(set(folds)):
            tr, te = np.flatnonzero(lab & (folds != f)), np.flatnonzero(lab & (folds == f))
            clf = build(a.model, cfg, a.n_jobs).fit(X[tr], y[tr].astype(str))
            P = clf.predict_proba(X[te])
            for j, c in enumerate(clf.classes_):
                probs[[pos[r] for r in ids[te]], classes.index(c)] = P[:, j]
            pred.loc[ids[te]] = [clf.classes_[k] for k in P.argmax(1)]
        f1 = f1_score(y[lab].astype(str), pred.loc[ids[lab]].astype(str), average="macro", zero_division=0)
        scores.append({"model": a.model, "task": a.task, "config": ci, "cfg": str(cfg), "f1_macro_oof": round(float(f1), 4), "minutes": round((time.time() - t0) / 60, 1)})
        oof_probs[ci] = probs
        logger.info("config %s -> pooled OOF f1_macro %.4f (%.0f s)", cfg, f1, time.time() - t0)
    S = pd.DataFrame(scores); S.to_csv(OUT / f"grid_{a.model}_{a.task}.tsv", sep="\t", index=False)
    best = int(S.f1_macro_oof.idxmax())
    d = pd.DataFrame(oof_probs[S.loc[best, "config"]], columns=[f"p_{c}" for c in classes])
    d.insert(0, "y_true", y[lab].astype(str)); d.insert(0, "fold", folds[lab]); d.insert(0, "Run_accession", ids[lab])
    d.to_csv(out, sep="\t", index=False)
    logger.info("selected %s; wrote %s", S.loc[best, "cfg"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
