#!/usr/bin/env python3
"""G8: tuned baselines on fraction vs presence/absence, one (rep, task, fold) per call.

The missing half of the representation comparison: G3 tested presence/absence on DIANA
only, G1 tested reduction on logistic regression only, so nothing covered both model
families on the same representation.

Why this is an array job rather than one process. The first attempt looped over
2 representations x 4 tasks x 5 folds x 5 models = 200 fits in a single process and ran
at 88 % CPU on a 16-core allocation, because `LinearSVC` is single-threaded and
`LogisticRegression`'s `n_jobs` does nothing under lbfgs. Only RandomForest used the
cores. Parallelism has to come from SLURM, so each array task does one
(representation, task, fold) and writes its own predictions; `--pool` then scores them.

Scoring is deliberately NOT per-fold: `--pool` concatenates the out-of-fold predictions,
applies the eligibility rule once over all training projects, and bootstraps the paired
difference over BioProjects. Comparing fold means was tried on the DIANA screens and was
blind to a real effect.

Standardisation is fitted on the training folds of each split only. Held-out is untouched.

    ./env/bin/python scripts/analysis/27_baselines_representation_devfold.py \
        --rep presence_absence --task feature --fold 0
    ./env/bin/python scripts/analysis/27_baselines_representation_devfold.py --pool
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
logger = logging.getLogger(__name__)

MATRICES = {"fraction": "data/matrices/matrix_v9_train/unitigs.frac.mat",
            "presence_absence": "data/matrices/matrix_v9_train/unitigs.pa.mat"}
TASKS = ["community_type", "feature", "sample_host", "material"]
OUT = ROOT / "results/baselines_representation"
N_BOOT = 2000
SEED = 42


def meta_table() -> tuple[pd.DataFrame, list[str]]:
    accs = [l.split(" : ")[0].strip() for l in open(ROOT / "data/train_samples_v9.fof") if l.strip()]
    m = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False)
    m = m.set_index("Run_accession").reindex(accs).reset_index()
    folds = pd.read_csv(ROOT / "data/splits_v9/dev_folds.tsv", sep="\t")
    return m.merge(folds[["Run_accession", "fold"]], on="Run_accession", how="left"), accs


def load_matrix(rel: str, n: int) -> np.ndarray:
    rows = [np.asarray(l.split()[1:], dtype=np.float32) for l in open(ROOT / rel) if l.strip()]
    X = np.vstack(rows).T
    if X.shape[0] != n:
        raise ValueError(f"{rel}: {X.shape[0]} sample columns, fof lists {n}")
    return X


def f1_elig(y, p, eligible) -> float:
    labs = sorted(c for c in set(y) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def run_one(rep: str, task: str, fold: int) -> int:
    from importlib import import_module
    bl = import_module("09_baselines_v9")
    meta, accs = meta_table()
    lab = meta[task].astype("string")
    X = load_matrix(MATRICES[rep], len(accs))
    logger.info("%s / %s / fold %d: matrix %s", rep, task, fold, X.shape)

    tr = ((meta.fold != fold) & lab.notna()).to_numpy()
    va = ((meta.fold == fold) & lab.notna()).to_numpy()
    if tr.sum() < 10 or va.sum() < 5 or lab[tr].nunique() < 2:
        logger.warning("fold %d unusable: %d train, %d val, %d classes",
                       fold, tr.sum(), va.sum(), lab[tr].nunique())
        return 0
    sc = StandardScaler().fit(X[tr])
    A, B = sc.transform(X[tr]), sc.transform(X[va])
    ytr = lab[tr].to_numpy()

    models = bl.build_models(SEED, bl.tuned_params(task))
    models.pop("MajorityClass", None)
    recs = []
    for name, proto in models.items():
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = clone(proto).fit(A, ytr)
            pred = clf.predict(B)
            converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        logger.info("   %-24s %6.1f s  converged=%s", name, time.time() - t0, converged)
        recs.append(pd.DataFrame({"Run_accession": meta.loc[va, "Run_accession"].to_numpy(),
                                  "model": name, "y_true": lab[va].to_numpy(),
                                  "y_pred": pred, "converged": converged}))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"pred_{rep}_{task}_fold{fold}.tsv"
    pd.concat(recs, ignore_index=True).to_csv(out, sep="\t", index=False)
    logger.info("wrote %s", out)
    return 0


def pool() -> int:
    meta, _ = meta_table()
    proj = meta.set_index("Run_accession")["archive_project"].to_dict()
    rng = np.random.default_rng(SEED)
    rows = []
    for task in TASKS:
        d = {}
        for rep in MATRICES:
            fs = sorted(OUT.glob(f"pred_{rep}_{task}_fold*.tsv"))
            if not fs:
                logger.warning("%s / %s: no predictions", rep, task)
                break
            d[rep] = pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)
        if len(d) != len(MATRICES):
            continue
        models = sorted(set(d["fraction"].model) & set(d["presence_absence"].model))
        for name in models:
            a = d["presence_absence"].query("model == @name").set_index("Run_accession")
            b = d["fraction"].query("model == @name").set_index("Run_accession")
            idx = sorted(set(a.index) & set(b.index))
            y = a.loc[idx, "y_true"].astype(str).to_numpy()
            pa = a.loc[idx, "y_pred"].astype(str).to_numpy()
            fr = b.loc[idx, "y_pred"].astype(str).to_numpy()
            g = np.array([proj.get(i) for i in idx])
            per = pd.DataFrame({"c": y, "g": g}).groupby("c").g.nunique()
            eligible = set(per[per >= 2].index)
            uniq = np.unique(g)
            idx_by = {p: np.where(g == p)[0] for p in uniq}
            obs = f1_elig(y, pa, eligible) - f1_elig(y, fr, eligible)
            dd = []
            for _ in range(N_BOOT):
                drawn = rng.choice(uniq, size=len(uniq), replace=True)
                s = np.concatenate([idx_by[q] for q in drawn])
                dd.append(f1_elig(y[s], pa[s], eligible) - f1_elig(y[s], fr[s], eligible))
            dd = np.asarray(dd, float); dd = dd[np.isfinite(dd)]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            rows.append({"task": task, "model": name, "n_runs": len(idx),
                         "n_projects": len(uniq), "n_eligible": len(eligible),
                         "f1_fraction": f1_elig(y, fr, eligible),
                         "f1_presence_absence": f1_elig(y, pa, eligible),
                         "delta": obs, "ci_low": lo, "ci_high": hi,
                         "frac_boot_positive": float((dd > 0).mean()),
                         "all_converged": bool(a.converged.all() and b.converged.all()),
                         "verdict": "tie" if lo <= 0 <= hi else
                                    ("PA better" if obs > 0 else "fraction better")})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pa_vs_fraction_baselines.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("\npresence/absence minus fraction, tuned baselines, paired over BioProjects:")
    print(res.round(4).to_string(index=False))
    print("\nverdicts:", res.verdict.value_counts().to_dict())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", choices=list(MATRICES))
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--pool", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.pool:
        return pool()
    if a.rep is None or a.task is None or a.fold is None:
        ap.error("need --rep, --task and --fold, or --pool")
    return run_one(a.rep, a.task, a.fold)


if __name__ == "__main__":
    raise SystemExit(main())
