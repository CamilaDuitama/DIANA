#!/usr/bin/env python3
"""S8.3a: the tuned baselines on fractions + off-list block, against the same models on fractions.

One array task per (representation, task, model). Same grid, same 5 BioProject-grouped
dev folds, same selection rule as 12_tune_baselines_v9.py (fold-mean f1_macro_eligible,
out-of-vocabulary rows excluded per fold), same raw inputs (12_ fits on unstandardised
fractions; the block is 0/1, so both live in [0, 1]). Every grid point's out-of-fold
predictions are kept, so `--pool` can pair the selected configuration of each model on
`s8a` against the selected configuration of the same model on `fraction`, pooled over
all training runs and resampled over BioProjects, whole task and by regime.

Two departures, both applied to BOTH representations and disclosed: RandomForest runs
only its `max_features="sqrt"` grid points (0.05 of 2.09 M columns per split is not
tractable), and inputs for `s8a` are sparse (scipy CSR), which every model here accepts.

    ./env/bin/python scripts/evaluation/14_tune_baselines_s8a_v9.py --rep s8a --task feature --model LogisticRegression_Bal
    ./env/bin/python scripts/evaluation/14_tune_baselines_s8a_v9.py --pool
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation.metrics import classification_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
SPLITS = ROOT / "data/splits_v9"
NPZ = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
OUT = ROOT / "results/baselines_s8a_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
# sig = fractions + per-fold class-signature columns; sigonly = the signature columns
# alone (a diagnostic: if 38 class scores match 110,202 fractions, the matrix is redundant)
REPS = ["fraction", "s8a", "sig", "sigonly", "hash"]
HASH_NPZ = None   # --hash-npz: H protocol, k-mer hash presence matrix (33_hash_matrix_v9.py)
SIG_DIR = ROOT / "results/s8_signatures_v9"
GRIDS = {
    "LogisticRegression_Bal": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "LinearSVM_Bal": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)],
    # RandomForest: `sqrt` only for the 2 M-column s8a run (0.05 of 2.09 M columns per
    # split is not tractable); --rf-grid full restores the v9 grid for normal widths.
    "RandomForest_Bal": [{"n_estimators": n, "max_features": "sqrt", "min_samples_leaf": l}
                         for n in (300, 800) for l in (1, 3)],
    "kNN": [{"n_neighbors": k, "weights": w} for k in (1, 3, 5, 10, 20) for w in ("uniform", "distance")],
}
RF_FULL = [{"n_estimators": n, "max_features": f, "min_samples_leaf": l}
           for n in (300, 800) for f in ("sqrt", 0.05) for l in (1, 3)]
MODELS = list(GRIDS)
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
N_BOOT, SEED = 2000, 42


def build(name: str, params: dict, seed: int, n_jobs: int):
    if name == "LogisticRegression_Bal":
        return LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=n_jobs, **params)
    if name == "LinearSVM_Bal":
        return LinearSVC(max_iter=8000, class_weight="balanced", **params)
    if name == "RandomForest_Bal":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed, **params)
    if name == "kNN":
        return KNeighborsClassifier(n_jobs=n_jobs, **params)
    raise ValueError(name)


def load(rep: str):
    if rep == "hash":
        if HASH_NPZ is None:
            raise SystemExit("--hash-npz is required for --rep hash")
        with np.load(HASH_NPZ, allow_pickle=False) as z:
            ids = list(z["sample_ids"].astype(str)); X = z["frac"].astype(np.float32)
        if X.mean() < 0.10:   # wide, sparse floors (14, 28): CSR, as the 2 M-column s8a input was
            X = sp.csr_matrix(X)
        meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
        folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
        return X, ids, meta.reset_index(), folds
    with np.load(NPZ, allow_pickle=False) as z:
        ids = list(z["sample_ids"].astype(str))
        F = z["frac"].astype(np.float32)
        if rep in ("fraction", "sig", "sigonly"):
            X = F          # for `sig`/`sigonly` the per-fold block is taken inside the fold loop
        else:
            X = sp.hstack([sp.csr_matrix(F), sp.csr_matrix(z["block"].astype(np.float32))]).tocsr()
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    return X, ids, meta.reset_index(), folds


def sig_block(fold: int, ids: list[str]) -> np.ndarray:
    """S8.8: the per-fold class-signature columns, built from training-fold runs only."""
    with np.load(SIG_DIR / f"sig_fold{fold}.npz", allow_pickle=False) as z:
        if list(z["sample_ids"].astype(str)) != ids:
            raise SystemExit(f"sig_fold{fold}.npz sample order differs from the fraction matrix")
        return z["block"].astype(np.float32)


def run_one(rep: str, task: str, model: str, n_jobs: int, out_dir: Path, rf_grid: str) -> int:
    global OUT
    OUT = out_dir
    grids = dict(GRIDS)
    if rf_grid == "full":
        grids["RandomForest_Bal"] = RF_FULL
    OUT.mkdir(parents=True, exist_ok=True)
    out_pred, out_score = OUT / f"pred_{rep}_{task}_{model}.tsv", OUT / f"score_{rep}_{task}_{model}.tsv"
    if out_pred.exists():
        raise SystemExit(f"{out_pred} exists; refusing to overwrite")
    t0 = time.time()
    X0, ids, meta, folds = load(rep)
    logger.info("%s / %s / %s: X %s (%s) in %.0f s", rep, task, model, X0.shape, type(X0).__name__, time.time() - t0)
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    eligible = set(elig[(elig.target == task) & elig.evaluable]["class"].astype(str))
    lab = meta[task].notna().to_numpy()
    y_all = meta[task].astype(object).to_numpy()
    preds, scores = [], []
    blocks = {f: sig_block(f, ids) for f in sorted(set(folds))} if rep in ("sig", "sigonly") else {}
    for cfg in grids[model]:
        fold_scores = []
        for f in sorted(set(folds)):
            tr, te = lab & (folds != f), lab & (folds == f)
            if te.sum() < 5 or tr.sum() < 20:
                continue
            X = np.hstack([X0, blocks[f]]) if rep == "sig" else (blocks[f] if rep == "sigonly" else X0)
            ytr, yte = y_all[tr].astype(str), y_all[te].astype(str)
            t1 = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                clf = build(model, cfg, SEED, n_jobs).fit(X[tr], ytr)
                pred = clf.predict(X[te])
                converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
            seen = set(ytr)
            keep = np.array([v in seen for v in yte])
            if keep.sum() >= 5:
                fold_scores.append(classification_metrics(yte[keep], pred[keep], eligible)["f1_macro_eligible"])
            preds.append(pd.DataFrame({"cfg": json.dumps(cfg), "fold": f, "Run_accession": np.array(ids)[te],
                                       "y_true": yte, "y_pred": pred, "converged": converged}))
            logger.info("   %s fold %d: %.0f s, converged=%s", json.dumps(cfg), f, time.time() - t1, converged)
        scores.append({"rep": rep, "task": task, "model": model, "cfg": json.dumps(cfg),
                       "mean_f1_eligible": float(np.nanmean(fold_scores)) if fold_scores else np.nan,
                       "n_folds": len(fold_scores)})
        logger.info("%s: mean f1_eligible %.4f over %d folds", json.dumps(cfg), scores[-1]["mean_f1_eligible"], len(fold_scores))
    pd.concat(preds, ignore_index=True).to_csv(out_pred, sep="\t", index=False)
    pd.DataFrame(scores).to_csv(out_score, sep="\t", index=False)
    logger.info("wrote %s and %s in %.0f s", out_pred.name, out_score.name, time.time() - t0)
    return 0


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()[:8], 16))


def f1_elig(y, p, eligible):
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def paired(y, pa, pb, g, eligible, key, rep_a: str = "s8a"):
    uniq = np.unique(g)
    by = {p: np.where(g == p)[0] for p in uniq}
    obs = f1_elig(y, pa, eligible) - f1_elig(y, pb, eligible)
    rng = rng_for(*key)
    dd = []
    for _ in range(N_BOOT):
        sel = np.concatenate([by[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        dd.append(f1_elig(y[sel], pa[sel], eligible) - f1_elig(y[sel], pb[sel], eligible))
    dd = np.asarray(dd, float)
    dd = dd[np.isfinite(dd)]
    lo, hi = (np.percentile(dd, [2.5, 97.5]) if len(dd) else (np.nan, np.nan))
    return {f"f1_{rep_a}": f1_elig(y, pa, eligible), "f1_fraction": f1_elig(y, pb, eligible), "delta": obs,
            "ci_low": lo, "ci_high": hi, "frac_boot_positive": float((dd > 0).mean()) if len(dd) else np.nan,
            "n_projects": len(uniq), "verdict": "tie" if lo <= 0 <= hi else (rep_a if obs > 0 else "fraction")}


def pool(out_dir: Path, rep_a: str = "s8a", rep_b: str = "fraction") -> int:
    """Selected configuration of `rep_a` against that of `rep_b`, per model, paired."""
    global OUT
    OUT = out_dir
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    rows, chosen = [], []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        counts = meta[task].dropna().astype(str).value_counts()
        for model in MODELS:
            best = {}
            for rep in (rep_a, rep_b):
                fs, fp = OUT / f"score_{rep}_{task}_{model}.tsv", OUT / f"pred_{rep}_{task}_{model}.tsv"
                if not (fs.exists() and fp.exists()):
                    logger.warning("%s / %s / %s: missing, skipped", rep, task, model)
                    break
                s = pd.read_csv(fs, sep="\t").sort_values("mean_f1_eligible", ascending=False).iloc[0]
                p = pd.read_csv(fp, sep="\t")
                p = p[p.cfg == s.cfg].set_index("Run_accession")
                best[rep] = (s, p)
                chosen.append({"task": task, "model": model, "rep": rep, "cfg": s.cfg,
                               "mean_f1_eligible_selection": s.mean_f1_eligible, "all_converged": bool(p.converged.all())})
            if len(best) != 2:
                continue
            (sa, pa_), (sb, pb_) = best[rep_a], best[rep_b]
            common = sorted(set(pa_.index) & set(pb_.index))
            y = pa_.loc[common, "y_true"].astype(str).to_numpy()
            if not np.array_equal(y, pb_.loc[common, "y_true"].astype(str).to_numpy()):
                raise SystemExit(f"{task} {model}: labels differ between representations")
            pa, pb = pa_.loc[common, "y_pred"].astype(str).to_numpy(), pb_.loc[common, "y_pred"].astype(str).to_numpy()
            g = meta.loc[common, "archive_project"].astype(str).to_numpy()
            r = paired(y, pa, pb, g, eligible, (task, model, "all"), rep_a)
            rows.append({"task": task, "model": model, "regime": "all", "n_runs": len(common), **r})
            for rn, lo_, hi_ in REGIMES:
                cls = {c for c, n in counts.items() if lo_ < n <= hi_} & eligible
                if cls:
                    r = paired(y, pa, pb, g, cls, (task, model, rn), rep_a)
                    rows.append({"task": task, "model": model, "regime": rn, "n_runs": len(common), **r})
    res, ch = pd.DataFrame(rows), pd.DataFrame(chosen)
    res.to_csv(OUT / f"paired_{rep_a}_vs_{rep_b}.tsv", sep="\t", index=False)
    ch.to_csv(OUT / f"selected_configs_{rep_a}_vs_{rep_b}.tsv", sep="\t", index=False)
    # the selections in the format 09_baselines_v9.py reads (--tuned), one file per representation
    for rep, blk in ch.groupby("rep"):
        blk.rename(columns={"cfg": "params", "mean_f1_eligible_selection": "mean_f1_eligible"}).assign(n_folds=5)[
            ["task", "model", "params", "mean_f1_eligible", "n_folds"]].to_csv(OUT / f"best_per_model_{rep}.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(ch.to_string(index=False))
    print(res.round(3).to_string(index=False))
    print("\nverdicts:", res.verdict.value_counts().to_dict())
    return 0


def main() -> int:
    global SPLITS, HASH_NPZ
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", type=Path, default=SPLITS, help="split directory (metadata, dev folds, eligibility); v13 = data/splits_v13")
    ap.add_argument("--rep", choices=REPS)
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT, help="results directory (S8.3a: baselines_s8a_v9)")
    ap.add_argument("--rf-grid", choices=["sqrt", "full"], default="sqrt")
    ap.add_argument("--rep-a", default="s8a", help="with --pool: the representation compared against --rep-b")
    ap.add_argument("--rep-b", default="fraction")
    ap.add_argument("--hash-npz", type=Path, default=None, help="with --rep hash: the npz from 33_hash_matrix_v9.py")
    a = ap.parse_args()
    SPLITS = a.splits
    HASH_NPZ = a.hash_npz
    if a.pool:
        return pool(a.out, a.rep_a, a.rep_b)
    if not (a.rep and a.task and a.model):
        raise SystemExit("--rep, --task and --model are required unless --pool")
    return run_one(a.rep, a.task, a.model, a.n_jobs, a.out, a.rf_grid)


if __name__ == "__main__":
    raise SystemExit(main())
