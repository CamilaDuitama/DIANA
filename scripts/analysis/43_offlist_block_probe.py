#!/usr/bin/env python3
"""S8.2b: does the off-list block carry class signal that transfers to unseen studies?

A linear probe on the 5 BioProject-grouped dev folds, before any DIANA search is paid
for. Three inputs per task, each a balanced logistic regression trained on four folds
and applied to the fifth: the off-list block alone (sparse 0/1), the 110,202 fractions
alone (standardised on the training folds), and both. Pooled out-of-fold predictions,
`f1_macro_eligible` with the eligibility table, paired over `archive_project` with 2,000
resamples, whole task and by regime. Nothing is selected from this; a null stops S8.3,
a signal does not promote anything.

    ./env/bin/python scripts/analysis/43_offlist_block_probe.py
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from diana.data.loader import MatrixLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/s8a_probe_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
N_BOOT, SEED, N_UNITIGS = 2000, 42, 110202


def rng_for(*parts) -> np.random.Generator:
    digest = hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()[:8]
    return np.random.default_rng(int(digest, 16))


def f1_elig(y, p, eligible) -> float:
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def paired(y, pa, pb, g, eligible, key) -> dict:
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
    lo, hi = np.percentile(dd, [2.5, 97.5]) if len(dd) else (np.nan, np.nan)
    return {"f1_a": f1_elig(y, pa, eligible), "f1_b": f1_elig(y, pb, eligible), "delta": obs,
            "ci_low": lo, "ci_high": hi, "frac_boot_positive": float((dd > 0).mean()) if len(dd) else np.nan,
            "n_projects": len(uniq), "verdict": "tie" if lo <= 0 <= hi else ("BETTER" if obs > 0 else "WORSE")}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    X, ids, _ = MatrixLoader(str(ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz")).load()
    ids = list(ids)
    F = X[:, :N_UNITIGS]
    B = sp.csr_matrix(X[:, N_UNITIGS:].astype(np.float32))
    del X
    logger.info("fractions %s, block %s (nnz %d) in %.0f s", F.shape, B.shape, B.nnz, time.time() - t0)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    project = meta["archive_project"].astype(str).to_numpy()
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    train_counts = {t: meta[t].dropna().astype(str).value_counts() for t in TASKS}

    rows, regime_rows, capped = [], [], []
    for task in TASKS:
        y_all = meta[task].astype(object).to_numpy()
        lab = pd.notna(y_all)
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        preds = {a: np.full(len(ids), None, dtype=object) for a in ("block", "fractions", "both")}
        for k in range(5):
            tr = (folds != k) & lab
            te = (folds == k) & lab
            if te.sum() == 0:
                continue
            sc = StandardScaler().fit(F[tr])
            Ftr, Fte = sp.csr_matrix(sc.transform(F[tr])), sp.csr_matrix(sc.transform(F[te]))
            inputs = {"block": (B[tr], B[te]), "fractions": (Ftr, Fte),
                      "both": (sp.hstack([Ftr, B[tr]]).tocsr(), sp.hstack([Fte, B[te]]).tocsr())}
            ytr = y_all[tr].astype(str)
            for arm, (Xtr, Xte) in inputs.items():
                # SGD on the log loss, not a batch solver. On 2026-09-28 saga hit its
                # 300-iteration cap on every fit at ~13 min each (the G8/S7b trap) and
                # liblinear converged on the block in 51 s but crawled on the 110,202
                # dense standardised fractions held as a sparse matrix. SGD costs one pass
                # over the non-zeros per epoch on both. alpha = 1/(C n) with C = 1 keeps
                # the regularisation comparable to the batch solvers. Non-convergence
                # (hitting max_iter) is logged and withholds the verdict.
                t1 = time.time()
                clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0 / Xtr.shape[0],
                                    class_weight="balanced", max_iter=200, tol=1e-4,
                                    n_iter_no_change=5, random_state=SEED, n_jobs=8)
                clf.fit(Xtr, ytr)
                preds[arm][te] = clf.predict(Xte)
                n_iter = int(clf.n_iter_)
                capped.append(n_iter >= 200)
                logger.info("%s fold %d %s: %d epochs, %.0f s%s", task, k, arm, n_iter, time.time() - t1,
                            "  NOT CONVERGED" if n_iter >= 200 else "")
        y = y_all[lab].astype(str)
        g = project[lab]
        P = {a: preds[a][lab].astype(str) for a in preds}
        for a in ("block", "fractions", "both"):
            rows.append({"task": task, "arm": a, "regime": "all", "n_runs": int(lab.sum()),
                         "f1_macro_eligible": f1_elig(y, P[a], eligible)})
        for a, b in (("block", "fractions"), ("both", "fractions")):
            r = paired(y, P[a], P[b], g, eligible, (task, a, b, "all"))
            regime_rows.append({"task": task, "comparison": f"{a} - {b}", "regime": "all", **r})
            for name, lo_, hi_ in REGIMES:
                cls = {c for c, n in train_counts[task].items() if lo_ < n <= hi_} & eligible
                if not cls:
                    continue
                r = paired(y, P[a], P[b], g, cls, (task, a, b, name))
                regime_rows.append({"task": task, "comparison": f"{a} - {b}", "regime": name, **r})
        pd.DataFrame({"Run_accession": np.array(ids)[lab], "y_true": y, **{f"pred_{a}": P[a] for a in P}}
                     ).to_csv(OUT / f"oof_{task}.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(OUT / "f1_by_arm.tsv", sep="\t", index=False)
    res = pd.DataFrame(regime_rows)
    res.to_csv(OUT / "paired.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(res.round(4).to_string(index=False))
    n_cap = int(sum(capped))
    (OUT / "convergence.txt").write_text(f"{n_cap} of {len(capped)} fits hit the iteration cap\n")
    if n_cap:
        print(f"\n{n_cap} of {len(capped)} fits did NOT converge: no verdict from this probe")
    logger.info("done in %.0f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
