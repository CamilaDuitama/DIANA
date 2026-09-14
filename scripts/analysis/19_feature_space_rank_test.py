#!/usr/bin/env python
"""Is the v9 feature space low-rank, and does feature reduction help?

Two questions this answers, both on the BioProject-grouped dev folds only --
the held-out set is never touched here.

  1. LOW-RANK?  If logistic regression on r SVD components matches logistic
     regression on all 110,202 unitig fractions, then the signal lives in an
     r-dimensional subspace that a linear model already reaches. DIANA's
     192-unit trunk would then be compressing to a space the baseline has for
     free, which would explain why the bottleneck buys nothing.

  2. DOES REDUCTION HELP AT ALL?  Supervised top-k selection and SVD are the
     two cheap feature-space interventions nobody has tried on this corpus.
     If either lifts the linear baseline, it must be applied to every model
     before any DIANA-vs-baseline claim is made.

Protocol. Every transform is fit on the four training folds and applied to the
held-out fold, so nothing leaks across the fold boundary. One SVD per fold,
reused across the four tasks, because the features do not depend on the task.
C is fixed at 1.0 for every arm: this compares feature spaces, not tuning.

Metric is macro-F1 over the eligible classes (>=2 BioProjects) present in the
validation fold, matching the baseline scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))

TASKS = ["community_type", "feature", "sample_host", "material"]
RANKS = [64, 192, 1000]
TOPK = [5000, 20000]
OUT = ROOT / "results/feature_space_rank"


def canon(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.where(~s.str.lower().isin(["nan", "none", "null", ""]))


def fit_score(Xtr, ytr, Xva, yva, elig):
    """One logistic-regression fit, scored over eligible classes present."""
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    labs = sorted(set(yva) & elig)
    return f1_score(yva, pred, labels=labs, average="macro", zero_division=0)


def main() -> int:
    from load_v9_features import load_features

    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t",
                       low_memory=False)
    folds = pd.read_csv(ROOT / "data/splits_v9/dev_folds.tsv", sep="\t")
    elig_tbl = pd.read_csv(ROOT / "data/splits_v9/class_eligibility.tsv", sep="\t")
    meta = meta.merge(folds[["Run_accession", "fold"]], on="Run_accession")

    X, kept = load_features(meta.Run_accession.tolist())
    meta = meta[meta.Run_accession.isin(kept)].reset_index(drop=True)
    X = np.asarray(X, dtype=np.float32)
    print(f"X = {X.shape}, folds = {sorted(meta.fold.unique())}", flush=True)

    rows = []
    for f in sorted(meta.fold.unique()):
        tr_m = (meta.fold != f).values
        va_m = ~tr_m
        sc = StandardScaler().fit(X[tr_m])
        Xtr_f, Xva_f = sc.transform(X[tr_m]), sc.transform(X[va_m])

        reps = {"full": (Xtr_f, Xva_f)}
        for r in RANKS:
            sv = TruncatedSVD(n_components=r, random_state=0).fit(Xtr_f)
            reps[f"svd{r}"] = (sv.transform(Xtr_f), sv.transform(Xva_f))
            print(f"  fold {f} svd{r}: explained var "
                  f"{sv.explained_variance_ratio_.sum():.3f}", flush=True)

        for t in TASKS:
            elig = set(elig_tbl[(elig_tbl.target == t) & elig_tbl.evaluable]["class"]
                       .astype(str))
            lab = canon(meta[t])
            ok_tr = tr_m & lab.notna().values
            ok_va = va_m & lab.notna().values
            if ok_va.sum() < 5 or lab[ok_tr].nunique() < 2:
                continue
            ytr = lab[ok_tr].values
            yva = lab[ok_va].values
            sub_tr = ok_tr[tr_m]           # index within the fold's train block
            sub_va = ok_va[va_m]

            for name, (A, B) in reps.items():
                rows.append({"fold": f, "task": t, "representation": name,
                             "n_features": A.shape[1], "n_val": int(ok_va.sum()),
                             "f1": fit_score(A[sub_tr], ytr, B[sub_va], yva, elig)})
            # supervised selection: fit on the training folds only
            for k in TOPK:
                kk = min(k, Xtr_f.shape[1])
                sel = SelectKBest(f_classif, k=kk).fit(Xtr_f[sub_tr], ytr)
                rows.append({"fold": f, "task": t, "representation": f"top{k}",
                             "n_features": kk, "n_val": int(ok_va.sum()),
                             "f1": fit_score(sel.transform(Xtr_f[sub_tr]), ytr,
                                             sel.transform(Xva_f[sub_va]), yva, elig)})
            print(f"  fold {f} {t}: done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rank_test_folds.tsv", sep="\t", index=False)
    piv = df.pivot_table(index="representation", columns="task", values="f1",
                         aggfunc="mean")
    piv.to_csv(OUT / "rank_test_mean.tsv", sep="\t")
    print("\nmean macro-F1 across dev folds:\n" + piv.round(4).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
