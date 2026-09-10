#!/usr/bin/env python
"""
Is the v9 baseline ranking robust? Three checks on the held-out set.

Replicates 09_baselines_v9.py exactly -- same models, same eligible-class
denominator, same out-of-vocabulary exclusion -- so `f1_macro_eligible` here
must reproduce that script's summary.csv. Then adds:

  1. Leave-one-BioProject-out. One project (PRJEB41240) is 26 % of the held-out
     set. Recompute f1_macro_eligible with each project dropped in turn; a
     ranking that survives only with one project present is not a ranking.
  2. Extraction route. 829 held-out runs were projected from Logan unitigs
     (threshold 1, matching training); 93 from raw FASTQ (threshold 2). Training
     was 100 % unitig-route, so the raw-reads runs sit in a feature regime the
     models never saw. Report both strata.
  3. BioProject-level bootstrap. Runs within a study are correlated, so
     resampling runs (what 09_baselines_v9.py does) understates uncertainty.
     Resample the 34 held-out BioProjects instead and compare interval widths.

Usage:
    ./env/bin/python scripts/evaluation/10_baseline_robustness_v9.py

Output: results/baseline_robustness_v9/{predictions,per_project,route,bootstrap}.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from diana.evaluation.metrics import classification_metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/analysis"))
from load_v9_features import load_features  # noqa: E402

SPLITS = PROJECT_ROOT / "data/splits_v9"
TARGETS = ["community_type", "feature", "sample_host", "material"]


def build_models(seed: int) -> dict:
    return {
        "MajorityClass": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression_Bal": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
        "LinearSVM_Bal": LinearSVC(max_iter=5000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                               n_jobs=-1, random_state=seed),
        "RandomForest_Bal": RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                   class_weight="balanced_subsample",
                                                   n_jobs=-1, random_state=seed),
        "kNN_5": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def f1_elig(y, p, elig) -> float:
    """Delegates to diana.evaluation.metrics: one definition for the baselines,
    diana-test and this script, so the three cannot drift apart."""
    return classification_metrics(y, p, elig)["f1_macro_eligible"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/baseline_robustness_v9")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    te = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t", low_memory=False)
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    uni_route = set(Path(PROJECT_ROOT / "data/project_v9_from_unitigs.txt").read_text().split())

    Xtr_all, ktr = load_features(tr.Run_accession)
    Xte_all, kte = load_features(te.Run_accession)
    tr = tr.set_index("Run_accession").loc[ktr].reset_index()
    te = te.set_index("Run_accession").loc[kte].reset_index()
    print(f"train {Xtr_all.shape}  held-out {Xte_all.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    pred_rows, proj_rows, route_rows, boot_rows = [], [], [], []

    for target in TARGETS:
        elig = set(elig_tbl[(elig_tbl.target == target) & elig_tbl.evaluable]["class"])
        mtr = tr[target].notna().to_numpy()
        mte = te[target].notna().to_numpy()
        ytr = tr.loc[mtr, target].astype(str).to_numpy()
        yte_all = te.loc[mte, target].astype(str).to_numpy()
        in_vocab = np.array([v in set(ytr) for v in yte_all])

        Xtr, Xte = Xtr_all[mtr], Xte_all[mte][in_vocab]
        yte = yte_all[in_vocab]
        sub = te.loc[mte].loc[in_vocab]
        proj = sub.archive_project.to_numpy()
        acc_ids = sub.Run_accession.to_numpy()
        route = np.where(np.isin(acc_ids, list(uni_route)), "unitig", "raw_reads")
        projects = sorted(set(proj))

        for name, model in build_models(args.seed).items():
            t0 = time.time()
            model.fit(Xtr, ytr)
            p = model.predict(Xte).astype(str)
            base = f1_elig(yte, p, elig)
            print(f"{target:15s} {name:24s} f1_elig={base:.4f}  ({time.time()-t0:.0f}s)", flush=True)

            pred_rows.append(pd.DataFrame({"target": target, "model": name,
                                           "Run_accession": acc_ids, "archive_project": proj,
                                           "route": route, "y_true": yte, "y_pred": p}))

            # 1. leave-one-BioProject-out
            for pj in projects:
                keep = proj != pj
                proj_rows.append(dict(target=target, model=name, dropped_project=pj,
                                      n_dropped=int((~keep).sum()),
                                      f1_without=f1_elig(yte[keep], p[keep], elig),
                                      f1_full=base))

            # 2. extraction route
            for r in ["unitig", "raw_reads"]:
                k = route == r
                if k.sum() == 0:
                    continue
                route_rows.append(dict(target=target, model=name, route=r, n=int(k.sum()),
                                       accuracy=float(accuracy_score(yte[k], p[k])),
                                       f1_macro_eligible=f1_elig(yte[k], p[k], elig)))

            # 3. bootstrap: runs vs BioProjects
            run_vals, proj_vals = [], []
            idx_by_proj = {pj: np.flatnonzero(proj == pj) for pj in projects}
            for _ in range(args.n_boot):
                i = rng.integers(0, len(yte), size=len(yte))
                run_vals.append(f1_elig(yte[i], p[i], elig))
                drawn = rng.choice(projects, size=len(projects), replace=True)
                j = np.concatenate([idx_by_proj[pj] for pj in drawn])
                proj_vals.append(f1_elig(yte[j], p[j], elig))
            boot_rows.append(dict(
                target=target, model=name, f1_full=base,
                run_ci_low=float(np.nanpercentile(run_vals, 2.5)),
                run_ci_high=float(np.nanpercentile(run_vals, 97.5)),
                proj_ci_low=float(np.nanpercentile(proj_vals, 2.5)),
                proj_ci_high=float(np.nanpercentile(proj_vals, 97.5))))

    pd.concat(pred_rows, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(proj_rows).to_csv(args.output / "per_project.csv", index=False)
    pd.DataFrame(route_rows).to_csv(args.output / "route.csv", index=False)
    b = pd.DataFrame(boot_rows)
    b["run_width"] = b.run_ci_high - b.run_ci_low
    b["proj_width"] = b.proj_ci_high - b.proj_ci_low
    b.to_csv(args.output / "bootstrap.csv", index=False)
    print("\nwrote", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
