#!/usr/bin/env python3
"""Classification by three metrics, whole field and by regime, with paired differences (v19).

Macro-F1 (as 43_/68_), balanced accuracy (mean recall over the eligible classes present; promised in
the Methods) and per-run accuracy (descriptive), on the stored predictions of the held-out reads 1
and 5 (--heldout) or of the dev folds; nothing is refitted. DIANA defaults to the released network alone
(PROTOCOLS.md v19 amendment 2026-10-26); --diana points elsewhere. Intervals: 2,000 BioProject resamples with
the same keys as 43_ (whole field: cls|<task>, pair|<task>) and 68_ (regime: cls|<task>|<regime>,
regpair|<task>|<regime>), so the macro-F1 values reproduce those tables. Pairs: DIANA minus each
other model. p_perm: paired BioProject permutation test (PROTOCOLS.md v19 amendment 2026-10-26), macro-F1 and
balanced accuracy only. Writes <out>/scores.tsv and <out>/paired.tsv.

    ./env/bin/python scripts/analysis/75_classification_metrics_v19.py --heldout --out results/classification_metrics_v19/heldout
    ./env/bin/python scripts/analysis/75_classification_metrics_v19.py --out results/classification_metrics_v19/dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
SPLIT = ROOT / "data/splits_v19"
HELDOUT = {"DIANA": "results/heldout_v19/diana_net/probs_DIANA_{task}.tsv",
           "LogReg": "results/heldout_v19/baselines/{task}/probs_LogisticRegression_Bal.tsv",
           "RandomForest": "results/heldout_v19/baselines/{task}/probs_RandomForest_Bal.tsv",
           "kNN": "results/heldout_v19/baselines/{task}/probs_kNN.tsv",
           "Majority": "results/heldout_v19/baselines/{task}/probs_MajorityClass.tsv"}
DEV = {"DIANA": "results/devfold_probs_v19net/probs_DIANA_{task}.tsv",
       "LogReg": "results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv",
       "RandomForest": "results/baselines_v19/{task}/oof_probs_RandomForest_Bal.tsv",
       "kNN": "results/baselines_v19/{task}/oof_probs_kNN.tsv",
       "Majority": "results/baselines_v19/{task}/oof_probs_MajorityClass.tsv"}


def argmax(path: str, task: str) -> pd.Series:
    d = pd.read_csv(ROOT / path.format(task=task), sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
    return d[[c for c in d.columns if c.startswith("p_")]].idxmax(axis=1).str[2:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--diana", default=None, help="DIANA probability template with {task} instead of the default")
    a = ap.parse_args()
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    ev = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession") if a.heldout else tr
    models = dict(HELDOUT if a.heldout else DEV)
    if a.diana:
        models["DIANA"] = a.diana
    rows, pairs = [], []
    for task in TASKS:
        eligible = S.eligible_classes(SPLIT / "class_eligibility.tsv", task)
        scored = S.scored_heldout(ev[task], set(tr[task].dropna())) if a.heldout else S.scored_devfold(tr[task], folds)
        runs = scored[scored].index
        y = ev.loc[runs, task].astype(str).to_numpy()
        g = ev.loc[runs, "archive_project"].astype(str).to_numpy()
        preds = {}
        for m, path in models.items():
            p = argmax(path, task)
            if len(runs.difference(p.index)):
                raise ValueError(f"{m}/{task}: scored runs without a prediction")
            preds[m] = p.loc[runs].to_numpy()
        scopes = [("all", eligible, f"cls|{task}", f"pair|{task}")]
        scopes += [(reg, cls & set(y), f"cls|{task}|{reg}", f"regpair|{task}|{reg}")
                   for reg, cls in S.regime_classes(tr[task], eligible).items() if cls & set(y)]
        for regime, cls, key, pkey in scopes:
            for metric, fn in S.METRICS.items():
                for m in models:
                    r = S.score_with_interval(y, preds[m], g, cls, key, metric=fn)
                    rows.append({"task": task, "regime": regime, "metric": metric, "model": m, "value": r["value"],
                                 "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n_classes": len(cls)})
                for m in models:
                    if m == "DIANA":
                        continue
                    r = S.paired_difference(runs, runs, y, preds["DIANA"], preds[m], g, cls, pkey, metric=fn)
                    pp = (S.paired_permutation_p(y, preds["DIANA"], preds[m], g, cls, f"perm|{pkey}|{metric}|{m}", metric=fn)
                          if metric in ("macro_f1", "balanced_accuracy") else float("nan"))
                    pairs.append({"task": task, "regime": regime, "metric": metric, "a": "DIANA", "b": m, "delta": r["delta"],
                                  "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_boot": r["p_boot"], "p_perm": pp,
                                  "verdict": {"a": "DIANA", "b": m}.get(r["verdict"], "tie")})
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / "scores.tsv", sep="\t", index=False)
    pd.DataFrame(pairs).to_csv(a.out / "paired.tsv", sep="\t", index=False)
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(r[r.regime == "all"].pivot_table(index=["task", "metric"], columns="model", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
