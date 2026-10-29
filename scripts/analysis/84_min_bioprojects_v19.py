#!/usr/bin/env python3
"""How many held-out BioProjects does a paired comparison need? (v19; PROTOCOLS.md amendment 2026-10-26)

For one field: random subsets of k held-out BioProjects (200 per k); in each, DIANA minus each baseline (balanced accuracy
and macro-F1) with the paper's 95 % BioProject percentile interval (500 resamples of the subset). Reference: the same
difference on all held-out BioProjects. Writes results/min_bioprojects_v19/<task>.tsv with, per k, baseline and metric:
coverage (interval contains the reference difference) and spurious significance (interval excludes zero while the
reference interval includes zero). Stored held-out predictions only.

    ./env/bin/python scripts/analysis/84_min_bioprojects_v19.py --task sample_host
"""
from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

SPLIT = ROOT / "data/splits_v19"
MODELS = {"DIANA": "results/heldout_v19/diana_refit/probs_DIANA_{t}.tsv",
          "LogReg": "results/heldout_v19/baselines/{t}/probs_LogisticRegression_Bal.tsv",
          "RandomForest": "results/heldout_v19/baselines/{t}/probs_RandomForest_Bal.tsv",
          "kNN": "results/heldout_v19/baselines/{t}/probs_kNN.tsv"}
METRICS = {"balanced_accuracy": S.balanced_accuracy_eligible, "macro_f1": S.f1_macro_eligible}
KS, N_SUB, N_BOOT = (2, 3, 4, 5, 6, 8, 10), 200, 500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True)
    t = ap.parse_args().task
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    el = S.eligible_classes(SPLIT / "class_eligibility.tsv", t)
    sc = S.scored_heldout(te[t], set(tr[t].dropna()))
    runs = sc[sc].index
    y = te.loc[runs, t].astype(str).to_numpy()
    g = te.loc[runs, "archive_project"].astype(str).to_numpy()
    pred = {}
    for m, p in MODELS.items():
        d = pd.read_csv(ROOT / p.format(t=t), sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
        pred[m] = d.loc[runs, [c for c in d.columns if c.startswith("p_")]].idxmax(axis=1).str[2:].to_numpy()
    projects = np.unique(g)
    rng = np.random.default_rng(20261026)
    rows = []
    for b in ("LogReg", "RandomForest", "kNN"):
        for metric, fn in METRICS.items():
            ref = S.paired_difference(runs, runs, y, pred["DIANA"], pred[b], g, el & set(y), f"minbp|{t}|{b}|{metric}", metric=fn)
            ref_sig = ref["ci_low"] > 0 or ref["ci_high"] < 0
            for k in [k for k in KS if k < len(projects)]:
                cover = spurious = n = 0
                for s in range(N_SUB):
                    pick = rng.choice(projects, size=k, replace=False)
                    i = np.flatnonzero(np.isin(g, pick))
                    cls = el & set(y[i])
                    if not cls:
                        continue
                    r = S.paired_difference(runs[i], runs[i], y[i], pred["DIANA"][i], pred[b][i], g[i], cls,
                                            f"minbp|{t}|{b}|{metric}|{k}|{s}", n_boot=N_BOOT, metric=fn)
                    if not np.isfinite(r["ci_low"]):
                        continue
                    n += 1
                    cover += r["ci_low"] <= ref["delta"] <= r["ci_high"]
                    spurious += (not ref_sig) and (r["ci_low"] > 0 or r["ci_high"] < 0)
                rows.append({"task": t, "baseline": b, "metric": metric, "k": k, "subsets": n, "distinct_resamples": comb(2 * k - 1, k),
                             "coverage": cover / n if n else np.nan, "spurious_significance": spurious / n if n else np.nan,
                             "reference_delta": ref["delta"], "reference_significant": ref_sig})
                print(rows[-1], flush=True)
    out = ROOT / "results/min_bioprojects_v19"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{t}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
