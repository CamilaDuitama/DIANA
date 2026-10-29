#!/usr/bin/env python3
"""Permutation bands for the held-out whole-field comparisons of Figure 3 (v19; DIANA = the shared network alone).

For each field and baseline (logistic regression, random forest, k-NN, majority class): DIANA minus the baseline in
balanced accuracy, its BioProject permutation p (recomputed with the keys of 75_ and checked against
results/multiplicity_v19/net_classification_heldout/paired.tsv), Benjamini-Hochberg q over the 16 comparisons, and the
band of differences that are not significant at the Benjamini-Hochberg threshold of the 16 (scoring
nonsignificant_halfwidth): a difference is significant exactly when it lies outside its band. Writes
results/multiplicity_v19/net_classification_heldout/figure3_bands.tsv.

    ./env/bin/python scripts/analysis/87_permutation_bands_v19.py
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

SPLIT = ROOT / "data/splits_v19"
OUT = ROOT / "results/multiplicity_v19/net_classification_heldout"
DIANA = "results/heldout_v19/diana_net/probs_DIANA_{task}.tsv"
BASELINES = ["LogReg", "RandomForest", "kNN", "Majority"]
METRIC, ALPHA = "balanced_accuracy", 0.05
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def bh_threshold(p: np.ndarray, alpha: float) -> float:
    """Benjamini-Hochberg: p-values at or below the returned value are significant (alpha / m when none is)."""
    s = np.sort(p)
    ok = np.flatnonzero(s <= alpha * np.arange(1, len(s) + 1) / len(s))
    return float(alpha * (ok[-1] + 1) / len(s)) if len(ok) else alpha / len(s)


def main() -> None:
    spec = importlib.util.spec_from_file_location("m75", ROOT / "scripts/analysis/75_classification_metrics_v19.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    stored = pd.read_csv(OUT / "paired.tsv", sep="\t")
    stored = stored[(stored.regime == "all") & (stored.metric == METRIC)].set_index(["task", "b"])
    rows, nulls = [], {}
    for task in M.TASKS:
        el = S.eligible_classes(SPLIT / "class_eligibility.tsv", task)
        sc = S.scored_heldout(te[task], set(tr[task].dropna()))
        runs = sc[sc].index
        y = te.loc[runs, task].astype(str).to_numpy()
        g = te.loc[runs, "archive_project"].astype(str).to_numpy()
        pa = M.argmax(DIANA, task).loc[runs].to_numpy()
        for b in BASELINES:
            pb = M.argmax(M.HELDOUT[b], task).loc[runs].to_numpy()
            obs, perm, exact = S.paired_permutation_null(y, pa, pb, g, el, f"perm|pair|{task}|{METRIC}|{b}",
                                                         metric=S.METRICS[METRIC])
            p = S.permutation_p(obs, perm, exact)
            ref = stored.loc[(task, b)]
            if not (np.isclose(obs, ref.delta) and np.isclose(p, ref.p_perm)):
                raise ValueError(f"{task}/{b}: delta {obs} / p {p} differ from paired.tsv {ref.delta} / {ref.p_perm}")
            nulls[(task, b)] = (perm, exact)
            rows.append({"task": task, "b": b, "metric": METRIC, "delta": obs, "p_perm": p, "bioprojects": len(np.unique(g)),
                         "exact": exact, "patterns": len(perm)})
    d = pd.DataFrame(rows)
    d["q"] = S.bh_adjust(d.p_perm.to_numpy())
    t = bh_threshold(d.p_perm.to_numpy(), ALPHA)
    d["threshold"] = t
    d["halfwidth"] = [S.nonsignificant_halfwidth(*nulls[(r.task, r.b)], t) for r in d.itertuples()]
    d["significant"] = d.q < ALPHA
    if not ((d.delta.abs() > d.halfwidth + 1e-12) == d.significant).all():
        raise ValueError("band and Benjamini-Hochberg verdict disagree")
    d.to_csv(OUT / "figure3_bands.tsv", sep="\t", index=False)
    log.info("BH threshold %.4f; %d of %d significant", t, int(d.significant.sum()), len(d))
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
