#!/usr/bin/env python3
"""Figure 1's detection half: paired ROC-AUC difference, DIANA minus the strongest baseline.

Until 2026-09-25 the four detection intervals in `plot_paired_deltas.py` were literals
with no producer in the repo. This writes them, on the same footing as the
classification half: both models score the SAME held-out runs against the SAME planted
flags (`--encoders` gives every training class its probability column, the defect
15_anomaly_detection_roc.py had), and the difference in ROC-AUC of the detector score
1 - P(stated label) is bootstrapped over whole BioProjects, 2,000 resamples.

Comparator, fixed before running: per task, the baseline with the highest held-out
detection AUC among those that can produce a probability (linear SVM cannot). Picked on
the test set, so biased in the baseline's favour, as Table 2's comparator is.

    ./env/bin/python scripts/analysis/39_detector_paired_test.py \\
        --eval-dir results/final_eval_budget_v9 --single-pattern "heldout_{task}" \\
        --encoders-pattern results/final_budget_v9/{task}/final_model/label_encoders.json \\
        --out results/final_eval_budget_v9/detector_paired.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
N_BOOT = 2000
SEED = 42


def rng_for(task: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}|{task}|detector-paired".encode()).hexdigest()[:8]
    return np.random.default_rng(int(digest, 16))


def diana_scores(pred: pd.DataFrame, task: str, stated: np.ndarray, classes: list[str]) -> np.ndarray:
    cols = sorted((c for c in pred.columns if c.startswith(f"{task}_prob_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    if len(cols) != len(classes):
        raise SystemExit(f"{task}: {len(classes)} encoder classes, {len(cols)} probability columns")
    P = pred[cols].to_numpy()
    idx = {c: i for i, c in enumerate(classes)}
    return np.asarray([P[i, idx[s]] if s in idx else np.nan for i, s in enumerate(stated)], float)


def baseline_scores(blk: pd.DataFrame, stated: np.ndarray) -> np.ndarray:
    cols = {c[2:]: c for c in blk.columns if c.startswith("p_")}
    return np.asarray([blk[cols[s]].iloc[i] if s in cols else np.nan
                       for i, s in enumerate(stated)], float)


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "results/final_eval_v9")
    ap.add_argument("--single-pattern", default="heldout_single_{task}")
    ap.add_argument("--encoders-pattern", required=True,
                    help="label_encoders.json of the single-task model, with {task}")
    ap.add_argument("--baseline-dir", type=Path, default=ROOT / "results/baseline_predictions_v9")
    ap.add_argument("--planted", type=Path,
                    default=ROOT / "results/planted_mislabels_v9/planted_test_mixed_r0.1.tsv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    plant = pd.read_csv(a.planted, sep="\t")
    project = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t", low_memory=False
                          ).set_index("Run_accession")["archive_project"]
    rows = []
    for task in TASKS:
        flag = f"{task}_planted"
        pl = plant[["Run_accession", task, flag]].dropna(subset=[task, flag])
        pred = pd.read_csv(a.eval_dir / a.single_pattern.format(task=task) / "test_predictions.tsv",
                           sep="\t")
        classes = json.load(open(a.encoders_pattern.format(task=task)))[task]["classes"]
        m = pred.merge(pl, on="Run_accession", how="inner", suffixes=("", "_stated"))
        stated = m[task].astype(str).to_numpy()
        s_diana = pd.Series(1.0 - diana_scores(m, task, stated, classes), index=m.Run_accession)

        prob = pd.read_csv(a.baseline_dir / f"heldout_probabilities_{task}.tsv", sep="\t")
        s_base = {}
        for model, blk in prob.groupby("model"):
            if model == "MajorityClass":
                continue
            mb = blk.merge(pl, on="Run_accession", how="inner", suffixes=("", "_stated"))
            s_base[model] = pd.Series(1.0 - baseline_scores(mb, mb[task].astype(str).to_numpy()),
                                      index=mb.Run_accession)

        # the common row set: runs every model could score
        common = s_diana.dropna().index
        for s in s_base.values():
            common = common.intersection(s.dropna().index)
        common = sorted(common)
        y = pl.set_index("Run_accession").loc[common, flag].astype(int).to_numpy()
        g = project.loc[common].to_numpy()
        sd = s_diana.loc[common].to_numpy()
        aucs = {mdl: safe_auc(y, s.loc[common].to_numpy()) for mdl, s in s_base.items()}
        best = max(aucs, key=aucs.get)
        sb = s_base[best].loc[common].to_numpy()

        uniq = np.unique(g)
        by = {p: np.where(g == p)[0] for p in uniq}
        rng = rng_for(task)
        dd = []
        for _ in range(N_BOOT):
            sel = np.concatenate([by[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            dd.append(safe_auc(y[sel], sd[sel]) - safe_auc(y[sel], sb[sel]))
        dd = np.asarray(dd, float)
        dd = dd[np.isfinite(dd)]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        rows.append({"task": task, "baseline": best, "n_scored": len(common), "n_planted": int(y.sum()),
                     "n_projects": len(uniq), "auc_diana": safe_auc(y, sd), "auc_baseline": aucs[best],
                     "delta": safe_auc(y, sd) - aucs[best], "ci_low": lo, "ci_high": hi,
                     "frac_boot_positive": float((dd > 0).mean()), "n_boot_finite": len(dd)})
        logger.info("%s: DIANA %.3f vs %s %.3f on %d runs / %d planted; delta %+.3f [%+.3f, %+.3f]",
                    task, rows[-1]["auc_diana"], best, aucs[best], len(common), int(y.sum()),
                    rows[-1]["delta"], lo, hi)
    res = pd.DataFrame(rows)
    res["verdict"] = np.where((res.ci_low > 0), "DIANA", np.where(res.ci_high < 0, "baseline", "tie"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
