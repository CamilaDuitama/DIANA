#!/usr/bin/env python3
"""W4: DIANA and the baselines on new runs from SEEN studies (within-study folds) beside
their held-out numbers on runs from UNSEEN studies (read 9).

Within-study side: pooled out-of-fold predictions on the 2,716 training runs from the
sample-grouped folds (34_): DIANA single-task and multi-task arms at the read-9 epoch
budgets (04_ run with --max-epochs = budget; the last epoch is taken), the baselines at
their read-9 configurations (16_). Across-study side: read 9, results/final_eval_budget_v9
and results/baseline_predictions_v9, already taken. Same metric on both sides,
f1_macro_eligible; intervals and the DIANA-vs-best-baseline paired test bootstrap over
BioProjects on both sides (within-study, a project sits in several folds; it is still the
unit whose runs are not independent).

The within-study side is a cross-validation on the training set, not a held-out result; it
is reported as the "new run, known study" setting, labelled as such.

Writes results/within_study_v9/{within_study_all_models.tsv, within_vs_across.tsv,
within_study_paired.tsv}.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
SPLITS, W = ROOT / "data/splits_v9", ROOT / "results/within_study_v9"
ACROSS = ROOT / "results/final_eval_budget_v9/heldout_all_models.tsv"
ACROSS_PAIRED = ROOT / "results/final_eval_budget_v9/regime_paired.tsv"
TASKS = ["community_type", "feature", "sample_host", "material"]
ARMS = {"multitask": TASKS, **{t: [t] for t in TASKS}}
BASELINES = ["LogisticRegression_Bal", "LinearSVM_Bal", "RandomForest_Bal", "kNN"]
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
N_BOOT, SEED = 2000, 42


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()[:8], 16))


def f1_elig(y, p, eligible):
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def boot_ci(y, p, g, eligible, key):
    uniq = np.unique(g); by = {q: np.where(g == q)[0] for q in uniq}; rng = rng_for(*key); vals = []
    for _ in range(N_BOOT):
        sel = np.concatenate([by[q] for q in rng.choice(uniq, size=len(uniq), replace=True)])
        vals.append(f1_elig(y[sel], p[sel], eligible))
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    return np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)


def paired(y, pa, pb, g, eligible, key):
    uniq = np.unique(g); by = {q: np.where(g == q)[0] for q in uniq}; rng = rng_for(*key); dd = []
    for _ in range(N_BOOT):
        sel = np.concatenate([by[q] for q in rng.choice(uniq, size=len(uniq), replace=True)])
        dd.append(f1_elig(y[sel], pa[sel], eligible) - f1_elig(y[sel], pb[sel], eligible))
    dd = np.asarray(dd, float); dd = dd[np.isfinite(dd)]
    lo, hi = np.percentile(dd, [2.5, 97.5]) if len(dd) else (np.nan, np.nan)
    obs = f1_elig(y, pa, eligible) - f1_elig(y, pb, eligible)
    return obs, lo, hi, float((dd > 0).mean()) if len(dd) else np.nan


def diana_oof(arm: str, task: str) -> pd.DataFrame:
    runs, preds = [], []
    for k in range(5):
        d = W / "diana" / arm / f"fold{k}"
        classes = json.load(open(d / "label_classes.json"))[task]
        r = [l.strip() for l in open(d / "eval_runs.txt") if l.strip()]
        z = np.load(d / "preds_by_epoch.npz")
        preds += [classes[i] for i in z[task][-1]]     # last epoch = the read-9 budget
        runs += r
    return pd.DataFrame({"Run_accession": runs, "y_pred": preds}).set_index("Run_accession")


def main() -> int:
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    across = pd.read_csv(ACROSS, sep="\t")
    rows, paired_rows = [], []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        counts = meta[task].dropna().astype(str).value_counts()
        labelled = meta.index[meta[task].notna()]
        preds = {"DIANA single-task": diana_oof(task, task), "DIANA multi-task": diana_oof("multitask", task)}
        for m in BASELINES:
            b = pd.read_csv(W / f"pred_{m}_{task}.tsv", sep="\t").set_index("Run_accession")
            preds[m] = b[["y_pred"]]
        common = sorted(set(labelled) & set.intersection(*[set(p.index) for p in preds.values()]))
        y = meta.loc[common, task].astype(str).to_numpy(); g = meta.loc[common, "archive_project"].astype(str).to_numpy()
        P = {m: p.loc[common, "y_pred"].astype(str).to_numpy() for m, p in preds.items()}
        for m, p in P.items():
            f1 = f1_elig(y, p, eligible); lo, hi = boot_ci(y, p, g, eligible, (task, m))
            rows.append({"task": task, "model": m, "setting": "within-study (new run, seen study; training-set CV)",
                         "f1_macro_eligible": f1, "ci_low": lo, "ci_high": hi, "n_runs": len(common), "n_projects": len(set(g))})
        best = max(BASELINES, key=lambda m: f1_elig(y, P[m], eligible))
        for rn, lo_, hi_ in [("all", 0, np.inf)] + REGIMES:
            cls = ({c for c, n in counts.items() if lo_ < n <= hi_} & eligible) if rn != "all" else eligible
            if not cls:
                continue
            obs, lo, hi, frac = paired(y, P["DIANA single-task"], P[best], g, cls, (task, rn))
            paired_rows.append({"task": task, "regime": rn, "classes": len(cls), "diana_f1": f1_elig(y, P["DIANA single-task"], cls),
                                "best_baseline": best, "baseline_f1": f1_elig(y, P[best], cls), "delta": obs, "ci_low": lo, "ci_high": hi,
                                "frac_boot_positive": frac, "verdict": "tie" if lo <= 0 <= hi else ("DIANA" if obs > 0 else "baseline")})
    within = pd.DataFrame(rows)
    within.to_csv(W / "within_study_all_models.tsv", sep="\t", index=False)
    pd.DataFrame(paired_rows).to_csv(W / "within_study_paired.tsv", sep="\t", index=False)
    ac = across.rename(columns={"f1": "f1_macro_eligible", "lo": "ci_low", "hi": "ci_high"})
    ac["setting"] = "across-study (unseen study; held-out, read 9)"
    side = pd.concat([within[["task", "model", "setting", "f1_macro_eligible", "ci_low", "ci_high"]],
                      ac[["task", "model", "setting", "f1_macro_eligible", "ci_low", "ci_high"]]])
    side = side[side.model.isin(list(preds))]
    side.to_csv(W / "within_vs_across.tsv", sep="\t", index=False)
    pd.set_option("display.width", 240)
    print("\nf1_macro_eligible, within-study (new run, seen study) vs across-study (held-out read 9):")
    print(side.pivot_table(index=["task", "model"], columns="setting", values="f1_macro_eligible").round(3).to_string())
    print("\nwithin-study: DIANA single-task vs best baseline, paired over BioProjects:")
    print(pd.DataFrame(paired_rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
