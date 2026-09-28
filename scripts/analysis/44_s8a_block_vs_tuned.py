#!/usr/bin/env python3
"""S8.2b, the fair reading: the off-list block alone against the TUNED fraction models.

43_offlist_block_probe.py compared the block against a fractions arm fitted by the same
SGD probe, and that arm came out far below the tuned logistic regression on the same
fractions and folds (0.189 / 0.130 / 0.147 / 0.136 against 0.462 / 0.147 / 0.218 /
0.316), so its "block better than fractions" verdicts are against a crippled reference.
Here the block's out-of-fold predictions are paired, on the same runs, against (a) the
tuned LogisticRegression_Bal on the fractions (results/baselines_representation/) and
(b) DIANA single-task at its dev-fold epoch budget (results/epoch_budget_v9/). Whole
task and by training-support regime, 2,000 BioProject resamples. Scaffolding, not a
result; held-out untouched.

    ./env/bin/python scripts/analysis/44_s8a_block_vs_tuned.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
PROBE = ROOT / "results/s8a_probe_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
N_BOOT, SEED = 2000, 42


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()[:8], 16))


def f1_elig(y, p, eligible):
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def paired(y, pa, pb, g, eligible, key):
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
    return {"f1_block": f1_elig(y, pa, eligible), "f1_ref": f1_elig(y, pb, eligible), "delta": obs,
            "ci_low": lo, "ci_high": hi, "frac_boot_positive": float((dd > 0).mean()) if len(dd) else np.nan,
            "n_projects": len(uniq), "verdict": "tie" if lo <= 0 <= hi else ("block" if obs > 0 else "reference")}


def main() -> int:
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    rows = []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        counts = meta[task].dropna().astype(str).value_counts()
        probe = pd.read_csv(PROBE / f"oof_{task}.tsv", sep="\t").set_index("Run_accession")
        logreg = pd.concat([pd.read_csv(ROOT / f"results/baselines_representation/pred_fraction_{task}_fold{k}.tsv", sep="\t")
                            for k in range(5)])
        logreg = logreg[logreg.model == "LogisticRegression_Bal"].set_index("Run_accession")
        diana = pd.read_csv(ROOT / f"results/epoch_budget_v9/oof_{task}_{task}.tsv", sep="\t")
        diana = diana[diana[f"{task}_true"].notna()].set_index("Run_accession")
        refs = {"tuned LogReg (fractions)": logreg["y_pred"], "DIANA single at budget": diana[f"{task}_pred"]}
        for name, ref in refs.items():
            common = sorted(set(probe.index) & set(ref.index))
            y = probe.loc[common, "y_true"].astype(str).to_numpy()
            pa = probe.loc[common, "pred_block"].astype(str).to_numpy()
            pb = ref.loc[common].astype(str).to_numpy()
            g = meta.loc[common, "archive_project"].astype(str).to_numpy()
            r = paired(y, pa, pb, g, eligible, (task, name, "all"))
            rows.append({"task": task, "reference": name, "regime": "all", "n_runs": len(common), **r})
            for rn, lo_, hi_ in REGIMES:
                cls = {c for c, n in counts.items() if lo_ < n <= hi_} & eligible
                if cls:
                    r = paired(y, pa, pb, g, cls, (task, name, rn))
                    rows.append({"task": task, "reference": name, "regime": rn, "n_runs": len(common), **r})
    res = pd.DataFrame(rows)
    res.to_csv(PROBE / "block_vs_tuned.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print("\nverdicts:", res.verdict.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
