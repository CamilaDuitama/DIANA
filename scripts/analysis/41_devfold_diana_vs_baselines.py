#!/usr/bin/env python3
"""DIANA against the tuned baselines on the pooled dev folds: 68 projects instead of 34.

What this is. The five BioProject-grouped dev folds already hold an out-of-fold
prediction for every training run from every model: DIANA single-task at its epoch
budget (`results/epoch_budget_v9/oof_<task>_<task>.tsv`, 37_epoch_budget_select.py)
and the five tuned baselines (`results/baselines_representation/
pred_fraction_<task>_fold<k>.tsv`, 27_baselines_representation_devfold.py). Pooling
them gives the same paired comparison as Table 2's whole-task test, resampled over
68 projects rather than held-out's 34.

What this is NOT. A result. The vocabulary was built from all 2,716 training runs, so
each fold's test runs helped choose the k-mers (a mild R3.4-type leak, identical for
every model); DIANA's hyperparameters and epoch budgets and the baselines' settings
were all tuned on these same folds. It is scaffolding: a cheap preview of what the
repeated-CV design (PROJECT.md, 2026-09-28) would change, namely the interval width.
Held-out is untouched.

    ./env/bin/python scripts/analysis/41_devfold_diana_vs_baselines.py
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
DIANA = ROOT / "results/epoch_budget_v9"
BASE = ROOT / "results/baselines_representation"
OUT = DIANA / "devfold_vs_baselines.tsv"
TASKS = ["community_type", "feature", "sample_host", "material"]
N_BOOT = 2000
SEED = 42


def rng_for(task: str, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}|{task}|{name}".encode()).hexdigest()[:8]
    return np.random.default_rng(int(digest, 16))


def f1_elig(y, p, eligible) -> float:
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    project = meta.set_index("Run_accession")["archive_project"]
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    rows = []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        d = pd.read_csv(DIANA / f"oof_{task}_{task}.tsv", sep="\t")
        d = d[d[f"{task}_true"].notna()].set_index("Run_accession")
        b = pd.concat([pd.read_csv(BASE / f"pred_fraction_{task}_fold{k}.tsv", sep="\t")
                       for k in range(5)], ignore_index=True)
        common = sorted(set(d.index) & set(b.Run_accession))
        y = d.loc[common, f"{task}_true"].astype(str).to_numpy()
        pd_ = d.loc[common, f"{task}_pred"].astype(str).to_numpy()
        g = project.loc[common].to_numpy()
        uniq = np.unique(g)
        by = {p: np.where(g == p)[0] for p in uniq}
        for model, blk in b.groupby("model"):
            bb = blk.set_index("Run_accession").loc[common]
            if not np.array_equal(bb.y_true.astype(str).to_numpy(), y):
                raise SystemExit(f"{task} {model}: true labels differ between files")
            pb = bb.y_pred.astype(str).to_numpy()
            obs = f1_elig(y, pd_, eligible) - f1_elig(y, pb, eligible)
            rng = rng_for(task, model)
            dd = []
            for _ in range(N_BOOT):
                sel = np.concatenate([by[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
                dd.append(f1_elig(y[sel], pd_[sel], eligible) - f1_elig(y[sel], pb[sel], eligible))
            dd = np.asarray(dd, float)
            dd = dd[np.isfinite(dd)]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            rows.append({"task": task, "baseline": model, "n_runs": len(common), "n_projects": len(uniq),
                         "f1_diana": f1_elig(y, pd_, eligible), "f1_baseline": f1_elig(y, pb, eligible),
                         "delta": obs, "ci_low": lo, "ci_high": hi, "ci_width": hi - lo,
                         "frac_boot_positive": float((dd > 0).mean()),
                         "verdict": "DIANA" if lo > 0 else ("baseline" if hi < 0 else "tie")})
        logger.info("%s: %d runs, %d projects", task, len(common), len(uniq))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print("\nverdicts:", res.verdict.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
