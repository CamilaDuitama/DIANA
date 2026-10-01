#!/usr/bin/env python3
"""S8.8 / B3: DIANA on fractions + class-signature columns against DIANA on fractions.

Both sides are the pooled out-of-fold predictions at each arm's own dev-fold epoch
budget (37_epoch_budget_select.py, run with --base for each set): the sig arms from
results/epoch_budget_sig_v9/oof_<arm>_<task>.tsv, the fraction arms from
results/epoch_budget_v9/oof_<arm>_<task>.tsv. Same runs, f1_macro_eligible, paired over
BioProjects with 2,000 resamples, whole task and by training-support regime, for the
single-task arm of each task and for the multi-task net. Scaffolding, held-out untouched.

    ./env/bin/python scripts/analysis/48_paired_sig_vs_fraction.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
SIG, FRAC = ROOT / "results/epoch_budget_sig_v9", ROOT / "results/epoch_budget_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
N_BOOT, SEED = 2000, 42
SINGLE_ONLY = False
NAME = "sig"   # --name-a: label of the compared arm in columns, verdicts and the output file name


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
    dd = np.asarray(dd, float); dd = dd[np.isfinite(dd)]
    lo, hi = (np.percentile(dd, [2.5, 97.5]) if len(dd) else (np.nan, np.nan))
    return {f"f1_{NAME}": f1_elig(y, pa, eligible), "f1_fraction": f1_elig(y, pb, eligible), "delta": obs,
            "ci_low": lo, "ci_high": hi, "frac_boot_positive": float((dd > 0).mean()) if len(dd) else np.nan,
            "n_projects": len(uniq), "verdict": "tie" if lo <= 0 <= hi else (NAME if obs > 0 else "fraction")}


def main() -> int:
    global SIG, FRAC, SPLITS, NAME
    import argparse
    ap = argparse.ArgumentParser(description="DIANA sig vs fraction, paired over BioProjects on the dev folds")
    ap.add_argument("--sig", type=Path, default=SIG, help="epoch-budget directory of the sig arms")
    ap.add_argument("--frac", type=Path, default=FRAC, help="epoch-budget directory of the fraction arms")
    ap.add_argument("--splits", type=Path, default=SPLITS, help="split directory; v13 = data/splits_v13")
    ap.add_argument("--name-a", default=NAME, help="label of the --sig arm (sig, hash1000, hash100, ...)")
    ap.add_argument("--single-only", action="store_true", help="compare the single-task arms only (no multitask files needed)")
    args = ap.parse_args()
    SIG, FRAC, SPLITS, NAME = args.sig, args.frac, args.splits, args.name_a
    global SINGLE_ONLY
    SINGLE_ONLY = args.single_only
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    bud_s = pd.read_csv(SIG / "epoch_budget.tsv", sep="\t").set_index("arm")["budget"]
    bud_f = pd.read_csv(FRAC / "epoch_budget.tsv", sep="\t").set_index("arm")["budget"]
    rows = []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        counts = meta[task].dropna().astype(str).value_counts()
        for arm in ((task,) if SINGLE_ONLY else (task, "multitask")):
            a = pd.read_csv(SIG / f"oof_{arm}_{task}.tsv", sep="\t"); a = a[a[f"{task}_true"].notna()].set_index("Run_accession")
            b = pd.read_csv(FRAC / f"oof_{arm}_{task}.tsv", sep="\t"); b = b[b[f"{task}_true"].notna()].set_index("Run_accession")
            common = sorted(set(a.index) & set(b.index))
            y = a.loc[common, f"{task}_true"].astype(str).to_numpy()
            if not np.array_equal(y, b.loc[common, f"{task}_true"].astype(str).to_numpy()):
                raise SystemExit(f"{task} {arm}: labels differ")
            pa, pb = a.loc[common, f"{task}_pred"].astype(str).to_numpy(), b.loc[common, f"{task}_pred"].astype(str).to_numpy()
            g = meta.loc[common, "archive_project"].astype(str).to_numpy()
            base = {"task": task, "arm": "single-task" if arm == task else "multi-task", "n_runs": len(common),
                    f"budget_{NAME}": int(bud_s[arm]), "budget_fraction": int(bud_f[arm])}
            rows.append({**base, "regime": "all", **paired(y, pa, pb, g, eligible, (task, arm, "all"))})
            for rn, lo_, hi_ in REGIMES:
                cls = {c for c, n in counts.items() if lo_ < n <= hi_} & eligible
                if cls:
                    rows.append({**base, "regime": rn, **paired(y, pa, pb, g, cls, (task, arm, rn))})
    res = pd.DataFrame(rows)
    res.to_csv(SIG / f"paired_{NAME}_vs_fraction.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print("\nverdicts:", res.verdict.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
