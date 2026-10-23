#!/usr/bin/env python3
"""Final A6 for v17: single-task vs multi-task at the final settings, on the
5-seed ensemble out-of-fold predictions (results/seeds_ensemble_v17).

Same rule and same bootstrap as 37_'s A6 section (fixed 2026-09-25): per task,
the paired difference single minus multi, 2,000 resamples over archive_project;
single if the interval is above zero, multi if below, tie otherwise. One
architecture for all four tasks: an arm that wins at least one task and loses
none is chosen; all ties or a split keeps the published single-task.

    ./env/bin/python scripts/analysis/58_final_a6_v17.py
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
N_BOOT = 2000


def rng_for(task: str, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{task}:{name}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(digest, 16))


def f1_elig(y: np.ndarray, p: np.ndarray, eligible: set) -> float:
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, default=ROOT / "results/seeds_ensemble_v17")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits_v17")
    a = ap.parse_args()
    meta = pd.read_csv(a.splits / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    elig_tbl = pd.read_csv(a.splits / "class_eligibility.tsv", sep="\t")
    rows = []
    for t in TASKS:
        single = pd.read_csv(a.base / f"oof_{t}_{t}.tsv", sep="\t").set_index("Run_accession")
        multi = pd.read_csv(a.base / f"oof_multitask_{t}.tsv", sep="\t").set_index("Run_accession")
        if sorted(single.index) != sorted(multi.index):
            raise SystemExit(f"{t}: single and multi arms scored different runs")
        multi = multi.loc[single.index]
        lab = single[f"{t}_true"].notna()
        yy = single.loc[lab, f"{t}_true"].astype(str).to_numpy()
        ps = single.loc[lab, f"{t}_pred"].astype(str).to_numpy()
        pm = multi.loc[lab, f"{t}_pred"].astype(str).to_numpy()
        g = meta.loc[single.index[lab], "archive_project"].to_numpy()
        elig = set(elig_tbl[(elig_tbl.target == t) & elig_tbl.evaluable]["class"].astype(str))
        uniq = np.unique(g)
        by = {p: np.where(g == p)[0] for p in uniq}
        obs = f1_elig(yy, ps, elig) - f1_elig(yy, pm, elig)
        rng = rng_for(t, "a6:single-multi")
        dd = []
        for _ in range(N_BOOT):
            sel = np.concatenate([by[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            dd.append(f1_elig(yy[sel], ps[sel], elig) - f1_elig(yy[sel], pm[sel], elig))
        dd = np.asarray(dd, float)
        dd = dd[np.isfinite(dd)]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        rows.append({"task": t, "n_runs": int(lab.sum()), "n_projects": len(uniq),
                     "f1_single": f1_elig(yy, ps, elig), "f1_multi": f1_elig(yy, pm, elig),
                     "delta_single_minus_multi": obs, "ci_low": lo, "ci_high": hi,
                     "frac_boot_positive": float((dd > 0).mean()),
                     "verdict": "single" if lo > 0 else ("multi" if hi < 0 else "tie")})
    a6 = pd.DataFrame(rows)
    a6.to_csv(a.base / "a6_final_paired.tsv", sep="\t", index=False)
    wins = a6.verdict.value_counts().to_dict()
    n_single, n_multi = wins.get("single", 0), wins.get("multi", 0)
    if n_single and not n_multi:
        decision = "single-task"
    elif n_multi and not n_single:
        decision = "multi-task"
    else:
        decision = "single-task (published architecture stands: no arm wins without losing)"
    (a.base / "a6_final_decision.txt").write_text(decision + "\n")
    pd.set_option("display.width", 220)
    print(a6.round(4).to_string(index=False))
    print("\ndecision:", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
