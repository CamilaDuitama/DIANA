#!/usr/bin/env python3
"""Paired held-out comparison of one arm against DIANA, bootstrapped over BioProjects.

Point estimates are not a comparison. Both models predict the same 922 runs, so the
difference is paired and the resampling unit is the BioProject, of which held-out has 34.
`CLAUDE.md` requires this form: pairing cancels the between-project difficulty that puts
+/- 0.18 on each arm's own interval, and a run-level interval would be 2.2 to 5.1x too
narrow because runs from one study share protocol, lab and sometimes specimen.

The metric is the one Table 2 uses, reproduced by the same rule this project's regime table
was reverse-engineered to: the classes are those with held-out support > 0, and the score is
the mean of their per-class F1. Within a bootstrap resample the per-class F1 is recomputed
from the resampled rows, so the interval reflects the same quantity the point estimate does.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
N_BOOT, SEED = 2000, 42


def load(dirpath: Path, task: str) -> pd.DataFrame | None:
    for name in (f"heldout_{task}", f"heldout_single_{task}"):
        p = dirpath / name / "test_predictions.tsv"
        if p.exists():
            d = pd.read_csv(p, sep="\t")
            return d.dropna(subset=[f"{task}_true"]).set_index("Run_accession")
    return None


def score(y: pd.Series, pred: pd.Series, classes: list[str]) -> float:
    if not classes:
        return np.nan
    per = f1_score(y.astype(str), pred.astype(str), labels=classes, average=None, zero_division=0)
    return float(np.mean(per))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--reference", type=Path, default=Path("results/final_eval_v9"),
                    help="the arm to compare against. DIANA by default; point it at the "
                         "scrambled arm to isolate whether the DNA content contributes, "
                         "which is the only comparison that holds the 768 extra columns "
                         "fixed and varies nothing but the letters.")
    a = ap.parse_args()

    meta = pd.read_csv(ROOT / "data/splits_v9/test_metadata.tsv", sep="\t", low_memory=False)
    proj = meta.set_index("Run_accession")["archive_project"].to_dict()
    train = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False)

    rows = []
    for task in TASKS:
        base = load(ROOT / a.reference, task)
        arm = load(ROOT / a.arm, task)
        if base is None or arm is None:
            logger.warning("%s: missing predictions", task); continue
        idx = base.index.intersection(arm.index)
        y = base.loc[idx, f"{task}_true"].astype(str)
        pb = base.loc[idx, f"{task}_pred"].astype(str)
        pa = arm.loc[idx, f"{task}_pred"].astype(str)
        g = pd.Series([proj.get(r) for r in idx], index=idx)
        sup = train[task].dropna().astype(str).value_counts()
        # The class set must be the one Table 2 uses, or the deltas are on a different
        # denominator than the table they are meant to qualify: classes in the held-out
        # classification report with support > 0. Scoring every class present instead
        # changed `feature` few-shot from 2 classes to 4 and its delta by 0.02.
        mp = ROOT / "results/final_eval_v9" / f"heldout_single_{task}/test_metrics.json"
        rep = json.load(mp.open())[task]["classification_report"]
        i2n = (pd.read_csv(ROOT / "results/final_eval_v9" /
                           f"heldout_single_{task}/test_predictions.tsv", sep="\t")
               [[f"{task}_true_idx", f"{task}_true"]].dropna().drop_duplicates()
               .set_index(f"{task}_true_idx")[f"{task}_true"].to_dict())
        present = []
        for k, v in rep.items():
            if not k.isdigit() or v["support"] <= 0:
                continue
            cn = i2n.get(int(k)) or i2n.get(float(k))
            if cn is not None:
                present.append(str(cn))

        groups = sorted(set(g.dropna()))
        rng = np.random.default_rng(SEED)
        draws = [rng.choice(groups, size=len(groups), replace=True) for _ in range(N_BOOT)]

        for rname, lo, hi in [("whole-task", -1, np.inf)] + REGIMES:
            if rname == "whole-task":
                cls = sorted(present)
            else:
                cls = sorted(c for c in present
                             if (sup.get(c, 0) < hi if lo == 0 else lo < sup.get(c, 0) <= hi))
            if not cls:
                continue
            d0 = score(y, pa, cls) - score(y, pb, cls)
            deltas = []
            for drawn in draws:
                take = np.concatenate([np.flatnonzero((g == p).to_numpy()) for p in drawn])
                yy = y.iloc[take]; 
                deltas.append(score(yy, pa.iloc[take], cls) - score(yy, pb.iloc[take], cls))
            deltas = np.array([x for x in deltas if np.isfinite(x)])
            lo_ci, hi_ci = np.percentile(deltas, [2.5, 97.5])
            verdict = ("WORSE" if hi_ci < 0 else "BETTER" if lo_ci > 0 else "tie")
            rows.append({"task": task, "regime": rname, "n_classes": len(cls),
                         "n_runs": int(y.isin(cls).sum()), "delta": d0,
                         "ci_low": lo_ci, "ci_high": hi_ci,
                         "frac_positive": float((deltas > 0).mean()), "verdict": verdict})
    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    dest = ROOT / a.arm / f"heldout_paired_vs_{a.reference.name}.tsv"
    out.to_csv(dest, sep="\t", index=False)
    logger.info("wrote %s  (%d projects resampled, %d draws)", dest, len(groups), N_BOOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
