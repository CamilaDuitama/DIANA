#!/usr/bin/env python3
"""Per-regime paired superiority test, replacing an invented tie margin.

`23_regime_stratified_table.py` compares DIANA against the best baseline per
(task, regime) cell from two point estimates, which cannot say whether a
difference is real. This runs the project's own rule instead: bootstrap the
**paired** difference over whole BioProjects, restricting the macro-F1 to the
eligible classes whose training support falls in that regime, and call the cell
a tie when the interval on the difference includes zero.

Pairing is what makes this work: both models predict the same runs, so the
between-project variance that puts +/- 0.18 on each model's own interval cancels.

The comparator is the baseline with the highest point estimate in that cell,
which is selected on the test set and therefore biased in the baseline's favour.
That bias is left in deliberately: it makes every DIANA win conservative.

    ./env/bin/python scripts/analysis/25_regime_paired_test.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from importlib import import_module

_sup = import_module("18_paired_superiority")
paired, f1_elig = _sup.paired, _sup.f1_elig

logger = logging.getLogger(__name__)
SPLITS = ROOT / "data/splits_v9"
EVAL = ROOT / "results/final_eval_v9"
BASE = ROOT / "results/baseline_predictions_v9/heldout_predictions.tsv"
OUT = ROOT / "results/final_eval_v9/regime_paired.tsv"

TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("many-shot (>100)", 100, np.inf), ("medium-shot (20-100)", 20, 100),
           ("few-shot (<20)", 0, 20)]
N_BOOT = 2000
SEED = 42


def regime_of(n: float) -> str | None:
    for name, lo, hi in REGIMES:
        if lo < n <= hi:
            return name
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    meta = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t", low_memory=False)
    base = pd.read_csv(BASE, sep="\t")
    if "model" not in base.columns:
        raise SystemExit(f"{BASE} has no model column: {list(base.columns)[:8]}")

    rows = []
    for task in TASKS:
        eligible = set(elig[(elig.target == task) & elig.evaluable]["class"].astype(str))
        by_class = {str(c): regime_of(n) for c, n in train[task].value_counts().items()
                    if regime_of(n)}

        d = pd.read_csv(EVAL / f"heldout_single_{task}/test_predictions.tsv", sep="\t")
        d = d[["Run_accession", f"{task}_pred", f"{task}_true"]].rename(
            columns={f"{task}_pred": "diana", f"{task}_true": "y"})
        d = d.merge(meta[["Run_accession", "archive_project"]], on="Run_accession")
        b = base[base.task == task] if "task" in base.columns else base
        cols = {"Run_accession", "model"}
        pcol = next(c for c in ("y_pred", "pred", f"{task}_pred") if c in b.columns)
        wide = b.pivot_table(index="Run_accession", columns="model", values=pcol,
                             aggfunc="first")
        m = d.merge(wide, on="Run_accession", how="inner")
        m = m[m.y.notna()]

        models = [c for c in wide.columns if c != "MajorityClass"]
        for name, _, _ in REGIMES:
            keep = {c for c in eligible if by_class.get(c) == name}
            if not keep:
                continue
            # Standard long-tail convention (ImageNet-LT / OLTR): score every
            # row, average macro-F1 over only this regime's classes. Filtering
            # rows to the regime instead would discard the false positives that
            # rare classes attract from elsewhere, which is the main thing that
            # makes them hard.
            sub = m
            n_reg = int(m.y.astype(str).isin(keep).sum())
            if n_reg == 0 or sub.archive_project.nunique() < 3:
                logger.info("%s %s: %d runs in regime, %d projects — skipped",
                            task, name, n_reg, sub.archive_project.nunique())
                continue
            y = sub.y.astype(str).to_numpy()
            g = sub.archive_project.to_numpy()
            dia = sub.diana.astype(str).to_numpy()
            f_d = f1_elig(y, dia, keep)
            # comparator: highest point estimate in this cell (test-set selected)
            scored = [(mo, f1_elig(y, sub[mo].astype(str).to_numpy(), keep))
                      for mo in models if sub[mo].notna().all()]
            scored = [(mo, v) for mo, v in scored if np.isfinite(v)]
            if not scored:
                continue
            best, f_b = max(scored, key=lambda t: t[1])
            r = paired(y, dia, sub[best].astype(str).to_numpy(), g, keep, N_BOOT, SEED)
            tie = r["ci_low"] <= 0.0 <= r["ci_high"]
            rows.append({"task": task, "regime": name.split()[0],
                         "classes": len(keep), "runs": n_reg,
                         "projects": r["n_projects"], "diana_f1": f_d,
                         "best_baseline": best, "baseline_f1": f_b,
                         "delta": r["delta"], "ci_low": r["ci_low"],
                         "ci_high": r["ci_high"],
                         "frac_boot_positive": r["frac_boot_positive"],
                         "verdict": "tie" if tie else
                                    ("DIANA" if r["delta"] > 0 else "baseline")})

    res = pd.DataFrame(rows)
    res.to_csv(OUT, sep="\t", index=False)
    logger.info("wrote %s", OUT)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print("\nverdicts:", res.verdict.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
