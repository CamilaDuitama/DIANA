#!/usr/bin/env python3
"""Table 2's classification half, for any arm, on held-out and by training-support regime.

The rule is not invented here. `results/final_eval_v9/stratified_by_shot.tsv` has no
producer anywhere in the repo, so it was reverse-engineered on 2026-09-16 and this script
**asserts** it reproduces DIANA single's 12 published cells before reporting any new arm.
It does, to 1.11e-16.

The rule, in the order that matters:

1. **Classes** are those in the held-out `classification_report` with support > 0. Classes
   the model can emit but that have no held-out run are excluded, which is what made the
   few-shot counts 5/2/4 rather than 7/4/5.
2. **Regime** is the class's **training** run count: few-shot < 20, medium-shot 20-100,
   many-shot > 100, the ImageNet-LT / OLTR split points.
3. **The value is the MEAN of those classes' per-class F1 from the report**, not a macro-F1
   recomputed from the predictions. This is the step that is easy to get wrong and it is not
   cosmetic: recomputing the obvious way disagrees by up to 0.047, larger than most effects
   being chased. The two differ because `diana-test` scores a slightly different row set
   than a naive `dropna` on the true label.

    ./env/bin/python scripts/analysis/30_heldout_regime_table.py --arm results/dnamax_final_v9 --label dna_max
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot (<20)", 0, 20), ("medium-shot (20-100)", 20, 100),
           ("many-shot (>100)", 100, np.inf)]
REFERENCE = "results/final_eval_v9/stratified_by_shot.tsv"


def training_support() -> dict[str, pd.Series]:
    meta = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False)
    return {t: meta[t].dropna().astype(str).value_counts() for t in TASKS}


def arm_table(arm: Path, support: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        base = arm / f"heldout_{task}"
        if not base.exists():
            base = arm / f"heldout_single_{task}"
        mp, pp = base / "test_metrics.json", base / "test_predictions.tsv"
        if not (mp.exists() and pp.exists()):
            logger.warning("%s: no held-out output under %s", task, arm)
            continue
        m = json.load(mp.open())[task]
        d = pd.read_csv(pp, sep="\t")
        idx2name = (d[[f"{task}_true_idx", f"{task}_true"]].dropna().drop_duplicates()
                     .set_index(f"{task}_true_idx")[f"{task}_true"].to_dict())
        ts = support[task]
        rep = {int(k): v for k, v in m["classification_report"].items() if k.isdigit()}
        for name, lo, hi in REGIMES:
            f1s, runs = [], 0
            for k, v in rep.items():
                if v["support"] <= 0:
                    continue
                cname = idx2name.get(k) or idx2name.get(float(k))
                if cname is None:
                    continue
                n = ts.get(str(cname), 0)
                if (n < hi) if lo == 0 else (lo < n <= hi):
                    f1s.append(v["f1-score"]); runs += int(v["support"])
            rows.append({"task": task, "stratum": name, "n_classes": len(f1s),
                         "n_runs": runs, "f1": float(np.mean(f1s)) if f1s else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--label", default="arm")
    a = ap.parse_args()
    support = training_support()

    # Refuse to report a new arm unless the rule still reproduces the published table.
    ref = pd.read_csv(ROOT / REFERENCE, sep="\t")
    ref = ref[ref.model == "DIANA single"].set_index(["task", "stratum"]).f1
    mine = arm_table(ROOT / "results/final_eval_v9", support).set_index(["task", "stratum"]).f1
    both = [k for k in mine.index if k in ref.index and pd.notna(ref[k]) and pd.notna(mine[k])]
    err = max(abs(mine[k] - ref[k]) for k in both)
    logger.info("reproduces DIANA single on %d of %d cells, worst error %.2e", len(both), len(mine), err)
    if err > 1e-9:
        logger.error("the rule no longer reproduces the published table; refusing to report")
        return 1

    new = arm_table(ROOT / a.arm, support)
    old = arm_table(ROOT / "results/final_eval_v9", support)
    m = old.merge(new, on=["task", "stratum"], suffixes=("_diana", f"_{a.label}"))
    m["delta"] = m[f"f1_{a.label}"] - m["f1_diana"]
    print()
    print(m.to_string(index=False))
    out = ROOT / a.arm / "heldout_regime_table.tsv"
    m.to_csv(out, sep="\t", index=False)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
