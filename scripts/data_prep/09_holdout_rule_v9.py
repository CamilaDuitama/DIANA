#!/usr/bin/env python3
"""Does a pre-declared rule select the held-out size we actually used?

The problem this addresses
-------------------------
The v9 held-out fraction was chosen by looking. `n_holdout_splits=5` (~25 %) was
picked over 7 (~14 %) because at 14 % the `feature` head fell to about 75 held-out
runs. That is a defensible engineering judgement and a bad methods sentence: a
referee can say the test set was sized after seeing what each size gave.

The fix is not to argue for 5. It is to declare a rule that is computable without
looking at any model output, run it over the candidate sizes, and report which size
the rule selects.

    RULE: choose the SMALLEST held-out fraction (largest n_splits) such that every
          evaluable class -- a class occurring in >= 2 BioProjects -- retains at
          least MIN_PER_CLASS held-out runs, for every task.

Smallest held-out is the right direction: it leaves the most data for training, so
the rule cannot be accused of inflating the test set to stabilise the numbers. The
only free parameter is MIN_PER_CLASS, declared up front.

Honest note on the consequence
------------------------------
If the rule selects something other than 5, following it means regenerating the
split -- and the split and the feature matrix are locked together (see the warning
at the top of PROJECT.md), so it also means a ~16 h MUSET rebuild and redoing every
number downstream. This script only reports what the rule selects. Acting on it is
a separate decision.

    ./env/bin/python scripts/data_prep/09_holdout_rule_v9.py --min-per-class 10
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
GROUP_COL = "archive_project"
TASKS = ["community_type", "feature", "sample_host", "material"]
SEED = 42
MIN_STRATUM = 10

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def stratify_key(df: pd.DataFrame, min_count: int) -> np.ndarray:
    key = df[TASKS].fillna("_").agg("|".join, axis=1)
    vc = key.value_counts()
    return key.where(key.map(vc) >= min_count, "__rare__").to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-per-class", type=int, default=10,
                    help="the rule's only free parameter, declared up front")
    ap.add_argument("--candidates", type=int, nargs="+", default=[4, 5, 6, 7, 8, 10],
                    help="n_holdout_splits values to test; larger = smaller holdout")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/holdout_rule")
    args = ap.parse_args()

    # The whole corpus as it stood before partitioning, so the rule is evaluated on
    # the same population the split was drawn from.
    full = pd.concat([
        pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False),
        pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t", low_memory=False),
    ], ignore_index=True).drop_duplicates("Run_accession").reset_index(drop=True)
    logger.info("corpus: %d runs, %d BioProjects",
                len(full), full[GROUP_COL].nunique())

    # Evaluable classes come from the corpus, not from any split.
    eligible = {}
    for t in TASKS:
        sub = full[[t, GROUP_COL]].dropna()
        nproj = sub.groupby(t)[GROUP_COL].nunique()
        eligible[t] = set(nproj[nproj >= 2].index)
    logger.info("evaluable classes: %s", {t: len(v) for t, v in eligible.items()})

    groups = full[GROUP_COL].astype(str)
    y = stratify_key(full, MIN_STRATUM)

    rows = []
    for n in args.candidates:
        if n > full[GROUP_COL].nunique():
            continue
        sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
        _, hold_idx = next(iter(sgkf.split(full, y, groups=groups)))
        held = full.iloc[hold_idx]
        rec = {"n_splits": n, "n_holdout": len(held),
               "pct_holdout": round(100 * len(held) / len(full), 1),
               "n_bioprojects": held[GROUP_COL].nunique()}
        passes = True
        for t in TASKS:
            vc = held[t].dropna().value_counts()
            counts = [int(vc.get(c, 0)) for c in sorted(eligible[t])]
            worst = min(counts) if counts else 0
            rec[f"{t}_min_per_evaluable_class"] = worst
            rec[f"{t}_n_holdout_labelled"] = int(held[t].notna().sum())
            if worst < args.min_per_class:
                passes = False
        rec["passes"] = passes
        rows.append(rec)

    tbl = pd.DataFrame(rows)
    ok = tbl[tbl.passes]
    selected = int(ok.n_splits.max()) if len(ok) else None

    args.output.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.output / "holdout_rule.tsv", sep="\t", index=False)

    cols = ["n_splits", "pct_holdout", "n_holdout", "n_bioprojects"] + \
           [f"{t}_min_per_evaluable_class" for t in TASKS] + ["passes"]
    logger.info("\nRULE: smallest holdout (largest n_splits) such that every "
                "evaluable class keeps >= %d held-out runs\n", args.min_per_class)
    logger.info(tbl[cols].to_string(index=False))

    used = json.loads((SPLITS / "split_config.json").read_text()).get("n_holdout_splits")
    logger.info("\nrule selects : n_splits = %s", selected)
    logger.info("actually used: n_holdout_splits = %s", used)
    if selected is None:
        logger.info("VERDICT: no candidate satisfies the rule; lower --min-per-class "
                    "or accept that some evaluable classes are thinly tested.")
    elif selected == used:
        logger.info("VERDICT: the rule reproduces the split we already have. The "
                    "choice can be stated as a rule rather than as a judgement, at "
                    "no cost.")
    else:
        logger.info("VERDICT: the rule does NOT select the split in use. Following "
                    "it means regenerating the split, which means rebuilding the "
                    "matrix (~16 h) and redoing every downstream number. Decide "
                    "deliberately -- do not quietly keep %s while claiming the rule.",
                    used)
    json.dump({"min_per_class": args.min_per_class, "selected": selected,
               "in_use": used, "candidates": rows},
              open(args.output / "holdout_rule.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
