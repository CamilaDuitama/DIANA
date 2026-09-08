#!/usr/bin/env python
"""Learn which label combinations are possible, and flag predictions that are not.

Question answered
-----------------
An anomaly flag that needs no ground truth: if a sample's *predicted* labels form
a combination that never occurs in the training corpus -- oral tissue paired with
a lake sediment material -- something is wrong, either with the prediction or with
the metadata. This is the detector arm PROJECT.md step 18 relies on, and it costs
nothing at inference.

Why not scripts/calibration/04_cross_task_consistency.py
-------------------------------------------------------
That script hard-codes which materials are "environmental" and which are "host",
so it has to be maintained by hand and silently misses any vocabulary it was not
told about. This learns the lookup from training co-occurrence instead.

What the measurements say (v9, BioProject-disjoint)
---------------------------------------------------
A flag is only usable if genuine data almost never trips it. Measuring the rate at
which *true* test pairs are unseen in training:

    pair                          all classes   evaluable classes only
    community_type x sample_host       0.23 %                   0.00 %  (n=429)
    feature x material                 8.33 %                   0.00 %  (n=220)
    community_type x material         20.62 %                  19.08 %
    sample_host x material            21.09 %                  19.37 %

So only the first two are usable, and only when both sides are restricted to
classes occurring in >=2 BioProjects. The material pairings fail for a legitimate
reason rather than a modelling one: `oral + birch pitch` (79 test runs) and
`gut + latrine` (8) are real combinations that happen to be absent from the
training split. On a BioProject-disjoint partition, new studies bring new
combinations, so "unseen" does not imply "impossible" for loosely-coupled pairs.

An earlier measurement on the v7 split suggested community_type x material was
tight (0.6 % floor). That split leaked BioProjects between train and validation;
the number did not survive an honest partition.

Inputs : data/splits_v9/{train,test}_metadata.tsv, class_eligibility.tsv
Outputs: results/cooccurrence_flag/{lookup.json,report.txt}
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Only pairs whose false-positive floor is acceptable. See the module docstring.
USABLE_PAIRS = [("community_type", "sample_host"), ("feature", "material")]
ALL_PAIRS = USABLE_PAIRS + [("community_type", "material"), ("sample_host", "material")]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def learn_lookup(train: pd.DataFrame, pairs: list, eligible: dict) -> dict:
    """Observed co-occurrences, restricted to classes evaluable out-of-project."""
    lookup = {}
    for a, b in pairs:
        sub = train[[a, b]].dropna()
        sub = sub[sub[a].isin(eligible.get(a, set())) & sub[b].isin(eligible.get(b, set()))]
        lookup[f"{a}|{b}"] = sorted({f"{x}\t{y}" for x, y in sub.drop_duplicates().values})
    return lookup


def false_positive_floor(test: pd.DataFrame, lookup: dict, pairs: list,
                         eligible: dict) -> dict:
    """How often genuine data trips the flag. This bounds its usefulness."""
    out = {}
    for a, b in pairs:
        obs = set(lookup[f"{a}|{b}"])
        sub = test[[a, b]].dropna()
        sub = sub[sub[a].isin(eligible.get(a, set())) & sub[b].isin(eligible.get(b, set()))]
        if not len(sub):
            continue
        unseen = [f"{x} + {y}" for x, y in sub.values if f"{x}\t{y}" not in obs]
        out[f"{a}|{b}"] = {
            "n_evaluated": int(len(sub)),
            "n_flagged": len(unseen),
            "false_positive_rate": len(unseen) / len(sub),
            # 0 of n observed is not proof of 0; report the 95 % upper bound.
            "upper_95_if_zero": (3.0 / len(sub)) if not unseen else None,
            "examples": pd.Series(unseen).value_counts().head(5).to_dict() if unseen else {},
        }
    return out


def flag(predictions: pd.DataFrame, lookup: dict, pairs: list) -> pd.Series:
    """True where a predicted combination never occurs in training."""
    flagged = pd.Series(False, index=predictions.index)
    for a, b in pairs:
        key = f"{a}|{b}"
        if key not in lookup or a not in predictions or b not in predictions:
            continue
        obs = set(lookup[key])
        both = predictions[a].notna() & predictions[b].notna()
        combo = predictions[a].astype(str) + "\t" + predictions[b].astype(str)
        flagged |= both & ~combo.isin(obs)
    return flagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data/splits_v9")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/cooccurrence_flag")
    ap.add_argument("--all-pairs", action="store_true",
                    help="also report the two pairs that are too loose to use")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.splits / "train_metadata.tsv", sep="\t")
    test = pd.read_csv(args.splits / "test_metadata.tsv", sep="\t")
    el = pd.read_csv(args.splits / "class_eligibility.tsv", sep="\t")
    eligible = {t: set(g[g.evaluable]["class"]) for t, g in el.groupby("target")}

    pairs = ALL_PAIRS if args.all_pairs else USABLE_PAIRS
    lookup = learn_lookup(train, pairs, eligible)
    floor = false_positive_floor(test, lookup, pairs, eligible)

    lines = ["Co-occurrence flag — learned from v9 training labels", ""]
    for a, b in pairs:
        key = f"{a}|{b}"
        n_obs = len(lookup[key])
        na = len(eligible.get(a, set()))
        nb = len(eligible.get(b, set()))
        lines.append(f"{a} x {b}")
        lines.append(f"    observed combinations : {n_obs} of {na * nb} possible "
                     f"({100 * (1 - n_obs / (na * nb)):.1f}% of the space is empty)")
        if key in floor:
            f = floor[key]
            rate = f"{100 * f['false_positive_rate']:.2f}%"
            if f["upper_95_if_zero"] is not None:
                rate += f" (0 of {f['n_evaluated']}; 95% upper bound {100 * f['upper_95_if_zero']:.2f}%)"
            lines.append(f"    false-positive floor  : {rate}")
            if f["examples"]:
                lines.append(f"    tripped by            : {f['examples']}")
        lines.append("")
    report = "\n".join(lines)
    print(report)

    json.dump({"lookup": lookup, "false_positive_floor": floor,
               "usable_pairs": [list(p) for p in USABLE_PAIRS]},
              open(args.output / "lookup.json", "w"), indent=2)
    (args.output / "report.txt").write_text(report)
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
