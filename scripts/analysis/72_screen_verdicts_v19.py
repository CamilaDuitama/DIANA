#!/usr/bin/env python3
"""Verdicts of the v19 dev-fold screens under the rule of PROTOCOLS.md (v19 amendment of 2026-10-21).

Per arm and field, from results/screens_v19/scores/<arm>/ (run_screen_score_v19.sbatch): the paired
choosing-score difference against DIANA (41_) and the paired macro-F1 difference (43_). An arm wins
a field only if the choosing-score interval is above zero and macro-F1 does not clearly drop
(interval entirely below zero and the drop at least the seed noise floor of CLAUDE.md). The same
two comparisons are listed for logistic regression on each new input against logistic regression
on the fractions. Writes results/screens_v19/verdicts.tsv and prints it.

    ./env/bin/python scripts/analysis/72_screen_verdicts_v19.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results/screens_v19/scores"
FLOOR = {"community_type": 0.080, "feature": 0.042, "sample_host": 0.028, "material": 0.017}


def verdict(row) -> str:
    if row.flag_verdict == "can't check" or pd.isna(row.flag_low):
        return "can't check"
    clear_drop = row.f1_high < 0 and -row.f1_delta >= FLOOR[row.task]
    if row.flag_low > 0 and not clear_drop:
        return "win"
    if row.flag_high < 0 or clear_drop:
        return "loss"
    return "tie"


def main() -> None:
    global SCORES
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=None, help="scores folder (default results/screens_v19/scores)")
    ap.add_argument("--out", type=Path, default=None, help="verdict table (default <scores>/../verdicts.tsv)")
    a = ap.parse_args()
    SCORES = a.scores or SCORES
    rows = []
    for d in sorted(p for p in SCORES.iterdir() if p.is_dir()):
        c = pd.read_csv(d / "classification_paired.tsv", sep="\t")
        f = pd.read_csv(d / "flag_single/flagging_paired.tsv", sep="\t")
        m = c.merge(f, on=["task", "a", "b"], suffixes=("_f1", "_flag"), validate="one_to_one")
        if len(m) != len(c) or len(m) != len(f):
            raise ValueError(f"{d.name}: classification and detection pairs differ")
        for r in m.itertuples(index=False):
            rows.append({"arm": d.name, "task": r.task, "a": r.a, "b": r.b,
                         "flag_delta": r.delta_choosing, "flag_low": r.ci_low_flag, "flag_high": r.ci_high_flag,
                         "flag_verdict": r.verdict_flag, "f1_delta": r.delta, "f1_low": r.ci_low_f1,
                         "f1_high": r.ci_high_f1, "f1_verdict": r.verdict_f1})
    v = pd.DataFrame(rows)
    v["rule"] = v.apply(verdict, axis=1)
    v.to_csv(a.out or SCORES.parent / "verdicts.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(v.round(3).to_string(index=False))
    print("\nwins under the rule:")
    print(v[v.rule == "win"][["arm", "task", "a", "b", "flag_delta", "flag_low", "f1_delta"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
