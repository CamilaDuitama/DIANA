#!/usr/bin/env python3
"""Does the shared trunk earn its place? Paired dev-fold test. (R1.3, R3.12)

The question
------------
One multi-task net with four heads, or four independent single-task nets? The
manuscript assumes the former; **R3.12** asks why, and **R1.3** asks us to start
simple and add complexity only when a limitation is demonstrated.

Why the earlier answer did not hold
-----------------------------------
The 25-fold search suggested multi-task won on all four heads, but that compared
*means across folds*. Per fold it wins only **12 of 20** comparisons (sign test
p = 0.50), and the means were carried by fold 4 -- the easiest fold, at 98.8 %
in-vocabulary coverage and a 75.3 % majority class. Worse, each outer fold ran its
own Optuna search, so its multi-task and single-task models differed in
hyperparameters as well as architecture. Architecture was never the only variable.

What this does instead
----------------------
Hyperparameters are **fixed** to those chosen by the all-train searches, and both
arms are refitted across the 5 dev folds with several seeds. Architecture becomes
the only difference; seeds separate the architecture effect from run-to-run noise.
For each (fold, seed) the two arms see identical data, so the differences are
**paired** and fold difficulty cancels -- the same argument as Table 3.

Pre-committed rule, fixed before any number was seen: adopt multi-task **only** if
the 95 % CI on the paired mean Δ excludes 0. Otherwise adopt four single-task nets,
because adopting a trunk we cannot show helps is exactly what R3.12 criticises.
Equivalent accuracy plus deployment convenience is a valid reason to ship one model,
but it is a deployment argument and must be reported as one, not as evidence.

Held-out is never touched: this is a dev-fold decision.

    ./env/bin/python scripts/analysis/14_architecture_paired_test.py \\
        --multitask-dir results/arch_test/multitask \\
        --single-dir    results/arch_test/single
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
METRIC = "f1_macro"          # what the fold trainer writes; eligible-only needs labels


def collect(root: Path, task: str) -> dict[tuple[int, int], float]:
    """(fold, seed) -> score for one task, from the fold trainer's result JSONs."""
    out = {}
    for f in glob.glob(str(root / "**" / "multitask_fold_*_results_*.json"), recursive=True):
        j = json.load(open(f))
        m = j.get("test_metrics", j)
        if task not in m or not isinstance(m[task], dict):
            continue
        v = m[task].get(METRIC)
        if v is None:
            continue
        # seed is encoded in the directory name by 01_train_..._single_fold.py
        parent = Path(f).parent.name
        seed = int(parent.split("seed")[1]) if "seed" in parent else 0
        out[(int(j["fold_id"]), seed)] = float(v)
    return out


def paired_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple:
    """Bootstrap CI on the mean paired difference."""
    rng = np.random.default_rng(seed)
    if len(diffs) < 2:
        return float("nan"), float("nan")
    means = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multitask-dir", type=Path, required=True)
    ap.add_argument("--single-dir", type=Path, required=True,
                    help="parent holding one subdirectory per task")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/arch_test")
    args = ap.parse_args()

    rows, verdicts = [], {}
    for task in TASKS:
        mt = collect(args.multitask_dir, task)
        st = collect(args.single_dir / task, task)
        shared = sorted(set(mt) & set(st))
        if not shared:
            print(f"  {task}: no paired (fold, seed) results yet — skipping")
            continue
        d = np.array([mt[k] - st[k] for k in shared])
        lo, hi = paired_ci(d)
        wins = int((d > 0).sum())
        verdicts[task] = {
            "n_pairs": len(d), "mean_delta": float(d.mean()),
            "ci_low": lo, "ci_high": hi, "wins": wins,
            "multitask_favoured": bool(lo > 0),
        }
        for (f, sd), dv in zip(shared, d):
            rows.append({"task": task, "fold": f, "seed": sd,
                         "multitask": mt[(f, sd)], "single": st[(f, sd)], "delta": dv})

    if not rows:
        raise SystemExit("no paired results found — run the fits first")

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "paired_scores.tsv", sep="\t", index=False)

    lines = ["Architecture: multi-task vs single-task, paired over dev folds and seeds",
             "", "Hyperparameters fixed per arm, so architecture is the only difference.",
             "Delta = multi-task - single-task on the SAME (fold, seed).", "",
             f"{'task':<16}{'pairs':>7}{'mean Δ':>9}{'95 % CI on Δ':>22}{'wins':>8}  verdict"]
    for task, v in verdicts.items():
        ci = f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
        verdict = "multi-task" if v["multitask_favoured"] else "not demonstrated"
        lines.append(f"{task:<16}{v['n_pairs']:>7}{v['mean_delta']:>+9.4f}{ci:>22}"
                     f"{v['wins']}/{v['n_pairs']:<5}  {verdict}")

    n_yes = sum(v["multitask_favoured"] for v in verdicts.values())
    lines += ["", f"Multi-task demonstrated on {n_yes} of {len(verdicts)} tasks.", ""]
    if n_yes == 0:
        lines += ["PRE-COMMITTED RULE -> adopt FOUR SINGLE-TASK NETS.",
                  "No task shows a trunk benefit whose CI excludes 0. This is the",
                  "simple-first progression R1.3 asks for, and the honest answer to",
                  "R3.12 is that the shared trunk is not needed.",
                  "If the multi-task net is shipped anyway for deployment reasons -- one",
                  "model, one forward pass, one calibration -- say exactly that, and do",
                  "not present it as an accuracy result."]
    elif n_yes == len(verdicts):
        lines += ["PRE-COMMITTED RULE -> adopt the MULTI-TASK net.",
                  "Every head benefits from the trunk with the CI excluding 0, which",
                  "answers R3.12 directly: the trunk learns structure independent heads",
                  "cannot."]
    else:
        lines += ["PRE-COMMITTED RULE -> MIXED, so adopt multi-task only if the tasks it",
                  "helps are the ones the paper leads with; otherwise take single-task.",
                  "Report per-task rather than claiming a global architecture win, and",
                  "state which heads gain and which do not."]

    report = "\n".join(lines)
    print("\n" + report)
    (args.output / "summary.txt").write_text(report + "\n")
    json.dump(verdicts, open(args.output / "verdicts.json", "w"), indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
