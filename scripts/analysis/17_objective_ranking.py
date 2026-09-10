#!/usr/bin/env python3
"""Do the candidate search objectives rank trials differently?

The v9 searches maximise `(balanced_accuracy + f1_macro) / 2` over all classes with
out-of-vocabulary rows scored wrong. The baselines are selected on
`f1_macro_eligible` with those rows dropped (`12_tune_baselines_v9.py`), so the two
arms of the R1.3 comparison are tuned toward different targets. Switching the search
costs a full rerun, because Optuna's TPE chooses each trial from the previous
trials' objective values: selection can be re-ranked from a record, exploration
cannot.

So the rerun is only worth it if the objectives actually disagree about which trials
are good. This reads the per-trial record, which holds every candidate computed from
the same trained models, and reports Spearman rank correlation plus whether the
top-ranked trial is the same one.

    ./env/bin/python scripts/analysis/17_objective_ranking.py \\
        --trials results/objective_diagnostic_v9/cv_results/optuna_trials.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/objective_diagnostic_v9")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.trials.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("candidates")]
    if len(rows) < 4:
        raise SystemExit(f"only {len(rows)} trials recorded; need at least 4")

    keys = list(rows[0]["candidates"])
    vals = {k: np.array([r["candidates"][k] for r in rows], dtype=float) for k in keys}
    ok = {k: v for k, v in vals.items() if np.isfinite(v).all()}

    lines = [f"Candidate search objectives over {len(rows)} trials", "",
             "Every candidate is computed from the SAME trained models, so any",
             "disagreement is the metric's, not training noise.", "",
             f"{'candidate':<26}{'mean':>8}{'min':>8}{'max':>8}{'best trial':>12}"]
    for k, v in ok.items():
        lines.append(f"{k:<26}{v.mean():>8.4f}{v.min():>8.4f}{v.max():>8.4f}"
                     f"{rows[int(np.argmax(v))]['number']:>12}")

    lines += ["", "Spearman rank correlation between candidates:", "",
              f"{'pair':<52}{'rho':>8}{'p':>10}  same best?"]
    verdict = []
    for a, b in itertools.combinations(ok, 2):
        rho, p = spearmanr(ok[a], ok[b])
        same = int(np.argmax(ok[a])) == int(np.argmax(ok[b]))
        lines.append(f"{a + ' vs ' + b:<52}{rho:>8.3f}{p:>10.4f}  {'yes' if same else 'NO'}")
        verdict.append({"a": a, "b": b, "rho": float(rho), "p": float(p),
                        "same_best_trial": bool(same)})

    key_pair = [v for v in verdict
                if {v["a"], v["b"]} == {"legacy_foldmean", "eligible_pooled"}]
    lines += ["", "The decision:"]
    if key_pair:
        rho = key_pair[0]["rho"]
        lines.append(f"  legacy_foldmean vs eligible_pooled: rho = {rho:+.3f}, "
                     f"same best trial = {key_pair[0]['same_best_trial']}")
        lines += ["", "  rho near 1 and the same best trial means the running searches",
                  "  would land in much the same place under the pooled objective, so a",
                  "  rerun buys little. A low rho means the objective is choosing the",
                  "  hyperparameters and the rerun is necessary.",
                  "",
                  "  Read with the sample size in mind: this is a handful of trials at",
                  "  reduced epochs, enough to separate 'essentially the same ordering'",
                  "  from 'unrelated' and not enough for anything finer."]
    report = "\n".join(lines)
    print(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "objective_ranking.txt").write_text(report + "\n")
    json.dump({"n_trials": len(rows), "pairs": verdict},
              open(args.output / "objective_ranking.json", "w"), indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
