#!/usr/bin/env python3
"""Is DIANA better than the baselines? Paired bootstrap over BioProjects. (A4)

Why a paired test and not two intervals
---------------------------------------
Each model's own held-out interval is roughly +/- 0.18, because the 34 held-out
BioProjects differ enormously in difficulty. Two such intervals overlap almost
entirely and say nothing about which model is better. But both models predict the
*same* runs, so the difference can be bootstrapped directly: resample the projects,
score both models on that same resample, and take
delta = F1(DIANA) - F1(baseline). A hard project drags both models down together,
so the shared difficulty cancels and delta is far more stable than either term.

**Superiority is demonstrated only if the 95 % CI on delta excludes 0.**

Rows scored
-----------
Only rows both models scored, which is the baselines' in-vocabulary subset: a run
whose class has no training support is excluded rather than counted wrong (R3.5).
The eligible-class denominator is then intersected with the classes actually present
in those rows, exactly as `classification_metrics` does elsewhere.

    ./env/bin/python scripts/analysis/18_paired_superiority.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
GROUP = "archive_project"


def f1_elig(y_true, y_pred, eligible) -> float:
    seen = sorted(set(y_true))
    e = sorted(c for c in seen if c in eligible)
    return f1_score(y_true, y_pred, labels=e, average="macro", zero_division=0) if e else np.nan


def paired(y_true, a_pred, b_pred, groups, eligible, n_boot, seed):
    """delta = F1(a) - F1(b), resampling whole BioProjects."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    obs = f1_elig(y_true, a_pred, eligible) - f1_elig(y_true, b_pred, eligible)
    deltas = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in drawn])
        deltas.append(f1_elig(y_true[idx], a_pred[idx], eligible)
                      - f1_elig(y_true[idx], b_pred[idx], eligible))
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    # how many individual projects favour a, as a descriptive companion
    wins = sum(1 for g in uniq
               if (lambda i: f1_elig(y_true[i], a_pred[i], eligible)
                   > f1_elig(y_true[i], b_pred[i], eligible))(idx_by[g]))
    return {"delta": float(obs),
            "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            "frac_boot_positive": float((d > 0).mean()),
            "projects_favouring_a": int(wins), "n_projects": int(len(uniq))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", type=Path,
                    default=PROJECT_ROOT / "results/baseline_predictions_v9/heldout_predictions.tsv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/paired_superiority_v9")
    args = ap.parse_args()

    base = pd.read_csv(args.baselines, sep="\t")
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    diana = {
        "DIANA multi-task": pd.read_csv(
            PROJECT_ROOT / "results/final_eval_v9/heldout_multitask/test_predictions.tsv", sep="\t"),
    }
    for t in TASKS:
        diana[f"DIANA single-task:{t}"] = pd.read_csv(
            PROJECT_ROOT / f"results/final_eval_v9/heldout_single_{t}/test_predictions.tsv", sep="\t")

    rows = []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        bt = base[base.task == task]
        # the common row set: what the baselines scored (in-vocabulary only)
        ref = bt[bt.model == bt.model.iloc[0]][["Run_accession", "y_true", GROUP]]
        order = ref.Run_accession.to_numpy()
        y_true = ref.y_true.astype(str).to_numpy()
        groups = ref[GROUP].astype(str).to_numpy()

        preds = {}
        for m, g in bt.groupby("model"):
            s = g.set_index("Run_accession").y_pred.astype(str)
            preds[m] = s.reindex(order).to_numpy()
        for label, df in diana.items():
            if label.startswith("DIANA single-task:") and not label.endswith(task):
                continue
            name = "DIANA single-task" if label.startswith("DIANA single-task") else label
            s = df.set_index("Run_accession")[f"{task}_pred"].astype(str)
            preds[name] = s.reindex(order).to_numpy()

        missing = {m: int(pd.isna(v).sum()) for m, v in preds.items() if pd.isna(v).any()}
        if missing:
            raise SystemExit(f"{task}: predictions missing for some scored rows: {missing}")

        for d_name in ("DIANA multi-task", "DIANA single-task"):
            if d_name not in preds:
                continue
            for b_name, b_pred in preds.items():
                if b_name.startswith("DIANA"):
                    continue
                r = paired(y_true, preds[d_name], b_pred, groups, eligible,
                           args.n_boot, args.seed)
                rows.append({"task": task, "model_a": d_name, "model_b": b_name,
                             "f1_a": f1_elig(y_true, preds[d_name], eligible),
                             "f1_b": f1_elig(y_true, b_pred, eligible),
                             "n_scored": len(y_true), **r})
        # does task sharing help? multi-task against its own single-task twin
        if "DIANA single-task" in preds:
            r = paired(y_true, preds["DIANA multi-task"], preds["DIANA single-task"],
                       groups, eligible, args.n_boot, args.seed)
            rows.append({"task": task, "model_a": "DIANA multi-task",
                         "model_b": "DIANA single-task",
                         "f1_a": f1_elig(y_true, preds["DIANA multi-task"], eligible),
                         "f1_b": f1_elig(y_true, preds["DIANA single-task"], eligible),
                         "n_scored": len(y_true), **r})

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "paired_deltas.tsv", sep="\t", index=False)

    L = [f"Paired superiority over {df.n_projects.iloc[0]} held-out BioProjects "
         f"({args.n_boot} resamples)", "",
         "delta = F1_eligible(A) - F1_eligible(B) on the same rows and the same",
         "resampled projects. Superiority is demonstrated only where the 95 % CI on",
         "delta excludes 0; 'sig' marks those.", ""]
    for task, g in df.groupby("task", sort=False):
        L.append(f"{task}   (n={g.n_scored.iloc[0]} scored)")
        L.append(f"   {'A':<20}{'B':<24}{'F1 A':>7}{'F1 B':>7}{'delta':>8}"
                 f"{'95 % CI on delta':>22}{'P(d>0)':>8}{'proj':>7}  sig")
        for _, r in g.sort_values("delta", ascending=False).iterrows():
            sig = "YES" if r.ci_low > 0 else ("(B better)" if r.ci_high < 0 else "no")
            ci = f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
            L.append(f"   {r.model_a:<20}{r.model_b:<24}{r.f1_a:>7.3f}{r.f1_b:>7.3f}"
                     f"{r.delta:>+8.3f}{ci:>22}{r.frac_boot_positive:>8.2f}"
                     f"{f'{r.projects_favouring_a}/{r.n_projects}':>7}  {sig}")
        L.append("")
    n_sig = int((df.ci_low > 0).sum())
    n_neg = int((df.ci_high < 0).sum())
    L += [f"{n_sig} of {len(df)} comparisons favour DIANA with the CI excluding 0; "
          f"{n_neg} favour the other model.", "",
          "Where the CI spans 0 the two models are not distinguished by this test. That",
          "is not evidence they perform the same, only that 34 projects cannot separate",
          "them."]
    report = "\n".join(L)
    print(report)
    (args.output / "summary.txt").write_text(report + "\n")
    json.dump(rows, open(args.output / "paired_deltas.json", "w"), indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
