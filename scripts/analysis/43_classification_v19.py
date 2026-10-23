#!/usr/bin/env python3
"""v19 dev-fold classification: macro-F1 with BioProject intervals and paired differences.

Every model is scored on the same runs: the labelled training runs whose true class has
fit rows in their fold (C2a), pooled out-of-fold, macro-F1 over the eligible classes
present (C2b), 2,000 BioProject resamples (C2c). Paired a - b on the identical runs and
shared draws; verdict a / b / tie.

    ./env/bin/python scripts/analysis/43_classification_v19.py \\
        --model Separate=results/devfold_probs_v19/probs_Separate_{task}.tsv \\
        --pred LinearSVM=results/baselines_v19/{task}/oof_pred_LinearSVM_Bal.tsv \\
        --pair Separate:Shared --out results/classification_v19/dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]


def predictions(spec: str, task: str, kind: str) -> pd.Series:
    d = pd.read_csv(spec.format(task=task), sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
    if kind == "pred":
        return d["y_pred"].astype(str)
    pc = [c for c in d.columns if c.startswith("p_")]
    return d[pc].idxmax(axis=1).str[2:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split-dir", type=Path, default=ROOT / "data/splits_v19")
    ap.add_argument("--model", action="append", default=[], help="name=probability table template with {task}")
    ap.add_argument("--pred", action="append", default=[], help="name=prediction table template with {task}")
    ap.add_argument("--pair", action="append", default=[], help="a:b, a minus b")
    ap.add_argument("--heldout", action="store_true",
                    help="score the held-out set: a run counts if its class occurs among the training labels")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(a.split_dir / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(a.split_dir / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    ev = (pd.read_csv(a.split_dir / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
          if a.heldout else tr)
    specs = [(m.split("=", 1), "probs") for m in a.model] + [(m.split("=", 1), "pred") for m in a.pred]
    rows, pairs, regime_rows = [], [], []
    for task in TASKS:
        eligible = S.eligible_classes(a.split_dir / "class_eligibility.tsv", task)
        scored = (S.scored_heldout(ev[task], set(tr[task].dropna())) if a.heldout
                  else S.scored_devfold(tr[task], folds))
        runs = scored[scored].index
        y = ev.loc[runs, task].astype(str).to_numpy()
        g = ev.loc[runs, "archive_project"].astype(str).to_numpy()
        by_regime = S.regime_classes(tr[task], eligible)
        preds = {}
        for (name, spec), kind in specs:
            p = predictions(spec, task, kind)
            missing = runs.difference(p.index)
            if len(missing):
                raise ValueError(f"{name}/{task}: {len(missing)} scored runs have no prediction")
            preds[name] = p.loc[runs].to_numpy()
            rows.append({"task": task, "model": name, "n_not_scored": int(ev[task].notna().sum() - len(runs)),
                         **S.score_with_interval(y, preds[name], g, eligible, f"cls|{task}")})
            for reg, cls in by_regime.items():
                cls = cls & set(y)
                if cls:
                    regime_rows.append({"task": task, "regime": reg, "model": name, "classes": ";".join(sorted(cls)),
                                        **S.score_with_interval(y, preds[name], g, cls, f"cls|{task}|{reg}")})
        for spec in a.pair:
            ma, mb = spec.split(":")
            r = S.paired_difference(runs, runs, y, preds[ma], preds[mb], g, eligible, f"pair|{task}")
            r["verdict"] = {"a": ma, "b": mb}.get(r["verdict"], "tie")
            pairs.append({"task": task, "a": ma, "b": mb, **r})
    pd.DataFrame(rows).to_csv(a.out / "classification.tsv", sep="\t", index=False)
    pd.DataFrame(pairs).to_csv(a.out / "classification_paired.tsv", sep="\t", index=False)
    pd.DataFrame(regime_rows).to_csv(a.out / "classification_by_regime.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    if pairs:
        print(pd.DataFrame(pairs).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
