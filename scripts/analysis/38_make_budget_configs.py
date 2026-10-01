#!/usr/bin/env python3
"""Write the final-fit configs for the epoch-budget arms (X1).

Each config is the arm's `configs/final_fixed_v9/<arm>.json` (the searched
hyperparameters, features, ids and seed) plus `epoch_budget`, the number of epochs
`37_epoch_budget_select.py` assigned to the arm on the grouped dev folds. Nothing
else changes. The script refuses to write anything if any arm has not peaked, so a
budget sitting at the cap can never reach a final fit.

    ./env/bin/python scripts/analysis/38_make_budget_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUDGET = ROOT / "results/epoch_budget_v9/epoch_budget.tsv"
SRC = ROOT / "configs/final_fixed_v9"
DST = ROOT / "configs/final_budget_v9"
ARMS = ["multitask", "community_type", "feature", "sample_host", "material"]


def main() -> int:
    global BUDGET, SRC, DST
    import argparse
    ap = argparse.ArgumentParser(description="write the epoch-budget configs from the dev-fold budgets")
    ap.add_argument("--budget", type=Path, default=BUDGET, help="epoch_budget.tsv from 37_")
    ap.add_argument("--src", type=Path, default=SRC, help="fixed configs to copy (default configs/final_fixed_v9)")
    ap.add_argument("--dst", type=Path, default=DST, help="where the budget configs go (default configs/final_budget_v9)")
    ap.add_argument("--results", default="results/final_budget_v9", help="output_dir prefix written into the configs")
    args = ap.parse_args()
    # relative paths are taken from the project root (the defaults are absolute)
    BUDGET, SRC, DST = ((q if q.is_absolute() else ROOT / q) for q in (args.budget, args.src, args.dst))
    bud = pd.read_csv(BUDGET, sep="\t").set_index("arm")
    missing = [a for a in ARMS if a not in bud.index]
    if missing:
        raise SystemExit(f"no budget for {missing}; run 37_epoch_budget_select.py")
    if not bud.loc[ARMS, "peaked"].all():
        raise SystemExit("an arm did not peak within the cap: "
                         f"{bud.index[~bud.peaked].tolist()}; raise the cap and re-run")
    DST.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        cfg = json.load(open(SRC / f"{arm}.json"))
        budget = int(bud.loc[arm, "budget"])
        cfg["epoch_budget"] = budget
        cfg["output_dir"] = f"{args.results}/{arm}/final_model"
        for key in ("validation_split", "early_stopping_patience", "early_stopping_monitor"):
            cfg.pop(key, None)
        cfg["_provenance"] = {
            **cfg.get("_provenance", {}),
            "budget_source": str(BUDGET.relative_to(ROOT)),
            "budget_selected_on": "5 BioProject-grouped dev folds, pooled f1_macro_eligible, "
                                  "centred 5-epoch smoothing (37_epoch_budget_select.py)",
            "refit_reason": "X1, 2026-09-25: no inner validation split; the epoch budget "
                            "replaces early stopping. Hyperparameters, features, train ids "
                            f"and seed unchanged from {SRC.relative_to(ROOT)}/.",
        }
        out = DST / f"{arm}.json"
        json.dump(cfg, open(out, "w"), indent=2)
        print(f"{arm:15s} epoch_budget={budget:4d}  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
