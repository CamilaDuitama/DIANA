#!/usr/bin/env python3
"""X6: pick, per single-task arm, the learning rate and schedule with the best pooled
dev-fold criterion at its own budget, and assemble the selected arms into one directory.

Input: <base>/<cfg>/ for every grid point, each already run through
37_epoch_budget_select.py --arms <task> (so it holds epoch_budget.tsv and oof_<task>_<task>.tsv).
<cfg> is lr<rate>_<schedule>, for example lr1e-03_cosine. Output, in --out:
oof_<task>_<task>.tsv and epoch_budget.tsv for the selected point of each arm, plus
x6_grid.tsv with every point's budget and criterion. Selection uses the smoothed pooled
criterion only (the same objective as the hyperparameter search); the paired test against
the current arm is run afterwards by 48_ and is reported, with the note that a
selected-then-tested comparison on the same folds is optimistic.

    ./env/bin/python scripts/analysis/55_x6_select.py --base results/x6_grid_v9 --out results/x6_selected_v9
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

TASKS = ["community_type", "feature", "sample_host", "material"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    grid, chosen = [], []
    for cfg in sorted(p for p in a.base.iterdir() if p.is_dir() and (p / "epoch_budget.tsv").exists()):
        b = pd.read_csv(cfg / "epoch_budget.tsv", sep="\t")
        for _, r in b.iterrows():
            grid.append({"arm": r.arm, "cfg": cfg.name, "budget": int(r.budget), "criterion_smoothed": float(r.criterion_smoothed),
                         "criterion_raw": float(r.criterion_raw), "peaked": bool(r.peaked), "capped": bool(r.get("capped", False))})
    g = pd.DataFrame(grid)
    g.to_csv(a.out / "x6_grid.tsv", sep="\t", index=False)
    for arm in TASKS:
        sub = g[(g.arm == arm) & g.peaked]
        if sub.empty:
            raise SystemExit(f"{arm}: no grid point peaked below the cap")
        best = sub.sort_values("criterion_smoothed", ascending=False).iloc[0]
        shutil.copy2(a.base / best.cfg / f"oof_{arm}_{arm}.tsv", a.out / f"oof_{arm}_{arm}.tsv")
        chosen.append({"arm": arm, "budget": int(best.budget), "peaked": True, "cfg": best.cfg,
                       "criterion_smoothed": float(best.criterion_smoothed)})
    pd.DataFrame(chosen).to_csv(a.out / "epoch_budget.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(g.pivot_table(index="cfg", columns="arm", values="criterion_smoothed").round(4).to_string())
    print("\nselected:")
    print(pd.DataFrame(chosen).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
