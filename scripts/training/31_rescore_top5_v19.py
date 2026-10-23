#!/usr/bin/env python3
"""v19 C4: re-score one of an arm's five best search trials with five seeds.

The trial's configuration is refitted on every dev fold with seeds 42, 1, 2, 3, 4 at
the search's cap; the five seeds' probabilities are averaged per epoch before the
argmax (the deployed five-seed ensemble), and the pooled criterion curve gives the
re-scored value and epoch count under the same rule as the search (30_search_v19.py).

    ./env/bin/python scripts/training/31_rescore_top5_v19.py --arm feature --rank 0 \\
        --search-dir results/search_v19/feature --split-dir data/splits_v19 \\
        --matrix data/matrices/matrix_v17_train/unitigs.frac.mat
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 1, 2, 3, 4]


def search_module():
    spec = importlib.util.spec_from_file_location("search_v19", ROOT / "scripts/training/30_search_v19.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["search_v19"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rank", type=int, required=True, choices=range(5))
    ap.add_argument("--search-dir", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=None, help="override the search cap (the doubled-cap rerun)")
    a = ap.parse_args()
    sm = search_module()
    if a.arm not in sm.ARMS:
        raise SystemExit(f"unknown arm {a.arm}")
    top = json.load(open(a.search_dir / "top5.json"))
    entry = next(e for e in top if e["rank"] == a.rank)
    cap = a.cap or entry["cap"]
    out = a.search_dir / "rescored" / f"rank{a.rank}_cap{cap}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(exist_ok=True)
    _, data, eligible, scored = sm.prepare(a.arm, a.split_dir, a.matrix)
    res = sm.evaluate_config(entry["config"], data, cap, SEEDS, eligible, scored)
    sm.write_curve(out.with_suffix(".tsv"), res)
    json.dump({"arm": a.arm, "rank": a.rank, "trial": entry["trial"], "search_score": entry["score"],
               "search_epochs": entry["epochs"], "cap": cap, "seeds": SEEDS,
               "selection": res["selection"], "f1_at_epoch": res["f1_at_epoch"], "config": entry["config"]},
              open(out, "w"), indent=1)
    sm.log.info("%s rank %d (trial %d): five-seed score %.4f at %d epochs (search %.4f at %d)", a.arm, a.rank,
                entry["trial"], res["selection"]["criterion_smoothed"], res["selection"]["epochs"],
                entry["score"], entry["epochs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
