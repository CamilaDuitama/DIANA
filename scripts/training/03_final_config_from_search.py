#!/usr/bin/env python3
"""Turn the all-train search result into a final training config.

`01_train_multitask_single_fold.py --run-config ... (search_only)` writes
`search_all_train_best_params.json`. The only tool that builds a
`final_training_config.json` is `aggregate_cv_results.py`, which globs
`multitask_fold_*_results_*.json` and aggregates *across outer folds* -- a shape the
single all-train search does not have, and an aggregation it does not need. Nothing
connected the two.

It reuses `create_final_training_config` so the final config is assembled by exactly
one function, and so the keys that were being dropped there -- `logit_adjust_tau`
and per-head label smoothing -- stay carried.

    ./env/bin/python scripts/training/03_final_config_from_search.py \\
        --search-dir results/search_v9_final \\
        --run-config configs/train_config_v9_final_search.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from importlib import import_module

_agg = import_module("aggregate_cv_results")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search-dir", type=Path, required=True,
                    help="output_dir of the search_only run")
    ap.add_argument("--run-config", type=Path, required=True,
                    help="the run config the search used, for paths and task types")
    ap.add_argument("--output", type=Path, default=None,
                    help="default: <search-dir>/final_training_config.json")
    args = ap.parse_args()

    hits = list(args.search_dir.rglob("search_all_train_best_params.json"))
    if not hits:
        raise FileNotFoundError(
            f"no search_all_train_best_params.json under {args.search_dir}. "
            "Did the search_only run finish?")
    payload = json.loads(hits[0].read_text())
    best_params = payload["best_params"]
    print(f"search result: {hits[0]}")
    print(f"  trained on {payload.get('n_train')} runs, "
          f"{payload.get('n_inner_folds')} inner folds")

    run_cfg = json.loads(args.run_config.read_text())

    # Sanity: the searched tau must be present, or the final fit silently falls back
    # to the config's starting value -- the exact bug this pipeline already had.
    if "logit_adjust_tau" not in best_params:
        raise ValueError(
            "best_params has no logit_adjust_tau. The search was configured with "
            "search_tau, so this means the value was not recorded and the final fit "
            "would use the starting tau instead.")

    cfg = _agg.create_final_training_config(
        cv_dir=args.search_dir / "cv_results",
        best_params=best_params,
        features_path=run_cfg["features_path"],
        metadata_path=run_cfg["metadata_path"],
        train_ids_path=run_cfg["train_ids_path"],
        task_types=run_cfg.get("task_types"),
        extra={"class_imbalance": run_cfg.get("class_imbalance", {}) or {}},
    )
    cfg["output_dir"] = str(args.search_dir / "final_model")
    cfg["_provenance"] = {
        "source": str(hits[0]),
        "selection": "single Optuna search over all of train, dev folds as inner CV",
        "why": "per-outer-fold hyperparameters would be selected on fold difficulty; "
               "fold 4 scored 0.453 against 0.154-0.238 because it had 98.8 % "
               "in-vocabulary coverage and a 75.3 % majority class",
    }

    out = args.output or (args.search_dir / "final_training_config.json")
    out.write_text(json.dumps(cfg, indent=4) + "\n")

    ls = cfg.get("label_smoothing_per_task") or {}
    print(f"\nwrote {out}")
    print(f"  hidden_dims : {cfg['hyperparameters']['model_params']['hidden_dims']}")
    print(f"  batch_norm  : {cfg['hyperparameters']['model_params']['use_batch_norm']}")
    print(f"  lr          : {cfg['hyperparameters']['trainer_params']['learning_rate']:.2e}")
    print(f"  tau         : {cfg['class_imbalance']['logit_adjust_tau']:.4f}")
    print(f"  label smooth: { {k: round(v, 3) for k, v in ls.items()} }")
    print(f"  tasks       : {cfg['task_names']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
