#!/usr/bin/env python3
"""Freeze each arm's searched hyperparameters into a config for the paired test.

The architecture question (multi-task vs four single-task nets) can only be answered
if architecture is the *only* difference between the arms. In the 25-fold search it
was not: every outer fold ran its own Optuna search, so each fold's two arms
differed in hyperparameters as well as structure.

This reads what the all-train searches chose for each arm and writes it back as a
`fixed_hyperparameters` block, so `run_arch_test_v9.sbatch` refits both arms over
the dev folds with no search and nothing else varying but the seed.

    ./env/bin/python scripts/training/prepare_arch_test_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]

# (search output dir, source run config, config to write, output dir for the fits)
ARMS = [("results/search_v9_final",
         "configs/train_config_v9_final_search.json",
         "configs/arch_test_multitask.json",
         "results/arch_test/multitask")]
ARMS += [(f"results/search_v9_final_single_{t}",
          f"configs/train_config_v9_final_search_single_{t}.json",
          f"configs/arch_test_single_{t}.json",
          f"results/arch_test/single/{t}") for t in TASKS]


def main() -> int:
    missing, written = [], []
    for search_dir, src_cfg, out_cfg, out_dir in ARMS:
        hits = list((PROJECT_ROOT / search_dir).rglob("search_all_train_best_params.json"))
        if not hits:
            missing.append(search_dir)
            continue
        best = json.loads(hits[0].read_text())["best_params"]
        cfg = json.loads((PROJECT_ROOT / src_cfg).read_text())

        cfg.pop("search_only", None)
        cfg["fixed_hyperparameters"] = best
        cfg["output_dir"] = out_dir
        # tau was searched; pin it to the selected value and stop searching it
        imb = dict(cfg.get("class_imbalance") or {})
        imb["search_tau"] = False
        if "logit_adjust_tau" in best:
            imb["logit_adjust_tau"] = float(best["logit_adjust_tau"])
        cfg["class_imbalance"] = imb
        cfg["_comment"] = ("Frozen hyperparameters for the architecture paired test. "
                           "Architecture is the only difference between arms; seeds "
                           "separate it from run-to-run noise. Held-out untouched.")
        (PROJECT_ROOT / out_cfg).write_text(json.dumps(cfg, indent=2) + "\n")
        written.append((out_cfg, cfg["task_names"], imb.get("logit_adjust_tau")))

    for c, tasks, tau in written:
        print(f"  wrote {c:<48} tasks={tasks} tau={tau}")
    if missing:
        print("\nSTILL SEARCHING (no config written):")
        for m in missing:
            print(f"  {m}")
        print("\nRe-run this once those searches finish.")
        return 1
    print(f"\nall {len(written)} arms frozen — submit run_arch_test_v9.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
