#!/usr/bin/env python3
"""S7f: per-fold configs for the dev-fold screen, from each arm's own searched settings.

One config per task and fold, built from the `final_training_config.json` that
`03_final_config_from_search.py` produced out of that arm's Optuna search. Each fold points
at its own matrix, whose appended block was standardised on the other four folds, and trains
on that fold's `cal_train` ids so the held-out fold is never seen.

The comparison this feeds is against `results/calibration_v9`, the fraction arm refitted on
exactly these folds and ids. Both sides therefore differ only in the 3,072 appended columns
and in each arm having been tuned for its own input, which is the point: S6's null was
uninformative precisely because only one side had been tuned.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scrambled", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arm = "offlistshuf" if a.scrambled else "offlist"
    tag = "shuf" if a.scrambled else ""

    out_root = f"results/{arm}_test_v9"
    cfg_dir = ROOT / out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for task in TASKS:
        src = ROOT / f"results/{arm}_search_v9/search_{task}/final_training_config.json"
        base = json.loads(src.read_text())
        for fold in range(5):
            cfg = json.loads(json.dumps(base))
            matrix = f"data/matrices/matrix_v9_train/unitigs.frac.off{tag}.fold{fold}.mat"
            if not (ROOT / matrix).exists():
                raise SystemExit(f"missing {matrix}")
            cfg["features_path"] = matrix
            cfg["train_ids_path"] = f"results/calibration_v9/ids/cal_train_fold{fold}.txt"
            cfg["output_dir"] = f"{out_root}/fit_{task}_fold{fold}"
            cfg["_comment"] = (
                f"S7f dev-fold screen, {task} fold {fold}, {'scrambled' if a.scrambled else 'real'} "
                f"arm. 110,202 fractions plus 3,072 columns summarising the sequence the k-mer "
                f"filter deleted. Hyperparameters from this arm's OWN Optuna search (100 trials, "
                f"eligible_pooled objective), which is what S6 lacked. The appended block is "
                f"standardised on this fold's four training folds only. Compare against "
                f"results/calibration_v9, the fraction arm on identical folds and ids. Held-out "
                f"untouched.")
            (cfg_dir / f"off{tag}_{task}_fold{fold}.json").write_text(json.dumps(cfg, indent=2))
            n += 1
    logger.info("wrote %d configs to %s", n, cfg_dir)


if __name__ == "__main__":
    main()
