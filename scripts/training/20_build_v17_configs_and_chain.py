#!/usr/bin/env python3
"""v17 driver: build the fixed configs from the finished searches and submit the
budget stage. Runs as a SLURM job dependent on the four Optuna searches, so the
chain continues without manual handoff (PROTOCOLS.md v17).

Builds configs/final_fixed_v17/<task>.json in the final_fixed_v9 shape from each
search's best_params, then submits:
  run_epoch_budget_v17.sbatch  (5 arms x 5 folds: 4 tasks + multitask control)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def build_config(task: str) -> dict:
    src = ROOT / f"results/search_v17_final_single_{task}/cv_results/search_all_train_best_params.json"
    bp = json.load(open(src))["best_params"]
    dims = [bp[f"hidden_dim_{i}"] for i in range(bp["n_layers"])]
    return {
        "features_path": "data/matrices/matrix_v17_train/unitigs.frac.mat",
        "metadata_path": "data/splits_v17/train_metadata.tsv",
        "train_ids_path": "data/splits_v17/train_accessions.txt",
        "output_dir": f"results/final_fixed_v17/{task}/final_model",
        "hyperparameters": {
            "model_params": {
                "hidden_dims": dims,
                "dropout": bp["dropout"],
                "activation": bp["activation"],
                "use_batch_norm": bp["use_batch_norm"],
            },
            "trainer_params": {
                "learning_rate": bp["learning_rate"],
                "weight_decay": bp["weight_decay"],
                "task_weights": {task: 1.0},
            },
            "batch_size": bp["batch_size"],
        },
        "task_names": [task],
        "task_types": {task: "classification"},
        "task_weights": {task: 1.0},
        "label_smoothing_per_task": {task: bp[f"ls_{task}"]},
        "validation_split": 0.1,
        "max_epochs": 200,
        "early_stopping_patience": 20,
        "random_seed": 42,
        "class_imbalance": {
            "method": "logit_adjustment",
            "logit_adjust_tau": bp["logit_adjust_tau"],
            "search_tau": False,
            "class_weights": None,
        },
        "_provenance": {
            "source": str(src.relative_to(ROOT)),
            "selection": "v17 Optuna search (100 trials, eligible_pooled) on the v17 dev folds",
            "protocol": "PROTOCOLS.md v17",
        },
    }


def main() -> None:
    out = ROOT / "configs/final_fixed_v17"
    out.mkdir(exist_ok=True)
    for t in TASKS:
        cfg = build_config(t)
        (out / f"{t}.json").write_text(json.dumps(cfg, indent=1) + "\n")
        log.info("%s: dims %s lr %.3g tau %.3f smoothing %.3f batch %d",
                 t, cfg["hyperparameters"]["model_params"]["hidden_dims"],
                 cfg["hyperparameters"]["trainer_params"]["learning_rate"],
                 cfg["class_imbalance"]["logit_adjust_tau"],
                 cfg["label_smoothing_per_task"][t],
                 cfg["hyperparameters"]["batch_size"])
    # the equal-study-weight feature arm reuses feature's config
    sw = ROOT / "configs/step4_2_v17"
    sw.mkdir(exist_ok=True)
    cfg = build_config("feature")
    cfg["output_dir"] = "results/step4_2_v17/feature"
    (sw / "feature.json").write_text(json.dumps(cfg, indent=1) + "\n")

    # multitask control arm: v9 multitask architecture carried (a stated design
    # control for the single-vs-multi re-test), v17 paths.
    mt = json.load(open(ROOT / "configs/final_fixed_v9/multitask.json"))
    mt_s = json.dumps(mt, indent=1)
    mt_s = mt_s.replace("data/splits_v9/", "data/splits_v17/")
    mt_s = mt_s.replace("data/matrices/matrix_v9_train/", "data/matrices/matrix_v17_train/")
    mt_s = mt_s.replace("results/final_fixed_v9/", "results/final_fixed_v17/")
    (out / "multitask.json").write_text(mt_s + "\n")
    log.info("multitask control config written (v9 architecture carried)")

    r = subprocess.run(["sbatch", "--parsable", "scripts/training/run_epoch_budget_v17.sbatch"],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        log.error("budget submission failed: %s", r.stderr.strip())
        sys.exit(1)
    log.info("budget stage submitted: %s", r.stdout.strip())


if __name__ == "__main__":
    main()
