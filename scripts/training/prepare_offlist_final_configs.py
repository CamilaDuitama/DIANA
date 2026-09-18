#!/usr/bin/env python3
"""S7: final configs — one model per task fitted on all 2,716 training runs.

Takes this arm's own searched settings and points them at the **all-train** matrix, whose
appended block is standardised on all 2,716 training samples. That matters because the
held-out matrix is standardised on the same statistics: S6's first final fit trained on a
fold matrix (80 % of train) and was tested on the all-train one, straddling two scalings,
and those results were discarded rather than adjusted.

Produces the two evaluations asked for: train and held-out, on the same frozen model.
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
    matrix = f"data/matrices/matrix_v9_train/unitigs.frac.off{tag}.alltrain.mat"
    if not (ROOT / matrix).exists():
        raise SystemExit(f"missing {matrix}")

    cfg_dir = ROOT / f"results/{arm}_final_v9/configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        src = ROOT / f"results/{arm}_search_v9/search_{task}/final_training_config.json"
        cfg = json.loads(src.read_text())
        cfg["features_path"] = matrix
        cfg["train_ids_path"] = "data/splits_v9/train_accessions.txt"
        cfg["output_dir"] = f"results/{arm}_final_v9/fit_{task}"
        cfg["_comment"] = (
            f"S7 FINAL fit, {task}, {'scrambled' if a.scrambled else 'real'} arm. All 2,716 "
            f"training runs, 110,202 fractions plus 3,072 columns summarising the sequence the "
            f"k-mer filter deleted, hyperparameters from this arm's own 100-trial search. "
            f"Training and held-out matrices are BOTH standardised on all 2,716 training runs, "
            f"so the model is evaluated on the scaling it was fitted on; held-out's own "
            f"statistics are never used.")
        (cfg_dir / f"off{tag}_final_{task}.json").write_text(json.dumps(cfg, indent=2))
    logger.info("wrote %d final configs to %s", len(TASKS), cfg_dir)


if __name__ == "__main__":
    main()
