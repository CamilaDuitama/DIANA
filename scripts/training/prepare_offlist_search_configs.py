#!/usr/bin/env python3
"""S7e: search configs for the appended-off-list input, one per task.

Copies the config that searched the fraction arm (`train_config_v9_final_search_single_*`)
and changes only the input matrix and the output directory, so the trial count, search space,
inner-CV structure, seed, imbalance handling and grouping are identical to what the fraction
arm received. That equality is the point: it is why a difference can be attributed to the
columns rather than to a luckier search.

`max_epochs` is raised from 200. On the fraction arm `feature` hit the 200-epoch cap with its
best epoch AT the cap, meaning validation loss was still improving when training stopped, and
an arm with 3,072 extra columns to fit is penalised more by that truncation than the arm it is
compared against. Patience still stops runs that have genuinely converged.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
MAX_EPOCHS = 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scrambled", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tag = "shuf" if a.scrambled else ""

    out_dir = ROOT / f"results/offlist{tag}_search_v9/configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = f"data/matrices/matrix_v9_train/unitigs.frac.off{tag}.fold0.mat"
    if not (ROOT / matrix).exists():
        raise SystemExit(f"missing {matrix}")

    for task in TASKS:
        src = ROOT / f"configs/train_config_v9_final_search_single_{task}.json"
        cfg = json.loads(src.read_text())
        cfg["features_path"] = matrix
        cfg["output_dir"] = f"results/offlist{tag}_search_v9/search_{task}"
        cfg["max_epochs"] = MAX_EPOCHS
        # Required by the trainer for any searching run. The fraction arm's final search
        # predates the requirement, so its config omits it; PROJECT.md records that all four
        # candidate objectives rank trials at Spearman 0.80 to 0.96 and pick the same best
        # trial, so setting it does not advantage this arm. `eligible_pooled` is the sound
        # choice: the five folds' out-of-fold predictions are concatenated and scored once,
        # so every class keeps its weight and no fold can degenerate to a single eligible
        # class and saturate at 1.0.
        cfg["search_objective"] = "eligible_pooled"
        cfg["class_eligibility_path"] = "data/splits_v9/class_eligibility.tsv"
        cfg["_comment"] = (
            f"S7e: Optuna search for the appended off-list input, {task}. Identical to the "
            f"fraction arm's search in trial count, search space, inner CV, seed, imbalance "
            f"handling and grouping; only the matrix differs, so a difference is attributable "
            f"to the 3,072 appended columns rather than to a luckier search. max_epochs raised "
            f"from 200 to {MAX_EPOCHS} because `feature` previously hit the 200 cap with its "
            f"best epoch AT the cap, and an arm with more columns to fit is penalised more by "
            f"that truncation than the arm it is compared against. "
            + ("SCRAMBLED CONTROL: the columns describe sequences whose bases were permuted, "
               "so this arm keeps the extra capacity and removes the sequence content."
               if a.scrambled else ""))
        (out_dir / f"search_{task}.json").write_text(json.dumps(cfg, indent=2))
    logger.info("wrote %d search configs to %s (matrix %s, max_epochs %d, n_trials %d)",
                len(TASKS), out_dir, matrix, MAX_EPOCHS,
                json.loads((ROOT / f"configs/train_config_v9_final_search_single_feature.json").read_text())["n_trials"])


if __name__ == "__main__":
    main()
