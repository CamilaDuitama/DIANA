#!/usr/bin/env python3
"""Final S6 configs: one model per task, fitted on all 2,716 training runs.

Copies that task's dev-fold S6 config so the hyperparameters, seed and appended columns are
identical, and changes only what makes it a final fit: the full training id list instead of a
fold's, and a fresh output directory. The matrix is fold 0's appended matrix, whose appended
block was standardised on folds 1-4; that is the same handicap every other final fit in this
project carries and is stated rather than hidden.

The held-out matrix built alongside it is standardised on all 2,716 training samples, never
on held-out's own statistics.

    ./env/bin/python scripts/training/prepare_dnaappend_final_configs.py --summary max
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
    ap.add_argument("--summary", choices=["max", "mean"], required=True)
    ap.add_argument("--table", choices=["real", "shuffled"], default="real")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    suffix = "" if a.table == "real" else "shuf"
    out_root = f"results/dna{a.summary}{suffix}_final_v9"
    cfg_dir = ROOT / out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        src = ROOT / (f"results/dna{a.summary}{suffix}_test_v9/configs/"
                      f"dna{a.summary}{suffix}_{task}_fold0.json")
        cfg = json.loads(src.read_text())
        # The final fit MUST use the all-train scaling, because the held-out matrix is scaled
        # on all 2,716 training runs. Training on a fold matrix (scaled on 80 % of train) and
        # testing on the all-train one straddles two scalings: measured 2026-09-16, that
        # shifts the appended columns by a mean 0.09 of their own sd. The first final fit had
        # this defect and its results were discarded, not adjusted.
        matrix = (f"data/matrices/matrix_v9_train/"
                  f"unitigs.frac.dna{a.summary}{suffix}.alltrain.mat")
        if not (ROOT / matrix).exists():
            raise SystemExit(f"missing {matrix}; build it with --all-train")
        cfg["features_path"] = matrix
        cfg["train_ids_path"] = "data/splits_v9/train_accessions.txt"
        cfg["output_dir"] = f"{out_root}/fit_{task}"
        cfg["_comment"] = (
            f"S6 FINAL fit for the held-out read: all 2,716 training runs, the 110,202 "
            f"fractions with 768 DNABERT-2 {a.summary} columns appended, every private "
            f"per-unitig weight vector kept. Hyperparameters, seed and appended columns "
            f"identical to the dev-fold arm. Training and held-out matrices are BOTH "
            f"standardised on all 2,716 training runs, so the model is tested on the same "
            f"scaling it was fitted on; held-out's own statistics are never used.")
        prov = cfg.setdefault("_provenance", {})
        prov["final_fit_copied_from"] = str(src.relative_to(ROOT))
        dest = cfg_dir / f"dna{a.summary}{suffix}_final_{task}.json"
        if dest.exists() and not a.overwrite:
            raise SystemExit(f"{dest} exists; pass --overwrite")
        dest.write_text(json.dumps(cfg, indent=2))
    logger.info("wrote %d final configs to %s", len(TASKS), cfg_dir)


if __name__ == "__main__":
    main()
