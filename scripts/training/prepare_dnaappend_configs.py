#!/usr/bin/env python3
"""Write the S6 configs by copying S1's and pointing them at the appended matrices.

S6 differs from S1 only in which columns are appended, so the configs are copied rather
than written: each is the matching S1 config, which already carries that task's
Optuna-selected hyperparameters, the same five dev folds and the same training ids, with
`features_path` pointed at that fold's appended matrix and `output_dir` moved.

The model is unchanged. Its input layer simply widens from 110,202 to 110,970, and every
private per-unitig weight vector stays, which is the entire difference from S4 and S5.

    ./env/bin/python scripts/training/prepare_dnaappend_configs.py --summary max
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/kmer_test_v9/configs"
TASKS = ["community_type", "feature", "sample_host", "material"]

COMMENT = {
 "max": (
  "S6 (max): the 110,202 fractions with 768 columns APPENDED, being the per-dimension "
  "maximum over the frozen DNABERT-2 descriptions of the unitigs the sample contains. "
  "Non-linear in the fractions, which is the point: appending a fraction-weighted SUM would "
  "span only functions the first layer could already express. Every private per-unitig "
  "weight vector is kept, unlike S4 and S5, which replaced them and lost 0.208 to 0.359 on "
  "held-out. The appended block is standardised on this fold's four TRAINING folds only and "
  "rescaled to the fraction block's spread. Acceptance checks before the run: 122 structural "
  "and leakage checks passed, and the block carries about 12 effective dimensions of "
  "between-sample variation with 0 dead columns. Held-out untouched."),
 "mean": (
  "S6 (mean): as the max arm but appending the fraction-weighted AVERAGE of the descriptions. "
  "This is the literal form of 'append a sequence embedding', and it is close to redundant by "
  "construction: a weighted sum of per-unitig embeddings is a linear function of the input, "
  "so only the division by the sample's total fraction escapes, the same and only escape S1 "
  "had. It carries about 5 effective dimensions against the max arm's 12. Run as the "
  "comparison for the max arm, with that caveat attached to any result. Held-out untouched."),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", choices=["max", "mean"], required=True)
    ap.add_argument("--table", choices=["real", "shuffled"], default="real")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    suffix = "" if a.table == "real" else "shuf"
    out_root = f"results/dna{a.summary}{suffix}_test_v9"
    cfg_dir = ROOT / out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        for fold in range(5):
            src = SOURCE / f"kmer_{task}_fold{fold}.json"
            cfg = json.loads(src.read_text())
            matrix = f"data/matrices/matrix_v9_train/unitigs.frac.dna{a.summary}{suffix}.fold{fold}.mat"
            if not (ROOT / matrix).exists():
                raise SystemExit(f"missing {matrix}")
            cfg["features_path"] = matrix
            cfg["output_dir"] = f"{out_root}/fit_{task}_fold{fold}"
            cfg["_comment"] = COMMENT[a.summary] + ("" if a.table == "real" else
                " SCRAMBLED CONTROL: the descriptions were built from sequences whose bases"
                " were permuted within each unitig under seed 42, so composition, length, the"
                " fractions and the labels are untouched and only sequence ORDER is removed."
                " This arm is what makes a clearing cell reportable: the pre-registered"
                " condition is that the cell must NOT clear here.")
            prov = cfg.setdefault("_provenance", {})
            prov["hyperparameters_copied_from"] = str(src.relative_to(ROOT))
            prov["representation"] = matrix
            prov["appended"] = f"768 DNABERT-2 {a.summary} columns; fractions unchanged"
            dest = cfg_dir / f"dna{a.summary}{suffix}_{task}_fold{fold}.json"
            if dest.exists() and not a.overwrite:
                raise SystemExit(f"{dest} exists; pass --overwrite")
            dest.write_text(json.dumps(cfg, indent=2))
    logger.info("wrote 20 configs to %s", cfg_dir)


if __name__ == "__main__":
    main()
