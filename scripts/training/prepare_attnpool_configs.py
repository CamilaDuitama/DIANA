#!/usr/bin/env python3
"""Write the S5 configs by copying the S4 arm's and swapping the input layer for attention.

Only the first layer differs from the fraction baseline, so these are not written from
scratch: each is the matching `results/seqenc_test_v9/configs/seqenc_<task>_fold<f>.json`,
which already carries that task's Optuna-selected hyperparameters, the same five dev folds
and the same training ids, with the `sequence_encoder` block replaced by `attention_pool`.

Unlike S4 there is no reader learning rate and no epoch budget to size: the description
table is frozen, so the only trained parameters are the small projections in the pooling
layer, and the task's searched settings apply to them as they do to any other layer. The
epoch budget therefore returns to the baseline's own (200 epochs, patience 20), which is
what every other arm used.

    ./env/bin/python scripts/training/prepare_attnpool_configs.py --arm real
    ./env/bin/python scripts/training/prepare_attnpool_configs.py --arm shuffled
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/seqenc_test_v9/configs"
TASKS = ["community_type", "feature", "sample_host", "material"]
FOLDS = range(5)

ARMS = {
    "real":     ("results/attnpool_test_v9", "data/sequences_v9/unitig_dnabert2.npy"),
    "shuffled": ("results/attnpool_shuf_v9", "data/sequences_v9/unitig_dnabert2.shuffled.npy"),
}
POOL = {"n_heads": 4, "value_dim": 128, "key_dim": 128}
BASELINE_EPOCHS, BASELINE_PATIENCE = 200, 20

COMMENT = {
    "real": (
        "S5: the input layer's fixed weighted sum is replaced by attention pooling over a "
        "FROZEN DNABERT-2 description of each unitig (both orientations averaged; see "
        "data/sequences_v9/unitig_dnabert2.json). The attention weights compete across only "
        "the unitigs a sample contains, so the same unitig can matter in one sample and not "
        "another, which no previous arm could express: every one of them was a fixed weighted "
        "sum. Coverage enters twice, shifting the attention logit and multiplying the value, so "
        "with one head, a flat score and gamma=0 this reduces to the fraction baseline. Folds, "
        "ids, hyperparameters and every layer after the first are unchanged, so "
        "results/calibration_v9 is the paired baseline on exactly these folds. Score with "
        "26_pooled_representation_test.py. A gain is only a SEQUENCE result if it does not also "
        "appear in results/attnpool_shuf_v9. Held-out untouched."
    ),
    "shuffled": (
        "S5 control: identical to results/attnpool_test_v9 except the description table was "
        "built from sequences whose bases were permuted within each unitig under seed 42. Base "
        "composition, length, the feature matrix and the labels are untouched, so the only thing "
        "removed is sequence ORDER. Separates 'the model reads DNA' from 'attention pooling "
        "beats a weighted sum', which are two different claims and this arm holds the second "
        "fixed. Held-out untouched."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_root, table = ARMS[args.arm]
    if not (ROOT / table).exists():
        raise SystemExit(f"missing {table}; run scripts/data_prep/18_embed_unitigs_dnabert2.py")
    cfg_dir = ROOT / out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        for fold in FOLDS:
            src = SOURCE / f"seqenc_{task}_fold{fold}.json"
            cfg = json.loads(src.read_text())
            cfg.pop("sequence_encoder", None)
            cfg["attention_pool"] = {"table": table, **POOL}
            cfg["output_dir"] = f"{out_root}/fit_{task}_fold{fold}"
            cfg["max_epochs"] = BASELINE_EPOCHS
            cfg["early_stopping_patience"] = BASELINE_PATIENCE
            prov = cfg.setdefault("_provenance", {})
            prov.pop("reader_learning_rate", None)
            prov.pop("epoch_budget", None)
            prov["description_table"] = table
            prov["pooled_width"] = (
                f"hidden_dims[0] = {cfg['hyperparameters']['model_params']['hidden_dims'][0]}, "
                "read off the replaced layer rather than configured")
            cfg["_comment"] = COMMENT[args.arm]

            dest = cfg_dir / f"attnpool_{task}_fold{fold}.json"
            if dest.exists() and not args.overwrite:
                raise SystemExit(f"{dest} exists; pass --overwrite to replace it")
            dest.write_text(json.dumps(cfg, indent=2))

    logger.info("wrote %d configs to %s (table: %s)", len(TASKS) * len(FOLDS), cfg_dir, table)


if __name__ == "__main__":
    main()
