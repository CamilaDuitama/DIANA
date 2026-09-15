#!/usr/bin/env python3
"""Write the S4 configs by copying the S1 arm's and changing only the input layer.

The point of the arm is that nothing except the input layer differs from the fraction
baseline, so the configs are not written from scratch. Each one is the matching
`results/kmer_test_v9/configs/kmer_<task>_fold<f>.json`, which already carries that task's
Optuna-selected hyperparameters, the same 5 dev folds and the same training ids, with
three changes:

1. `features_path` goes back to the plain fraction matrix (S1 appended 136 columns; S4
   appends nothing and substitutes the layer instead);
2. `output_dir` moves to this arm;
3. a `sequence_encoder` block is added, which is the only thing that alters the model.

The card length is deliberately absent from that block: `attach_sequence_encoder` reads it
off the layer it replaces, so it cannot drift from `hidden_dims[0]`.

    ./env/bin/python scripts/training/prepare_seqenc_configs.py --arm real
    ./env/bin/python scripts/training/prepare_seqenc_configs.py --arm shuffled
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
FOLDS = range(5)
FRACTION_MATRIX = "data/matrices/matrix_v9_train/unitigs.frac.mat"

ARMS = {
    "real":     ("results/seqenc_test_v9", "data/sequences_v9/unitig_bases.npy"),
    "shuffled": ("results/seqenc_shuf_v9", "data/sequences_v9/unitig_bases.shuffled.npy"),
}

READER = {"embed_dim": 32, "channels": [96, 96], "kernel": 9, "reader_activation": "relu"}

# The reader gets its own learning rate, decided by the dev-fold sweep in
# `results/seqenc_lrsweep_v9/` (job 14802341) before this arm was run. The searched rates
# belong to an input layer of up to 56.4 M free weights and are the wrong scale for a
# 211 k-weight function shared across all 110,202 unitigs: at `material`'s inherited
# 5.57e-03 the reader wrote effectively the same card for every unitig, the model emitted
# one identical output for all 671 samples (PC1 100.0 %) and predicted a single class, while
# at 1e-04 it predicted four classes with PC1 81.1 %. 1e-03 also avoided the collapse but
# did slightly worse (three classes), so 1e-04 it is.
READER_LR = 1e-4

# Reader updates equal optimiser steps, and no run so far has had more than about 500
# before early stopping ended it: on `feature` the MLP reaches its best validation loss by
# epoch 3 using near-random cards, patience 20 then expires around epoch 24, and 30 updates
# is nothing for learning to read DNA. Raising a higher reader rate did not help there
# because the run was over, not because the steps were too small. So each task gets an
# epoch budget sized to a fixed number of reader updates, and patience is set equal to it so
# nothing is cut short. This is not a free pass for the encoder: the trainer still evaluates
# the best-validation checkpoint, so each arm is judged at its own convergence rather than
# at a fixed epoch count. A reader update costs about 1.56 s, so this is roughly 2.2 h a fit.
TARGET_READER_UPDATES = 5000
SUB_TRAIN = 1840          # 90 % of a fold's training ids, from the run logs
# One reader update per optimiser step. This was 8 in the first submission, which was a
# mistake: the reader is updated once per cache rebuild, so the interval is the reader's
# update budget, and at batch 256 on 1,840 training runs an epoch is 8 steps. The reader
# got one update per epoch, nine before early stopping, and the real and shuffled arms came
# out byte-identical because neither had trained (jobs 14474106 and 14482844, cancelled
# 2026-09-15).
REFRESH_EVERY = 1
MAX_PADDED_PER_CHUNK = 1_000_000

COMMENT = {
    "real": (
        "S4: the input layer's 110,202 private per-unitig weight vectors are replaced by "
        "vectors computed from each unitig's bases by one small CNN shared across the whole "
        "vocabulary, with two gates per unitig (one scaled by the fraction, one by presence). "
        "The feature matrix, folds, ids, hyperparameters and every layer after the first are "
        "unchanged from the S1/calibration arms, so `results/calibration_v9` is the paired "
        "fraction baseline on exactly these folds. Score with 26_pooled_representation_test.py. "
        "A gain here is only a SEQUENCE result if it does not also appear in the shuffled arm "
        "(results/seqenc_shuf_v9), which preserves each unitig's base composition and length "
        "and destroys only order; a gain in both arms is a capacity or regularisation effect "
        "and must be reported as one. Held-out untouched."
    ),
    "shuffled": (
        "S4 control: identical in every respect to results/seqenc_test_v9 except that each "
        "unitig's bases were permuted within that unitig under seed 42 before encoding. Base "
        "composition, length, the feature matrix and the labels are untouched, so the only "
        "thing removed is sequence ORDER. This arm exists to stop a capacity effect being "
        "reported as a sequence effect. Held-out untouched."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_root, bases = ARMS[args.arm]
    if not (ROOT / bases).exists():
        raise SystemExit(f"missing {bases}; run scripts/data_prep/17_unitig_sequence_tensor.py")
    cfg_dir = ROOT / out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        for fold in FOLDS:
            src = SOURCE / f"kmer_{task}_fold{fold}.json"
            cfg = json.loads(src.read_text())

            steps_per_epoch = -(-SUB_TRAIN // cfg["hyperparameters"]["batch_size"])
            epochs = -(-TARGET_READER_UPDATES // steps_per_epoch)
            cfg["max_epochs"] = epochs
            cfg["early_stopping_patience"] = epochs
            cfg["features_path"] = FRACTION_MATRIX
            cfg["output_dir"] = f"{out_root}/fit_{task}_fold{fold}"
            cfg["sequence_encoder"] = {
                "bases": bases,
                "offsets": "data/sequences_v9/unitig_offsets.npy",
                "refresh_every": REFRESH_EVERY,
                "learning_rate": READER_LR,
                "max_padded_per_chunk": MAX_PADDED_PER_CHUNK,
                **READER,
            }
            prov = cfg.setdefault("_provenance", {})
            prov["hyperparameters_copied_from"] = str(src.relative_to(ROOT))
            prov["representation"] = FRACTION_MATRIX
            prov["reader_learning_rate"] = (
                f"{READER_LR:.0e}, chosen by the dev-fold sweep in results/seqenc_lrsweep_v9/ "
                "before this arm ran; the MLP keeps its searched rate")
            prov["epoch_budget"] = (
                f"{epochs} epochs at {steps_per_epoch} steps each = about "
                f"{epochs * steps_per_epoch} reader updates, patience equal to the budget")
            prov["card_length"] = (
                f"hidden_dims[0] = {cfg['hyperparameters']['model_params']['hidden_dims'][0]}, "
                "read off the replaced layer rather than configured"
            )
            cfg["_comment"] = COMMENT[args.arm]

            dest = cfg_dir / f"seqenc_{task}_fold{fold}.json"
            if dest.exists() and not args.overwrite:
                raise SystemExit(f"{dest} exists; pass --overwrite to replace it")
            dest.write_text(json.dumps(cfg, indent=2))

    n = len(TASKS) * len(FOLDS)
    logger.info("wrote %d configs to %s (bases: %s)", n, cfg_dir, bases)


if __name__ == "__main__":
    main()
