#!/usr/bin/env python3
"""Dev-fold sweep of the reader's learning rate, to test why the cards went uniform.

This is tuning, not a result. It exists to decide one thing: whether `material`'s collapse
to a single constant output came from the reader being driven too hard. The searched rates
were chosen for an input layer of up to 56.4 M free weights and differ 230-fold between
tasks, and the reader is a 211 k-weight function shared across all 110,202 unitigs, so
neither inherited rate need be anywhere near right for it.

Two directions are tested, because the two tasks failed in opposite ways:

  material  inherited 5.57e-03 -> trained the reader and collapsed to one constant output;
            sweep DOWN to see whether a gentler rate keeps the output sample-specific.
  feature   inherited 2.41e-05 -> 24 updates, reader unmoved, predictions near-degenerate;
            sweep UP to see whether it can learn anything at all.

The MLP keeps its searched rate throughout, so the only thing changing is the reader's.

    ./env/bin/python scripts/training/prepare_seqenc_lrsweep.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = "results/seqenc_lrsweep_v9"
GRID = {"material": [1e-5, 1e-4, 1e-3], "feature": [1e-4, 1e-3]}
FOLD = 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg_dir = ROOT / OUT_ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task, rates in GRID.items():
        base = json.loads((ROOT / f"results/seqenc_test_v9/configs/seqenc_{task}_fold{FOLD}.json").read_text())
        for lr in rates:
            cfg = json.loads(json.dumps(base))
            tag = f"{task}_fold{FOLD}_lr{lr:.0e}".replace("-0", "-")
            cfg["sequence_encoder"]["learning_rate"] = lr
            cfg["output_dir"] = f"{OUT_ROOT}/fit_{tag}"
            cfg["_comment"] = (
                "Reader learning-rate sweep, dev fold only, TUNING NOT A RESULT. The MLP keeps "
                f"its searched rate of {cfg['hyperparameters']['trainer_params']['learning_rate']:.2e} "
                f"and only the reader runs at {lr:.0e}. Decides whether the uniform-card collapse "
                "on `material` (identical output for all 671 samples, PC1 100.0 %) is a "
                "too-high-rate effect, and whether `feature` can learn at all above 2.41e-05. "
                "Held-out untouched."
            )
            (cfg_dir / f"{tag}.json").write_text(json.dumps(cfg, indent=2))
            written.append(tag)
    (cfg_dir / "sweep_order.txt").write_text("\n".join(written) + "\n")
    logger.info("wrote %d sweep configs to %s", len(written), cfg_dir)
    for w in written:
        logger.info("  %s", w)


if __name__ == "__main__":
    main()
