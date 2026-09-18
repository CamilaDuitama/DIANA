#!/usr/bin/env python3
"""S7d: pool each sample's off-list descriptions into a fixed-size row, by quantiles.

Quantiles, not the mean. The mean is what emptied S1 and left S6's appended block with about
12 effective dimensions out of 768: averaging many vectors converges to the corpus mean, so
every sample lands in nearly the same place. A tail statistic moves when the mean does not.

Per dimension: p50, p90, p99 and max, giving 3,072 columns from 768 dimensions.

Empty samples are recorded rather than silently zeroed. A sample with no off-list sequence
would otherwise get an all-zero row that standardisation turns into a distinctive vector, an
identifier the model can recognise; 5 samples were in that state in S6.
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "data/sequences_v9/offlist"
QUANTILES = (50, 90, 99)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scrambled", action="store_true")
    a = ap.parse_args()
    tag = ".shuffled" if a.scrambled else ""

    rows, ids, empty = [], [], []
    for sf in sorted(glob.glob(str(OFF / "shard*.npz"))):
        shard = Path(sf).stem                      # shard000
        idx = shard.replace("shard", "")
        z = np.load(sf, allow_pickle=True)
        E = np.load(OFF / f"emb{tag}_shard{idx}.npy", mmap_mode="r")
        counts, samples = z["counts"], z["samples"]
        if int(counts.sum()) != E.shape[0]:
            logger.error("%s: %d sequences but %d embeddings", shard, counts.sum(), E.shape[0])
            return 1
        at = 0
        for s, c in zip(samples, counts):
            if c == 0:
                rows.append(np.zeros(768 * (len(QUANTILES) + 1), dtype=np.float32))
                ids.append(str(s)); empty.append(str(s)); continue
            block = np.asarray(E[at:at + c], dtype=np.float32)
            at += c
            qs = np.percentile(block, QUANTILES, axis=0)
            rows.append(np.concatenate([qs.ravel(), block.max(axis=0)]).astype(np.float32))
            ids.append(str(s))
        logger.info("%s: %d samples pooled", shard, len(samples))

    X = np.vstack(rows)
    np.save(OFF / f"pooled{tag}.npy", X)
    np.savetxt(OFF / f"pooled{tag}.samples.txt", np.asarray(ids), fmt="%s")
    logger.info("wrote pooled%s.npy %s from %d quantiles + max; %d empty samples",
                tag, X.shape, len(QUANTILES), len(empty))
    if empty:
        logger.warning("empty samples (recorded, must be masked downstream): %s", empty[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
