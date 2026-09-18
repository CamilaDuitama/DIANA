#!/usr/bin/env python3
"""S7 step 2: describe the off-list sequences with DNABERT-2.

Reads the shards written by `22_extract_offlist_sequences.py` and writes one 768-vector per
off-list sequence, keeping the sample boundaries so step 3 can pool per sample.

Inherits every correction S6 cost us:

* **canonical orientation**, never averaging the two orientations, which costs 22 % of the
  description's informative magnitude (mean deviation norm 1.014 against 1.325);
* `--scrambled` permutes each sequence's bases under a seed derived from the sequence's
  position, giving the control that separates "the DNA content matters" from "768 more
  columns matter" — the comparison without which a clearing cell is unreportable;
* it **refuses to overwrite** an existing output, because a stale table propagates silently
  into every matrix built from it.
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SHARDS = ROOT / "data/sequences_v9/offlist"
BASES = np.array(list("ACGT"))
SEED = 42


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT / "scripts/data_prep"))
    import importlib
    mod = importlib.import_module("18_embed_unitigs_dnabert2")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--scrambled", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=24_000,
                    help="Padded token budget per batch. A fixed sequence count is wrong "
                         "here: off-list lengths run from 31 bp to 11,526 bp (p50 38, p99 "
                         "144), attention is O(L^2), and 256 x the longest sequence "
                         "exhausted a 40 GB GPU on shard 21. Batching by budget keeps peak "
                         "memory flat whatever the length mix.")
    a = ap.parse_args()

    tag = ".shuffled" if a.scrambled else ""
    out = SHARDS / f"emb{tag}_shard{a.shard:03d}.npy"
    if out.exists():
        logger.error("%s exists; refusing to overwrite", out)
        return 1

    z = np.load(SHARDS / f"shard{a.shard:03d}.npz", allow_pickle=True)
    bases, offs = z["bases"], z["offsets"]
    n = len(offs) - 1
    seqs = []
    rng = np.random.default_rng(SEED + a.shard)
    for i in range(n):
        codes = bases[offs[i]:offs[i + 1]].copy()
        if a.scrambled:
            rng.shuffle(codes)
        seqs.append("".join(BASES[codes]))
    logger.info("shard %d: %d sequences, %d bases, scrambled=%s",
                a.shard, n, len(bases), a.scrambled)

    tok, model = mod.load_model(a.device)
    # canonical orientation, one fixed choice per sequence, nothing averaged away
    seqs = [mod.canonical(s) for s in seqs]
    # Long sequences are windowed like the unitig embedder does, then averaged, so one
    # 11 kb outlier cannot dictate the batch shape for the 38 bp median.
    order = np.argsort([len(s) for s in seqs], kind="stable")
    table = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    batch, batch_idx, done = [], [], 0

    def flush():
        nonlocal batch, batch_idx, done
        if not batch:
            return
        table[batch_idx] = mod.embed_batch(tok, model, batch, a.device).numpy()
        done += len(batch)
        if done % 20000 < len(batch):
            logger.info("  %d / %d sequences", done, n)
        batch, batch_idx = [], []

    for j in order:
        s = seqs[j]
        if len(s) > mod.WINDOW_BP:
            flush()
            parts = mod.embed_batch(tok, model, mod.windows(s), a.device)
            table[j] = parts.mean(0).numpy()
            done += 1
            continue
        # BPE gives roughly 5 bases a token; keep the PADDED budget under max-tokens
        longest = max([len(s)] + [len(x) for x in batch])
        if batch and (len(batch) + 1) * (longest // 5 + 8) > a.max_tokens:
            flush()
        batch.append(s); batch_idx.append(int(j))
    flush()
    np.save(out, table)
    logger.info("wrote %s %s", out.name, table.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
