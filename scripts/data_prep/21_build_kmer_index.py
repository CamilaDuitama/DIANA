#!/usr/bin/env python3
"""Sorted array of the vocabulary's canonical 31-mers, for the S7 off-list test.

A sample's sequence is "off-list" when it shares no canonical 31-mer with the 110,202-unitig
vocabulary. Testing that exactly needs a membership structure over the vocabulary's k-mers;
`unitigs.sshash.dict` exists but querying it means a MUSET dependency, whereas the set is
small enough to hold outright: 17,498,477 bases minus 30 per unitig leaves about 14.2 M
k-mers, which is 114 MB as sorted uint64 and answers membership by binary search.

Canonical means the smaller of a k-mer's 2-bit code and its reverse complement's, matching
what kmtricks and ggcat do, so orientation cannot cause a false "off-list".
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/sequences_v9/vocab_kmers_k31.npy"
K = 31


def encode_canonical(seq_codes: np.ndarray, k: int = K) -> np.ndarray:
    """Canonical 2-bit codes of every k-mer in one sequence of base codes 0-3."""
    n = len(seq_codes) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(seq_codes.astype(np.uint64), k)
    shifts = (np.arange(k, dtype=np.uint64) * np.uint64(2))[::-1]
    fwd = (w << shifts).sum(axis=1, dtype=np.uint64)
    # reverse complement: complement is 3 - base, then reverse the order
    rcw = (np.uint64(3) - w)[:, ::-1]
    rev = (rcw << shifts).sum(axis=1, dtype=np.uint64)
    return np.minimum(fwd, rev)


def main() -> None:
    bases = np.load(ROOT / "data/sequences_v9/unitig_bases.npy")
    offsets = np.load(ROOT / "data/sequences_v9/unitig_offsets.npy")
    logger.info("vocabulary: %d unitigs, %d bases", len(offsets) - 1, len(bases))

    chunks = []
    for i in range(len(offsets) - 1):
        s = bases[offsets[i]:offsets[i + 1]]
        if len(s) >= K:
            chunks.append(encode_canonical(s))
        if i % 20000 == 0 and i:
            logger.info("  %d / %d unitigs", i, len(offsets) - 1)
    kmers = np.unique(np.concatenate(chunks))
    logger.info("%d canonical %d-mers, %.0f MB", len(kmers), K, kmers.nbytes / 1e6)
    np.save(OUT, kmers)
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
