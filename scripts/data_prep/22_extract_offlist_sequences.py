#!/usr/bin/env python3
"""S7 step 1: pull each sample's off-list sequences, the ones the matrix threw away.

A sequence in a sample's own unitig file is **off-list** when it shares no canonical 31-mer
with the 110,202-unitig vocabulary. Those are the sequences the k-mer filter deleted:
20,623,622 of 308,662,143,339 k-mers survived it, and a class with 15 training runs has its
k-mers in 0.55 % of samples against a 10 % threshold, so rare-class signal was removed by
construction.

Two design choices that matter, both fixed before running:

**Sample before testing, and sample by reservoir.** Drawing a fixed number of sequences per
file and then testing them costs two orders of magnitude fewer k-mer lookups than testing
everything. The draw is a seeded reservoir over the whole file rather than the first N,
because the assembler writes in an order that correlates with coverage, so "the first N"
would be a biased slice of every sample.

**Per-sample seed.** The seed is derived from the accession, so a sample's draw is
reproducible on its own and does not depend on which shard it landed in or how many tasks
the array had.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
UNITIGS = ROOT / "data/unitigs"
OUT = ROOT / "data/sequences_v9/offlist"
K = 31
DRAW = 20_000          # sequences sampled per file
KEEP = 2_000           # off-list sequences kept per sample
CODES = np.full(256, 255, dtype=np.uint8)
for i, b in enumerate("ACGT"):
    CODES[ord(b)] = i
    CODES[ord(b.lower())] = i


def sample_seed(acc: str) -> int:
    return int(hashlib.sha256(acc.encode()).hexdigest()[:8], 16)


def reservoir(path: Path, n: int, rng: np.random.Generator) -> list[str]:
    """Uniform sample of n sequences from a gzipped fasta, one streaming pass."""
    keep: list[str] = []
    seen = 0
    cur: list[str] = []
    def emit(s):
        nonlocal seen
        if not s:
            return
        seen += 1
        if len(keep) < n:
            keep.append(s)
        else:
            j = rng.integers(0, seen)
            if j < n:
                keep[int(j)] = s
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                emit("".join(cur)); cur = []
            else:
                cur.append(line.strip())
    emit("".join(cur))
    return keep


def canonical_kmers(seq: str) -> np.ndarray:
    codes = CODES[np.frombuffer(seq.encode(), dtype=np.uint8)]
    if len(codes) < K or (codes == 255).any():
        codes = codes[codes != 255]
    if len(codes) < K:
        return np.empty(0, dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(codes.astype(np.uint64), K)
    shifts = (np.arange(K, dtype=np.uint64) * np.uint64(2))[::-1]
    fwd = (w << shifts).sum(axis=1, dtype=np.uint64)
    rev = ((np.uint64(3) - w)[:, ::-1] << shifts).sum(axis=1, dtype=np.uint64)
    return np.minimum(fwd, rev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--accessions", type=Path, default=None,
                    help="Process only these accessions, as one shard. Used to backfill the "
                         "93 held-out samples whose unitig files arrived later: the held-out "
                         "matrix was built from diana-predict fraction vectors, which never "
                         "keep the sample's own assembly, so those samples had no off-list "
                         "data and were scored on a placeholder equal to the training mean.")
    a = ap.parse_args()

    vocab = np.load(ROOT / "data/sequences_v9/vocab_kmers_k31.npy")
    accs = sorted({l.split(" : ")[0].strip()
                   for l in (ROOT / "data/train_samples_v9.fof").open() if l.strip()} |
                  {l.strip() for l in (ROOT / "data/splits_v9/test_accessions.txt").open()
                   if l.strip()})
    if a.accessions:
        mine = [l.strip() for l in a.accessions.open() if l.strip()]
    else:
        mine = accs[a.shard::a.n_shards]
    logger.info("shard %d/%d: %d samples, vocabulary %d k-mers",
                a.shard, a.n_shards, len(mine), len(vocab))

    OUT.mkdir(parents=True, exist_ok=True)
    flat, offsets, kept_ids, counts, missing = [], [0], [], [], []
    for n, acc in enumerate(mine, 1):
        f = UNITIGS / f"{acc}.unitigs.fa.gz"
        if not f.exists() or f.stat().st_size == 0:
            missing.append(acc); continue
        rng = np.random.default_rng(sample_seed(acc))
        drawn = reservoir(f, DRAW, rng)
        off = 0
        for s in drawn:
            if off >= KEEP:
                break
            km = canonical_kmers(s)
            if len(km) == 0:
                continue
            pos = np.searchsorted(vocab, km)
            pos[pos >= len(vocab)] = 0
            if (vocab[pos] == km).any():        # shares a k-mer -> on-list
                continue
            codes = CODES[np.frombuffer(s.encode(), dtype=np.uint8)]
            codes = codes[codes != 255]
            flat.append(codes); offsets.append(offsets[-1] + len(codes)); off += 1
        kept_ids.append(acc); counts.append(off)
        if n % 20 == 0:
            logger.info("  %d/%d samples, %d off-list sequences so far", n, len(mine), len(offsets) - 1)

    np.savez(OUT / f"shard{a.shard:03d}.npz",
             bases=np.concatenate(flat) if flat else np.empty(0, np.uint8),
             offsets=np.asarray(offsets, dtype=np.int64),
             samples=np.asarray(kept_ids), counts=np.asarray(counts, dtype=np.int32))
    logger.info("shard %d: %d samples, %d sequences, %d files missing or empty",
                a.shard, len(kept_ids), len(offsets) - 1, len(missing))
    if missing:
        logger.warning("  missing/empty: %s", ", ".join(missing[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
