#!/usr/bin/env python3
"""Encode the 110,202 unitig sequences as a ragged integer array for the S4 encoder.

Why ragged and not a padded matrix
----------------------------------
Unitig lengths run from 61 bp to 40,205 bp (p25 71, p50 95, p75 153, p99 966). Padding
every unitig to the maximum would store and process 110,202 x 40,205 = 4.43 billion
positions against 17,498,477 real ones, a 253x waste. So the bases are stored flat with
an offset per unitig, and the encoder pads only inside a length-sorted chunk.

Why the order is asserted rather than assumed
---------------------------------------------
`MatrixLoader.load` takes the first whitespace field of each `.mat` row as the feature id
and never sorts, so unitig column `j` of the feature matrix is row `j` of the matrix file.
The encoder indexes its cards by that same `j`, so the fasta must be in the same order.
A silent mismatch here would attach every card to the wrong unitig and still train, so the
ids are compared position by position and the script refuses to write on any difference.

The shuffled control
--------------------
`--shuffle` permutes bases *within* each unitig under a fixed seed. Each unitig's base
composition, its length and the matrix are untouched, so the only thing destroyed is
order. An S4 gain that survives this control is not a sequence effect.

    ./env/bin/python scripts/data_prep/17_unitig_sequence_tensor.py
    ./env/bin/python scripts/data_prep/17_unitig_sequence_tensor.py --shuffle
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MATRIX_DIR = ROOT / "data/matrices/matrix_v9_train"
OUT_DIR = ROOT / "data/sequences_v9"

# A, C, G, T then one shared index for everything else (N and any IUPAC code). The
# encoder embeds 5 symbols and masks nothing: an N is a symbol the reader can learn to
# ignore, which is honest, whereas dropping it would shift every downstream position.
BASES = {"A": 0, "C": 1, "G": 2, "T": 3}
OTHER = 4
SHUFFLE_SEED = 42


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    """Ids and sequences in file order. One record's sequence may span several lines."""
    ids: list[str] = []
    seqs: list[str] = []
    chunks: list[str] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if chunks:
                    seqs.append("".join(chunks))
                    chunks = []
                # ">2624173 LN:i:61" -> "2624173"
                ids.append(line[1:].split()[0])
            else:
                chunks.append(line.strip())
    if chunks:
        seqs.append("".join(chunks))
    if len(ids) != len(seqs):
        raise ValueError(f"{path}: {len(ids)} headers but {len(seqs)} sequences")
    return ids, seqs


def matrix_feature_ids(path: Path) -> list[str]:
    """First whitespace field of every row, which is what MatrixLoader uses as the id."""
    with path.open() as fh:
        return [line.split(" ", 1)[0] for line in fh]


def encode(seqs: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """Flat base codes, offsets (n+1 entries), and the count of non-ACGT symbols."""
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    table = np.full(256, OTHER, dtype=np.uint8)
    for base, code in BASES.items():
        table[ord(base)] = code
        table[ord(base.lower())] = code

    flat = np.empty(int(offsets[-1]), dtype=np.uint8)
    for i, s in enumerate(seqs):
        flat[offsets[i]:offsets[i + 1]] = table[np.frombuffer(s.encode(), dtype=np.uint8)]
    return flat, offsets, int((flat == OTHER).sum())


def shuffle_within(flat: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Permute each unitig's bases in place in a copy, preserving composition exactly."""
    rng = np.random.default_rng(SHUFFLE_SEED)
    out = flat.copy()
    for i in range(len(offsets) - 1):
        rng.shuffle(out[offsets[i]:offsets[i + 1]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", type=Path, default=MATRIX_DIR / "unitigs.fa")
    ap.add_argument("--matrix", type=Path, default=MATRIX_DIR / "unitigs.frac.mat",
                    help="only its first column is read, to assert the unitig order")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--shuffle", action="store_true",
                    help="write the within-unitig shuffled control instead")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ids, seqs = read_fasta(args.fasta)
    logger.info("fasta: %d records, %d bases", len(ids), sum(len(s) for s in seqs))

    mat_ids = matrix_feature_ids(args.matrix)
    if len(mat_ids) != len(ids):
        raise SystemExit(f"{len(mat_ids)} matrix rows but {len(ids)} fasta records")
    mismatch = [k for k, (a, b) in enumerate(zip(ids, mat_ids)) if a != b]
    if mismatch:
        raise SystemExit(
            f"unitig order differs from {args.matrix.name} at {len(mismatch)} positions, "
            f"first at row {mismatch[0]} (fasta {ids[mismatch[0]]} vs matrix {mat_ids[mismatch[0]]})"
        )
    logger.info("unitig order matches %s at all %d rows", args.matrix.name, len(ids))

    flat, offsets, n_other = encode(seqs)
    lengths = np.diff(offsets)
    logger.info("non-ACGT symbols: %d of %d (%.4f %%)", n_other, len(flat),
                100 * n_other / len(flat))
    logger.info("lengths: min %d p50 %d p99 %d max %d",
                lengths.min(), int(np.percentile(lengths, 50)),
                int(np.percentile(lengths, 99)), lengths.max())

    if args.shuffle:
        flat = shuffle_within(flat, offsets)
        suffix = ".shuffled"
        logger.info("bases permuted within each unitig, seed %d", SHUFFLE_SEED)
    else:
        suffix = ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f"unitig_bases{suffix}.npy", flat)
    np.save(args.out_dir / "unitig_offsets.npy", offsets)
    meta = {
        "n_unitigs": len(ids),
        "n_bases": int(len(flat)),
        "n_non_acgt": n_other,
        "order_asserted_against": str(args.matrix.relative_to(ROOT)),
        "shuffled": bool(args.shuffle),
        "shuffle_seed": SHUFFLE_SEED if args.shuffle else None,
        "base_codes": {**BASES, "other": OTHER},
        "length_percentiles": {str(q): int(np.percentile(lengths, q))
                               for q in (0, 25, 50, 75, 90, 99, 100)},
    }
    (args.out_dir / f"sequences{suffix}.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s", args.out_dir / f"unitig_bases{suffix}.npy")


if __name__ == "__main__":
    main()
