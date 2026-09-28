#!/usr/bin/env python3
"""S8.2 (S8a): append the kept off-list hashes as 0/1 columns to the fraction matrices.

For train (2,716 runs) and held-out (922 runs): the 110,202 fractions unchanged, then
one column per hash in `data/sketches_v9/offlist_shared_hashes.tsv` (K = 1,984,422 at
>= 14 training samples and >= 2 training projects), 1 if the run's sketch contains the
hash. The list is defined on training sketches only; a held-out run's column values
come from its own sketch, so held-out never touches which columns exist.

Rules carried over from S7 (25_append_offlist_columns.py): sample order is taken from
the matrix directory's kmtricks.fof and asserted, never assumed; an existing output is
refused, not reused. Difference from S7: the block is NOT standardised (0/1 as is), so a
run with none of the hashes is "nothing present" rather than a distinctive vector.

Output: <matrix dir>/unitigs.frac.s8a.npz with frac (float32), block (uint8),
feature_ids, sample_ids; read by MatrixLoader via its .npz path.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from diana.data.loader import MatrixLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SK = ROOT / "data/sketches_v9"
DIRS = {"train": ROOT / "data/matrices/matrix_v9_train",
        "heldout": ROOT / "data/matrices/matrix_v9_heldout"}
NAME = "unitigs.frac.s8a.npz"


def read_sketches(zip_path: Path, wanted: set[str], ksize: int = 31) -> dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            n = info.filename
            if not (n.endswith(".sig") or n.endswith(".sig.gz")):
                continue
            raw = z.read(info)
            if n.endswith(".gz"):
                raw = gzip.decompress(raw)
            for rec in json.load(io.BytesIO(raw)):
                if rec["name"] not in wanted:
                    continue
                for s in rec.get("signatures", []):
                    if s.get("ksize") == ksize:
                        out[rec["name"]] = np.array(s["mins"], dtype=np.uint64)
    return out


def fof_order(d: Path) -> list[str]:
    fof = d / "kmer_matrix/kmtricks.fof"
    return [l.split(" : ")[0].strip() if " : " in l else l.split()[0] for l in fof.open() if l.strip()]


def block_for(order: list[str], sk: dict[str, np.ndarray], K: np.ndarray) -> np.ndarray:
    B = np.zeros((len(order), len(K)), dtype=np.uint8)
    for i, acc in enumerate(order):
        m = sk[acc]
        idx = np.searchsorted(K, m)
        idx[idx == len(K)] = 0
        hit = K[idx] == m
        B[i, idx[hit]] = 1
    return B


def main() -> int:
    t0 = time.time()
    kept = pd.read_csv(SK / "offlist_shared_hashes.tsv", sep="\t")
    K = np.sort(kept.hash.to_numpy().astype(np.uint64))
    logger.info("K = %d kept off-list hashes", len(K))
    orders = {k: fof_order(d) for k, d in DIRS.items()}
    sk = read_sketches(SK / "unitigs_split_v9.zip", set(orders["train"]) | set(orders["heldout"]))
    logger.info("sketches read for %d runs in %.0f s", len(sk), time.time() - t0)

    feature_ids = None
    for part, d in DIRS.items():
        out = d / NAME
        if out.exists():
            raise SystemExit(f"{out} exists; refusing to reuse a possibly stale matrix")
        order = orders[part]
        missing = [a for a in order if a not in sk]
        if missing:
            raise SystemExit(f"{part}: {len(missing)} runs without a sketch, e.g. {missing[:5]}")
        F, ids, _ = MatrixLoader(str(d / "unitigs.frac.mat")).load()
        if list(ids) != order:
            raise SystemExit(f"{part}: matrix sample order differs from kmtricks.fof")
        uids = [l.split()[0] for l in (d / "unitigs.frac.mat").open()]
        if feature_ids is None:
            feature_ids = uids
        elif uids != feature_ids:
            raise SystemExit("held-out unitig rows differ from train")
        B = block_for(order, sk, K)
        density = B.mean()
        logger.info("%s: fractions %s, block %s, density %.4f, runs with no kept hash %d",
                    part, F.shape, B.shape, density, int((B.sum(1) == 0).sum()))
        np.savez(out, frac=F.astype(np.float32), block=B,
                 feature_ids=np.array(list(feature_ids) + [f"h{h}" for h in K], dtype=str),
                 sample_ids=np.array(order, dtype=str))
        (d / f"{NAME}.samples.txt").write_text("\n".join(order) + "\n")
        logger.info("wrote %s (%.1f GB) in %.0f s", out, out.stat().st_size / 1e9, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
