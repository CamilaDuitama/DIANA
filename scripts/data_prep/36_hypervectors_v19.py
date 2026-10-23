#!/usr/bin/env python3
"""Hypervectors of every run from our own sketches (v19 HV test, PROTOCOLS.md amendments of 2026-10-14).

Input: data/sketches_v9/unitigs_split_v9.zip (sourmash, k = 31, scaled = 1000, presence;
all 3,638 runs, sketched from the same Logan unitig files as the matrix). Projection of
Faure et al. 2026: for a hash x and 64-bit word k, SplitMix64 of x + k * step; bit n of word
k gives dimension 64 k + n the value 1 - 2 bit; summed over the sketch; d = 2,048.
--projection builder: step 2,048 (Similarity-Matrix/src/random_projection.cpp, their
repository's database builder); --projection database: step 64
(Hypervectors/src/random_projection.cpp, which reproduces their released database). Then
divided by sqrt(d), as their database stores it, and by the vector's Euclidean length (a run
with an empty sketch keeps the zero vector and is logged).

Writes <out>: hv_train.npz and hv_heldout.npz (`frac` = the normalised vectors, float32,
`feature_ids` hv_0..hv_2047, `sample_ids` in split order; readable by MatrixLoader),
hv_raw.npz (integer sums of all runs) and sketch_sizes.tsv.

    ./env/bin/python scripts/data_prep/36_hypervectors_v19.py --projection database --out results/hypervectors_v19/database_projection
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SKETCHES = ROOT / "data/sketches_v9/unitigs_split_v9.zip"
SPLIT = ROOT / "data/splits_v19"
OUT = ROOT / "results/hypervectors_v19"
D = 2048
STEPS = {"builder": 64 * (D // 64), "database": 64}
CHUNK = 20_000
GOLDEN, M1, M2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def read_sketches(zip_path: Path, ksize: int = 31) -> dict[str, np.ndarray]:
    """Run accession -> sorted unique sketch hashes (uint64)."""
    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if not info.filename.endswith((".sig", ".sig.gz")):
                continue
            raw = z.read(info)
            if info.filename.endswith(".gz"):
                raw = gzip.decompress(raw)
            for rec in json.load(io.BytesIO(raw)):
                for s in rec.get("signatures", []):
                    if s.get("ksize") == ksize:
                        if rec["name"] in out:
                            raise ValueError(f"{rec['name']}: two k={ksize} sketches")
                        out[rec["name"]] = np.unique(np.array(s["mins"], dtype=np.uint64))
    return out


def project(hashes: np.ndarray, d: int = D, step: int = STEPS["builder"]) -> np.ndarray:
    """Integer hypervector of a hash set; `step` is the seed offset between 64-bit words."""
    words = d // 64
    offsets = np.arange(words, dtype=np.uint64) * np.uint64(step)
    ones = np.zeros(d, dtype=np.int64)
    with np.errstate(over="ignore"):
        for start in range(0, len(hashes), CHUNK):
            x = hashes[start:start + CHUNK, None] + offsets[None, :]
            x = x + GOLDEN
            x = (x ^ (x >> np.uint64(30))) * M1
            x = (x ^ (x >> np.uint64(27))) * M2
            x = x ^ (x >> np.uint64(31))
            bits = np.unpackbits(x.astype("<u8").view(np.uint8), axis=1, bitorder="little")
            ones += bits.sum(axis=0, dtype=np.int64)
    return len(hashes) - 2 * ones


def normalise(raw: np.ndarray) -> np.ndarray:
    v = raw.astype(np.float64) / np.sqrt(D)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


def write(path: Path, ids: list[str], vecs: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, frac=np.stack([vecs[r] for r in ids]),
                        feature_ids=np.array([f"hv_{i}" for i in range(D)]), sample_ids=np.array(ids))
    log.info("%s: %d runs", path.name, len(ids))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--projection", choices=list(STEPS), default="builder")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    out, step = a.out, STEPS[a.projection]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "hv_train.npz").exists():
        raise SystemExit(f"{out}/hv_train.npz exists; refusing to overwrite")
    log.info("projection %s (word offset %d)", a.projection, step)
    train = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).Run_accession.tolist()
    test = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).Run_accession.tolist()
    sk = read_sketches(SKETCHES)
    missing = sorted(set(train + test) - set(sk))
    if missing:
        raise SystemExit(f"{len(missing)} split runs without a sketch, e.g. {missing[:3]}")
    raw, norm = {}, {}
    for i, run in enumerate(train + test):
        raw[run] = project(sk[run], step=step)
        norm[run] = normalise(raw[run])
        if not sk[run].size:
            log.warning("%s: empty sketch, zero vector", run)
        if (i + 1) % 500 == 0:
            log.info("%d / %d runs", i + 1, len(train) + len(test))
    write(out / "hv_train.npz", train, norm)
    write(out / "hv_heldout.npz", test, norm)
    ids = train + test
    np.savez_compressed(out / "hv_raw.npz", raw=np.stack([raw[r] for r in ids]).astype(np.int32), sample_ids=np.array(ids))
    pd.DataFrame({"Run_accession": ids, "split": ["train"] * len(train) + ["held-out"] * len(test),
                  "sketch_hashes": [len(sk[r]) for r in ids]}).to_csv(out / "sketch_sizes.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
