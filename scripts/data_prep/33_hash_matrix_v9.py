#!/usr/bin/env python3
"""H1: a k-mer hash matrix from the sourmash sketches, columns chosen from the training runs.

Each run's FracMinHash sketch (k = 31, scaled 1,000 from data/sketches_v9/unitigs_split_v9.zip
for all 3,638 runs, or scaled 100 from unitigs_train_v9_s100.zip for the 2,716 training runs)
is a fixed subsample of its k-mers, the same rule for every run. A column is one hash. The
columns kept are the hashes present in at least --min-samples of the 2,716 training runs
and in at most 90 % of them (MUSET's -f 0.1 ceiling), computed on all training runs as the
unitig vocabulary was. Held-out runs never take part in the choice. Values are 0/1 presence.

Floors (PROJECT.md, "Read first" item 9): 272 = MUSET's default -F 0.1 on 2,716 runs, the
v9 rule, under which only 5 of 38 eligible classes can keep a private k-mer; 28 = 1 %;
14 = the v11 rule (-N 14), the highest floor that lets the 15- and 16-run classes keep one.

Outputs, results/hash_matrix_v9/:
  hash_s<scaled>_f<floor>_train.npz     frac (2,716 x K uint8 0/1), feature_ids (hash as text), sample_ids
  hash_s<scaled>_f<floor>_heldout.npz   the 922 held-out runs on the same columns (scaled 1,000 only)
  hash_s<scaled>_f<floor>_summary.json  counts: distinct hashes, kept, density
and the DIANA configs configs/hash_v9/<arm>_s<scaled>_f<floor>.json (configs/final_fixed_v9
with the features path swapped; hyperparameters unchanged).

    ./env/bin/python scripts/data_prep/33_hash_matrix_v9.py --scaled 1000 --min-samples 14
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import math
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SK, SPLITS = ROOT / "data/sketches_v9", ROOT / "data/splits_v9"
OUT, CFG_SRC, CFG_DST = ROOT / "results/hash_matrix_v9", ROOT / "configs/final_fixed_v9", ROOT / "configs/hash_v9"
ZIPS = {1000: SK / "unitigs_split_v9.zip", 100: SK / "unitigs_train_v9_s100.zip"}
CAP = 0.90
ARMS = ["multitask", "community_type", "feature", "sample_host", "material"]


def read_sketches(zip_path: Path, ksize: int) -> dict[str, np.ndarray]:
    """run accession -> sorted uint64 hashes (same reader as 46_class_signature_ceiling.py)."""
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
                for s in rec.get("signatures", []):
                    if s.get("ksize") == ksize:
                        out[rec["name"]] = np.sort(np.array(s["mins"], dtype=np.uint64))
    return out


def presence(keep: np.ndarray, sk: dict[str, np.ndarray], runs: list[str]) -> np.ndarray:
    X = np.zeros((len(runs), len(keep)), dtype=np.uint8)
    for i, r in enumerate(runs):
        h = sk[r]
        pos = np.searchsorted(h, keep)
        pos[pos == len(h)] = 0
        X[i] = (h[pos] == keep)
    return X


def write_configs(tag: str, k_cols: int, floor: int) -> None:
    CFG_DST.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        cfg = json.load(open(CFG_SRC / f"{arm}.json"))
        cfg["features_path"] = f"results/hash_matrix_v9/hash_{tag}_train.npz"
        cfg["output_dir"] = f"results/epoch_budget_hash_{tag}_v9/{arm}"
        cfg["_provenance"] = {**cfg.get("_provenance", {}),
                              "h_protocol": f"H ({date.today().isoformat()}): k-mer hash presence matrix {tag}, {k_cols} columns "
                                            f"present in >= {floor} and <= 90 % of the 2,716 training runs; hyperparameters "
                                            "unchanged from configs/final_fixed_v9 (the one stated inequality: input width)."}
        json.dump(cfg, open(CFG_DST / f"{arm}_{tag}.json", "w"), indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scaled", type=int, choices=sorted(ZIPS), required=True)
    ap.add_argument("--min-samples", type=int, required=True, help="floor: present in at least this many training runs")
    ap.add_argument("--ksize", type=int, default=31)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"s{a.scaled}_f{a.min_samples}"
    out_train = OUT / f"hash_{tag}_train.npz"
    if out_train.exists():
        raise SystemExit(f"{out_train} exists; refusing to overwrite")
    train = [l.strip() for l in open(SPLITS / "train_accessions.txt") if l.strip()]
    test = [l.strip() for l in open(SPLITS / "test_accessions.txt") if l.strip()]
    t0 = time.time()
    sk = read_sketches(ZIPS[a.scaled], a.ksize)
    missing = [r for r in train if r not in sk]
    if missing:
        raise SystemExit(f"{len(missing)} training runs without a sketch, e.g. {missing[:3]}")
    logger.info("%s: %d sketches read in %.0f s", tag, len(sk), time.time() - t0)

    n = len(train)
    allh = np.concatenate([sk[r] for r in train])
    uniq, counts = np.unique(allh, return_counts=True)
    del allh
    lo, hi = a.min_samples, math.floor(CAP * n)
    keep = uniq[(counts >= lo) & (counts <= hi)]
    summary = {"tag": tag, "scaled": a.scaled, "ksize": a.ksize, "n_training_runs": n,
               "n_distinct_hashes_train": int(len(uniq)), "floor_present_in_runs": lo, "cap_present_in_runs": hi,
               "n_kept": int(len(keep)), "n_above_floor": int((counts >= lo).sum()), "n_above_cap": int((counts > hi).sum()),
               "kept_by_floor": {str(f): int(((counts >= f) & (counts <= hi)).sum()) for f in (14, 20, 28, 272)},
               "median_hashes_per_train_run": float(np.median([len(sk[r]) for r in train])),
               "rule": f"present in >= {lo} and <= {hi} (90 %) of the 2,716 training runs", "built_on": date.today().isoformat()}
    logger.info("kept %d of %d distinct hashes (floor %d, cap %d runs)", len(keep), len(uniq), lo, hi)

    X = presence(keep, sk, train)
    summary["train_density"] = float(X.mean())
    fid = np.array([str(int(h)) for h in keep])
    np.savez(out_train, frac=X, feature_ids=fid, sample_ids=np.array(train))
    logger.info("wrote %s: %s, density %.4f", out_train, X.shape, X.mean())
    have_test = [r for r in test if r in sk]
    if len(have_test) == len(test):
        Xt = presence(keep, sk, test)
        np.savez(OUT / f"hash_{tag}_heldout.npz", frac=Xt, feature_ids=fid, sample_ids=np.array(test))
        summary["heldout_density"] = float(Xt.mean())
        logger.info("wrote held-out matrix %s", Xt.shape)
    else:
        summary["heldout"] = f"not written: {len(test) - len(have_test)} held-out runs have no sketch at this scale"
    json.dump(summary, open(OUT / f"hash_{tag}_summary.json", "w"), indent=2)
    write_configs(tag, int(len(keep)), lo)
    logger.info("done in %.0f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
