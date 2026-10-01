#!/usr/bin/env python3
"""S8.8: one feature per class, the share of the class's signature k-mers a sample carries.

Rule (PROJECT.md section 4, S8.8, fixed 2026-09-29 before running): for dev fold k, a
class's signature is the set of off-list k-mers (scaled = 100 sketches, k = 31,
vocabulary removed) present in >= 1/3 of the class's TRAINING-FOLD runs and in < 5 % of
the other classes' training-fold runs of the same task. No floor on run count, no
cross-study requirement inside the fold (a two-study class loses one study to the test
fold; recognising that study from the other is exactly what the fold tests). Classes
with fewer than 2 training-fold runs get an empty signature (column stays 0).

Every run, training and test fold alike, then gets one column per eligible class: the
fraction of that class's signature present in the run's sketch. The test fold's runs are
never read while a signature is built; that is asserted, not assumed.

Output, per fold: results/s8_signatures_v9/sig_fold{k}.npz with `frac` (the 110,202
fractions, from unitigs.frac.s8a.npz), `block` (2,716 x n_classes float32),
`feature_ids`, `sample_ids`, readable by MatrixLoader's .npz path; plus
signature_sizes_fold{k}.tsv.

    ./env/bin/python scripts/data_prep/31_signature_features_v9.py --fold 0
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from importlib import import_module  # noqa: E402

read_sketches = import_module("46_class_signature_ceiling").read_sketches

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
SK = ROOT / "data/sketches_v9"
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/s8_signatures_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
FRAC_IN_CLASS, FRAC_OUTSIDE, MIN_RUNS, KSIZE = 1 / 3, 0.05, 2, 31


def counts_over(runs: list[str], off: dict) -> tuple[np.ndarray, np.ndarray]:
    """Distinct off-list hashes over `runs` and in how many of them each occurs."""
    if not runs:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)
    return np.unique(np.concatenate([off[r] for r in runs]), return_counts=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"sig_fold{a.fold}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")

    with np.load(ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz", allow_pickle=False) as z:
        ids = list(z["sample_ids"].astype(str))
        F = z["frac"].astype(np.float32)
        unitig_ids = [str(x) for x in z["feature_ids"][:F.shape[1]]]
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession").loc[ids, "fold"].to_numpy()
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    elig = elig[elig.evaluable]

    V = np.sort(next(iter(read_sketches(SK / "vocab_v9_s100.sig.zip", KSIZE).values())))
    sk = read_sketches(SK / "unitigs_train_v9_s100.zip", KSIZE)
    off = {r: sk[r][~np.isin(sk[r], V, assume_unique=True)] for r in ids}
    del sk
    logger.info("fold %d: %d runs, sketches read in %.0f s", a.fold, len(ids), time.time() - t0)

    train_rows = np.where(folds != a.fold)[0]
    test_rows = set(np.where(folds == a.fold)[0])
    columns, sizes, block = [], [], []
    for task in TASKS:
        lab = meta[task].astype(object).to_numpy()
        labelled_train = [i for i in train_rows if pd.notna(lab[i])]
        u_all, c_all = counts_over([ids[i] for i in labelled_train], off)
        logger.info("%s: %d labelled training-fold runs, %d distinct off-list hashes (%.0f s)",
                    task, len(labelled_train), len(u_all), time.time() - t0)
        sig_hash, sig_cls = [], []
        classes = sorted(set(elig[elig.target == task]["class"].astype(str)))
        for ci, cls in enumerate(classes):
            rows_c = [i for i in labelled_train if str(lab[i]) == cls]
            assert not (set(rows_c) & test_rows), "test-fold run used in a signature"
            n_out = len(labelled_train) - len(rows_c)
            if len(rows_c) < MIN_RUNS:
                sizes.append({"fold": a.fold, "task": task, "class": cls, "train_fold_runs": len(rows_c),
                              "train_fold_projects": len(set(meta.iloc[rows_c]["archive_project"])) if rows_c else 0,
                              "consistent": 0, "signature": 0})
                columns.append(f"sig__{task}__{cls}")
                continue
            u_c, c_c = counts_over([ids[i] for i in rows_c], off)
            cand = u_c[c_c >= max(1, FRAC_IN_CLASS * len(rows_c))]
            pos = np.searchsorted(u_all, cand)
            total = c_all[pos]                          # every candidate is in u_all by construction
            in_cls = c_c[np.searchsorted(u_c, cand)]
            outside = total - in_cls
            keep = cand[outside < FRAC_OUTSIDE * max(1, n_out)]
            sig_hash.append(keep); sig_cls.append(np.full(len(keep), ci, dtype=np.int32))
            sizes.append({"fold": a.fold, "task": task, "class": cls, "train_fold_runs": len(rows_c),
                          "train_fold_projects": len(set(meta.iloc[rows_c]["archive_project"])),
                          "consistent": int(len(cand)), "signature": int(len(keep))})
            columns.append(f"sig__{task}__{cls}")
        # features for EVERY run: share of each class's signature present
        H = np.concatenate(sig_hash) if sig_hash else np.array([], dtype=np.uint64)
        C = np.concatenate(sig_cls) if sig_cls else np.array([], dtype=np.int32)
        order = np.argsort(H, kind="stable"); H, C = H[order], C[order]
        sizes_by_cls = np.bincount(C, minlength=len(classes)) if len(C) else np.zeros(len(classes), int)
        feat = np.zeros((len(ids), len(classes)), dtype=np.float32)
        if len(H):
            for i, r in enumerate(ids):
                m = off[r]
                lo = np.searchsorted(H, m, side="left"); hi = np.searchsorted(H, m, side="right")
                hit = hi > lo
                if not hit.any():
                    continue
                # expand duplicates (a hash in more than one class's signature)
                idx = np.concatenate([np.arange(l, h) for l, h in zip(lo[hit], hi[hit])])
                counts = np.bincount(C[idx], minlength=len(classes))
                with np.errstate(divide="ignore", invalid="ignore"):
                    feat[i] = np.where(sizes_by_cls > 0, counts / np.maximum(sizes_by_cls, 1), 0.0)
        block.append(feat)
        logger.info("%s: signatures %s (%.0f s)", task, dict(zip(classes, sizes_by_cls.tolist())), time.time() - t0)

    B = np.hstack(block).astype(np.float32)
    np.savez(out, frac=F, block=B, feature_ids=np.array(unitig_ids + columns, dtype=str), sample_ids=np.array(ids, dtype=str))
    pd.DataFrame(sizes).to_csv(OUT / f"signature_sizes_fold{a.fold}.tsv", sep="\t", index=False)
    logger.info("wrote %s: block %s, non-zero entries %.1f %%; %.0f s", out.name, B.shape, 100 * (B > 0).mean(), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
