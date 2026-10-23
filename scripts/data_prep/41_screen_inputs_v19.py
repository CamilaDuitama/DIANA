#!/usr/bin/env python3
"""Inputs of the v19 dev-fold screens (PROTOCOLS.md, v19 amendment of 2026-10-21), from the v17
training matrix, in the MatrixLoader npz layout (frac, optional block, feature_ids, sample_ids).

Anything fitted to the data (the PCA basis and its centring, the standardisation of appended
columns) uses the four training folds of each dev fold only, so those inputs come one file per
fold; the held-back fold is only projected. Written to results/screens_v19/inputs/:

  frac.npz                        the fractions (input of 18_/19_ and of the log-depth arm)
  pa.npz                          presence/absence, fraction > 0 (as 12_binarise_fraction_matrix.py)
  abund_log1p.npz                 log1p of the mean k-mer abundance (as 11_log1p_abundance_matrix.py)
  frac_logdepth.npz               fractions + log10(1 + non-zero unitigs) / log10(1 + n unitigs) (as 35_)
  {frac,pa}_pca192_fold{k}.npz    centred PCA, 192 components (as 13_ with --center)
  {frac,pa}_pca191nd_fold{k}.npz  the same without the first component (the depth axis)
  frac_kmer4_fold{k}.npz          fractions + 136 canonical 4-mer composition columns (as 14_)
  frac_clust_fold{k}.npz          fractions + cluster-summed fractions of multi-member clusters (as 16_;
                                  labels from 15_unitig_clusters.py --fasta <v17 unitigs.fa>)

    ./env/bin/python scripts/data_prep/41_screen_inputs_v19.py --clusters <cluster_labels.npy>
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data/matrices/matrix_v17_train"
SPLIT = ROOT / "data/splits_v19"
OUT = ROOT / "results/screens_v19/inputs"
RANK = 192

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(samples x features float32, feature ids) of a text .mat (features as rows)."""
    from diana.data.loader import MatrixLoader
    X, ids, _ = MatrixLoader(str(path)).load()
    fid = np.array([line.split(" ", 1)[0] for line in open(path)])
    if len(fid) != X.shape[1]:
        raise ValueError(f"{path}: {len(fid)} row ids for {X.shape[1]} features")
    return X.astype(np.float32), np.asarray(ids, dtype=str), fid


def save(name: str, frac: np.ndarray, fid: np.ndarray, ids: np.ndarray, block: np.ndarray | None = None,
         block_ids: list[str] | None = None) -> None:
    out = OUT / name
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    arrays = {"frac": frac.astype(np.float32), "sample_ids": ids,
              "feature_ids": fid if block is None else np.concatenate([fid, np.array(block_ids)])}
    if block is not None:
        arrays["block"] = block.astype(np.float32)
    np.savez_compressed(out, **arrays)
    log.info("wrote %s: %s%s", name, frac.shape, "" if block is None else f" + block {block.shape}")


def pca(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Centred PCA with RANK components, means and basis from the training rows only."""
    mu = X[train].mean(axis=0, keepdims=True)
    sv = TruncatedSVD(n_components=RANK, random_state=0).fit(X[train] - mu)
    log.info("  PCA on %d training rows: explained variance %.3f", int(train.sum()), sv.explained_variance_ratio_.sum())
    return sv.transform(X - mu).astype(np.float32)


def scaled_block(B: np.ndarray, F: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Appended columns standardised on the training rows, then put on the fractions' scale (as 14_/16_)."""
    mu, sd = B[train].mean(axis=0), B[train].std(axis=0)
    sd[sd == 0] = 1.0
    return (B - mu) / sd * float(F[train].std()) + float(F[train].mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clusters", type=Path, required=True, help="cluster labels of the v17 unitigs (15_)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    F, ids, fid = load_mat(MAT / "unitigs.frac.mat")
    A, ids_a, fid_a = load_mat(MAT / "unitigs.abundance.mat")
    if not (np.array_equal(ids, ids_a) and np.array_equal(fid, fid_a)):
        raise ValueError("fraction and abundance matrices differ in runs or unitigs")
    train_ids = set(open(SPLIT / "train_accessions.txt").read().split())
    if set(ids) != train_ids:
        raise ValueError("matrix runs differ from the v19 training runs")
    fold = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(ids).to_numpy()
    if np.isnan(fold.astype(float)).any():
        raise ValueError("runs without a dev fold")
    comp = module("comp14", "scripts/data_prep/14_kmer_composition_features.py")
    seqs = comp.read_unitigs(MAT / "unitigs.fa")
    heads = np.array([line[1:].split()[0] for line in open(MAT / "unitigs.fa") if line.startswith(">")])
    if not np.array_equal(heads, fid):
        raise ValueError("unitigs.fa order differs from the matrix rows")
    lab = np.load(a.clusters)
    if lab.size != F.shape[1]:
        raise ValueError(f"{lab.size} cluster labels for {F.shape[1]} unitigs")

    P = (F > 0).astype(np.float32)
    save("frac.npz", F, fid, ids)
    save("pa.npz", P, fid, ids)
    save("abund_log1p.npz", np.log1p(A), fid, ids)
    nz = P.sum(axis=1)
    depth = (np.log10(1.0 + nz) / np.log10(1.0 + F.shape[1])).astype(np.float32)
    save("frac_logdepth.npz", F, fid, ids, depth[:, None], ["log_nonzero_scaled"])
    del A

    C, kmers = comp.composition(seqs, 4)
    w = F.sum(axis=1, keepdims=True); w[w == 0] = 1.0
    K = (F @ C) / w                                    # fraction-weighted composition per run
    sizes = np.bincount(lab)
    multi = np.flatnonzero(sizes > 1)
    pos = {c: j for j, c in enumerate(multi)}
    S = np.zeros((F.shape[0], multi.size), dtype=np.float32)
    for u in range(lab.size):
        j = pos.get(lab[u])
        if j is not None:
            S[:, j] += F[:, u]
    log.info("%d canonical 4-mers; %d multi-member clusters covering %d unitigs", len(kmers), multi.size, int(sizes[multi].sum()))

    for k in range(5):
        train = fold != k
        log.info("fold %d: %d training-fold runs, %d held-back runs", k, int(train.sum()), int((~train).sum()))
        for tag, X in (("frac", F), ("pa", P)):
            Z = pca(X, train)
            comps = np.array([f"pc{j}" for j in range(RANK)])
            save(f"{tag}_pca192_fold{k}.npz", Z, comps, ids)
            save(f"{tag}_pca191nd_fold{k}.npz", Z[:, 1:], comps[1:], ids)
        save(f"frac_kmer4_fold{k}.npz", F, fid, ids, scaled_block(K, F, train), [f"kmer_{c}" for c in kmers])
        save(f"frac_clust_fold{k}.npz", F, fid, ids, scaled_block(S, F, train), [f"clust_{c}" for c in multi])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
