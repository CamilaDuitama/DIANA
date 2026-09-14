#!/usr/bin/env python3
"""Scatter the first two SVD components of a v9 matrix, coloured by task label.

One file per task, per the figure conventions. Uses the fold-0 basis, which was fitted
on that fold's four training folds only.

Two components out of 192 cannot establish whether the classes are linearly separable,
so `--report-separability` additionally fits a linear model in the full 192-dimensional
space and prints its TRAINING accuracy, which is the direct test: a linear model that
reproduces its own training labels exactly has found a separating hyperplane.

    ./env/bin/python scripts/paper/plot_svd_labels.py --tag frac
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]


def load_svd(path: Path) -> np.ndarray:
    rows = [np.asarray(l.split()[1:], dtype=np.float64) for l in open(path) if l.strip()]
    return np.vstack(rows).T          # samples x components


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="frac", choices=("frac", "pa"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--rank", type=int, default=192)
    ap.add_argument("--outdir", type=Path, default=ROOT / "results/paper")
    ap.add_argument("--report-separability", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)

    mat = (ROOT / "data/matrices/matrix_v9_train" /
           f"unitigs.{args.tag}.svd{args.rank}.fold{args.fold}.mat")
    Z = load_svd(mat)
    accs = [l.split(" : ")[0].strip() for l in
            open(ROOT / "data/train_samples_v9.fof") if l.strip()]
    meta = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False)
    meta = meta.set_index("Run_accession").reindex(accs)
    logger.info("%s: %s", mat.name, Z.shape)

    name = {"frac": "fraction", "pa": "presence/absence"}[args.tag]
    for task in TASKS:
        lab = meta[task].astype("string")
        keep = lab.notna().to_numpy()
        y = lab[keep].to_numpy()
        P = Z[keep]
        classes = pd.Series(y).value_counts()

        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(classes.index):
            m = y == c
            ax.scatter(P[m, 0], P[m, 1], s=14, alpha=0.65, color=cmap(i % 20),
                       label=f"{c} (n={int(classes[c])})", linewidths=0)
        ax.set_xlabel("SVD component 1")
        ax.set_ylabel("SVD component 2")
        ax.set_title(f"SVD components 1-2 of the v9 {name} matrix, by {task}", fontsize=11)
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, ncol=1 if len(classes) <= 14 else 2)
        fig.tight_layout()
        out = args.outdir / f"svd_{args.tag}_{task}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)
        logger.info("  wrote %s (%d classes, %d runs)", out.name, len(classes), int(keep.sum()))

        if args.report_separability:
            from sklearn.svm import LinearSVC
            from sklearn.preprocessing import StandardScaler
            Xs = StandardScaler().fit_transform(P)
            clf = LinearSVC(C=1.0, max_iter=20000, dual="auto").fit(Xs, y)
            tr = float((clf.predict(Xs) == y).mean())
            print(f"  {task:16s} classes={len(classes):3d} runs={int(keep.sum()):5d} "
                  f"linear TRAIN accuracy in {args.rank}-d = {tr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
