#!/usr/bin/env python3
"""Write a log1p-transformed copy of the training abundance matrix.

DIANA is trained on `unitigs.frac.mat`, the k-mer completeness fraction: MUSET's
f(u,S) = mean over the unitig's k-mers of a BINARY present/absent indicator, bounded
in [0,1]. `unitigs.abundance.mat` is a different quantity, A(u,S) = mean over the same
k-mers of their COUNT, unbounded. R3.7 makes exactly this distinction, and abundance
has never been fed to a model.

Raw abundance cannot be swapped in directly: it reaches 8,391 with 77.7 % zeros and a
p99 of 53.7, while the pipeline has no standardisation step at all because bounded
features never needed one. log1p brings it to mean 0.276 and sd 0.681, comparable to
fraction's mean of about 0.16, so the existing trainer can consume it unchanged and no
statistic is fitted across folds. Transforming the data rather than the code keeps the
comparison clean and adds no leakage surface.

    ./env/bin/python scripts/data_prep/11_log1p_abundance_matrix.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=PROJECT_ROOT / "data/matrices/matrix_v9_train/unitigs.abundance.mat")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "data/matrices/matrix_v9_train/unitigs.abundance.log1p.mat")
    args = ap.parse_args()

    n = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            p = line.split()
            fout.write(p[0] + " " + " ".join(f"{math.log1p(float(v)):.4f}" for v in p[1:]) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"  {n} rows", flush=True)
    print(f"wrote {args.output} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
