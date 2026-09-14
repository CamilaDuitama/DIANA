#!/usr/bin/env python3
"""Write a presence/absence copy of the training fraction matrix.

DIANA reads `unitigs.frac.mat`, MUSET's f(u,S) = mean over the unitig's k-mers of a
binary present/absent indicator, bounded in [0,1]. That is a *graded* signal: measured
over the matrix, 75.9 % of entries are exactly 0, 4.9 % are exactly 1, and 19.2 % lie
strictly between, so **80 % of the non-zero entries are partial** with a median of 0.59.

Binarising asks whether that gradation carries anything. It is not a rounding tweak:
it moves each present entry by 0.42 on average. Two things it could do, in opposite
directions, which is why it is worth measuring rather than arguing about:

* a partial fraction is partly a sequencing-depth artefact (the same unitig reads
  lower in a shallow sample), so removing it could cut study-driven variance;
* a partial fraction is also real biology (a genome present at partial coverage), so
  removing it could throw away most of the signal.

Threshold is `> 0`, i.e. exactly MUSET's own definition of a k-mer being present, so
no new cutoff is introduced and nothing is fitted from the data. Column 1 is the
unitig id and is copied through unchanged.

    ./env/bin/python scripts/data_prep/12_binarise_fraction_matrix.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=PROJECT_ROOT / "data/matrices/matrix_v9_train/unitigs.frac.mat")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "data/matrices/matrix_v9_train/unitigs.pa.mat")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.output.exists():
        logger.error("%s exists; refusing to overwrite", args.output)
        return 1

    rows = nonzero = total = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            parts = line.split()
            if not parts:
                continue
            out = [parts[0]]
            for v in parts[1:]:
                present = float(v) > 0.0
                nonzero += present
                out.append("1" if present else "0")
            total += len(parts) - 1
            fout.write(" ".join(out) + "\n")
            rows += 1
            if rows % 20000 == 0:
                logger.info("  %d rows", rows)

    logger.info("wrote %s: %d unitigs, %d entries, %.2f %% present",
                args.output, rows, total, 100.0 * nonzero / total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
