"""Rewrite back_to_sequences k-mer positions as Logan coverage counts.

`01_count_kmers.sh` counts how many times each reference k-mer occurs in the
input sequences. For Logan unitig input every k-mer occurs exactly once, so the
count is 1 and the downstream `kmat_tools unitig` abundance collapses to the
completeness fraction. Logan does record coverage, in the `ka:f:` header field
("the average count of all k-mers over the entire unitig"), and
`back_to_sequences --output-kmer-positions` reports which input sequence each
reference k-mer was found in. Joining the two recovers a per-k-mer count.

Reads the positions file, writes the two-column `<kmer>\t<count>` file
`kmat_tools unitig` expects. A reference k-mer found in several sample unitigs
gets the sum of their coverages, matching MUSET's A(u,S) = (sum_i c_i)/N.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

KA_RE = re.compile(r"\bka:f:([0-9.eE+-]+)")
POS_RE = re.compile(r"\((\d+),\s*-?\d+,\s*(?:true|false)\)")


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_coverage(unitigs: Path) -> list[float]:
    """Coverage per input sequence, indexed by order of appearance."""
    cov: list[float] = []
    missing = 0
    with _open(unitigs) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m = KA_RE.search(line)
            if m is None:
                missing += 1
                cov.append(1.0)
            else:
                cov.append(float(m.group(1)))
    if missing:
        logger.warning("%d of %d sequences carry no ka:f: field; counted as 1", missing, len(cov))
    return cov


def rewrite(positions: Path, cov: list[float], out: Path,
            min_abundance: float = 2.0, rule: str = "drop") -> tuple[int, int, int]:
    """Write <kmer>\\t<count>, weighting each k-mer by its unitig's coverage.

    `min_abundance` must match the `-a` MUSET was run with (`c_ab_min` in the
    kmtricks options), because the training matrix applied that threshold to the
    same `ka:f:` values. `rule` selects how the threshold is applied:

      drop       -- coverage below the threshold is absent (kmtricks hard-min)
      round      -- sum the occurrences, round, then threshold
      round_each -- round each occurrence, then sum, then threshold
      floor      -- no threshold, every present k-mer counts at least 1
    """
    written = oob = dropped = 0
    with open(positions) as fh, open(out, "w") as o:
        for line in fh:
            kmer, _, rest = line.partition(" ")
            if not rest:
                raise ValueError(f"no positions on line: {line!r}")
            total = 0.0
            for m in POS_RE.finditer(rest):
                idx = int(m.group(1))
                if idx >= len(cov):
                    oob += 1
                    continue
                total += round(cov[idx]) if rule == "round_each" else cov[idx]
            if rule == "floor":
                count = max(1, round(total))
            elif rule in ("round", "round_each"):
                count = round(total)
                if count < min_abundance:
                    dropped += 1
                    continue
            else:
                if total < min_abundance:
                    dropped += 1
                    continue
                count = round(total)
            if count <= 0:
                dropped += 1
                continue
            o.write(f"{kmer}\t{count}\n")
            written += 1
    return written, oob, dropped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("positions", type=Path, help="back_to_sequences --output-kmer-positions output")
    p.add_argument("unitigs", type=Path, help="the sample's own Logan unitigs FASTA (.gz ok)")
    p.add_argument("out", type=Path, help="two-column kmer/count file for kmat_tools unitig")
    p.add_argument("--min-abundance", type=float, default=2.0,
                   help="must match MUSET's -a / kmtricks c_ab_min (default 2)")
    p.add_argument("--rule", choices=("drop", "round", "round_each", "floor"),
                   default="round",
                   help="how the threshold is applied; see rewrite()")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cov = read_coverage(args.unitigs)
    if not cov:
        logger.error("no sequences read from %s", args.unitigs)
        return 1
    logger.info("read coverage for %d sample unitigs", len(cov))
    written, oob, dropped = rewrite(args.positions, cov, args.out,
                                   args.min_abundance, args.rule)
    if oob:
        logger.error("%d positions referenced a sequence index beyond the FASTA", oob)
        return 1
    logger.info("rule=%s min_abundance=%s: wrote %d k-mers, dropped %d below threshold",
                args.rule, args.min_abundance, written, dropped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
